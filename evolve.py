"""Differentiation loop. Same code, three task classes, three specialists.

The meta-agent reads three demo tasks, the current spec, and the most recent
dev failures. It returns a JSON edit over four spec fields. Each child is
checked by parse, smoke, and a Pareto-dominance rule before being added to
the frontier.

Pareto frontier:
    Each accepted spec carries (dev_score, mean_in_tokens_per_eval).
    Spec A dominates B iff A.score >= B.score and A.tokens <= B.tokens with
    at least one strict. The frontier is the set of non-dominated specs
    seen so far. The loop stops when N consecutive iterations fail to add
    a new frontier point. The reported specialist is the highest-scoring
    frontier entry, ties broken by lower tokens.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent import SeedSpec, run
from eval import EvalResult, evaluate, load, split
from llm import chat


META_MODEL = "gpt-4o"
META_TEMPERATURE = 0.7
PLATEAU_PATIENCE = 3
DEFAULT_ITERS = 8


META_SYSTEM = """You evolve a small worker agent's spec one structured edit at a time.

The worker spec has exactly four fields, no others:
  system_prompt: free text, the worker's system message.
  validation:    one of "none", "schema", "testcases", "results".
                 schema only matches the json class (parses output, checks fields).
                 testcases only matches the regex class (compiles the regex, runs against given examples).
                 results only matches the sql class (runs the SQL against a fixture and compares its result set to the gold result set).
  tool_policy:   one of "no_tools", "validate_retry", "code_exec".
                 no_tools: one LLM call.
                 validate_retry: run the validator after the LLM call; on failure feed the
                                 reason back and retry up to max_retries times.
                 code_exec: the worker is given a python_exec tool and may use it during the loop.
  max_retries:   integer in [0, 4].

You will see the task class, three demo tasks, the current spec, and a list of dev tasks
that the worker just failed. Each failure carries a tag in square brackets:
  [parse-fail]         output didn't parse as the expected format
  [missing-field]      JSON output missing a required key
  [value-not-allowed]  JSON value outside the allowed set for that field
  [value-mismatch]     output is well-formed but the value doesn't match gold (e.g. wrong category)
Validation+retry only helps the first three. [value-mismatch] is a judgement error and only
prompt-level rules or worked examples can move it.

Propose ONE structured edit. Output ONLY a JSON object:

  {"reason": "<one sentence why this should help>", "edit": {<field>: <new value>, ...}}

Only include fields you change. Do not invent fields. Do not exceed max_retries=4.

Heuristics that matter:
  - If outputs frequently fail to parse or fail their tests, validate_retry with the matching
    validator is the obvious move. validate_retry helps only if the validator can recognise the
    error; it cannot recognise "wrong category" or "wrong number".
  - If the bottleneck is the worker's own arithmetic or multi-step reasoning, code_exec is the
    move. The worker can call python_exec to compute values precisely.
  - If outputs are well-formed but make wrong judgements (severity, kind), the system_prompt
    needs to carry rules and worked examples. Validation cannot help that case.
  - Larger max_retries costs tokens with diminishing returns past 2 unless the worker is close
    to the answer and just needs a nudge.
  - When you turn on code_exec, the worker still picks whether to call the tool. Edit
    system_prompt at the same time so it actually uses python_exec; otherwise you'll get
    a spec that *can* compute but doesn't.
  - Under code_exec, max_retries caps the number of tool-call rounds (turn budget = 2 + max_retries,
    capped at 8). Higher values let the worker iterate when its first python attempt has a bug.
"""


# ----- meta-agent -------------------------------------------------------

@dataclass
class Proposal:
    reason: str
    edit: dict
    raw: str


def _propose(task_class: str, demos: list[dict], current_spec: dict,
             last_failures: list[dict] | None,
             show_demos: bool = True,
             show_feedback: bool = True) -> Proposal:
    user_lines = [f"Task class: {task_class}"]
    if show_demos:
        user_lines += ["", "Three demo tasks:"]
        for d in demos[:3]:
            user_lines.append(f"  id={d['id']}")
            prompt = d['prompt'].replace("\n", " ")
            user_lines.append(f"  prompt: {prompt[:240]}...")
            if "gold" in d:
                user_lines.append(f"  gold: {json.dumps(d['gold'])[:120]}")
            if "positives" in d:
                user_lines.append(f"  positives: {d['positives']}")
                user_lines.append(f"  negatives: {d['negatives']}")
    user_lines += ["", f"Current spec: {json.dumps(current_spec, indent=2)}"]
    if show_feedback and last_failures:
        user_lines += ["", "Last dev-eval failures:"]
        for f in last_failures[:8]:
            user_lines.append(f"  {f['task_id']}: output={f['output'][:120]!r} feedback={f['feedback']!r}")
    user_lines += ["", "Output the JSON edit now."]

    r = chat(
        [{"role": "system", "content": META_SYSTEM},
         {"role": "user", "content": "\n".join(user_lines)}],
        model=META_MODEL, temperature=META_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    obj = json.loads(r.content)
    return Proposal(reason=obj.get("reason", ""), edit=obj.get("edit", {}), raw=r.content)


# random-edit baseline: replace _propose() with a random valid edit.
# Tests how much of the gain comes from gpt-4o vs from the structured
# search itself plus the acceptance rule.
def _propose_random(current_spec: dict) -> Proposal:
    """Pick one of the four spec fields and assign a random valid value."""
    field = random.choice(list(ALLOWED_FIELDS))
    if field == "validation":
        new = random.choice(list(ALLOWED_VALIDATIONS))
    elif field == "tool_policy":
        new = random.choice(list(ALLOWED_TOOL_POLICIES))
    elif field == "max_retries":
        new = random.randint(0, 4)
    elif field == "system_prompt":
        # Pick from a small pool of generic alternates so this baseline
        # gets to touch the prompt axis at all.
        pool = [
            "Read the task and return only the requested output.",
            "Read the task carefully. Produce the answer in the format requested.",
            "Solve the task and emit the answer alone, no explanation.",
            "Think briefly, then produce the requested output and nothing else.",
            "You are a careful assistant. Return the requested output.",
        ]
        new = random.choice(pool)
    else:
        new = current_spec[field]
    return Proposal(reason=f"random-edit baseline picked {field}={new!r}", edit={field: new}, raw="")


# ----- safeguards -------------------------------------------------------

ALLOWED_FIELDS = {"system_prompt", "validation", "tool_policy", "max_retries"}
ALLOWED_VALIDATIONS = {"none", "schema", "testcases", "results"}
ALLOWED_TOOL_POLICIES = {"no_tools", "validate_retry", "code_exec"}


def merge(parent: dict, edit: dict) -> dict:
    child = copy.deepcopy(parent)
    for k, v in edit.items():
        child[k] = v
    return child


def parse_check(spec: dict) -> tuple[bool, str]:
    extras = set(spec) - ALLOWED_FIELDS
    if extras:
        return False, f"unknown fields: {sorted(extras)}"
    if spec["validation"] not in ALLOWED_VALIDATIONS:
        return False, f"validation={spec['validation']!r}"
    if spec["tool_policy"] not in ALLOWED_TOOL_POLICIES:
        return False, f"tool_policy={spec['tool_policy']!r}"
    if not isinstance(spec["max_retries"], int) or not 0 <= spec["max_retries"] <= 4:
        return False, f"max_retries={spec['max_retries']!r}"
    if not isinstance(spec["system_prompt"], str) or len(spec["system_prompt"]) < 5:
        return False, "system_prompt too short"
    return True, ""


def smoke_check(spec: dict, smoke_task: dict, task_class: str) -> tuple[bool, str]:
    try:
        out = run(smoke_task, spec, task_class)
    except Exception as e:
        return False, f"crashed: {type(e).__name__}: {e}"
    return (bool(out.output), "" if out.output else "empty output")


# ----- Pareto frontier --------------------------------------------------

@dataclass
class Point:
    spec: dict
    spec_hash: str
    iter: int
    score: float
    tokens: float  # mean in_tokens per eval

    def dominates(self, other: "Point") -> bool:
        ge_score = self.score >= other.score
        le_tok = self.tokens <= other.tokens
        strict = self.score > other.score or self.tokens < other.tokens
        return ge_score and le_tok and strict


def update_frontier(frontier: list[Point], cand: Point) -> tuple[bool, list[Point]]:
    """Add cand if not dominated; mutates frontier in place. Returns (added, removed)."""
    if any(p.dominates(cand) for p in frontier):
        return False, []
    removed = [p for p in frontier if cand.dominates(p)]
    frontier[:] = [p for p in frontier if not cand.dominates(p)]
    frontier.append(cand)
    return True, removed


# ----- run --------------------------------------------------------------

@dataclass
class Node:
    iter: int
    spec_hash: str
    parent_hash: str | None
    edit: dict
    reason: str
    accepted: bool
    rejection: str  # 'parse' | 'smoke' | 'regression' | 'dominated' | '' (accepted)
    on_frontier: bool
    dev_score: float
    n_pass: int
    n_total: int
    in_tokens: int
    out_tokens: int
    spec: dict


def spec_hash(spec: dict) -> str:
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


def evolve(
    task_class: str,
    iters: int = DEFAULT_ITERS,
    out_dir: Path = Path("runs"),
    seed: int = 0,
    skip_rollback: bool = False,
    propose_mode: str = "full",  # 'full' | 'no_demos' | 'no_feedback' | 'random'
) -> dict:
    random.seed(seed)
    tasks = split(load(task_class))
    demos, dev, test = tasks["demo"], tasks["dev"], tasks["test"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / task_class
    run_dir.mkdir(exist_ok=True)
    save_name = f"{propose_mode}_seed{seed}.json" if propose_mode != "full" else f"seed{seed}.json"

    lineage: list[Node] = []
    frontier: list[Point] = []

    parent_spec = copy.deepcopy(SeedSpec)
    parent_eval = evaluate(parent_spec, dev, task_class, label=f"{task_class}/seed dev")
    seed_point = Point(parent_spec, spec_hash(parent_spec), 0,
                       parent_eval.score, parent_eval.in_tokens / max(1, len(dev)))
    frontier.append(seed_point)

    lineage.append(Node(
        iter=0, spec_hash=seed_point.spec_hash, parent_hash=None, edit={}, reason="seed",
        accepted=True, rejection="", on_frontier=True,
        dev_score=parent_eval.score, n_pass=parent_eval.n_pass, n_total=parent_eval.n_total,
        in_tokens=parent_eval.in_tokens, out_tokens=parent_eval.out_tokens, spec=parent_spec,
    ))

    iters_since_frontier_grew = 0
    stop_cause = "iter_cap"
    stopped_at = iters

    for i in range(1, iters + 1):
        last_fails = [
            {"task_id": t.task_id, "output": t.output[:160], "feedback": t.feedback}
            for t in parent_eval.per_task if not t.passed
        ]
        try:
            if propose_mode == "random":
                prop = _propose_random(parent_spec)
            else:
                prop = _propose(task_class, demos, parent_spec, last_fails,
                                show_demos=(propose_mode != "no_demos"),
                                show_feedback=(propose_mode != "no_feedback"))
        except Exception as e:
            print(f"  iter {i}: meta-agent error: {e}")
            iters_since_frontier_grew += 1
            continue
        child = merge(parent_spec, prop.edit)
        h = spec_hash(child)

        ok, why = parse_check(child)
        if not ok:
            lineage.append(Node(
                iter=i, spec_hash=h, parent_hash=spec_hash(parent_spec),
                edit=prop.edit, reason=prop.reason, accepted=False,
                rejection="parse", on_frontier=False,
                dev_score=0.0, n_pass=0, n_total=len(dev),
                in_tokens=0, out_tokens=0, spec=child,
            ))
            print(f"  iter {i}: APOPTOSIS parse: {why}")
            iters_since_frontier_grew += 1
            if iters_since_frontier_grew >= PLATEAU_PATIENCE:
                stop_cause, stopped_at = "plateau", i; break
            continue

        ok, why = smoke_check(child, demos[0], task_class)
        if not ok:
            lineage.append(Node(
                iter=i, spec_hash=h, parent_hash=spec_hash(parent_spec),
                edit=prop.edit, reason=prop.reason, accepted=False,
                rejection="smoke", on_frontier=False,
                dev_score=0.0, n_pass=0, n_total=len(dev),
                in_tokens=0, out_tokens=0, spec=child,
            ))
            print(f"  iter {i}: APOPTOSIS smoke: {why}")
            iters_since_frontier_grew += 1
            if iters_since_frontier_grew >= PLATEAU_PATIENCE:
                stop_cause, stopped_at = "plateau", i; break
            continue

        ce = evaluate(child, dev, task_class, label=f"{task_class}/iter{i} {h}")

        if (not skip_rollback) and ce.score < parent_eval.score:
            lineage.append(Node(
                iter=i, spec_hash=h, parent_hash=spec_hash(parent_spec),
                edit=prop.edit, reason=prop.reason, accepted=False,
                rejection="regression", on_frontier=False,
                dev_score=ce.score, n_pass=ce.n_pass, n_total=ce.n_total,
                in_tokens=ce.in_tokens, out_tokens=ce.out_tokens, spec=child,
            ))
            print(f"  iter {i}: ROLLBACK ({ce.score:.0%} < {parent_eval.score:.0%})")
            iters_since_frontier_grew += 1
            if iters_since_frontier_grew >= PLATEAU_PATIENCE:
                stop_cause, stopped_at = "plateau", i; break
            continue

        cand = Point(child, h, i, ce.score, ce.in_tokens / max(1, len(dev)))
        added, removed = update_frontier(frontier, cand)
        on_frontier = added
        if added:
            iters_since_frontier_grew = 0
        else:
            iters_since_frontier_grew += 1

        lineage.append(Node(
            iter=i, spec_hash=h, parent_hash=spec_hash(parent_spec),
            edit=prop.edit, reason=prop.reason, accepted=True,
            rejection="dominated" if not added else "", on_frontier=on_frontier,
            dev_score=ce.score, n_pass=ce.n_pass, n_total=ce.n_total,
            in_tokens=ce.in_tokens, out_tokens=ce.out_tokens, spec=child,
        ))
        tag = ("FRONTIER+" if added else "ACCEPT-dominated")
        print(f"  iter {i}: {tag} ({parent_eval.score:.0%} -> {ce.score:.0%}, {cand.tokens:.0f} tok/eval)")

        parent_spec = child
        parent_eval = ce

        if iters_since_frontier_grew >= PLATEAU_PATIENCE:
            stop_cause, stopped_at = "plateau", i
            break

    # The reported specialist is the highest-scoring frontier point. Ties
    # broken by lowest token count.
    specialist_pt = max(frontier, key=lambda p: (p.score, -p.tokens))
    seed_test = evaluate(SeedSpec, test, task_class, label=f"{task_class}/seed test")
    spec_test = evaluate(specialist_pt.spec, test, task_class, label=f"{task_class}/specialist test")

    summary: dict[str, Any] = {
        "task_class": task_class,
        "seed_index": seed,
        "propose_mode": propose_mode,  # 'full' is the canonical run; others are ablations
        "iters_planned": iters,
        "iters_run": stopped_at,
        "stop_cause": stop_cause,
        "seed_test_score": seed_test.score,
        "specialist_test_score": spec_test.score,
        "specialist_spec": specialist_pt.spec,
        "specialist_dev_score": specialist_pt.score,
        "frontier": [
            {"hash": p.spec_hash, "iter": p.iter, "score": p.score, "tokens": p.tokens, "spec": p.spec}
            for p in sorted(frontier, key=lambda p: (-p.score, p.tokens))
        ],
        "lineage": [asdict(n) for n in lineage],
    }
    (run_dir / save_name).write_text(json.dumps(summary, indent=2, default=str))
    return summary
