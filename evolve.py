"""Differentiation loop.

The meta-agent (gpt-4o) reads three demo tasks plus the most recent dev
failures and proposes a JSON edit to the worker spec. Children are
checked for parse, smoke, and regression; the parent advances when the
child is accepted.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent import SeedSpec, run
from eval import EvalResult, evaluate, load, split
from llm import chat


META_MODEL = "gpt-4o"
META_TEMPERATURE = 0.7


META_SYSTEM = """You evolve a small worker agent's spec one structured edit at a time.

The spec has exactly four fields: system_prompt (free text), validation (none/schema/testcases),
tool_policy (no_tools/validate_retry), max_retries (int 0..4). No other fields.

You see the task class, three demo tasks, the current spec, and which dev tasks just failed.
Propose ONE structured edit. Output ONLY a JSON object:

  {"reason": "<one sentence>", "edit": {<field>: <new value>, ...}}

Only include fields you change. Do not invent fields. Do not exceed max_retries=4.
"""


@dataclass
class Proposal:
    reason: str
    edit: dict


def _propose(task_class: str, demos: list[dict], current_spec: dict,
             last_failures: list[dict] | None) -> Proposal:
    user_lines = [f"Task class: {task_class}", "", "Three demo tasks:"]
    for d in demos[:3]:
        user_lines.append(f"  id={d['id']}")
        user_lines.append(f"  prompt: {d['prompt'][:200]}...")
        if "positives" in d:
            user_lines.append(f"  positives: {d['positives']}")
            user_lines.append(f"  negatives: {d['negatives']}")
    user_lines += ["", f"Current spec: {json.dumps(current_spec, indent=2)}"]
    if last_failures:
        user_lines += ["", "Last dev-eval failures:"]
        for f in last_failures[:8]:
            user_lines.append(f"  {f['task_id']}: feedback={f['feedback']!r}")
    r = chat(
        [{"role": "system", "content": META_SYSTEM},
         {"role": "user", "content": "\n".join(user_lines)}],
        model=META_MODEL, temperature=META_TEMPERATURE,
    )
    raw = r.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    obj = json.loads(raw.strip())
    return Proposal(reason=obj.get("reason", ""), edit=obj.get("edit", {}))


ALLOWED_FIELDS = {"system_prompt", "validation", "tool_policy", "max_retries"}
ALLOWED_VALIDATIONS = {"none", "schema", "testcases"}
ALLOWED_TOOL_POLICIES = {"no_tools", "validate_retry"}


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


def spec_hash(spec: dict) -> str:
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]


def evolve(task_class: str, iters: int = 6, out_dir: Path = Path("runs"), seed: int = 0) -> dict:
    tasks = split(load(task_class))
    demos, dev, test = tasks["demo"], tasks["dev"], tasks["test"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / task_class
    run_dir.mkdir(exist_ok=True)

    parent_spec = copy.deepcopy(SeedSpec)
    parent_eval = evaluate(parent_spec, dev, task_class, label=f"{task_class}/seed dev")
    history: list[dict] = [{"iter": 0, "spec": parent_spec, "dev_score": parent_eval.score, "accepted": True}]

    for i in range(1, iters + 1):
        last_fails = [{"task_id": t.task_id, "feedback": t.feedback} for t in parent_eval.per_task if not t.passed]
        prop = _propose(task_class, demos, parent_spec, last_fails)
        child = merge(parent_spec, prop.edit)
        ok, why = parse_check(child)
        if not ok:
            history.append({"iter": i, "spec": child, "rejected": "parse", "why": why})
            print(f"  iter {i}: APOPTOSIS parse: {why}")
            continue
        ok, why = smoke_check(child, demos[0], task_class)
        if not ok:
            history.append({"iter": i, "spec": child, "rejected": "smoke", "why": why})
            print(f"  iter {i}: APOPTOSIS smoke: {why}")
            continue
        ce = evaluate(child, dev, task_class, label=f"{task_class}/iter{i}")
        # Tried strict improvement (`>`) here first. Killed the regex specialist
        # because validate_retry can fix a *test* item without moving the *dev*
        # score on a 12-task dev set. Back to monotone (`>=`).
        if ce.score < parent_eval.score:
            history.append({"iter": i, "spec": child, "rejected": "regression", "child_score": ce.score})
            print(f"  iter {i}: ROLLBACK ({ce.score:.0%} < {parent_eval.score:.0%})")
            continue
        history.append({"iter": i, "spec": child, "dev_score": ce.score, "accepted": True})
        print(f"  iter {i}: ACCEPT ({parent_eval.score:.0%} -> {ce.score:.0%})")
        parent_spec = child
        parent_eval = ce

    seed_test = evaluate(SeedSpec, test, task_class, label=f"{task_class}/seed test")
    spec_test = evaluate(parent_spec, test, task_class, label=f"{task_class}/specialist test")
    summary = {
        "task_class": task_class,
        "seed_test_score": seed_test.score,
        "specialist_test_score": spec_test.score,
        "specialist_spec": parent_spec,
        "history": history,
    }
    (run_dir / f"seed{seed}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
