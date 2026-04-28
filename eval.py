"""Eval harness. Loads tasks, splits demo/dev/test, scores a spec on a task list."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent import Run, run, validate_regex, validate_schema, validate_sql, _strip_fence


# Hand-picked splits keep the dev set heavy on items the seed fails on,
# which is what the meta-agent needs to see useful signal. The split is
# 6 demo / 12 dev / rest as test for each class.
_SPLITS: dict[str, dict[str, set[str]]] = {
    "regex": {
        "demo": {"us-phone", "strong-pw", "email-edu", "json-num", "comment-cpp", "no-three-same"},
        "dev":  {"ipv4", "html-attr", "float-sci", "csv-field", "ipv6-simple", "tag-balanced",
                 "iso-time", "no-tabs", "abba-pal", "long-double", "iso-datetime", "kebab-id"},
    },
    "json": {
        # demo is a cross-section, deliberately not all the hardest items.
        # dev keeps the harder regressions and crits for signal but mixes in
        # low/med items so the meta-agent doesn't learn "everything is critical".
        "demo": {"feat-dark-mode", "support-q", "ci-flaky", "api-500", "prod-down", "regress-search-empty"},
        "dev":  {"crash-startup", "auth-bypass", "billing-double-charge", "data-loss",
                 "memory-leak-worker", "regress-pasted-code",
                 "typo-readme", "how-to-config", "lint-rule", "broken-link",
                 "node-upgrade", "feat-keyboard-shortcuts"},
    },
    "math": {
        "demo": {"train-speed", "recipe-scale", "book-ratio", "discount-stack", "perc-of-perc", "circle-area"},
        "dev":  {"compound-int", "interest-monthly", "two-trains-meet", "binom-prob", "loan-emi",
                 "compound-3-rates", "three-discounts", "clock-angle", "speed-vary", "work-vary",
                 "age-puzzle", "sphere-vol"},
    },
    "sql": {
        # demo covers easy/med/hard so the meta-agent sees the range.
        # dev mixes joins, group-by, and a couple of subquery tasks.
        "demo": {"count-users", "uk-users", "count-orders-per-user", "revenue-by-category",
                 "top-spender-country", "second-most-popular"},
        "dev":  {"electronics-prices", "avg-order-value", "top-3-products", "users-no-orders",
                 "orders-per-month", "category-counts", "big-spenders", "first-orders",
                 "users-bought-electronics", "product-never-ordered", "q4-revenue",
                 "category-share"},
    },
}


def _detect_class(tasks: list[dict]) -> str:
    if "gold_sql" in tasks[0]:
        return "sql"
    if "schema" in tasks[0]:
        return "json"
    if "positives" in tasks[0]:
        return "regex"
    if "gold" in tasks[0]:
        return "math"
    raise ValueError("cannot detect task class")


def split(tasks: list[dict]) -> dict[str, list[dict]]:
    cls = _detect_class(tasks)
    cfg = _SPLITS[cls]
    demo = [t for t in tasks if t["id"] in cfg["demo"]]
    dev = [t for t in tasks if t["id"] in cfg["dev"]]
    test = [t for t in tasks if t["id"] not in cfg["demo"] and t["id"] not in cfg["dev"]]
    return {"demo": demo, "dev": dev, "test": test}


def random_split(tasks: list[dict], split_seed: int) -> dict[str, list[dict]]:
    """Same shape as the hand-picked split (6 demo / 12 dev / rest test) but
    random task assignment, deterministic per seed. Use for split-sensitivity:
    if the conclusions only hold with the hand-picked split, that's a finding."""
    import random as _r
    rng = _r.Random(split_seed)
    shuffled = sorted(tasks, key=lambda t: t["id"])  # stable starting order
    rng.shuffle(shuffled)
    return {"demo": shuffled[:6], "dev": shuffled[6:18], "test": shuffled[18:]}


def load(task_class: str) -> list[dict]:
    return json.loads(Path(f"tasks/{task_class}/tasks.json").read_text())


# ----- scoring ----------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    passed: bool
    output: str
    feedback: str
    n_steps: int
    in_tokens: int
    out_tokens: int


@dataclass
class EvalResult:
    score: float
    n_pass: int
    n_total: int
    per_task: list[TaskResult] = field(default_factory=list)
    in_tokens: int = 0
    out_tokens: int = 0
    duration_s: float = 0.0


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_FRAC_RE = re.compile(r"-?\d+/\d+")


def _norm_math_answer(s: str) -> str:
    """Pull a clean numeric or fractional or HH:MM answer out of free-form text.

    Order of preference: HH:MM, a/b fraction, last number in the string.
    Numbers are stripped of $ , and trailing zeros are normalised so that
    '7.50' equals '7.5'.
    """
    s = s.strip().lower().replace("$", "").replace(",", "")
    if (m := re.search(r"\b\d{1,2}:\d{2}\b", s)):
        return m.group(0)
    if (m := _FRAC_RE.search(s)):
        return m.group(0)
    nums = _NUM_RE.findall(s)
    if not nums:
        return s
    last = nums[-1]
    if "." in last:
        last = last.rstrip("0").rstrip(".")
        if last == "" or last == "-":
            last = "0"
    return last


def _score_math(task: dict, output: str) -> tuple[bool, str]:
    pred = _norm_math_answer(_strip_fence(output))
    gold = _norm_math_answer(task["gold"])
    if pred == gold:
        return True, ""
    return False, f"pred={pred!r} gold={gold!r}"


def _score_json_full(task: dict, output: str) -> tuple[bool, str]:
    """validate_schema first, then gold-match. The two failure modes get
    different tags so the meta-agent can tell which lever to pull."""
    ok, fb = validate_schema(output, task)
    if not ok:
        return False, fb  # already carries [parse-fail] / [missing-field] / [value-not-allowed]
    obj = json.loads(_strip_fence(output))
    diffs = [(k, obj[k], task["gold"][k]) for k in task["gold"] if obj[k] != task["gold"][k]]
    if diffs:
        return False, "[value-mismatch] " + "; ".join(f"{k}: predicted {a!r} but gold is {g!r}" for k, a, g in diffs)
    return True, ""


def score_one(task: dict, run_out: Run, task_class: str) -> tuple[bool, str]:
    if task_class == "regex":
        return validate_regex(run_out.output, task)
    if task_class == "json":
        return _score_json_full(task, run_out.output)
    if task_class == "math":
        return _score_math(task, run_out.output)
    if task_class == "sql":
        return validate_sql(run_out.output, task)
    raise ValueError(f"unknown class: {task_class}")


def evaluate(spec: dict, tasks: list[dict], task_class: str, label: str = "") -> EvalResult:
    t0 = time.time()
    res = EvalResult(score=0.0, n_pass=0, n_total=len(tasks))
    for t in tasks:
        out = run(t, spec, task_class)
        ok, fb = score_one(t, out, task_class)
        res.per_task.append(TaskResult(
            task_id=t["id"], passed=ok, output=out.output, feedback=fb,
            n_steps=len(out.steps), in_tokens=out.in_tokens, out_tokens=out.out_tokens,
        ))
        res.in_tokens += out.in_tokens
        res.out_tokens += out.out_tokens
        if ok:
            res.n_pass += 1
    res.score = res.n_pass / res.n_total if res.n_total else 0.0
    res.duration_s = time.time() - t0
    if label:
        print(f"[{label}] {res.n_pass}/{res.n_total}={res.score:.0%} ({res.in_tokens}+{res.out_tokens} tok, {res.duration_s:.1f}s)")
    return res
