"""Eval harness. Runs an agent under a spec on a list of tasks; reports pass rate."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent import Run, run, validate_regex, validate_schema, _strip_fence
import json


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    output: str
    feedback: str
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


def load(task_class: str) -> list[dict]:
    return json.loads(Path(f"tasks/{task_class}/tasks.json").read_text())


# Hand-picked splits per class. Dev keeps the harder regressions and crits but
# mixes in low/med items so the meta-agent doesn't learn "everything is critical".
_SPLITS = {
    "regex": {
        "demo": {"us-phone", "strong-pw", "email-edu"},
        "dev":  {"ipv4", "html-attr", "iso-time", "no-tabs", "iso-date", "csv-field"},
    },
    "json": {
        "demo": {"feat-dark-mode", "support-q", "ci-flaky", "api-500", "prod-down"},
        "dev":  {"crash-startup", "auth-bypass", "billing-double-charge", "data-loss",
                 "memory-leak-worker", "regress-pasted-code",
                 "typo-readme", "how-to-config", "lint-rule", "broken-link"},
    },
}


def _detect_class(tasks: list[dict]) -> str:
    if "schema" in tasks[0]:
        return "json"
    if "positives" in tasks[0]:
        return "regex"
    raise ValueError("cannot detect task class")


def split(tasks: list[dict]) -> dict[str, list[dict]]:
    cls = _detect_class(tasks)
    cfg = _SPLITS[cls]
    demo = [t for t in tasks if t["id"] in cfg["demo"]]
    dev = [t for t in tasks if t["id"] in cfg["dev"]]
    test = [t for t in tasks if t["id"] not in cfg["demo"] and t["id"] not in cfg["dev"]]
    return {"demo": demo, "dev": dev, "test": test}


def _score_json_full(task: dict, output: str) -> tuple[bool, str]:
    ok, fb = validate_schema(output, task)
    if not ok:
        return False, fb
    obj = json.loads(_strip_fence(output))
    diffs = [(k, obj[k], task["gold"][k]) for k in task["gold"] if obj[k] != task["gold"][k]]
    if diffs:
        return False, "; ".join(f"{k}: {a!r} vs gold {g!r}" for k, a, g in diffs)
    return True, ""


def score_one(task: dict, run_out: Run, task_class: str) -> tuple[bool, str]:
    if task_class == "regex":
        return validate_regex(run_out.output, task)
    if task_class == "json":
        return _score_json_full(task, run_out.output)
    raise ValueError(f"unknown class: {task_class}")


def evaluate(spec: dict, tasks: list[dict], task_class: str, label: str = "") -> EvalResult:
    t0 = time.time()
    res = EvalResult(score=0.0, n_pass=0, n_total=len(tasks))
    for t in tasks:
        out = run(t, spec, task_class)
        ok, fb = score_one(t, out, task_class)
        res.per_task.append(TaskResult(t["id"], ok, out.output, fb, out.in_tokens, out.out_tokens))
        res.in_tokens += out.in_tokens
        res.out_tokens += out.out_tokens
        if ok:
            res.n_pass += 1
    res.score = res.n_pass / res.n_total if res.n_total else 0.0
    res.duration_s = time.time() - t0
    if label:
        print(f"[{label}] {res.n_pass}/{res.n_total}={res.score:.0%}")
    return res
