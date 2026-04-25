"""Spec-driven worker.

Two tool policies now: no_tools and validate_retry. validate_retry runs the
class's validator, feeds the failure back, and retries up to max_retries.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from llm import chat


SeedSpec: dict[str, Any] = {
    "system_prompt": "Read the task and return only the requested output.",
    "validation":    "none",
    "tool_policy":   "no_tools",
    "max_retries":   0,
}


@dataclass
class Step:
    kind: str
    detail: str = ""
    ms: int = 0


@dataclass
class Run:
    output: str
    steps: list[Step] = field(default_factory=list)
    in_tokens: int = 0
    out_tokens: int = 0


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    if len(s) >= 2 and s.startswith("`") and s.endswith("`") and "\n" not in s:
        s = s[1:-1]
    return s.strip()


def validate_regex(text: str, task: dict) -> tuple[bool, str]:
    pat = _strip_fence(text)
    try:
        rx = re.compile(pat)
    except re.error as e:
        return False, f"regex did not compile: {e}"
    miss_pos = [p for p in task["positives"] if not rx.fullmatch(p)]
    miss_neg = [n for n in task["negatives"] if rx.fullmatch(n)]
    if miss_pos or miss_neg:
        parts = []
        if miss_pos: parts.append(f"should match but did not: {miss_pos[:3]}")
        if miss_neg: parts.append(f"should NOT match but did: {miss_neg[:3]}")
        return False, "; ".join(parts)
    return True, ""


# Pair the policy with the validator it implies. Mismatched pairs fall
# through as no-ops so the meta-agent gets a clean signal next iter.
_VALIDATOR = {
    ("testcases", "regex"): validate_regex,
}


def _run_no_tools(task: dict, spec: dict, model: str | None) -> Run:
    out = Run(output="")
    msgs = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": task["prompt"]},
    ]
    t0 = time.time()
    r = chat(msgs, model=model)
    out.steps.append(Step("llm", "initial", int((time.time() - t0) * 1000)))
    out.in_tokens += r.in_tokens
    out.out_tokens += r.out_tokens
    out.output = r.content
    return out


def _run_validate_retry(task: dict, spec: dict, task_class: str, model: str | None) -> Run:
    out = _run_no_tools(task, spec, model)
    validator = _VALIDATOR.get((spec["validation"], task_class))
    if validator is None:
        return out  # mismatched policy/class is a no-op
    msgs = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": task["prompt"]},
    ]
    for k in range(spec.get("max_retries", 0)):
        ok, fb = validator(out.output, task)
        out.steps.append(Step("validate", "ok" if ok else fb))
        if ok:
            return out
        msgs += [
            {"role": "assistant", "content": out.output},
            {"role": "user", "content": f"Previous answer failed: {fb}. Produce a corrected answer. Output only the answer."},
        ]
        t0 = time.time()
        r = chat(msgs, model=model)
        out.in_tokens += r.in_tokens
        out.out_tokens += r.out_tokens
        out.output = r.content
    return out


def run(task: dict, spec: dict, task_class: str, model: str | None = None) -> Run:
    if spec["tool_policy"] == "no_tools":
        return _run_no_tools(task, spec, model)
    if spec["tool_policy"] == "validate_retry":
        return _run_validate_retry(task, spec, task_class, model)
    raise ValueError(f"unknown tool_policy: {spec['tool_policy']!r}")
