"""Spec-driven worker.

Three knobs that change behaviour: system_prompt, validation, max_retries.
This first version supports only the no_tools branch.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

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


def run(task: dict, spec: dict, task_class: str, model: str | None = None) -> Run:
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
