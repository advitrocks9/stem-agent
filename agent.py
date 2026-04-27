"""Spec-driven worker. The spec is a small dict; behaviour is selected from it.

Four knobs.
  system_prompt   free text
  validation      'none' | 'schema' | 'testcases'
  tool_policy     'no_tools' | 'validate_retry' | 'code_exec'
  max_retries     int in [0, 4]

The seed is no_tools + none + 0. A specialist may add validation+retry, or
turn on the python_exec tool, or rewrite the system_prompt. Three knobs
that change behaviour, plus one numeric budget.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from llm import chat


SeedSpec: dict[str, Any] = {
    "system_prompt": "Read the task and return only the requested output.",
    "validation": "none",
    "tool_policy": "no_tools",
    "max_retries": 0,
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


# ----- validators -------------------------------------------------------

def _strip_fence(s: str) -> str:
    """Remove a ``` ... ``` block or single backticks around the whole thing."""
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
        if miss_pos:
            parts.append(f"should match but did not: {miss_pos[:3]}")
        if miss_neg:
            parts.append(f"should NOT match but did: {miss_neg[:3]}")
        return False, "; ".join(parts)
    return True, ""


def validate_schema(text: str, task: dict) -> tuple[bool, str]:
    raw = _strip_fence(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"not valid JSON: {e.msg}"
    schema = task["schema"]
    missing = [k for k in schema if k not in obj]
    if missing:
        return False, f"missing fields: {missing}"
    bad = [f"{k}={obj[k]!r} not in {v}" for k, v in schema.items() if obj[k] not in v]
    if bad:
        return False, "; ".join(bad)
    return True, ""


# (validation_policy, task_class) -> validator. Mismatched pairs fall through
# as no-ops, which gives the meta-agent a clean signal next iter.
_VALIDATOR = {
    ("testcases", "regex"): validate_regex,
    ("schema", "json"):     validate_schema,
}


# ----- python sandbox ---------------------------------------------------

PY_TIMEOUT = 6  # seconds; 6 is enough for the math class

def python_exec(code: str) -> str:
    """Run code in a fresh subprocess, return combined stdout/stderr (truncated).

    No network (best-effort: uses a short sys.path with no extras), short timeout,
    no shell. The agent only sees the program's printed output.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=PY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"<timeout after {PY_TIMEOUT}s>"
    out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    return out[:2000] if len(out) > 2000 else out


PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": "Run a short Python program and return its printed output. Use for arithmetic and step-by-step computation. No network, 6s timeout.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "A complete Python program. Print the final answer."},
            },
            "required": ["code"],
        },
    },
}


# ----- the three branches ----------------------------------------------

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
        return out  # mismatched policy/class: no-op pass-through
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
            {"role": "user", "content": f"Your previous answer failed validation: {fb}. Produce a corrected answer. Output only the answer."},
        ]
        t0 = time.time()
        r = chat(msgs, model=model)
        out.steps.append(Step("llm", f"retry {k+1}", int((time.time() - t0) * 1000)))
        out.in_tokens += r.in_tokens
        out.out_tokens += r.out_tokens
        out.output = r.content
    return out


def _run_code_exec(task: dict, spec: dict, model: str | None) -> Run:
    """Multi-turn ReAct with python_exec available. max_retries caps tool calls."""
    out = Run(output="")
    msgs = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": task["prompt"]},
    ]
    # 1 initial LLM call plus up to max_retries tool-call rounds, plus a final
    # call to consume the last tool result. Cap at 8 to keep token spend sane.
    turn_cap = min(8, 2 + spec.get("max_retries", 0))
    for turn in range(turn_cap):
        t0 = time.time()
        r = chat(msgs, model=model, tools=[PYTHON_TOOL_SCHEMA])
        out.in_tokens += r.in_tokens
        out.out_tokens += r.out_tokens
        out.steps.append(Step("llm", f"turn {turn}", int((time.time() - t0) * 1000)))

        if not r.tool_calls:
            out.output = r.content
            return out

        # The model called python_exec. Append its assistant message *with* the
        # tool_calls field (otherwise the API rejects the follow-up), then add
        # one tool message per call.
        msgs.append({
            "role": "assistant",
            "content": r.content or "",
            "tool_calls": [tc.raw for tc in r.tool_calls],
        })
        for tc in r.tool_calls:
            try:
                args = json.loads(tc.arguments_json or "{}")
            except json.JSONDecodeError:
                args = {}
            code = args.get("code", "")
            result = python_exec(code) if tc.name == "python_exec" else f"<unknown tool: {tc.name}>"
            out.steps.append(Step("tool", f"python_exec -> {result[:120]!r}"))
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Hit the turn cap; the last LLM reply (if any) is the output.
    out.output = out.output or "<turn cap reached>"
    return out


def run(task: dict, spec: dict, task_class: str, model: str | None = None) -> Run:
    """Pure function of (task, spec, class). Picks the branch by tool_policy."""
    tp = spec["tool_policy"]
    if tp == "no_tools":
        return _run_no_tools(task, spec, model)
    if tp == "validate_retry":
        return _run_validate_retry(task, spec, task_class, model)
    if tp == "code_exec":
        return _run_code_exec(task, spec, model)
    raise ValueError(f"unknown tool_policy: {tp!r}")
