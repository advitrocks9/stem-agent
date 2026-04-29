"""Spec-driven worker. The spec is a small dict; behaviour is selected from it.

Four knobs.
  system_prompt   free text
  validation      'none' | 'schema' | 'testcases' | 'results'
  tool_policy     'no_tools' | 'validate_retry' | 'code_exec'
  max_retries     int in [0, 4]

The seed is no_tools + none + 0. A specialist may add validation+retry, or
turn on the python_exec tool, or rewrite the system_prompt. Three knobs
that change behaviour, plus one numeric budget.
"""
from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    """Tagged feedback so the meta-agent can target the right fix.

    Three failure tags: [parse-fail], [missing-field], [value-not-allowed].
    A schema-only validator can never see [wrong-but-legal-value] errors;
    those surface only at scoring time against the gold (eval._score_json_full).
    """
    raw = _strip_fence(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"[parse-fail] not valid JSON: {e.msg}"
    schema = task["schema"]
    missing = [k for k in schema if k not in obj]
    if missing:
        return False, f"[missing-field] {missing}"
    bad = [f"{k}={obj[k]!r} not in {v}" for k, v in schema.items() if obj[k] not in v]
    if bad:
        return False, f"[value-not-allowed] {'; '.join(bad)}"
    return True, ""


_SQL_FIXTURE_DB = Path(__file__).resolve().parent / "tasks/sql/fixture.db"
_SQL_FIXTURE_SQL = Path(__file__).resolve().parent / "tasks/sql/fixture.sql"


def _ensure_sql_fixture() -> None:
    if _SQL_FIXTURE_DB.exists():
        return
    if not _SQL_FIXTURE_SQL.exists():
        return  # no fixture, validator will fail loudly when called
    con = sqlite3.connect(_SQL_FIXTURE_DB)
    con.executescript(_SQL_FIXTURE_SQL.read_text())
    con.commit()
    con.close()


def _run_sql(query: str) -> list[tuple]:
    _ensure_sql_fixture()
    con = sqlite3.connect(_SQL_FIXTURE_DB)
    try:
        return con.execute(query).fetchall()
    finally:
        con.close()


def validate_sql(text: str, task: dict) -> tuple[bool, str]:
    """Run predicted SQL against the fixture and compare its result set to gold's."""
    pred_sql = _strip_fence(text)
    if pred_sql.endswith(";"):
        pred_sql = pred_sql[:-1]
    try:
        pred_rows = _run_sql(pred_sql)
    except sqlite3.Error as e:
        return False, f"[sql-error] {e}"
    try:
        gold_rows = _run_sql(task["gold_sql"])
    except sqlite3.Error as e:
        return False, f"[gold-bug] {e}"  # shouldn't happen; gold is hand-checked
    # Apply _row_key on both branches: float/int parity (1.0 == 1) bit me on
    # top-3-products where the gold path returned ints but the predicted path
    # returned floats from a SUM(quantity) cast.
    pred_keyed = [_row_key(r) for r in pred_rows]
    gold_keyed = [_row_key(r) for r in gold_rows]
    pred_set = pred_keyed if task.get("ordered") else sorted(pred_keyed)
    gold_set = gold_keyed if task.get("ordered") else sorted(gold_keyed)
    if pred_set == gold_set:
        return True, ""
    if len(pred_rows) != len(gold_rows):
        return False, f"[row-count] predicted {len(pred_rows)} rows, gold has {len(gold_rows)}"
    return False, f"[result-mismatch] first 3 predicted={pred_rows[:3]}; first 3 gold={gold_rows[:3]}"


def _row_key(row: tuple) -> tuple:
    """Coerce floats to a stable string so 1.0 and 1 compare equal across SQLite casts."""
    return tuple((round(c, 4) if isinstance(c, float) else c) for c in row)


# (validation_policy, task_class) -> validator. Mismatched pairs fall through
# as no-ops, which gives the meta-agent a clean signal next iter.
_VALIDATOR = {
    ("testcases", "regex"): validate_regex,
    ("schema", "json"):     validate_schema,
    ("results", "sql"):     validate_sql,
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


# SQL tasks need the schema in context. Following Spider/BIRD convention,
# prepend the schema to the user message rather than baking it into the
# spec, so the seed stays class-agnostic.
SQL_SCHEMA_PREFIX = """The database has three tables:

    users(id, name, country, signup_date)
    products(id, name, category, price)
    orders(id, user_id, product_id, quantity, order_date)

Output only the SQL query, no explanation, no surrounding text or fences.

Question: """


def _user_msg(task: dict, task_class: str) -> str:
    if task_class == "sql":
        return SQL_SCHEMA_PREFIX + task["prompt"]
    return task["prompt"]


# ----- the three branches ----------------------------------------------

def _run_no_tools(task: dict, spec: dict, task_class: str, model: str | None) -> Run:
    out = Run(output="")
    msgs = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": _user_msg(task, task_class)},
    ]
    t0 = time.time()
    r = chat(msgs, model=model)
    out.steps.append(Step("llm", "initial", int((time.time() - t0) * 1000)))
    out.in_tokens += r.in_tokens
    out.out_tokens += r.out_tokens
    out.output = r.content
    return out


def _run_validate_retry(task: dict, spec: dict, task_class: str, model: str | None) -> Run:
    out = _run_no_tools(task, spec, task_class, model)
    validator = _VALIDATOR.get((spec["validation"], task_class))
    if validator is None:
        return out  # mismatched policy/class: no-op pass-through
    msgs = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": _user_msg(task, task_class)},
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


def _run_code_exec(task: dict, spec: dict, task_class: str, model: str | None) -> Run:
    """Multi-turn ReAct with python_exec available. max_retries caps tool calls."""
    out = Run(output="")
    msgs = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": _user_msg(task, task_class)},
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
        return _run_no_tools(task, spec, task_class, model)
    if tp == "validate_retry":
        return _run_validate_retry(task, spec, task_class, model)
    if tp == "code_exec":
        return _run_code_exec(task, spec, task_class, model)
    raise ValueError(f"unknown tool_policy: {tp!r}")
