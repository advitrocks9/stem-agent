"""Thin wrapper over the OpenAI chat API. One function plus a small reply type.

Kept deliberately minimal. There's one retry on transient connection errors,
nothing else: any other failure should propagate so the experiment surfaces
it loudly rather than silently logging.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

load_dotenv()

DEFAULT_MODEL = os.getenv("STEM_MODEL", "gpt-4o-mini")
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        # 60s read timeout so we don't hang forever on a stalled response.
        _client = OpenAI(timeout=60.0)
    return _client


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_json: str
    raw: dict  # the original .model_dump() so we can echo it back to the API


@dataclass
class Reply:
    content: str
    model: str
    in_tokens: int
    out_tokens: int
    tool_calls: list[ToolCall]


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    tools: list[dict] | None = None,
    response_format: dict | None = None,
) -> Reply:
    model = model or DEFAULT_MODEL
    kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(2):
        try:
            r = client().chat.completions.create(**kwargs)
            choice = r.choices[0]
            tcs: list[ToolCall] = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tcs.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments_json=tc.function.arguments or "",
                        raw=tc.model_dump(),
                    ))
            usage = r.usage
            return Reply(
                content=choice.message.content or "",
                model=model,
                in_tokens=usage.prompt_tokens if usage else 0,
                out_tokens=usage.completion_tokens if usage else 0,
                tool_calls=tcs,
            )
        except (APIConnectionError, APITimeoutError, RateLimitError):
            if attempt == 0:
                time.sleep(2)
                continue
            raise
