"""Thin wrapper over the OpenAI chat API."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError

load_dotenv()

DEFAULT_MODEL = os.getenv("STEM_MODEL", "gpt-4o-mini")
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@dataclass
class Reply:
    content: str
    model: str
    in_tokens: int
    out_tokens: int


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0) -> Reply:
    model = model or DEFAULT_MODEL
    for attempt in range(2):
        try:
            r = client().chat.completions.create(model=model, messages=messages, temperature=temperature)
            usage = r.usage
            return Reply(
                content=r.choices[0].message.content or "",
                model=model,
                in_tokens=usage.prompt_tokens if usage else 0,
                out_tokens=usage.completion_tokens if usage else 0,
            )
        except (APIConnectionError, RateLimitError):
            if attempt == 0:
                time.sleep(2); continue
            raise
