from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Protocol

from openai import OpenAI

from surf_rag.core.answer_prefix import strip_answer_prefix


class Generator(Protocol):
    """LLM consumer of rendered chat messages."""

    def generate(self, messages: List[Dict[str, str]]) -> "GenerationResult": ...


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_id: str
    latency_ms: float
    prompt_hash: str
    sampling: Dict[str, object]


def hash_messages(messages: List[Dict[str, str]]) -> str:
    """Stable hash over full message list (system + user)."""
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class OpenAIChatGenerator:
    """Synchronous OpenAI chat completions from pre-rendered messages."""

    model_id: str
    temperature: float = 0.0
    max_tokens: int = int(os.getenv("GENERATOR_MAX_TOKENS", 512))

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        """Call the API. Caller decides when to skip (e.g. NO_CONTEXT)."""
        start_time = time.perf_counter()
        prompt_hash = hash_messages(messages)

        has_user_content = any(
            (m.get("content") or "").strip() for m in messages if m.get("role") == "user"
        )
        if not has_user_content and len(messages) <= 1:
            return GenerationResult(
                text="No context provided.",
                model_id=self.model_id,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                prompt_hash=prompt_hash,
                sampling={"temperature": self.temperature, "max_tokens": self.max_tokens},
            )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.choices[0].message.content or ""
        final_answer = strip_answer_prefix(answer)

        latency_ms = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            text=final_answer,
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            sampling={"temperature": self.temperature, "max_tokens": self.max_tokens},
        )
