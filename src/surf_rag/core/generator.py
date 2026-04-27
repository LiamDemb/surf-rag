from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Protocol

from openai import OpenAI

from surf_rag.generation.batch import build_completion_body
from surf_rag.generation.generator_tool import parse_openai_sync_response


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
    """Synchronous OpenAI chat completions from pre-rendered messages (format_answer tool)."""

    model_id: str
    temperature: float = 0.0
    max_tokens: int = int(os.getenv("GENERATOR_MAX_TOKENS", 512))

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        """Call the API. Caller decides when to skip (e.g. NO_CONTEXT)."""
        start_time = time.perf_counter()
        prompt_hash = hash_messages(messages)

        has_user_content = any(
            (m.get("content") or "").strip()
            for m in messages
            if m.get("role") == "user"
        )
        if not has_user_content and len(messages) <= 1:
            return GenerationResult(
                text="No context provided.",
                model_id=self.model_id,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                prompt_hash=prompt_hash,
                sampling={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            )

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        body = build_completion_body(
            messages,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response = client.chat.completions.create(
            model=body["model"],
            messages=body["messages"],
            temperature=body["temperature"],
            max_tokens=body["max_tokens"],
            tools=body["tools"],
            tool_choice=body["tool_choice"],
            parallel_tool_calls=body.get("parallel_tool_calls", False),
        )
        parsed = parse_openai_sync_response(response)

        latency_ms = (time.perf_counter() - start_time) * 1000
        return GenerationResult(
            text=parsed.answer,
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            sampling={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "candidate_answer_span": parsed.candidate_answer_span,
                "support_quote": parsed.support_quote,
                "reasoning": parsed.reasoning,
                "generation_output_format": parsed.generation_output_format,
                "generation_parse_error": parsed.generation_parse_error,
            },
        )
