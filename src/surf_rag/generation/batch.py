"""OpenAI Batch API helpers: build chat completion bodies from rendered messages."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from surf_rag.generation.generator_tool import (
    FORMAT_ANSWER_TOOL,
    TOOL_CHOICE_FORCED,
    GenerationParseResult,
    parse_generation_output_line,
)


def build_completion_body(
    messages: List[Dict[str, str]],
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """Build request body for /v1/chat/completions (Batch API). Matches sync OpenAIChatGenerator."""
    model_id = model_id or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("GENERATOR_TEMPERATURE", "0"))
    )
    max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.getenv("GENERATOR_MAX_TOKENS", "512"))
    )
    return {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": [FORMAT_ANSWER_TOOL],
        "tool_choice": TOOL_CHOICE_FORCED,
        "parallel_tool_calls": False,
    }


def build_batch_line(custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single line for the Batch API input JSONL file."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def parse_generation_output(line: Dict[str, Any]) -> GenerationParseResult:
    """Parse a Batch API output line into structured generation fields."""
    return parse_generation_output_line(line)
