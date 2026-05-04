"""Build chat completion bodies and parse outputs for answerability audit batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def answerability_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answerability_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answerable": {"type": "boolean"},
                },
                "required": ["answerable"],
                "additionalProperties": False,
            },
        },
    }


def build_answerability_body(
    *,
    question: str,
    gold_answers: List[str],
    gold_support: str,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt_template: str,
) -> dict[str, Any]:
    ga = "\n".join(f"- {a}" for a in gold_answers if str(a).strip()) or "(none)"
    user = (
        prompt_template.replace("{question}", question)
        .replace("{gold_answers}", ga)
        .replace("{gold_support}", gold_support or "")
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You output only structured JSON matching the schema.",
            },
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": answerability_response_format(),
    }


def parse_answerability_batch_output_line(obj: Mapping[str, Any]) -> Optional[bool]:
    """Return answerable bool from one batch output line, or None if missing/error."""
    if obj.get("error"):
        return None
    resp = obj.get("response") or {}
    if int(resp.get("status_code", 0) or 0) != 200:
        return None
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if not content:
        return None
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    val = data.get("answerable")
    if val is True:
        return True
    if val is False:
        return False
    return None


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")
