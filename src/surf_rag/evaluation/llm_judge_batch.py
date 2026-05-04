"""OpenAI Batch request bodies and output parsing for LLM-as-judge."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

JUDGE_PREFIX = "llm_judge::"


def make_llm_judge_custom_id(question_id: str) -> str:
    safe = question_id.replace("::", "__COLON__")
    return f"{JUDGE_PREFIX}{safe}"


def parse_llm_judge_custom_id(custom_id: str) -> Optional[str]:
    if not str(custom_id).startswith(JUDGE_PREFIX):
        return None
    rest = str(custom_id)[len(JUDGE_PREFIX) :]
    return rest.replace("__COLON__", "::")


def judge_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "qa_judge_verdict",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "correct": {"type": "boolean"},
                },
                "required": ["correct"],
                "additionalProperties": False,
            },
        },
    }


def build_judge_body(
    *,
    question: str,
    gold_answers: list[str],
    prediction: str,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt_template: str,
) -> dict[str, Any]:
    ga = "\n".join(f"- {a}" for a in gold_answers if str(a).strip()) or "(none)"
    user = (
        prompt_template.replace("{question}", question)
        .replace("{gold_answers}", ga)
        .replace("{prediction}", prediction or "")
    )
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You judge whether the prediction correctly answers the question given the reference answers. Output only JSON matching the schema.",
            },
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": judge_response_format(),
    }


def parse_judge_batch_output_line(obj: Mapping[str, Any]) -> Optional[bool]:
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
    val = data.get("correct")
    if val is True:
        return True
    if val is False:
        return False
    return None
