"""Forced `format_answer` tool contract for QA generation (Batch + sync)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

FORMAT_ANSWER_FUNCTION_NAME = "format_answer"
GENERATION_OUTPUT_FORMAT = "format_answer_v2"

# Single forced function; grounded evidence fields + final short answer for metrics.
FORMAT_ANSWER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": FORMAT_ANSWER_FUNCTION_NAME,
        "description": (
            "Return grounded evidence fields, concise rationale, and the final short answer. "
            "The answer must be short and suitable for exact-match evaluation."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_answer_span": {
                    "type": "string",
                    "description": "Best candidate answer phrase copied from context.",
                },
                "support_quote": {
                    "type": "string",
                    "description": "Shortest direct quote from context supporting the answer.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief rationale grounded in the provided context only.",
                },
                "answer": {
                    "type": "string",
                    "description": "Final short answer only; no full sentences when a phrase "
                    "suffices.",
                },
            },
            "required": [
                "candidate_answer_span",
                "support_quote",
                "reasoning",
                "answer",
            ],
            "additionalProperties": False,
        },
    },
}

TOOL_CHOICE_FORCED: Dict[str, Any] = {
    "type": "function",
    "function": {"name": FORMAT_ANSWER_FUNCTION_NAME},
}


@dataclass(frozen=True)
class GenerationParseResult:
    """Result of parsing one chat completion (Batch line or sync response body)."""

    custom_id: str
    answer: str
    reasoning: str
    candidate_answer_span: str = ""
    support_quote: str = ""
    generation_output_format: str = GENERATION_OUTPUT_FORMAT
    generation_parse_error: Optional[str] = None

    def to_tuple(self) -> Tuple[str, str]:
        """Adapter for call sites that only need (custom_id, answer)."""
        return (self.custom_id, self.answer)

    @property
    def ok(self) -> bool:
        return self.generation_parse_error is None

    @classmethod
    def error(
        cls,
        custom_id: str,
        err: str,
        *,
        answer: str = "",
        reasoning: str = "",
        candidate_answer_span: str = "",
        support_quote: str = "",
    ) -> "GenerationParseResult":
        return cls(
            custom_id=custom_id,
            answer=answer,
            reasoning=reasoning,
            candidate_answer_span=candidate_answer_span,
            support_quote=support_quote,
            generation_output_format=GENERATION_OUTPUT_FORMAT,
            generation_parse_error=err,
        )


def _find_format_answer_tool_call(
    message: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not message:
        return None
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            continue
        if fn.get("name") == FORMAT_ANSWER_FUNCTION_NAME:
            return fn
    return None


def _parse_arguments_json(
    args_str: Optional[Union[str, Any]],
) -> Tuple[str, str, str, str, Optional[str]]:
    """Returns (answer, reasoning, candidate_answer_span, support_quote, parse_error)."""
    if args_str is None:
        return ("", "", "", "", "missing function arguments")
    s = str(args_str).strip() if not isinstance(args_str, str) else args_str.strip()
    if not s:
        s = "{}"
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        return ("", "", "", "", f"invalid tool arguments JSON: {e}")
    if not isinstance(data, dict):
        return ("", "", "", "", "tool arguments must be a JSON object")
    reasoning = str(data.get("reasoning", "") or "").strip()
    answer = str(data.get("answer", "") or "").strip()
    candidate_answer_span = str(data.get("candidate_answer_span", "") or "").strip()
    support_quote = str(data.get("support_quote", "") or "").strip()
    if "answer" not in data:
        return (
            "",
            reasoning,
            candidate_answer_span,
            support_quote,
            "missing required field: answer",
        )
    if "reasoning" not in data:
        return (
            answer,
            "",
            candidate_answer_span,
            support_quote,
            "missing required field: reasoning",
        )
    if "candidate_answer_span" not in data:
        return (
            answer,
            reasoning,
            "",
            support_quote,
            "missing required field: candidate_answer_span",
        )
    if "support_quote" not in data:
        return (
            answer,
            reasoning,
            candidate_answer_span,
            "",
            "missing required field: support_quote",
        )
    return (answer, reasoning, candidate_answer_span, support_quote, None)


def parse_message_for_format_answer(
    message: Optional[Dict[str, Any]], *, custom_id: str = ""
) -> GenerationParseResult:
    """
    Parse choices[0].message from a chat completion (dict / Batch shape).
    Ignores message.content; only tool_calls format_answer is used.
    """
    fn = _find_format_answer_tool_call(message)
    if fn is None:
        return GenerationParseResult.error(
            custom_id, "no format_answer tool call in response"
        )
    args_str = fn.get("arguments")
    answer, reasoning, candidate_answer_span, support_quote, err = _parse_arguments_json(
        args_str
    )
    if err:
        return GenerationParseResult.error(
            custom_id,
            err,
            answer=answer,
            reasoning=reasoning,
            candidate_answer_span=candidate_answer_span,
            support_quote=support_quote,
        )
    return GenerationParseResult(
        custom_id=custom_id,
        answer=answer,
        reasoning=reasoning,
        candidate_answer_span=candidate_answer_span,
        support_quote=support_quote,
    )


def parse_generation_output_line(
    line: Optional[Dict[str, Any]],
) -> GenerationParseResult:
    """
    Parse one Batch API output JSONL object into GenerationParseResult.

    Batch responses use the same body shape as /v1/chat/completions, with
    top-level `error` or `response` + `status_code`.
    """
    if not line or not isinstance(line, dict):
        return GenerationParseResult.error("", "empty or invalid line object")

    custom_id = (line.get("custom_id") or line_custom_id or "") or ""

    if line.get("error"):
        e = line.get("error")
        if isinstance(e, dict):
            err_msg = (
                e.get("message") or e.get("code") or json.dumps(e, ensure_ascii=False)
            )
        else:
            err_msg = str(e)
        return GenerationParseResult.error(custom_id, f"batch line error: {err_msg}")

    response = line.get("response")
    if not response or not isinstance(response, dict):
        return GenerationParseResult.error(
            custom_id, "missing or invalid response object"
        )
    if response.get("status_code") != 200:
        return GenerationParseResult.error(
            custom_id,
            f"non-200 status: {response.get('status_code')!r}",
        )

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return GenerationParseResult.error(custom_id, "no choices in completion body")

    first = choices[0] or {}
    msg = first.get("message") or {}
    return parse_message_for_format_answer(msg, custom_id=custom_id)


def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """OpenAI ChatCompletionMessage → dict for tool parsing."""
    if msg is None:
        return {}
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        out = msg.model_dump()
        return out if isinstance(out, dict) else {}
    out: Dict[str, Any] = {
        "content": getattr(msg, "content", None),
        "role": getattr(msg, "role", None),
    }
    tcs = getattr(msg, "tool_calls", None)
    if not tcs:
        return out
    dumped: List[Dict[str, Any]] = []
    for tc in tcs:
        if isinstance(tc, dict):
            dumped.append(tc)
        elif hasattr(tc, "model_dump"):
            d = tc.model_dump()
            if isinstance(d, dict):
                dumped.append(d)
        else:
            fn = getattr(tc, "function", None)
            dumped.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": (
                            (getattr(fn, "arguments", None) or "{}") if fn else "{}"
                        ),
                    },
                }
            )
    out["tool_calls"] = dumped
    return out


def parse_openai_sync_response(response: Any) -> GenerationParseResult:
    """
    Parse OpenAI SDK chat completion response (sync) to GenerationParseResult.
    """
    custom_id = ""
    try:
        if not response or not response.choices:
            return GenerationParseResult.error(
                custom_id, "no choices in completion response"
            )
        d = _message_to_dict(response.choices[0].message)
        return parse_message_for_format_answer(d, custom_id=custom_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("parse_openai_sync_response: %s", e)
        return GenerationParseResult.error(custom_id, f"sync parse error: {e!s}")
