"""Custom IDs for answerability audit OpenAI Batch lines."""

from __future__ import annotations

PREFIX = "answerability::"


def make_answerability_custom_id(question_id: str) -> str:
    safe = question_id.replace("::", "__COLON__")
    return f"{PREFIX}{safe}"


def parse_answerability_custom_id(custom_id: str) -> str | None:
    if not str(custom_id).startswith(PREFIX):
        return None
    rest = str(custom_id)[len(PREFIX) :]
    return rest.replace("__COLON__", "::")
