"""Strip leading model output before the final ``answer:`` marker (any casing)."""

from __future__ import annotations

import re

# Word-boundary so we don't match substrings like "counteranswer:".
# Optional whitespace before colon. Case-insensitive.
_ANSWER_LABEL = re.compile(r"(?i)\banswer\s*:")


def strip_answer_prefix(text: str) -> str:
    """
    If ``text`` contains ``answer:`` (any casing), return only the part after the
    **last** occurrence (CoT often has numbered steps, then a final answer line).

    If there is no such marker, return ``text`` stripped.
    """
    if not text:
        return ""
    matches = list(_ANSWER_LABEL.finditer(text))
    if not matches:
        return text.strip()
    return text[matches[-1].end() :].strip()
