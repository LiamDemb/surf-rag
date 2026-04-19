from __future__ import annotations

import re


def strip_answer_prefix(text: str) -> str:
    """Strip leading 'answer:' markers and surrounding whitespace."""
    if text is None:
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        candidate = lines[-1]
    else:
        candidate = cleaned

    # Remove common "Answer:" / "Final answer:" style prefixes.
    candidate = re.sub(
        r"^(?:final\s+)?answer\s*[:\-]\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    return candidate or cleaned
