"""Lightweight query / phrase normalization for dictionary matching.

Preserves leading articles inside multi-word titles (e.g. "The Hunger Games")
unlike :func:`normalize_key`, which strips leading ``the``/``a``/``an``.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%])")
_SPACE_AFTER_OPEN_BRACKET_RE = re.compile(r"([(\[{])\s+")
_SPACE_BEFORE_CLOSE_BRACKET_RE = re.compile(r"\s+([)\]}])")


def normalize_for_query_match(text: str) -> str:
    """Normalize text for substring matching against the phrase inventory.

    - Unicode NFKC
    - Lowercase (casefold)
    - Collapse whitespace
    - Light punctuation spacing cleanup (aligned with benchmark alignment helpers)

    Does **not** strip leading articles, so title-style entities stay matchable.
    """
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.casefold()
    s = s.replace("|", " ")
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _SPACE_AFTER_OPEN_BRACKET_RE.sub(r"\1", s)
    s = _SPACE_BEFORE_CLOSE_BRACKET_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    return s
