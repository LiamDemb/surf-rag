"""FlashText-based matcher for Wikipedia titles in chunk text.

Pre-scans chunk text to find which wiki titles (or their redirects/aliases)
appear, returning a deduped subset for LLM seed anchoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .enrich_entities import normalize_key

try:
    from flashtext import KeywordProcessor
except ImportError:
    KeywordProcessor = None  # type: ignore


def load_wiki_titles(path: Path) -> List[str]:
    """Load wiki titles from jsonl (one title per line as {"title": "..."}) or .txt (one per line)."""
    titles: List[str] = []
    if not path.exists():
        return titles
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                try:
                    obj = json.loads(line)
                    t = obj.get("title") or obj.get("name") or ""
                    if t:
                        titles.append(t)
                except json.JSONDecodeError:
                    continue
            else:
                titles.append(line)
    return list(dict.fromkeys(titles))


def _normalize_for_match(title: str) -> str:
    """Lightweight normalization for matched titles before deduping.

    We keep the original human-readable title, only stripping surrounding
    whitespace so seeds passed to the LLM remain unchanged.
    """
    return (title or "").strip()


def build_wiki_title_matcher(
    titles: Iterable[str],
    alias_map: Optional[Dict[str, str]] = None,
    max_keywords: int = 5000,
) -> "WikiTitleMatcher":
    """Build a matcher from wiki titles and optional alias map.

    alias_map: normalized alias -> normalized canonical. When an alias is found
    in text, the original title string is returned if its norm is in the corpus.
    """
    if KeywordProcessor is None:
        raise ImportError("flashtext is required. Install with: pip install flashtext")

    alias_map = alias_map or {}
    title_list: List[str] = []
    norm_to_original: Dict[str, str] = {}
    for t in titles:
        tt = (t or "").strip()
        if not tt:
            continue
        nn = normalize_key(tt)
        if nn and nn not in norm_to_original:
            norm_to_original[nn] = tt
            title_list.append(tt)

    kp = KeywordProcessor(case_sensitive=False)
    added = 0
    for title in sorted(title_list, key=lambda x: (normalize_key(x), x)):
        if added >= max_keywords:
            break
        kp.add_keyword(title, title)
        added += 1

    for alias_norm, canonical_norm in alias_map.items():
        if added >= max_keywords:
            break
        if canonical_norm not in norm_to_original:
            continue
        original_title = norm_to_original[canonical_norm]
        if alias_norm != canonical_norm:
            kp.add_keyword(alias_norm, original_title)
            added += 1

    return WikiTitleMatcher(kp, list(norm_to_original.values()))


class WikiTitleMatcher:
    """Matches chunk text against wiki titles and returns matched titles."""

    def __init__(
        self, keyword_processor: "KeywordProcessor", titles: List[str]
    ) -> None:
        self._kp = keyword_processor
        self._titles = titles

    def find_titles_in_text(self, text: str, max_results: int = 50) -> List[str]:
        """Return wiki titles (and canonicals for aliases) that appear in text.

        Deduped, ordered by first occurrence, capped at max_results.
        """
        if not text or not text.strip():
            return []
        matches = self._kp.extract_keywords(text)
        seen: Set[str] = set()
        result: List[str] = []
        for m in matches:
            m_clean = _normalize_for_match(m)
            if m_clean and m_clean not in seen:
                seen.add(m_clean)
                result.append(m_clean)
                if len(result) >= max_results:
                    break
        return result


def write_wiki_titles(titles: Iterable[str], path: Path, format: str = "jsonl") -> None:
    """Persist wiki titles to jsonl or txt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(t for t in titles if t and str(t).strip()))
    with path.open("w", encoding="utf-8") as f:
        for t in unique:
            if format == "jsonl":
                f.write(json.dumps({"title": t}, ensure_ascii=False) + "\n")
            else:
                f.write(str(t).strip() + "\n")
