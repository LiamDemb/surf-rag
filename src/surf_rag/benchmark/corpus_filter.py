from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?%])")
_SPACE_AFTER_OPEN_BRACKET_RE = re.compile(r"([(\[{])\s+")
_SPACE_BEFORE_CLOSE_BRACKET_RE = re.compile(r"\s+([)\]}])")


def normalize_for_matching(text: str) -> str:
    s = str(text or "").casefold()
    s = s.replace("|", " ")
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _SPACE_AFTER_OPEN_BRACKET_RE.sub(r"\1", s)
    s = _SPACE_BEFORE_CLOSE_BRACKET_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_for_matching_compact(text: str) -> str:
    """Whitespace-insensitive normalized form for fallback containment checks."""
    return _WS_RE.sub("", normalize_for_matching(text))


def contains_normalized(haystack: str, needle: str) -> bool:
    """Two-stage containment: normalized, then whitespace-insensitive fallback."""
    norm_h = normalize_for_matching(haystack)
    norm_n = normalize_for_matching(needle)
    if not norm_h or not norm_n:
        return False
    if norm_n in norm_h:
        return True
    compact_h = normalize_for_matching_compact(norm_h)
    compact_n = normalize_for_matching_compact(norm_n)
    return bool(compact_h and compact_n and compact_n in compact_h)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped: int = 0
    dropped_by_source: Dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.dropped_by_source is None:
            self.dropped_by_source = {}


class CorpusSentenceMatcher:
    def __init__(self, corpus_rows: Sequence[dict]) -> None:
        self._normalized_chunks: List[str] = []
        self._cache: Dict[str, bool] = {}
        for row in corpus_rows:
            text = normalize_for_matching(row.get("text", ""))
            if text:
                self._normalized_chunks.append(text)

    def sentence_found(self, sentence: str) -> bool:
        normalized = normalize_for_matching(sentence)
        if not normalized:
            return False
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        found = any(normalized in chunk for chunk in self._normalized_chunks)
        self._cache[normalized] = found
        return found


def _should_keep_nq(
    support_sentences: Sequence[str], matcher: CorpusSentenceMatcher
) -> bool:
    return any(matcher.sentence_found(s) for s in support_sentences)


def _should_keep_2wiki(
    support_sentences: Sequence[str], matcher: CorpusSentenceMatcher
) -> bool:
    return bool(support_sentences) and all(
        matcher.sentence_found(s) for s in support_sentences
    )


def should_keep_benchmark_row(row: dict, matcher: CorpusSentenceMatcher) -> bool:
    source = str(row.get("dataset_source", "")).strip().lower()
    support_sentences = row.get("gold_support_sentences", [])
    if not isinstance(support_sentences, list):
        support_sentences = []
    support_sentences = [str(s).strip() for s in support_sentences if str(s).strip()]
    if source == "nq":
        return _should_keep_nq(support_sentences, matcher)
    if source == "2wiki":
        return _should_keep_2wiki(support_sentences, matcher)
    # Conservative default for unknown sources.
    return bool(support_sentences) and all(
        matcher.sentence_found(s) for s in support_sentences
    )


def filter_benchmark_rows(
    benchmark_rows: Sequence[dict], corpus_rows: Sequence[dict]
) -> Tuple[List[dict], FilterStats]:
    matcher = CorpusSentenceMatcher(corpus_rows)
    kept: List[dict] = []
    stats = FilterStats(total=len(benchmark_rows))
    for row in benchmark_rows:
        source = str(row.get("dataset_source", "unknown")).strip().lower() or "unknown"
        if should_keep_benchmark_row(row, matcher):
            kept.append(row)
            continue
        stats.dropped += 1
        stats.dropped_by_source[source] = stats.dropped_by_source.get(source, 0) + 1
    stats.kept = len(kept)
    return kept, stats
