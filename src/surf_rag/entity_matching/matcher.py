"""Build phrase inventory and find longest-prefix greedy matches in normalized queries."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from surf_rag.core.alias_map import CURATED_ALIASES, normalize_alias_map
from surf_rag.core.enrich_entities import normalize_key
from surf_rag.core.entity_alias_resolver import EntityAliasResolver, _iter_surface_forms
from surf_rag.entity_matching.normalization import normalize_for_query_match
from surf_rag.entity_matching.types import PhraseRecord, PhraseSource, RawPhraseMatch

logger = logging.getLogger(__name__)

# Stronger (lower int) = higher priority for tie-breaks after ``df`` and span length
_SOURCE_RANK: Dict[PhraseSource, int] = {
    PhraseSource.REDIRECT_ALIAS: 0,
    PhraseSource.CURATED_ALIAS: 1,
    PhraseSource.SURFACE: 2,
    PhraseSource.CANONICAL: 3,
}


@dataclass
class _TrieNode:
    children: Dict[str, _TrieNode] = field(default_factory=dict)
    # Best payload at a word ending (deduped by our merge rules)
    best: Optional[PhraseRecord] = None


@dataclass
class PhraseMatcher:
    """Aho-style forward trie over match-normalized lowercase strings (character-wise)."""

    root: _TrieNode = field(default_factory=_TrieNode)

    def insert(self, record: PhraseRecord) -> None:
        if not record.match_key:
            return
        node = self.root
        for ch in record.match_key:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        node.best = _better_record(node.best, record)

    def longest_from(self, text: str, i: int) -> Optional[Tuple[int, PhraseRecord]]:
        """Return (end_exclusive, record) for the longest dictionary word starting at ``i``."""
        node = self.root
        best_end: Optional[int] = None
        best_rec: Optional[PhraseRecord] = None
        for j in range(i, len(text)):
            ch = text[j]
            if ch not in node.children:
                break
            node = node.children[ch]
            if node.best is not None:
                best_end = j + 1
                best_rec = node.best
        if best_end is None or best_rec is None:
            return None
        return best_end, best_rec


def _better_record(a: Optional[PhraseRecord], b: PhraseRecord) -> PhraseRecord:
    if a is None:
        return b
    if b.df != a.df:
        return a if a.df < b.df else b
    if _SOURCE_RANK[b.source] != _SOURCE_RANK[a.source]:
        return a if _SOURCE_RANK[a.source] < _SOURCE_RANK[b.source] else b
    return a


def _register(
    by_key: Dict[str, PhraseRecord],
    match_key: str,
    rec: PhraseRecord,
) -> None:
    if not (match_key and match_key.strip()):
        return
    mk = match_key.strip()
    existing = by_key.get(mk)
    if existing is None:
        by_key[mk] = rec
        return
    by_key[mk] = _better_record(existing, rec)


def build_phrase_records(
    output_dir: str,
) -> Tuple[Dict[str, int], List[PhraseRecord]]:
    """
    Load ``alias_map.json`` and ``entity_lexicon.parquet`` and emit phrase records
    for exact matching, with ``df`` from the lexicon.
    """
    out = Path(output_dir)
    alias_path = out / "alias_map.json"
    lexicon_path = out / "entity_lexicon.parquet"

    if not alias_path.exists():
        raise FileNotFoundError(f"alias_map.json not found under {out}")
    if not lexicon_path.exists():
        raise FileNotFoundError(f"entity_lexicon.parquet not found under {out}")

    with alias_path.open("r", encoding="utf-8") as f:
        raw_alias = json.load(f)
    if not isinstance(raw_alias, dict):
        raise ValueError("alias_map.json must be a JSON object")

    df_map = EntityAliasResolver.load_df_map_from_lexicon(lexicon_path.as_posix())

    by_key: Dict[str, PhraseRecord] = {}

    for k, v in raw_alias.items():
        v_norm = normalize_key(str(v))
        if not v_norm:
            continue
        mk = normalize_for_query_match(str(k))
        if not mk:
            continue
        rec = PhraseRecord(
            match_key=mk,
            canonical_norm=v_norm,
            source=PhraseSource.REDIRECT_ALIAS,
            df=int(df_map.get(v_norm, 0)),
        )
        _register(by_key, mk, rec)

    for k, v in normalize_alias_map(CURATED_ALIASES).items():
        mk = (
            normalize_for_query_match(k)
            if not isinstance(k, str)
            else normalize_for_query_match(k)
        )
        if not mk:
            continue
        rec = PhraseRecord(
            match_key=mk,
            canonical_norm=v,
            source=PhraseSource.CURATED_ALIAS,
            df=int(df_map.get(v, 0)),
        )
        _register(by_key, mk, rec)

    df_lex = pd.read_parquet(lexicon_path, columns=["norm", "surface_forms", "df"])
    for _, row in df_lex.iterrows():
        norm_value = normalize_key(str(row.get("norm", "") or ""))
        if not norm_value:
            continue
        try:
            dfi = int(row.get("df", 0) or 0)
        except (TypeError, ValueError):
            dfi = 0
        n_raw = str(row.get("norm", "") or "").strip()
        if n_raw:
            mkn = normalize_for_query_match(n_raw)
            if mkn:
                _register(
                    by_key,
                    mkn,
                    PhraseRecord(
                        match_key=mkn,
                        canonical_norm=norm_value,
                        source=PhraseSource.CANONICAL,
                        df=dfi,
                    ),
                )
        for surf in _iter_surface_forms(row.get("surface_forms")):
            mks = normalize_for_query_match(surf)
            if not mks:
                continue
            _register(
                by_key,
                mks,
                PhraseRecord(
                    match_key=mks,
                    canonical_norm=norm_value,
                    source=PhraseSource.SURFACE,
                    df=dfi,
                ),
            )

    records = list(by_key.values())
    logger.info(
        "Built %d unique match keys from lexicon+alias (output_dir=%s)",
        len(records),
        out,
    )
    return df_map, records


def records_to_matcher(records: List[PhraseRecord]) -> PhraseMatcher:
    m = PhraseMatcher()
    for r in records:
        m.insert(r)
    return m


def _spans_alnum_morpheme_boundary(s: str, start: int, end: int) -> bool:
    """True if [start, end) does not start/end inside a longer alphanumeric run."""
    if start < 0 or end > len(s) or start >= end:
        return False
    if start > 0 and s[start - 1].isalnum() and s[start].isalnum():
        return False
    if end < len(s) and s[end - 1].isalnum() and s[end].isalnum():
        return False
    return True


def greedy_nonoverlapping_matches(
    query_norm: str,
    matcher: PhraseMatcher,
) -> List[RawPhraseMatch]:
    """
    Left-to-right scan: at each unvisited index, take the longest trie match.
    Produces a non-overlapping set with locally longest prefixes (standard maximal munch).
    """
    if not query_norm:
        return []
    n = len(query_norm)
    out: List[RawPhraseMatch] = []
    i = 0
    while i < n:
        # Skip non-word-start if we only want to match at boundaries? v1: allow match anywhere
        if query_norm[i].isspace():
            i += 1
            continue
        result = matcher.longest_from(query_norm, i)
        if result is None:
            i += 1
            continue
        end, rec = result
        if not _spans_alnum_morpheme_boundary(query_norm, i, end):
            i += 1
            continue
        out.append(
            RawPhraseMatch(
                start=i,
                end=end,
                match_key=rec.match_key,
                canonical_norm=rec.canonical_norm,
                source=rec.source,
                df=rec.df,
            )
        )
        i = end
    return out
