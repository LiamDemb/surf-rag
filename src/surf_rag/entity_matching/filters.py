from __future__ import annotations

from typing import List, Set

from surf_rag.core.entity_alias_resolver import EntityAliasResolver
from surf_rag.entity_matching.types import FilteredEntity, PhraseSource, RawPhraseMatch

_SOURCE_RANK: dict[PhraseSource, int] = {
    PhraseSource.REDIRECT_ALIAS: 0,
    PhraseSource.CURATED_ALIAS: 1,
    PhraseSource.SURFACE: 2,
    PhraseSource.CANONICAL: 3,
}

# Standalone matches that are almost always structural English, not knowledge-graph
# entities, even if they slipped into the lexicon with low ``df``. Multi-word
# entity phrases that merely **contain** these as substrings are not affected.
_STANDALONE_GARBAGE: Set[str] = frozenset(
    s.casefold()
    for s in (
        "a an the and or but if in is it of on at to as by no we he be do me my up "
        "was were are for not with from that this when who whom which what how why "
        "her him his has had may out new can did its now one any all our she own too "
        "few per via etc age ago two six ten n d place s wh f c or b wha "
        "older earlier later first second third born die died death burial spouse "
        "performer director song country earring burial place which film which song "
        "which country where was where is when did what is who are which is film was "
        "which who what how year years month day date age released release "
        "paternal father mother husband wife spouse born study studied same more recently directors"
    ).split()
) | {
    "place of birth",
    "place of death",
    "film was released",
    "the same",
}

_TWO_CHAR_KNOWN_OK: Set[str] = frozenset(
    s.casefold() for s in ("us", "uk", "eu", "un", "up")
)


def _is_spurious_match(m: RawPhraseMatch, min_key_len: int) -> bool:
    """Filter pathological 1–2 char fragments and stand-alone function words."""
    mk = m.match_key.strip()
    n = len(mk)
    if n < 2:
        return True
    if n == 2 and mk not in _TWO_CHAR_KNOWN_OK:
        return True
    if n < min_key_len and not (n == 2 and mk in _TWO_CHAR_KNOWN_OK):
        return True
    if mk in _STANDALONE_GARBAGE:
        return True
    return False


def resolve_and_filter(
    matches: List[RawPhraseMatch],
    resolver: EntityAliasResolver,
    query_matched_text: str,
    max_df: int,
    min_match_key_len: int = 3,
) -> List[FilteredEntity]:
    """
    Map matched spans to canonical norms via the resolver, apply ``df`` hard max,
    then deduplicate by norm keeping the best record per the ranking rules.
    """
    if not matches:
        return []
    cands: List[FilteredEntity] = []
    for m in matches:
        if _is_spurious_match(m, min_match_key_len):
            continue
        span = query_matched_text[m.start : m.end]
        final_norm = resolver.normalize(span) if span else m.canonical_norm
        if not final_norm:
            continue
        dfi = m.df
        if dfi > max_df:
            continue
        cands.append(
            FilteredEntity(
                canonical_norm=final_norm,
                source=m.source,
                df=dfi,
                span_len=m.end - m.start,
                start=m.start,
                end=m.end,
            )
        )
    if not cands:
        return []

    by_norm: dict[str, FilteredEntity] = {}
    for fe in cands:
        ex = by_norm.get(fe.canonical_norm)
        if ex is None or _is_better_entity(fe, ex):
            by_norm[fe.canonical_norm] = fe
    return list(by_norm.values())


def _is_better_entity(a: FilteredEntity, b: FilteredEntity) -> bool:
    if a.span_len != b.span_len:
        return a.span_len > b.span_len
    if a.df != b.df:
        return a.df < b.df
    if _SOURCE_RANK[a.source] != _SOURCE_RANK[b.source]:
        return _SOURCE_RANK[a.source] < _SOURCE_RANK[b.source]
    if a.start != b.start:
        return a.start < b.start
    return False


def rank_and_cap(entities: List[FilteredEntity], max_count: int) -> List[str]:
    """Sort for stable prompt-friendly output, then return up to ``max_count`` norms."""
    if not entities:
        return []
    ordered = sorted(
        entities,
        key=lambda e: (
            -e.span_len,
            e.df,
            _SOURCE_RANK[e.source],
            e.start,
            e.canonical_norm,
        ),
    )
    out: List[str] = []
    seen: Set[str] = set()
    for e in ordered:
        if e.canonical_norm in seen:
            continue
        seen.add(e.canonical_norm)
        out.append(e.canonical_norm)
        if len(out) >= max_count:
            break
    return out
