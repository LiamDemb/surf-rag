"""Datatypes for lexicon/alias query entity extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class PhraseSource(str, Enum):
    """Source of a matchable phrase, for ranking after df-based filtering."""

    REDIRECT_ALIAS = "redirect_alias"
    CURATED_ALIAS = "curated_alias"
    SURFACE = "surface"
    CANONICAL = "canonical"


@dataclass(frozen=True, slots=True)
class PhraseRecord:
    """One registered phrase in the match dictionary merged with metadata."""

    match_key: str
    """Lowercase, match-normalized substring used for trie / substring search."""

    canonical_norm: str
    """Entity key aligned with graph nodes ``E:{norm}`` after resolver pass."""

    source: PhraseSource
    """Where this phrase came from (used for tie-breaking)."""

    df: int
    """Chunk frequency of the entity from the lexicon (0 if unknown)."""


@dataclass(frozen=True, slots=True)
class RawPhraseMatch:
    """A phrase matched in the query before df filter and deduplication."""

    start: int
    end: int
    match_key: str
    canonical_norm: str
    source: PhraseSource
    df: int


@dataclass(frozen=True, slots=True)
class FilteredEntity:
    """One entity after df filter and before final cap."""

    canonical_norm: str
    source: PhraseSource
    df: int
    span_len: int
    start: int
    end: int
    match_key: str = ""


@dataclass(frozen=True, slots=True)
class SeedCandidate:
    """Structured lexicon/vector seed with evidence preserved for graph-v03."""

    canonical_norm: str
    matched_text: str
    start: int
    end: int
    span_token_count: int
    df: int
    source: PhraseSource
    match_key: str
    node_id: Optional[str] = None
    vector_score: Optional[float] = None
    graph_present: bool = False
