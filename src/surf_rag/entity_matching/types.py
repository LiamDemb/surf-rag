"""Datatypes for lexicon/alias query entity extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


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
