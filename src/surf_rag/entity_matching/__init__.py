"""Lexicon / alias query entity extraction (lazy public exports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "LexiconAliasEntityPipeline",
    "PhraseSource",
    "RawPhraseMatch",
    "SeedCandidate",
]


def __getattr__(name: str) -> Any:
    if name == "LexiconAliasEntityPipeline":
        from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline

        return LexiconAliasEntityPipeline
    if name == "PhraseSource":
        from surf_rag.entity_matching.types import PhraseSource

        return PhraseSource
    if name == "RawPhraseMatch":
        from surf_rag.entity_matching.types import RawPhraseMatch

        return RawPhraseMatch
    if name == "SeedCandidate":
        from surf_rag.entity_matching.types import SeedCandidate

        return SeedCandidate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from surf_rag.entity_matching.pipeline import LexiconAliasEntityPipeline
    from surf_rag.entity_matching.types import (
        PhraseSource,
        RawPhraseMatch,
        SeedCandidate,
    )
