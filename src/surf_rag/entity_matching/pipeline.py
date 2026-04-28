"""High-level lexicon + alias exact-match pipeline for query entity strings."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from surf_rag.core.entity_alias_resolver import EntityAliasResolver
from surf_rag.entity_matching.artifacts import try_load_precomputed_matcher
from surf_rag.entity_matching.filters import (
    _SOURCE_RANK,
    rank_and_cap,
    resolve_and_filter,
)
from surf_rag.entity_matching.matcher import (
    PhraseMatcher,
    build_phrase_records,
    greedy_nonoverlapping_matches,
    records_to_matcher,
)
from surf_rag.entity_matching.normalization import normalize_for_query_match
from surf_rag.entity_matching.types import FilteredEntity, SeedCandidate

logger = logging.getLogger(__name__)


@dataclass
class LexiconAliasEntityPipeline:
    """
    Exact phrase match against ``alias_map.json`` + ``entity_lexicon.parquet``, then
    ``df`` filter and deduplication, without LLM calls.

    Aligned with graph node ids as ``E:{norm}`` after
    :meth:`EntityAliasResolver.normalize` on the matched span.
    """

    output_dir: str
    resolver: EntityAliasResolver
    matcher: PhraseMatcher
    max_df: int = 8
    max_entities_per_query: int = 12
    min_match_key_len: int = 3

    @classmethod
    def from_artifacts(
        cls,
        output_dir: str,
        *,
        max_df: int | None = None,
        max_entities_per_query: int | None = None,
        min_match_key_len: int | None = None,
    ) -> "LexiconAliasEntityPipeline":
        odir = str(output_dir)
        resolver = EntityAliasResolver.from_artifacts(output_dir=odir)
        matcher = try_load_precomputed_matcher(Path(odir))
        if matcher is None:
            logger.warning(
                "Building phrase matcher from alias_map + entity_lexicon under %s. "
                "For faster startup, run: python -m scripts.build_entity_matching_artifacts "
                "--corpus-dir %s",
                odir,
                odir,
            )
            _, records = build_phrase_records(odir)
            matcher = records_to_matcher(records)
        return cls(
            output_dir=odir,
            resolver=resolver,
            matcher=matcher,
            max_df=int(
                max_df if max_df is not None else os.getenv("ENTITY_MATCH_MAX_DF", "8")
            ),
            max_entities_per_query=int(
                max_entities_per_query
                if max_entities_per_query is not None
                else os.getenv("ENTITY_MATCH_MAX_PER_QUERY", "12")
            ),
            min_match_key_len=int(
                min_match_key_len
                if min_match_key_len is not None
                else os.getenv("ENTITY_MATCH_MIN_KEY_LEN", "3")
            ),
        )

    def extract(self, query: str) -> List[str]:
        """Return canonical entity norms in match-quality order (span, df, source)."""
        qn = normalize_for_query_match(query)
        raw = greedy_nonoverlapping_matches(qn, self.matcher)
        filtered = resolve_and_filter(
            raw,
            self.resolver,
            qn,
            max_df=self.max_df,
            min_match_key_len=self.min_match_key_len,
        )
        return rank_and_cap(filtered, max_count=self.max_entities_per_query)

    def extract_filtered_entities(
        self, query: str, *, soft_df: bool = False
    ) -> List[FilteredEntity]:
        """Return filtered entities in rank order (span, df, source), capped."""
        qn = normalize_for_query_match(query)
        raw = greedy_nonoverlapping_matches(qn, self.matcher)
        md = None if soft_df else self.max_df
        filtered = resolve_and_filter(
            raw,
            self.resolver,
            qn,
            max_df=md,
            min_match_key_len=self.min_match_key_len,
        )
        ordered = sorted(
            filtered,
            key=lambda e: (
                -e.span_len,
                e.df,
                _SOURCE_RANK[e.source],
                e.start,
                e.canonical_norm,
            ),
        )
        return ordered[: self.max_entities_per_query]

    def extract_candidates(
        self, query: str, *, soft_df: bool = False
    ) -> List[SeedCandidate]:
        """Structured seeds with spans for graph-v03; ``soft_df`` skips lexicon DF hard-drop."""
        qn = normalize_for_query_match(query)
        entities = self.extract_filtered_entities(query, soft_df=soft_df)
        out: List[SeedCandidate] = []
        for fe in entities:
            span_text = qn[fe.start : fe.end]
            tokens = [t for t in span_text.split() if t]
            mk = fe.match_key or fe.canonical_norm.casefold().replace(" ", "_")
            out.append(
                SeedCandidate(
                    canonical_norm=fe.canonical_norm,
                    matched_text=span_text,
                    start=fe.start,
                    end=fe.end,
                    span_token_count=max(1, len(tokens)),
                    df=fe.df,
                    source=fe.source,
                    match_key=mk,
                )
            )
        return out
