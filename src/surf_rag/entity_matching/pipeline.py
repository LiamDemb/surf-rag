"""High-level lexicon + alias exact-match pipeline for query entity strings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from surf_rag.core.entity_alias_resolver import EntityAliasResolver
from surf_rag.entity_matching.filters import rank_and_cap, resolve_and_filter
from surf_rag.entity_matching.matcher import (
    PhraseMatcher,
    build_phrase_records,
    greedy_nonoverlapping_matches,
    records_to_matcher,
)
from surf_rag.entity_matching.normalization import normalize_for_query_match


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
        _, records = build_phrase_records(odir)
        matcher = records_to_matcher(records)
        return cls(
            output_dir=odir,
            resolver=resolver,
            matcher=matcher,
            max_df=int(
                max_df
                if max_df is not None
                else os.getenv("ENTITY_MATCH_MAX_DF", "8")
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
        return rank_and_cap(
            filtered, max_count=self.max_entities_per_query
        )
