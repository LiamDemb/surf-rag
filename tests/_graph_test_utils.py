"""Shared stubs and toy graphs for graph retriever tests."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from unittest.mock import MagicMock

from surf_rag.entity_matching.types import PhraseSource, SeedCandidate


class CorpusStub:
    def __init__(self, texts: Dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


class GraphStoreStub:
    def __init__(self, graph):
        self._graph = graph

    def load(self):
        return self._graph


class StaticExtractor:
    """Returns fixed entity norms."""

    def __init__(self, norms: List[str]):
        self._norms = list(norms)

    def extract(self, query: str) -> List[str]:
        return list(self._norms)

    def extract_candidates(
        self, query: str, *, soft_df: bool = False
    ) -> List[SeedCandidate]:
        out: List[SeedCandidate] = []
        pos = 0
        for n in self._norms:
            ln = max(1, len(n))
            out.append(
                SeedCandidate(
                    canonical_norm=n,
                    matched_text=n,
                    start=pos,
                    end=pos + ln,
                    span_token_count=max(1, len(n.split())),
                    df=1,
                    source=PhraseSource.CANONICAL,
                    match_key=n.casefold().replace(" ", "_"),
                )
            )
            pos += ln + 1
        return out


def strategy_embedder():
    """Minimal embedder for score_bundle (384-d vectors)."""
    m = MagicMock()

    def embed_query(text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(384).astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    m.embed_query = MagicMock(side_effect=embed_query)
    return m


def toy_chunks() -> List[dict]:
    """Small two-chunk graph: A causes B, B causes C."""
    return [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [{"norm": "a"}, {"norm": "b"}],
                "relations": [
                    {"subj_norm": "a", "pred": "causes", "obj_norm": "b"},
                ],
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "entities": [{"norm": "b"}, {"norm": "c"}],
                "relations": [
                    {"subj_norm": "b", "pred": "causes", "obj_norm": "c"},
                ],
            },
        },
    ]


def comparison_chunks() -> List[dict]:
    """Single chunk graph for film vs director comparison regression."""
    return [
        {
            "chunk_id": "c_all",
            "metadata": {
                "entities": [
                    {"norm": "valentin the good"},
                    {"norm": "a daughter of two worlds"},
                    {"norm": "martin fric"},
                    {"norm": "james young"},
                ],
                "relations": [
                    {
                        "subj_norm": "valentin the good",
                        "pred": "directed_by",
                        "obj_norm": "martin fric",
                    },
                    {
                        "subj_norm": "a daughter of two worlds",
                        "pred": "directed_by",
                        "obj_norm": "james young",
                    },
                ],
            },
        },
    ]


def comparison_corpus() -> Dict[str, str]:
    return {
        "c_all": (
            "Valentin the Good was directed by Martin Fric. "
            "Martin Fric died on 1968-08-26. "
            "A Daughter of Two Worlds was directed by James Young. "
            "James Young died on 1948-06-09."
        )
    }
