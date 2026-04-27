"""Unit tests for sentence-level cross-encoder reranking."""

from __future__ import annotations

from surf_rag.reranking.sentence_reranker import (
    PROMPT_EVIDENCE_KEY,
    PROMPT_EVIDENCE_SENTENCE_SHORTLIST,
    apply_sentence_rerank,
)
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


class _FakeCE:
    def __init__(self, scores: list[float] | None = None) -> None:
        self._scores = scores
        self.pairs: list[list[str]] = []

    def predict(
        self, pairs: list[list[str]], show_progress_bar: bool = False
    ) -> list[float]:
        self.pairs = pairs
        if self._scores is not None:
            return self._scores
        return [float(i) for i in range(len(pairs))]


def test_apply_sentence_rerank_ranks_higher_score_first(monkeypatch) -> None:
    monkeypatch.setattr(
        "surf_rag.reranking.sentence_reranker.get_cross_encoder",
        lambda _model: _FakeCE(),
    )
    ch = RetrievedChunk(
        chunk_id="c1",
        text="First sentence. Second sentence is longer. Third one.",
        score=0.5,
        rank=0,
        metadata={"branch": "dense", "title": "T", "rerank_score": 2.0},
    )
    base = RetrievalResult(
        query="q",
        retriever_name="Fused+rerank",
        status="OK",
        chunks=[ch],
        latency_ms={},
    )
    out = apply_sentence_rerank(
        "What?",
        base,
        cross_encoder_model="any",
        top_k=2,
        max_sentences=20,
        max_words=200,
    )
    assert out.retriever_name.endswith("+sentence_rerank")
    assert (out.debug_info or {}).get(PROMPT_EVIDENCE_KEY) == PROMPT_EVIDENCE_SENTENCE_SHORTLIST
    assert len(out.chunks) >= 2
    # FakeCE returns ascending by pair index: last sentence in split order has highest index/score
    assert out.chunks[0].score >= out.chunks[1].score
    m0 = out.chunks[0].metadata
    assert m0.get("evidence_kind") == "sentence"
    assert m0.get("parent_chunk_id") == "c1"


def test_apply_sentence_rerank_no_op_when_not_ok() -> None:
    base = RetrievalResult(
        query="q",
        retriever_name="Fused",
        status="NO_CONTEXT",
        chunks=[],
        latency_ms={},
    )
    out = apply_sentence_rerank(
        "q",
        base,
        cross_encoder_model="any",
    )
    assert out is base


def test_apply_sentence_rerank_ignores_top_k_and_keeps_all_sentences(monkeypatch) -> None:
    monkeypatch.setattr(
        "surf_rag.reranking.sentence_reranker.get_cross_encoder",
        lambda _model: _FakeCE(),
    )
    base = RetrievalResult(
        query="q",
        retriever_name="Fused+rerank",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="First sentence. Second sentence! Third sentence?",
                score=1.0,
                rank=0,
                metadata={"branch": "dense"},
            )
        ],
        latency_ms={},
    )
    out_small = apply_sentence_rerank(
        "q",
        base,
        cross_encoder_model="any",
        top_k=1,
        max_sentences=20,
        max_words=1000,
    )
    out_large = apply_sentence_rerank(
        "q",
        base,
        cross_encoder_model="any",
        top_k=999,
        max_sentences=20,
        max_words=1000,
    )
    assert len(out_small.chunks) == len(out_large.chunks)
    assert len(out_small.chunks) >= 2
