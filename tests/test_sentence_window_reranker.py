"""Unit tests for sentence-window context selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from surf_rag.reranking.sentence_windows import (
    SentenceWindowConfig,
    SentenceWindowReranker,
)
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def _chunk(cid: str, text: str, score: float, **meta: object) -> RetrievedChunk:
    m = dict(meta)
    return RetrievedChunk(chunk_id=cid, text=text, score=score, rank=0, metadata=m)


@patch("surf_rag.reranking.sentence_windows.get_cross_encoder")
def test_sentence_window_preserves_prompt_order_via_selection_score(
    mock_get_ce: MagicMock,
) -> None:
    """Selection score decreases so after RetrievalResult sort, order is coverage then fill."""
    mock_model = MagicMock()
    # One sentence per chunk => one window each => two scores
    mock_model.predict.return_value = [0.5, 0.9]
    mock_get_ce.return_value = mock_model
    cfg = SentenceWindowConfig(
        radius=0,
        max_windows=4,
        min_windows=1,
        max_words=5000,
        max_subwindow_words=500,
        min_top_chunk_coverage=2,
        min_distinct_parent_chunks=2,
        max_windows_per_chunk=2,
        merge_overlaps=True,
        duplicate_filter=False,
        include_title=False,
    )
    sw = SentenceWindowReranker(model_name="dummy", config=cfg)
    r = RetrievalResult(
        query="q",
        retriever_name="Fused",
        status="OK",
        chunks=[
            _chunk("a", "First only.", 1.0),
            _chunk("b", "Second only.", 0.9),
        ],
    )
    out = sw.rerank("q", r, top_k=10)
    assert out.retriever_name.endswith("+sw")
    assert len(out.chunks) >= 1
    for i, ch in enumerate(out.chunks):
        assert ch.metadata.get("context_unit") == "sentence_window"
        assert "window_rerank_score" in ch.metadata
        assert float(ch.score) == float(len(out.chunks) - i)


@patch("surf_rag.reranking.sentence_windows.get_cross_encoder")
def test_build_reranker_sentence_window_kind(mock_get_ce: MagicMock) -> None:
    from surf_rag.reranking.reranker import build_reranker

    mock_get_ce.return_value = MagicMock()
    mock_get_ce.return_value.predict.return_value = [1.0]
    r = build_reranker(
        "sentence_window",
        cross_encoder_model="m",
        sentence_window_config=SentenceWindowConfig(
            max_windows=1,
            min_windows=0,
            min_top_chunk_coverage=0,
            min_distinct_parent_chunks=1,
            max_words=100,
            max_subwindow_words=80,
            include_title=False,
        ),
    )
    assert isinstance(r, SentenceWindowReranker)
    rr = RetrievalResult(
        query="q",
        retriever_name="Fused",
        status="OK",
        chunks=[_chunk("a", "Only.", 1.0)],
    )
    out = r.rerank("q", rr, top_k=5)
    assert len(out.chunks) == 1
