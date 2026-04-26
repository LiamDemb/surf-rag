from unittest.mock import MagicMock, patch

from surf_rag.reranking.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    build_reranker,
)
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def _result() -> RetrievalResult:
    ch = [
        RetrievedChunk(chunk_id="a", text="aa", score=0.1, rank=0, metadata={}),
        RetrievedChunk(chunk_id="b", text="bb", score=0.9, rank=1, metadata={}),
    ]
    return RetrievalResult(
        query="q",
        retriever_name="dense",
        status="OK",
        chunks=ch,
        latency_ms={},
    )


def test_noop_rerank_truncates() -> None:
    r = _result()
    out = NoOpReranker().rerank("q", r, top_k=1)
    assert len(out.chunks) == 1
    # RetrievalResult sorts chunks by score descending
    assert out.chunks[0].chunk_id == "b"


@patch("sentence_transformers.CrossEncoder")
def test_cross_encoder_rerank_order(mock_ce: MagicMock) -> None:
    # Chunks are score-sorted as [b, a]; pairs follow that order.
    mock_ce.return_value.predict.return_value = [0.0, 1.0]
    r = _result()
    ce = CrossEncoderReranker(model_name="dummy-model")
    out = ce.rerank("q", r, top_k=2)
    assert [c.chunk_id for c in out.chunks] == ["a", "b"]
    assert "rerank_score" in (out.chunks[0].metadata or {})


def test_build_reranker_none_alias() -> None:
    r = build_reranker("noop")
    assert isinstance(r, NoOpReranker)
