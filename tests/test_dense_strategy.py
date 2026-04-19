"""Tests for DenseRetriever."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import faiss
import numpy as np
import pandas as pd

from surf_rag.core.index_store import FaissIndexStore
from surf_rag.core.mapping import VectorMetaMapper
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.strategies.dense import DenseRetriever


def _make_tiny_index_and_meta(tmp_path: Path, dim: int = 384):
    """Build a minimal FAISS index + metadata for tests."""
    rng = np.random.default_rng(42)
    n = 5
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    idx_path = tmp_path / "vector_index.faiss"
    faiss.write_index(index, str(idx_path))
    meta = pd.DataFrame(
        [
            {"row_id": i, "chunk_id": f"c{i}", "year_min": None, "year_max": None}
            for i in range(n)
        ]
    )
    meta_path = tmp_path / "vector_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    return str(idx_path), str(meta_path)


class MockCorpus:
    """Minimal ChunkIdToText implementation for tests."""

    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


def test_retriever_returns_valid_retrieval_result(tmp_path):
    """DenseRetriever returns a RetrievalResult with correct shape and latency keys."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec

    corpus = MockCorpus({f"c{i}": f"text for chunk {i}" for i in range(5)})

    retriever = DenseRetriever(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=mock_embedder,
        corpus=corpus,
        top_k=3,
    )

    result = retriever.retrieve("What is the capital of France?")

    assert isinstance(result, RetrievalResult)
    assert result.status == "OK"
    assert len(result.chunks) > 0
    assert len(result.chunks) <= 3
    assert "retrieval" in result.latency_ms
    assert "total" in result.latency_ms
    assert result.latency_ms["total"] >= result.latency_ms["retrieval"]
    assert result.error is None


def test_chunks_sorted_descending_by_score(tmp_path):
    """Chunks are sorted by score descending (via RetrievalResult)."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec

    corpus = MockCorpus({f"c{i}": f"chunk {i}" for i in range(5)})

    retriever = DenseRetriever(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=mock_embedder,
        corpus=corpus,
        top_k=5,
    )

    result = retriever.retrieve("test query")
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_no_context_when_corpus_returns_empty(tmp_path):
    """When corpus returns no valid chunk texts, status is NO_CONTEXT."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec

    mock_corpus = MagicMock()
    mock_corpus.get_text.return_value = None

    retriever = DenseRetriever(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=mock_embedder,
        corpus=mock_corpus,
        top_k=5,
    )

    result = retriever.retrieve("test query")
    assert result.status == "NO_CONTEXT"
    assert result.chunks == []
    assert result.error is None
