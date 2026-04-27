from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from surf_rag.core.embedder import Embedder
from surf_rag.core.index_store import FaissIndexStore
from surf_rag.core.mapping import (
    ChunkIdToText,
    RowIdToChunkId,
    metadata_from_corpus_record,
)
from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


@dataclass
class DenseRetriever(BranchRetriever):
    """Dense vector retrieval over a FAISS index."""

    name = "Dense"
    index_store: FaissIndexStore = None  # type: ignore[assignment]
    meta: RowIdToChunkId = None  # type: ignore[assignment]
    embedder: Embedder = None  # type: ignore[assignment]
    corpus: ChunkIdToText = None  # type: ignore[assignment]
    top_k: int = 10

    _index: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = self.index_store.load()

    def retrieve(self, query: str, **kwargs) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        try:
            r0 = time.perf_counter()
            self._ensure_loaded()

            query_vector = self.embedder.embed_query(query)
            query_matrix = np.expand_dims(query_vector, axis=0)

            scores, row_ids = self._index.search(query_matrix, self.top_k)  # type: ignore[union-attr]
            scores = scores[0].tolist()
            row_ids = row_ids[0].tolist()

            chunks: List[RetrievedChunk] = []
            for score, row_id in zip(scores, row_ids):
                if row_id is None or int(row_id) < 0:
                    continue
                chunk_id = self.meta.row_to_chunk(int(row_id))
                if not chunk_id:
                    continue
                text = self.corpus.get_text(chunk_id)
                if not text:
                    continue
                meta: Dict[str, Any] = {"branch": "dense"}
                rec = (
                    self.corpus.get_record(chunk_id)
                    if hasattr(self.corpus, "get_record")
                    else None
                )
                meta.update(metadata_from_corpus_record(rec))
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        score=float(score),
                        rank=0,
                        metadata=meta,
                    )
                )

            timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
            timings["total"] = (time.perf_counter() - t0) * 1000.0

            if not chunks:
                return RetrievalResult(
                    query=query,
                    retriever_name=self.name,
                    status="NO_CONTEXT",
                    chunks=[],
                    latency_ms=timings,
                )

            return RetrievalResult(
                query=query,
                retriever_name=self.name,
                status="OK",
                chunks=chunks,
                latency_ms=timings,
            )

        except Exception as e:
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                query=query,
                retriever_name=self.name,
                status="ERROR",
                chunks=[],
                latency_ms=timings,
                error=(
                    f"DenseRetriever failed during retrieval: "
                    f"{type(e).__name__}: {e}"
                ),
            )
