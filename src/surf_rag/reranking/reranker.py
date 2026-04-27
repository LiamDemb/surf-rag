"""Cross-encoder and no-op rerankers over ``RetrievalResult``."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from surf_rag.core.model_cache import get_cross_encoder
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, result: RetrievalResult, top_k: int
    ) -> RetrievalResult: ...


class NoOpReranker:
    """Keeps retrieval order; only truncates to ``top_k``."""

    def rerank(
        self, query: str, result: RetrievalResult, top_k: int
    ) -> RetrievalResult:
        _ = query
        if result.status != "OK" or not result.chunks or top_k <= 0:
            return result
        chunks = result.chunks[:top_k]
        out_chunks: list[RetrievedChunk] = []
        for i, ch in enumerate(chunks):
            meta = dict(ch.metadata or {})
            meta.setdefault("rerank_score", float(ch.score))
            out_chunks.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    text=ch.text,
                    score=float(ch.score),
                    rank=i,
                    metadata=meta,
                )
            )
        return RetrievalResult(
            query=result.query,
            retriever_name=result.retriever_name,
            status="OK",
            chunks=out_chunks,
            latency_ms=dict(result.latency_ms),
            error=result.error,
            debug_info=dict(result.debug_info) if result.debug_info else None,
        )


class CrossEncoderReranker:
    """SentenceTransformers cross-encoder scores query-passage pairs."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> None:
        self._model = get_cross_encoder(model_name)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, result: RetrievalResult, top_k: int
    ) -> RetrievalResult:
        if result.status != "OK" or not result.chunks or top_k <= 0:
            return result
        pairs = [[query, ch.text or ""] for ch in result.chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        order = sorted(
            range(len(result.chunks)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:top_k]
        out_chunks: list[RetrievedChunk] = []
        for rank, idx in enumerate(order):
            ch = result.chunks[idx]
            sc = float(scores[idx])
            meta = dict(ch.metadata or {})
            meta["rerank_score"] = sc
            meta["retrieval_score"] = float(ch.score)
            out_chunks.append(
                RetrievedChunk(
                    chunk_id=ch.chunk_id,
                    text=ch.text,
                    score=sc,
                    rank=rank,
                    metadata=meta,
                )
            )
        lat = dict(result.latency_ms)
        return RetrievalResult(
            query=result.query,
            retriever_name=f"{result.retriever_name}+rerank",
            status="OK",
            chunks=out_chunks,
            latency_ms=lat,
            error=result.error,
            debug_info=dict(result.debug_info) if result.debug_info else None,
        )


RerankerKind = Literal["none", "cross_encoder"]


def build_reranker(
    kind: str,
    *,
    cross_encoder_model: str | None = None,
) -> Reranker:
    k = (kind or "cross_encoder").strip().lower()
    if k in ("none", "noop", "no-op"):
        return NoOpReranker()
    if k in ("cross_encoder", "cross-encoder", "ce"):
        return CrossEncoderReranker(cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL)
    raise ValueError(f"unknown reranker kind {kind!r}")
