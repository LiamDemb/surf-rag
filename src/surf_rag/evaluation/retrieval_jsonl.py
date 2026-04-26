"""Serialize RetrievalResult to JSON lines for evaluation runs."""

from __future__ import annotations

import json
from typing import Any, Dict

from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def dict_to_retrieval_result(row: dict) -> RetrievalResult:
    """Rebuild :class:`RetrievalResult` from :func:`retrieval_result_to_dict` JSON."""
    chunks: list[RetrievedChunk] = []
    for c in row.get("chunks") or []:
        if not isinstance(c, dict):
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=str(c.get("chunk_id", "") or ""),
                text=str(c.get("text", "") or ""),
                score=float(c.get("score", 0.0) or 0.0),
                rank=int(c.get("rank", 0) or 0),
                metadata=dict(c.get("metadata") or {}),
            )
        )
    return RetrievalResult(
        query=str(row.get("query", "") or ""),
        retriever_name=str(row.get("retriever_name", "") or ""),
        status=str(row.get("status", "") or ""),
        chunks=chunks,
        latency_ms=dict(row.get("latency_ms") or {}),
        error=row.get("error"),
        debug_info=row.get("debug_info"),
    )


def retrieval_result_to_dict(
    result: RetrievalResult, question_id: str, extra: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "question_id": question_id,
        "query": result.query,
        "retriever_name": result.retriever_name,
        "status": result.status,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "score": c.score,
                "rank": c.rank,
                "metadata": dict(c.metadata),
            }
            for c in result.chunks
        ],
        "latency_ms": dict(result.latency_ms),
        "error": result.error,
        "debug_info": result.debug_info,
    }
    if extra:
        payload.update(extra)
    return payload


def write_retrieval_line(f, result: RetrievalResult, question_id: str) -> None:
    row = retrieval_result_to_dict(result, question_id)
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
