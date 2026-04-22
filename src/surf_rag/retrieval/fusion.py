"""Weighted dense+graph fusion over RetrievalResult chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.types import RetrievalResult, RetrievedChunk

FUSED_RETRIEVER_NAME = "Fused"


def min_max_normalize(values: Iterable[float]) -> List[float]:
    """Min-max normalize a sequence of scores into [0.0, 1.0]."""
    xs = [float(v) for v in values]
    if not xs:
        return []
    lo = min(xs)
    hi = max(xs)
    # Degenerate case
    if hi == lo:
        return [1.0 for _ in xs]
    span = hi - lo
    return [(x - lo) / span for x in xs]


def _chunk_lookup(chunks: Iterable[RetrievedChunk]) -> Dict[str, RetrievedChunk]:
    """Map chunk_id -> chunk. Later duplicates (unexpected) overwrite earlier."""
    out: Dict[str, RetrievedChunk] = {}
    for ch in chunks:
        out[ch.chunk_id] = ch
    return out


def _normalized_scores(
    chunks: Iterable[RetrievedChunk],
) -> Dict[str, float]:
    """Return chunk_id -> normalized score for a single branch."""
    items = list(chunks)
    if not items:
        return {}
    norm = min_max_normalize(ch.score for ch in items)
    return {ch.chunk_id: n for ch, n in zip(items, norm)}


@dataclass(frozen=True)
class FusedCandidate:
    """One fused candidate with per-branch provenance."""

    chunk_id: str
    text: str
    dense_raw_score: Optional[float]
    graph_raw_score: Optional[float]
    dense_norm_score: float
    graph_norm_score: float
    fused_score: float
    dense_present: bool
    graph_present: bool
    source_metadata: Dict[str, Any]


def fuse_branch_results(
    dense: RetrievalResult,
    graph: RetrievalResult,
    dense_weight: float,
    fusion_keep_k: int,
) -> List[FusedCandidate]:
    """Merge dense+graph results into a ranked list of fused candidates."""
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError(f"dense_weight must be in [0.0, 1.0], got {dense_weight!r}")
    if fusion_keep_k <= 0:
        raise ValueError(f"fusion_keep_k must be > 0, got {fusion_keep_k!r}")

    graph_weight = 1.0 - dense_weight

    dense_chunks = list(dense.chunks) if dense.status == "OK" else []
    graph_chunks = list(graph.chunks) if graph.status == "OK" else []

    dense_by_id = _chunk_lookup(dense_chunks)
    graph_by_id = _chunk_lookup(graph_chunks)

    dense_norm = _normalized_scores(dense_chunks)
    graph_norm = _normalized_scores(graph_chunks)

    all_ids = list(dict.fromkeys([*dense_by_id.keys(), *graph_by_id.keys()]))
    candidates: List[FusedCandidate] = []

    for cid in all_ids:
        d_ch = dense_by_id.get(cid)
        g_ch = graph_by_id.get(cid)

        d_norm = dense_norm.get(cid, 0.0)
        g_norm = graph_norm.get(cid, 0.0)

        fused = dense_weight * d_norm + graph_weight * g_norm

        text = ""
        source_metadata: Dict[str, Any] = {}
        if d_ch is not None:
            text = d_ch.text
            source_metadata = dict(d_ch.metadata)
        if g_ch is not None:
            if not text:
                text = g_ch.text
            graph_path_lines = g_ch.metadata.get("graph_path_lines")
            if graph_path_lines is not None:
                source_metadata.setdefault("graph_path_lines", graph_path_lines)

        candidates.append(
            FusedCandidate(
                chunk_id=cid,
                text=text,
                dense_raw_score=float(d_ch.score) if d_ch is not None else None,
                graph_raw_score=float(g_ch.score) if g_ch is not None else None,
                dense_norm_score=float(d_norm),
                graph_norm_score=float(g_norm),
                fused_score=float(fused),
                dense_present=d_ch is not None,
                graph_present=g_ch is not None,
                source_metadata=source_metadata,
            )
        )

    candidates.sort(key=lambda c: (-c.fused_score, c.chunk_id))
    return candidates[:fusion_keep_k]


def fused_candidates_to_chunks(
    candidates: Iterable[FusedCandidate],
    dense_weight: float,
) -> List[RetrievedChunk]:
    """Convert FusedCandidates into RetrievedChunks with full fusion metadata."""
    chunks: List[RetrievedChunk] = []
    for cand in candidates:
        metadata: Dict[str, Any] = dict(cand.source_metadata)
        metadata.update(
            {
                "branch": "fused",
                "dense_present": cand.dense_present,
                "graph_present": cand.graph_present,
                "dense_raw_score": cand.dense_raw_score,
                "graph_raw_score": cand.graph_raw_score,
                "dense_norm_score": cand.dense_norm_score,
                "graph_norm_score": cand.graph_norm_score,
                "fused_score": cand.fused_score,
                "fusion_weight_dense": float(dense_weight),
            }
        )
        chunks.append(
            RetrievedChunk(
                chunk_id=cand.chunk_id,
                text=cand.text,
                score=cand.fused_score,
                rank=0,
                metadata=metadata,
            )
        )
    return chunks


def _combined_latency(
    *branches: RetrievalResult,
    fusion_ms: float,
    total_ms: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "fusion": float(fusion_ms),
        "total": float(total_ms),
    }
    for br in branches:
        prefix = (br.retriever_name or "branch").lower()
        for key, val in br.latency_ms.items():
            out[f"{prefix}_{key}"] = float(val)
    return out


def build_fused_retrieval_result(
    query: str,
    dense: RetrievalResult,
    graph: RetrievalResult,
    dense_weight: float,
    fusion_keep_k: int,
    fusion_ms: float,
    total_ms: float,
) -> RetrievalResult:
    """Build a fused RetrievalResult from two branch results."""
    both_error = dense.status == "ERROR" and graph.status == "ERROR"
    latency = _combined_latency(dense, graph, fusion_ms=fusion_ms, total_ms=total_ms)

    if both_error:
        error_msg = (
            "; ".join(e for e in (dense.error, graph.error) if e)
            or "Both dense and graph branches failed"
        )
        return RetrievalResult(
            query=query,
            retriever_name=FUSED_RETRIEVER_NAME,
            status="ERROR",
            chunks=[],
            latency_ms=latency,
            error=error_msg,
        )

    candidates = fuse_branch_results(
        dense=dense,
        graph=graph,
        dense_weight=dense_weight,
        fusion_keep_k=fusion_keep_k,
    )
    chunks = fused_candidates_to_chunks(candidates, dense_weight=dense_weight)

    if not chunks:
        return RetrievalResult(
            query=query,
            retriever_name=FUSED_RETRIEVER_NAME,
            status="NO_CONTEXT",
            chunks=[],
            latency_ms=latency,
        )

    return RetrievalResult(
        query=query,
        retriever_name=FUSED_RETRIEVER_NAME,
        status="OK",
        chunks=chunks,
        latency_ms=latency,
    )


@dataclass
class FusionPipeline:
    """Retrieval-stage pipeline that fuses dense and graph branches.

    Drop-in alongside :class:`SingleBranchPipeline`. Consumes the same
    :class:`BranchRetriever` contract and produces a standard
    :class:`RetrievalResult` whose ``retriever_name`` is ``"Fused"``.

    Attributes:
        dense_retriever: The dense branch retriever.
        graph_retriever: The graph branch retriever.
        dense_weight: Dense fusion weight in ``[0.0, 1.0]``.
        fusion_keep_k: Number of fused candidates to keep after scoring.
    """

    dense_retriever: BranchRetriever
    graph_retriever: BranchRetriever
    dense_weight: float = 0.5
    fusion_keep_k: int = 10

    def run(
        self,
        query: str,
        *,
        dense_result: Optional[RetrievalResult] = None,
        graph_result: Optional[RetrievalResult] = None,
        dense_weight: Optional[float] = None,
        fusion_keep_k: Optional[int] = None,
        **retriever_kwargs: Any,
    ) -> RetrievalResult:
        """Run both branches (or reuse provided results) and fuse.

        ``dense_result`` / ``graph_result`` allow callers (notably the
        oracle pipeline) to skip branch execution and reuse cached
        retrievals while sweeping ``dense_weight``.
        """
        t0 = time.perf_counter()
        w = self.dense_weight if dense_weight is None else dense_weight
        keep_k = self.fusion_keep_k if fusion_keep_k is None else fusion_keep_k

        if dense_result is None:
            dense_result = self.dense_retriever.retrieve(query, **retriever_kwargs)
        if graph_result is None:
            graph_result = self.graph_retriever.retrieve(query, **retriever_kwargs)

        f0 = time.perf_counter()
        fused = build_fused_retrieval_result(
            query=query,
            dense=dense_result,
            graph=graph_result,
            dense_weight=w,
            fusion_keep_k=keep_k,
            fusion_ms=(time.perf_counter() - f0) * 1000.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return fused


def fuse_cached_results(
    query: str,
    dense: RetrievalResult,
    graph: RetrievalResult,
    dense_weight: float,
    fusion_keep_k: int,
) -> RetrievalResult:
    """Convenience wrapper around :func:`build_fused_retrieval_result`.

    Used by the oracle pipeline when both branch results are already
    available from the retrieval cache and only the fusion weight is
    sweeping.
    """
    t0 = time.perf_counter()
    return build_fused_retrieval_result(
        query=query,
        dense=dense,
        graph=graph,
        dense_weight=dense_weight,
        fusion_keep_k=fusion_keep_k,
        fusion_ms=(time.perf_counter() - t0) * 1000.0,
        total_ms=(time.perf_counter() - t0) * 1000.0,
    )


Candidate = Tuple[str, float]  # (chunk_id, fused_score) for external consumers
