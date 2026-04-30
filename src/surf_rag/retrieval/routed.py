"""Routed dense+graph retrieval: policies and optional learned router."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.fusion import (
    FUSED_RETRIEVER_NAME,
    build_fused_retrieval_result,
)
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.router.policies import (
    RoutingPolicyName,
    decide_routing,
    decision_to_debug_info,
)

if TYPE_CHECKING:
    from surf_rag.router.inference import LoadedRouter


def trim_retrieval_top_k(result: RetrievalResult, k: int) -> RetrievalResult:
    """Keep top-``k`` chunks by score (already sorted in ``RetrievalResult``)."""
    if result.status != "OK" or k <= 0:
        return result
    if not result.chunks:
        return result
    chunks = result.chunks[:k]
    return RetrievalResult(
        query=result.query,
        retriever_name=result.retriever_name,
        status="OK",
        chunks=chunks,
        latency_ms=dict(result.latency_ms),
        error=result.error,
        debug_info=dict(result.debug_info) if result.debug_info else None,
    )


def _merge_debug(
    base: Optional[Dict[str, Any]], extra: Dict[str, Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base or {})
    out.update(extra)
    return out


@dataclass
class RoutedFusionPipeline:
    """Run dense/graph per routing policy; learned router optional."""

    dense_retriever: BranchRetriever
    graph_retriever: BranchRetriever
    fusion_keep_k: int = 25
    router: Optional["LoadedRouter"] = None

    def run(
        self,
        query: str,
        policy: RoutingPolicyName,
        *,
        dense_result: Optional[RetrievalResult] = None,
        graph_result: Optional[RetrievalResult] = None,
        query_embedding: Optional[np.ndarray] = None,
        feature_vector: Optional[np.ndarray] = None,
        **retriever_kwargs: Any,
    ) -> RetrievalResult:
        """Execute routing and return a ``RetrievalResult`` (fused or single-branch)."""
        from surf_rag.router.inference import predict_batch

        t0 = time.perf_counter()
        pred_weight: Optional[float] = None
        if policy in (
            RoutingPolicyName.LEARNED_SOFT,
            RoutingPolicyName.LEARNED_HARD,
        ):
            if self.router is None:
                raise ValueError("learned policies require a loaded router")
            if query_embedding is None or feature_vector is None:
                raise ValueError(
                    "learned policies require query_embedding and feature_vector"
                )
            pred = predict_batch(self.router, query_embedding, feature_vector)
            pred_weight = float(pred.reshape(-1)[0])

        decision = decide_routing(policy, predicted_weight=pred_weight)
        debug = decision_to_debug_info(decision)

        if not decision.run_graph:
            if dense_result is None:
                dense_result = self.dense_retriever.retrieve(query, **retriever_kwargs)
            out = trim_retrieval_top_k(dense_result, self.fusion_keep_k)
            di = _merge_debug(out.debug_info, {"routing": debug})
            return RetrievalResult(
                query=out.query,
                retriever_name=out.retriever_name,
                status=out.status,
                chunks=out.chunks,
                latency_ms={
                    **dict(out.latency_ms),
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
                error=out.error,
                debug_info=di,
            )

        if not decision.run_dense:
            if graph_result is None:
                graph_result = self.graph_retriever.retrieve(query, **retriever_kwargs)
            out = trim_retrieval_top_k(graph_result, self.fusion_keep_k)
            di = _merge_debug(out.debug_info, {"routing": debug})
            return RetrievalResult(
                query=out.query,
                retriever_name=out.retriever_name,
                status=out.status,
                chunks=out.chunks,
                latency_ms={
                    **dict(out.latency_ms),
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
                error=out.error,
                debug_info=di,
            )

        if dense_result is None:
            dense_result = self.dense_retriever.retrieve(query, **retriever_kwargs)
        if graph_result is None:
            graph_result = self.graph_retriever.retrieve(query, **retriever_kwargs)

        f0 = time.perf_counter()
        fused = build_fused_retrieval_result(
            query=query,
            dense=dense_result,
            graph=graph_result,
            dense_weight=decision.dense_weight,
            fusion_keep_k=self.fusion_keep_k,
            fusion_ms=(time.perf_counter() - f0) * 1000.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )
        di = _merge_debug(fused.debug_info, {"routing": debug})
        return RetrievalResult(
            query=fused.query,
            retriever_name=FUSED_RETRIEVER_NAME,
            status=fused.status,
            chunks=fused.chunks,
            latency_ms=fused.latency_ms,
            error=fused.error,
            debug_info=di,
        )
