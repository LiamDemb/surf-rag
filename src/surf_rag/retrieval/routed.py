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


@dataclass(frozen=True)
class RoutedRunOutput:
    """Retrieval outputs for evaluation and generation paths."""

    pretrunc_result: RetrievalResult
    generation_result: RetrievalResult


@dataclass
class RoutedFusionPipeline:
    """Run dense/graph per routing policy; learned router optional."""

    dense_retriever: BranchRetriever
    graph_retriever: BranchRetriever
    fusion_keep_k: int = 25
    router: Optional["LoadedRouter"] = None
    fallback_router: Optional["LoadedRouter"] = None
    router_confidence_threshold: float = 0.7

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
        """Execute routing and return generation-path retrieval."""
        out = self.run_with_pretrunc(
            query,
            policy,
            dense_result=dense_result,
            graph_result=graph_result,
            query_embedding=query_embedding,
            feature_vector=feature_vector,
            **retriever_kwargs,
        )
        return out.generation_result

    def run_with_pretrunc(
        self,
        query: str,
        policy: RoutingPolicyName,
        *,
        dense_result: Optional[RetrievalResult] = None,
        graph_result: Optional[RetrievalResult] = None,
        query_embedding: Optional[np.ndarray] = None,
        feature_vector: Optional[np.ndarray] = None,
        **retriever_kwargs: Any,
    ) -> RoutedRunOutput:
        """Execute routing and return pre-truncation + generation retrieval."""
        from surf_rag.router.inference import (
            predict_batch,
            predict_class_id_batch,
            predict_class_probs_batch,
        )

        t0 = time.perf_counter()
        pred_weight: Optional[float] = None
        fallback_weight: Optional[float] = None
        pred_class_id: Optional[int] = None
        pred_class_probs: Optional[tuple[float, float]] = None
        routing_predict_ms = 0.0
        if policy in (
            RoutingPolicyName.LEARNED_SOFT,
            RoutingPolicyName.HARD_ROUTING,
            RoutingPolicyName.HYBRID,
        ):
            if self.router is None:
                raise ValueError("learned policies require a loaded router")
            if query_embedding is None or feature_vector is None:
                raise ValueError(
                    "learned policies require query_embedding and feature_vector"
                )
            p0 = time.perf_counter()
            if policy in (RoutingPolicyName.HARD_ROUTING, RoutingPolicyName.HYBRID):
                pred = predict_class_id_batch(
                    self.router, query_embedding, feature_vector
                )
                pred_class_id = int(pred.reshape(-1)[0])
                probs = predict_class_probs_batch(
                    self.router, query_embedding, feature_vector
                )
                row = probs.reshape(-1, 2)[0]
                pred_class_probs = (float(row[0]), float(row[1]))
                if policy == RoutingPolicyName.HYBRID and max(pred_class_probs) < float(
                    self.router_confidence_threshold
                ):
                    if self.fallback_router is None:
                        raise ValueError(
                            "hybrid policy requires fallback_router for low-confidence cases"
                        )
                    fb = predict_batch(
                        self.fallback_router, query_embedding, feature_vector
                    )
                    fallback_weight = float(fb.reshape(-1)[0])
            else:
                pred = predict_batch(self.router, query_embedding, feature_vector)
                pred_weight = float(pred.reshape(-1)[0])
            routing_predict_ms = (time.perf_counter() - p0) * 1000.0

        decision = decide_routing(
            policy,
            predicted_weight=pred_weight,
            predicted_class_id=pred_class_id,
            predicted_class_probs=pred_class_probs,
            confidence_threshold=self.router_confidence_threshold,
            fallback_weight=fallback_weight,
        )
        debug = decision_to_debug_info(decision)

        if not decision.run_graph:
            if dense_result is None:
                dense_result = self.dense_retriever.retrieve(query, **retriever_kwargs)
            pre = RetrievalResult(
                query=dense_result.query,
                retriever_name=dense_result.retriever_name,
                status=dense_result.status,
                chunks=list(dense_result.chunks),
                latency_ms=dict(dense_result.latency_ms),
                error=dense_result.error,
                debug_info=(
                    dict(dense_result.debug_info) if dense_result.debug_info else None
                ),
            )
            gen = trim_retrieval_top_k(pre, self.fusion_keep_k)
            total_ms = (time.perf_counter() - t0) * 1000.0
            pre_di = _merge_debug(pre.debug_info, {"routing": debug})
            gen_di = _merge_debug(gen.debug_info, {"routing": debug})
            common = {"routing_predict_ms": routing_predict_ms, "total_ms": total_ms}
            return RoutedRunOutput(
                pretrunc_result=RetrievalResult(
                    query=pre.query,
                    retriever_name=pre.retriever_name,
                    status=pre.status,
                    chunks=pre.chunks,
                    latency_ms={**dict(pre.latency_ms), **common},
                    error=pre.error,
                    debug_info=pre_di,
                ),
                generation_result=RetrievalResult(
                    query=gen.query,
                    retriever_name=gen.retriever_name,
                    status=gen.status,
                    chunks=gen.chunks,
                    latency_ms={**dict(gen.latency_ms), **common},
                    error=gen.error,
                    debug_info=gen_di,
                ),
            )

        if not decision.run_dense:
            if graph_result is None:
                graph_result = self.graph_retriever.retrieve(query, **retriever_kwargs)
            pre = RetrievalResult(
                query=graph_result.query,
                retriever_name=graph_result.retriever_name,
                status=graph_result.status,
                chunks=list(graph_result.chunks),
                latency_ms=dict(graph_result.latency_ms),
                error=graph_result.error,
                debug_info=(
                    dict(graph_result.debug_info) if graph_result.debug_info else None
                ),
            )
            gen = trim_retrieval_top_k(pre, self.fusion_keep_k)
            total_ms = (time.perf_counter() - t0) * 1000.0
            pre_di = _merge_debug(pre.debug_info, {"routing": debug})
            gen_di = _merge_debug(gen.debug_info, {"routing": debug})
            common = {"routing_predict_ms": routing_predict_ms, "total_ms": total_ms}
            return RoutedRunOutput(
                pretrunc_result=RetrievalResult(
                    query=pre.query,
                    retriever_name=pre.retriever_name,
                    status=pre.status,
                    chunks=pre.chunks,
                    latency_ms={**dict(pre.latency_ms), **common},
                    error=pre.error,
                    debug_info=pre_di,
                ),
                generation_result=RetrievalResult(
                    query=gen.query,
                    retriever_name=gen.retriever_name,
                    status=gen.status,
                    chunks=gen.chunks,
                    latency_ms={**dict(gen.latency_ms), **common},
                    error=gen.error,
                    debug_info=gen_di,
                ),
            )

        if dense_result is None:
            dense_result = self.dense_retriever.retrieve(query, **retriever_kwargs)
        if graph_result is None:
            graph_result = self.graph_retriever.retrieve(query, **retriever_kwargs)

        f0 = time.perf_counter()
        fused_pre = build_fused_retrieval_result(
            query=query,
            dense=dense_result,
            graph=graph_result,
            dense_weight=decision.dense_weight,
            fusion_keep_k=None,
            fusion_ms=(time.perf_counter() - f0) * 1000.0,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )
        fused_gen = trim_retrieval_top_k(fused_pre, self.fusion_keep_k)
        pre_di = _merge_debug(fused_pre.debug_info, {"routing": debug})
        gen_di = _merge_debug(fused_gen.debug_info, {"routing": debug})
        common = {
            "routing_predict_ms": routing_predict_ms,
            "total_ms": float(
                dict(fused_pre.latency_ms).get(
                    "total_ms", fused_pre.latency_ms.get("total", 0.0)
                )
            ),
        }
        return RoutedRunOutput(
            pretrunc_result=RetrievalResult(
                query=fused_pre.query,
                retriever_name=FUSED_RETRIEVER_NAME,
                status=fused_pre.status,
                chunks=fused_pre.chunks,
                latency_ms={**dict(fused_pre.latency_ms), **common},
                error=fused_pre.error,
                debug_info=pre_di,
            ),
            generation_result=RetrievalResult(
                query=fused_gen.query,
                retriever_name=FUSED_RETRIEVER_NAME,
                status=fused_gen.status,
                chunks=fused_gen.chunks,
                latency_ms={**dict(fused_gen.latency_ms), **common},
                error=fused_gen.error,
                debug_info=gen_di,
            ),
        )
