"""Routed pipeline: frozen branch injection skips live retriever calls."""

from __future__ import annotations

from surf_rag.retrieval.routed import RoutedFusionPipeline
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult
from surf_rag.router.policies import RoutingPolicyName


class _BoomRetriever:
    name = "Boom"

    def retrieve(self, query: str, **_: object) -> RetrievalResult:
        raise RuntimeError(
            "live retrieve should not run when frozen results are injected"
        )


def _one_chunk(name: str, cid: str) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        retriever_name=name,
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id=cid, text="t", score=1.0, rank=0, metadata={"branch": name}
            )
        ],
        latency_ms={"total": 5.0},
    )


def test_equal_50_50_with_injection_skips_retrievers() -> None:
    d = _one_chunk("Dense", "a")
    g = _one_chunk("Graph", "b")
    pl = RoutedFusionPipeline(
        _BoomRetriever(),
        _BoomRetriever(),
        fusion_keep_k=5,
        router=None,
    )
    out = pl.run_with_pretrunc(
        "q",
        RoutingPolicyName.EQUAL_50_50,
        dense_result=d,
        graph_result=g,
    )
    assert out.generation_result.status == "OK"
    assert {c.chunk_id for c in out.generation_result.chunks} == {"a", "b"}
