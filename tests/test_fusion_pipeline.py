"""Tests for fusion primitives and the fused retrieval pipeline."""

from __future__ import annotations

from typing import List

import pytest

from surf_rag.evaluation.latency_metrics import (
    PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY,
    canonicalize_latency_ms,
)
from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.fusion import (
    FUSED_RETRIEVER_NAME,
    FusionPipeline,
    GRAPH_SCORE_LOG_EPS,
    branch_retrieval_wall_ms,
    build_fused_retrieval_result,
    build_rrf_fused_retrieval_result,
    fuse_branch_results,
    fuse_branch_results_rrf,
    fuse_cached_results,
    graph_score_log_transform,
    min_max_normalize,
)
from surf_rag.retrieval.routed import (
    dual_branch_rrf_fusion_output,
    dual_branch_weighted_fusion_output,
)
from surf_rag.retrieval.types import RetrievalResult, RetrievedChunk


def _mk_result(
    name: str,
    status: str,
    chunks: List[RetrievedChunk],
    error: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        retriever_name=name,
        status=status,
        chunks=chunks,
        latency_ms={"retrieval": 1.0, "total": 1.0},
        error=error,
    )


def _chunk(
    chunk_id: str, score: float, text: str = "", metadata=None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text or f"text-{chunk_id}",
        score=float(score),
        rank=0,
        metadata=dict(metadata or {}),
    )


class _StaticRetriever(BranchRetriever):
    def __init__(self, name: str, result: RetrievalResult) -> None:
        self.name = name
        self._result = result
        self.calls = 0

    def retrieve(self, query: str, **_: object) -> RetrievalResult:
        self.calls += 1
        return self._result


def test_min_max_normalize_basic_range():
    assert min_max_normalize([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]
    assert min_max_normalize([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_min_max_normalize_empty():
    assert min_max_normalize([]) == []


def test_min_max_normalize_tied_pool_maps_to_one():
    """Degenerate max==min case: every retrieved score maps to 1.0."""
    assert min_max_normalize([0.7]) == [1.0]
    assert min_max_normalize([0.3, 0.3, 0.3]) == [1.0, 1.0, 1.0]


def test_missing_branch_score_is_zero_and_fusion_uses_weights():
    """Chunks retrieved by only one branch get a 0.0 score for the missing branch."""
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0), _chunk("b", 0.0)])
    graph = _mk_result("Graph", "OK", [_chunk("c", 0.5)])
    # dense_weight=1.0 -> pure dense; 'c' has dense_norm=0.0.
    cands_dense = fuse_branch_results(dense, graph, dense_weight=1.0, fusion_keep_k=10)
    by_id = {c.chunk_id: c for c in cands_dense}
    assert by_id["a"].fused_score == pytest.approx(1.0)
    assert by_id["b"].fused_score == pytest.approx(0.0)
    assert by_id["c"].fused_score == pytest.approx(0.0)
    assert by_id["c"].dense_present is False
    assert by_id["c"].graph_present is True

    # dense_weight=0.0 -> pure graph; only c contributes (norm=1.0 tied pool).
    cands_graph = fuse_branch_results(dense, graph, dense_weight=0.0, fusion_keep_k=10)
    by_id = {c.chunk_id: c for c in cands_graph}
    assert by_id["c"].fused_score == pytest.approx(1.0)
    assert by_id["a"].fused_score == pytest.approx(0.0)


def test_graph_log_before_normalize_changes_pure_graph_ranking() -> None:
    """Log graph scores before min-max so steep decay spreads more linearly."""
    import math

    dense = _mk_result("Dense", "NO_CONTEXT", [])
    graph = _mk_result(
        "Graph",
        "OK",
        [
            _chunk("low", 1e-8),
            _chunk("mid", 0.01),
            _chunk("high", 1.0),
        ],
    )
    plain = fuse_branch_results(dense, graph, dense_weight=0.0, fusion_keep_k=10)
    logged = fuse_branch_results(
        dense,
        graph,
        dense_weight=0.0,
        fusion_keep_k=10,
        graph_log_before_normalize=True,
    )
    plain_order = [c.chunk_id for c in plain]
    logged_order = [c.chunk_id for c in logged]
    assert plain_order == ["high", "mid", "low"]
    assert logged_order == ["high", "mid", "low"]
    plain_scores = {c.chunk_id: c.graph_norm_score for c in plain}
    logged_scores = {c.chunk_id: c.graph_norm_score for c in logged}
    assert plain_scores["high"] == pytest.approx(1.0)
    assert logged_scores["high"] == pytest.approx(1.0)
    assert logged_scores["mid"] > plain_scores["mid"]
    assert plain_scores["low"] == pytest.approx(0.0)
    assert logged_scores["low"] == pytest.approx(0.0)
    for c in logged:
        assert c.graph_raw_score == pytest.approx(
            {"low": 1e-8, "mid": 0.01, "high": 1.0}[c.chunk_id]
        )
    assert graph_score_log_transform(0.0) == pytest.approx(
        math.log(GRAPH_SCORE_LOG_EPS)
    )


def test_fuse_dedupes_shared_chunk_and_sums_weighted_contributions():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0), _chunk("b", 0.0)])
    graph = _mk_result("Graph", "OK", [_chunk("a", 0.0), _chunk("b", 1.0)])
    # At 0.5/0.5, both a and b end up with 0.5.
    cands = fuse_branch_results(dense, graph, dense_weight=0.5, fusion_keep_k=10)
    assert {c.chunk_id for c in cands} == {"a", "b"}
    for c in cands:
        assert c.fused_score == pytest.approx(0.5)
        assert c.dense_present and c.graph_present


def test_fuse_sorts_descending_and_keeps_top_k():
    dense = _mk_result(
        "Dense",
        "OK",
        [_chunk("a", 0.2), _chunk("b", 0.8), _chunk("c", 0.5)],
    )
    graph = _mk_result("Graph", "NO_CONTEXT", [])
    cands = fuse_branch_results(dense, graph, dense_weight=1.0, fusion_keep_k=2)
    assert [c.chunk_id for c in cands] == ["b", "c"]
    assert cands[0].fused_score > cands[1].fused_score


def test_no_context_branches_produce_no_context_result():
    dense = _mk_result("Dense", "NO_CONTEXT", [])
    graph = _mk_result("Graph", "NO_CONTEXT", [])
    res = build_fused_retrieval_result(
        query="q",
        dense=dense,
        graph=graph,
        dense_weight=0.5,
        fusion_keep_k=5,
        fusion_ms=0.1,
        total_ms=0.2,
    )
    assert res.status == "NO_CONTEXT"
    assert res.retriever_name == FUSED_RETRIEVER_NAME
    assert res.chunks == []


def test_both_error_branches_produce_error_result():
    dense = _mk_result("Dense", "ERROR", [], error="boom-dense")
    graph = _mk_result("Graph", "ERROR", [], error="boom-graph")
    res = build_fused_retrieval_result(
        query="q",
        dense=dense,
        graph=graph,
        dense_weight=0.5,
        fusion_keep_k=5,
        fusion_ms=0.1,
        total_ms=0.2,
    )
    assert res.status == "ERROR"
    assert "boom-dense" in (res.error or "")
    assert "boom-graph" in (res.error or "")


def test_one_error_branch_does_not_poison_fusion():
    """An ERROR branch is treated as contributing nothing; the other branch still wins."""
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "ERROR", [], error="graph-crashed")
    res = build_fused_retrieval_result(
        query="q",
        dense=dense,
        graph=graph,
        dense_weight=0.5,
        fusion_keep_k=5,
        fusion_ms=0.1,
        total_ms=0.2,
    )
    assert res.status == "OK"
    assert [c.chunk_id for c in res.chunks] == ["a"]


def test_fused_chunk_metadata_is_complete():
    dense = _mk_result("Dense", "OK", [_chunk("a", 0.8, metadata={"branch": "dense"})])
    graph = _mk_result(
        "Graph",
        "OK",
        [
            _chunk(
                "a",
                0.4,
                metadata={"branch": "graph", "graph_path_lines": ["Path: X"]},
            )
        ],
    )
    res = build_fused_retrieval_result(
        query="q",
        dense=dense,
        graph=graph,
        dense_weight=0.5,
        fusion_keep_k=5,
        fusion_ms=0.1,
        total_ms=0.2,
    )
    assert res.status == "OK"
    md = res.chunks[0].metadata
    assert md["branch"] == "fused"
    assert md["dense_present"] is True
    assert md["graph_present"] is True
    assert md["dense_raw_score"] == pytest.approx(0.8)
    assert md["graph_raw_score"] == pytest.approx(0.4)
    assert md["dense_norm_score"] == pytest.approx(1.0)
    assert md["graph_norm_score"] == pytest.approx(1.0)
    assert md["fused_score"] == pytest.approx(1.0)
    assert md["fusion_weight_dense"] == pytest.approx(0.5)
    assert md.get("graph_path_lines") == ["Path: X"]


def test_fuse_rejects_invalid_weight_and_keep_k():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "NO_CONTEXT", [])
    with pytest.raises(ValueError):
        fuse_branch_results(dense, graph, dense_weight=1.5, fusion_keep_k=5)
    with pytest.raises(ValueError):
        fuse_branch_results(dense, graph, dense_weight=0.5, fusion_keep_k=0)


def test_fuse_rrf_tie_breaks_on_chunk_id_when_scores_equal():
    """k=1: ranks 1 and 2 give 0.5 and 1/3 each way; symmetric total → lex id order."""
    dense = _mk_result("Dense", "OK", [_chunk("b", 1.0), _chunk("a", 0.9)])
    graph = _mk_result("Graph", "OK", [_chunk("a", 1.0), _chunk("b", 0.9)])
    cands = fuse_branch_results_rrf(dense, graph, rrf_k=1, fusion_keep_k=10)
    assert [c.chunk_id for c in cands[:2]] == ["a", "b"]
    assert cands[0].rrf_score == pytest.approx(cands[1].rrf_score)


def test_fuse_rrf_rejects_invalid_k_and_keep_k():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    with pytest.raises(ValueError):
        fuse_branch_results_rrf(dense, graph, rrf_k=0, fusion_keep_k=5)
    with pytest.raises(ValueError):
        fuse_branch_results_rrf(dense, graph, rrf_k=60, fusion_keep_k=0)


def test_build_rrf_fused_metadata_and_one_error_branch():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "ERROR", [], error="x")
    res = build_rrf_fused_retrieval_result(
        "q",
        dense,
        graph,
        rrf_k=60,
        fusion_keep_k=5,
        fusion_ms=0.1,
        total_ms=0.2,
    )
    assert res.status == "OK"
    md = res.chunks[0].metadata
    assert md["fusion_method"] == "rrf"
    assert md["rrf_k"] == 60
    assert md["dense_rank"] == 1
    assert md["graph_rank"] is None


def test_build_rrf_both_error_branches():
    dense = _mk_result("Dense", "ERROR", [], error="d")
    graph = _mk_result("Graph", "ERROR", [], error="g")
    res = build_rrf_fused_retrieval_result(
        "q",
        dense,
        graph,
        rrf_k=60,
        fusion_keep_k=5,
        fusion_ms=0.0,
        total_ms=0.0,
    )
    assert res.status == "ERROR"


def test_dual_branch_rrf_sequential_fusion_total_matches_branch_sum():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    dense.latency_ms["total"] = 10.0
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    graph.latency_ms["total"] = 20.0
    out = dual_branch_rrf_fusion_output(
        query="q",
        dense_result=dense,
        graph_result=graph,
        fusion_keep_k=5,
        rrf_k=60,
        routing_predict_ms=0.0,
        t_route_start=0.0,
        debug={},
        sequential_fusion_total=True,
    )
    fusion_ms = float(out.pretrunc_result.latency_ms.get("fusion", 0.0))
    total_ms = float(out.pretrunc_result.latency_ms.get("total", 0.0))
    assert fusion_ms > 0.0
    assert total_ms == pytest.approx(10.0 + 20.0 + fusion_ms)
    assert (
        out.pretrunc_result.latency_ms.get(PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY)
        == 1.0
    )


def test_fusion_pipeline_runs_both_branches_by_default():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    d = _StaticRetriever("Dense", dense)
    g = _StaticRetriever("Graph", graph)
    pipeline = FusionPipeline(d, g, dense_weight=0.5, fusion_keep_k=5)

    res = pipeline.run("q")

    assert d.calls == 1 and g.calls == 1
    assert res.retriever_name == FUSED_RETRIEVER_NAME
    assert res.status == "OK"
    assert {c.chunk_id for c in res.chunks} == {"a", "b"}
    assert "fusion" in res.latency_ms
    assert "total" in res.latency_ms


def test_dual_branch_rrf_sequential_canonical_reported_adds_router_predict() -> None:
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    dense.latency_ms["total"] = 10.0
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    graph.latency_ms["total"] = 20.0
    out = dual_branch_rrf_fusion_output(
        query="q",
        dense_result=dense,
        graph_result=graph,
        fusion_keep_k=5,
        rrf_k=60,
        routing_predict_ms=2.5,
        t_route_start=0.0,
        debug={},
        sequential_fusion_total=True,
    )
    canon = canonicalize_latency_ms(
        retriever_name="Fused",
        latency_ms=dict(out.pretrunc_result.latency_ms),
        routing_input_ms=1.0,
    )
    assert PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY not in canon
    assert canon["retrieval_reported_total_ms"] == pytest.approx(
        canon["retrieval_stage_total_ms"] + 2.5
    )


def test_fusion_pipeline_can_reuse_provided_branch_results():
    """Oracle pipeline reuses cached branch results and sweeps weights without re-retrieving."""
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    d = _StaticRetriever("Dense", _mk_result("Dense", "NO_CONTEXT", []))
    g = _StaticRetriever("Graph", _mk_result("Graph", "NO_CONTEXT", []))
    pipeline = FusionPipeline(d, g, dense_weight=0.5, fusion_keep_k=5)

    res = pipeline.run("q", dense_result=dense, graph_result=graph, dense_weight=0.7)

    assert d.calls == 0 and g.calls == 0
    md = res.chunks[0].metadata
    assert md["fusion_weight_dense"] == pytest.approx(0.7)


def test_fuse_cached_results_wrapper_produces_fused_result():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0), _chunk("b", 0.5)])
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    res = fuse_cached_results(
        query="q", dense=dense, graph=graph, dense_weight=0.5, fusion_keep_k=5
    )
    assert res.retriever_name == FUSED_RETRIEVER_NAME
    assert res.status == "OK"


def test_branch_retrieval_wall_ms_prefers_total():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    dense.latency_ms.clear()
    dense.latency_ms["total"] = 12.5
    assert branch_retrieval_wall_ms(dense) == pytest.approx(12.5)


def test_dual_branch_sequential_fusion_total_matches_branch_sum():
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    dense.latency_ms["total"] = 10.0
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    graph.latency_ms["total"] = 20.0
    out = dual_branch_weighted_fusion_output(
        query="q",
        dense_result=dense,
        graph_result=graph,
        fusion_keep_k=5,
        dense_weight=0.5,
        routing_predict_ms=0.0,
        t_route_start=0.0,
        debug={},
        sequential_fusion_total=True,
    )
    fusion_ms = float(out.pretrunc_result.latency_ms.get("fusion", 0.0))
    total_ms = float(out.pretrunc_result.latency_ms.get("total", 0.0))
    assert fusion_ms > 0.0
    assert total_ms == pytest.approx(10.0 + 20.0 + fusion_ms)
    assert (
        out.pretrunc_result.latency_ms.get(PIPE_TOTAL_EXCLUDES_ROUTER_PREDICT_KEY)
        == 1.0
    )
