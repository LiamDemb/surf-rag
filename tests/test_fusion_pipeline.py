"""Tests for fusion primitives and the fused retrieval pipeline."""

from __future__ import annotations

from typing import List

import pytest

from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.fusion import (
    FUSED_RETRIEVER_NAME,
    FusionPipeline,
    build_fused_retrieval_result,
    fuse_branch_results,
    fuse_cached_results,
    min_max_normalize,
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


def _chunk(chunk_id: str, score: float, text: str = "", metadata=None) -> RetrievedChunk:
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
    dense = _mk_result(
        "Dense", "OK", [_chunk("a", 1.0), _chunk("b", 0.0)]
    )
    graph = _mk_result(
        "Graph", "OK", [_chunk("c", 0.5)]
    )
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
    dense = _mk_result(
        "Dense", "OK", [_chunk("a", 0.8, metadata={"branch": "dense"})]
    )
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


def test_fusion_pipeline_can_reuse_provided_branch_results():
    """Oracle pipeline reuses cached branch results and sweeps weights without re-retrieving."""
    dense = _mk_result("Dense", "OK", [_chunk("a", 1.0)])
    graph = _mk_result("Graph", "OK", [_chunk("b", 1.0)])
    d = _StaticRetriever("Dense", _mk_result("Dense", "NO_CONTEXT", []))
    g = _StaticRetriever("Graph", _mk_result("Graph", "NO_CONTEXT", []))
    pipeline = FusionPipeline(d, g, dense_weight=0.5, fusion_keep_k=5)

    res = pipeline.run(
        "q", dense_result=dense, graph_result=graph, dense_weight=0.7
    )

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
