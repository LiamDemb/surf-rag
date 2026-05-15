"""Tests for frozen branch cache resolution and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surf_rag.evaluation.frozen_branch_cache import (
    ResolvedFrozenBranchPaths,
    frozen_results_for_question,
    load_frozen_branch_bundle,
    validate_frozen_branch_bundle,
)
from surf_rag.evaluation.oracle_artifacts import OracleRunPaths, append_retrieval_line
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.router.policies import RoutingPolicyName


def _write_minimal_retrieval_jsonl(path: Path, qid: str) -> None:
    from surf_rag.retrieval.types import RetrievedChunk

    rr = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1", text="t", score=1.0, rank=0, metadata={"branch": "dense"}
            )
        ],
        latency_ms={"total": 3.0},
    )
    append_retrieval_line(path, rr, qid)


def test_validate_frozen_branch_strict_manifest_mismatch(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    paths = OracleRunPaths(run_root=oracle_root)
    paths.ensure_dirs()
    bench = tmp_path / "bench" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True)
    bench.touch()
    corp = tmp_path / "corpus"
    corp.mkdir()
    manifest = {
        "schema_version": 3,
        "benchmark_path": str(tmp_path / "other" / "benchmark.jsonl"),
        "retrieval_asset_dir": str(corp.resolve()),
        "branch_top_k": 10,
    }
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    _write_minimal_retrieval_jsonl(paths.retrieval_dense, "q1")
    _write_minimal_retrieval_jsonl(paths.retrieval_graph, "q1")
    resolved = ResolvedFrozenBranchPaths(
        mode="router_oracle",
        dense_jsonl=paths.retrieval_dense,
        graph_jsonl=paths.retrieval_graph,
        oracle_paths=paths,
        manifest={},
    )
    with pytest.raises(ValueError, match="benchmark_path"):
        validate_frozen_branch_bundle(
            resolved,
            benchmark_path=bench,
            retrieval_asset_dir=corp,
            expect_branch_top_k=10,
            strict_manifest=True,
        )


def test_frozen_results_for_question_dense_only() -> None:
    from surf_rag.retrieval.types import RetrievedChunk

    d = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="OK",
        chunks=[RetrievedChunk(chunk_id="a", text="t", score=1.0, rank=0, metadata={})],
        latency_ms={"total": 1.0},
    )
    dense_by = {"1": d}
    graph_by: dict = {}
    dr, gr = frozen_results_for_question(
        RoutingPolicyName.DENSE_ONLY.value,
        "1",
        dense_by_qid=dense_by,
        graph_by_qid=graph_by,
    )
    assert dr is not None and gr is None


def test_load_frozen_explicit_jsonl(tmp_path: Path) -> None:
    dpath = tmp_path / "d.jsonl"
    gpath = tmp_path / "g.jsonl"
    _write_minimal_retrieval_jsonl(dpath, "x1")
    _write_minimal_retrieval_jsonl(gpath, "x1")
    bench = tmp_path / "b.jsonl"
    bench.write_text('{"question_id":"x1"}\n', encoding="utf-8")
    corp = tmp_path / "c"
    corp.mkdir()
    bundle = load_frozen_branch_bundle(
        mode="explicit_jsonl",
        router_base=tmp_path,
        paths_router_id="",
        oracle_router_id=None,
        dense_jsonl=dpath,
        graph_jsonl=gpath,
        benchmark_path=bench,
        retrieval_asset_dir=corp,
        expect_branch_top_k=25,
        strict_manifest=False,
        question_ids={"x1"},
        policy=RoutingPolicyName.EQUAL_50_50.value,
    )
    assert "x1" in bundle.dense_by_qid
    assert "x1" in bundle.graph_by_qid
