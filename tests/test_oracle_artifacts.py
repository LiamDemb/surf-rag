"""Tests for oracle run paths, manifest, summary, and JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surf_rag.evaluation.oracle_artifacts import (
    DEFAULT_DENSE_WEIGHT_GRID,
    OracleRunPaths,
    OracleScoreRow,
    WeightBinScore,
    _format_beta,
    append_oracle_score_rows,
    append_retrieval_line,
    build_oracle_run_root,
    default_oracle_base,
    make_run_paths_for_cli,
    read_manifest,
    read_oracle_score_rows,
    read_question_ids,
    read_retrieval_cache,
    read_summary,
    update_manifest,
    write_manifest,
    write_questions_snapshot,
    write_summary,
)
from surf_rag.retrieval.types import RetrievalResult, RetrievedChunk


def _paths(tmp_path: Path) -> OracleRunPaths:
    base = tmp_path / "data" / "oracle"
    run_root = build_oracle_run_root(base, "nq", "dev", "runA")
    return OracleRunPaths(run_root=run_root)


def test_dense_weight_grid_has_11_bins_from_zero_to_one():
    assert len(DEFAULT_DENSE_WEIGHT_GRID) == 11
    assert DEFAULT_DENSE_WEIGHT_GRID[0] == 0.0
    assert DEFAULT_DENSE_WEIGHT_GRID[-1] == 1.0
    assert all(0.0 <= w <= 1.0 for w in DEFAULT_DENSE_WEIGHT_GRID)


def test_default_oracle_base_respects_env(monkeypatch):
    monkeypatch.delenv("ORACLE_BASE", raising=False)
    assert default_oracle_base() == Path("data/oracle")
    monkeypatch.setenv("ORACLE_BASE", "/tmp/something")
    assert default_oracle_base() == Path("/tmp/something")


def test_format_beta_filename_component_is_stable():
    assert _format_beta(1.0) == "1"
    assert _format_beta(2.0) == "2"
    assert _format_beta(0.5) == "0p5"
    assert _format_beta(10.25) == "10p25"


def test_make_run_paths_for_cli_builds_expected_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("ORACLE_BASE", str(tmp_path))
    paths = make_run_paths_for_cli("nq", "dev", "runA")
    assert paths.run_root == tmp_path / "nq" / "dev" / "runA"
    assert paths.manifest == paths.run_root / "manifest.json"
    assert paths.oracle_scores == paths.run_root / "oracle_scores.jsonl"
    assert paths.labels_selected == paths.run_root / "labels" / "selected.jsonl"


def test_write_and_read_manifest_round_trip(tmp_path):
    paths = _paths(tmp_path)
    write_manifest(
        paths,
        oracle_run_id="runA",
        benchmark="nq",
        split="dev",
        benchmark_path="data/benchmarks/nq_dev.jsonl",
        retrieval_asset_dir="data/processed/nq_dev",
        weight_grid=DEFAULT_DENSE_WEIGHT_GRID,
        branch_top_k=25,
        fusion_keep_k=25,
        oracle_metric="stateful_ndcg",
        oracle_metric_k=10,
        diagnostic_metric_ks=(5, 10, 20),
    )
    data = read_manifest(paths)
    assert data["oracle_run_id"] == "runA"
    assert data["benchmark"] == "nq"
    assert data["branch_top_k"] == 25
    assert data["oracle_metric_k"] == 10
    assert data["weight_grid"][0] == 0.0
    assert data["weight_grid"][-1] == 1.0
    assert data["artifacts"]["oracle_scores"] == "oracle_scores.jsonl"

    update_manifest(paths, {"note": "updated"})
    data2 = read_manifest(paths)
    assert data2["note"] == "updated"
    assert "updated_at" in data2


def test_write_and_read_summary(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_dirs()
    write_summary(paths, {"questions": 3, "oracle_scores": 3})
    assert read_summary(paths)["questions"] == 3
    # Empty read when missing:
    assert read_summary(OracleRunPaths(tmp_path / "missing")) == {}


def test_questions_snapshot_round_trips_rows(tmp_path):
    paths = _paths(tmp_path)
    rows = [
        {"question_id": "q1", "question": "A?"},
        {"question_id": "q2", "question": "B?"},
    ]
    n = write_questions_snapshot(paths, rows)
    assert n == 2
    assert read_question_ids(paths.questions_snapshot) == {"q1", "q2"}


def _retrieval_result(retriever_name: str, qid_chunks: dict) -> RetrievalResult:
    chunks = [
        RetrievedChunk(
            chunk_id=cid, text=txt, score=score, rank=0, metadata={"branch": retriever_name.lower()}
        )
        for cid, (txt, score) in qid_chunks.items()
    ]
    return RetrievalResult(
        query="q",
        retriever_name=retriever_name,
        status="OK" if chunks else "NO_CONTEXT",
        chunks=chunks,
        latency_ms={"retrieval": 1.0, "total": 1.0},
    )


def test_retrieval_cache_append_and_read_round_trip(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_dirs()

    r1 = _retrieval_result("Dense", {"c1": ("hello", 1.0), "c2": ("world", 0.5)})
    r2 = _retrieval_result("Dense", {"c3": ("another", 0.9)})

    append_retrieval_line(paths.retrieval_dense, r1, question_id="q1")
    append_retrieval_line(paths.retrieval_dense, r2, question_id="q2")

    cache = read_retrieval_cache(paths.retrieval_dense)
    assert set(cache.keys()) == {"q1", "q2"}
    assert cache["q1"].status == "OK"
    assert [c.chunk_id for c in cache["q1"].chunks] == ["c1", "c2"]
    assert cache["q2"].chunks[0].chunk_id == "c3"


def test_retrieval_cache_handles_no_context(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_dirs()
    r = _retrieval_result("Graph", {})
    append_retrieval_line(paths.retrieval_graph, r, question_id="q1")
    cache = read_retrieval_cache(paths.retrieval_graph)
    assert cache["q1"].status == "NO_CONTEXT"
    assert cache["q1"].chunks == []


def test_oracle_score_row_round_trips_through_jsonl(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_dirs()
    row = OracleScoreRow(
        question_id="q1",
        question="A?",
        dataset_source="nq",
        weight_grid=list(DEFAULT_DENSE_WEIGHT_GRID),
        oracle_metric="stateful_ndcg",
        oracle_metric_k=10,
        scores=[
            WeightBinScore(
                dense_weight=0.0,
                graph_weight=1.0,
                ndcg_primary=0.5,
                diagnostic_ndcg={5: 0.4, 10: 0.5, 20: 0.6},
                diagnostic_hit={5: 1.0, 10: 1.0, 20: 1.0},
                diagnostic_recall={5: 1.0, 10: 1.0, 20: 1.0},
                fused_chunk_ids=["c1", "c2"],
            )
        ],
        best_bin_index=0,
        best_dense_weight=0.0,
        best_score=0.5,
        dense_status="OK",
        graph_status="OK",
    )
    append_oracle_score_rows(paths, [row])

    rows = read_oracle_score_rows(paths)
    assert len(rows) == 1
    back = rows[0]
    assert back["question_id"] == "q1"
    assert back["oracle_metric_k"] == 10
    assert back["scores"][0]["fused_chunk_ids"] == ["c1", "c2"]
    assert back["scores"][0]["diagnostic_ndcg"]["10"] == pytest.approx(0.5)


def test_labels_for_beta_filename(tmp_path):
    paths = _paths(tmp_path)
    assert paths.labels_for_beta(2.0).name == "beta_2.jsonl"
    assert paths.labels_for_beta(0.5).name == "beta_0p5.jsonl"
    assert paths.labels_for_beta(1.25).name == "beta_1p25.jsonl"
