"""Small end-to-end test of the oracle pipeline with in-memory fakes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from surf_rag.evaluation.oracle_artifacts import (
    DEFAULT_DENSE_WEIGHT_GRID,
    OracleRunPaths,
    read_oracle_score_rows,
    read_retrieval_cache,
    read_summary,
)
from surf_rag.evaluation.oracle_pipeline import (
    OracleRunConfig,
    populate_retrieval_cache,
    prepare_oracle_run,
    sweep_missing_oracle_scores,
    sweep_weights_for_question,
)
from surf_rag.retrieval.base import BranchRetriever
from surf_rag.retrieval.types import RetrievalResult, RetrievedChunk


class _MapRetriever(BranchRetriever):
    """Return prebuilt RetrievalResult keyed by question text."""

    def __init__(self, name: str, by_question: Dict[str, RetrievalResult]) -> None:
        self.name = name
        self._map = by_question
        self.calls = 0

    def retrieve(self, query: str, **_: object) -> RetrievalResult:
        self.calls += 1
        return self._map[query]


def _chunk(cid: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=text, score=float(score), rank=0, metadata={}
    )


def _make_benchmark(
    tmp_path: Path, rows: List[dict]
) -> Path:
    p = tmp_path / "bench.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _paths(tmp_path: Path) -> OracleRunPaths:
    return OracleRunPaths(tmp_path / "run")


def test_sweep_weights_per_question_prefers_winning_branch():
    dense = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="OK",
        chunks=[_chunk("gold", "the target fact.", 1.0), _chunk("noise", "noise", 0.1)],
    )
    graph = RetrievalResult(
        query="q",
        retriever_name="Graph",
        status="OK",
        chunks=[_chunk("noise2", "other", 0.8)],
    )
    row = {
        "question_id": "q1",
        "question": "q",
        "dataset_source": "nq",
        "gold_support_sentences": ["the target fact."],
    }
    out = sweep_weights_for_question(
        row,
        dense,
        graph,
        weight_grid=DEFAULT_DENSE_WEIGHT_GRID,
        fusion_keep_k=5,
        oracle_metric_k=10,
        diagnostic_metric_ks=(5, 10, 20),
    )
    # Pure dense (1.0): gold is at rank 1 -> NDCG@10 = 1.0.
    # Pure graph (0.0): gold chunk still appears in fused union at score 0.0,
    # but graph's own chunk wins rank 1, so gold falls to rank >=2 -> NDCG<1.
    by_w = {round(s.dense_weight, 2): s.ndcg_primary for s in out.scores}
    assert by_w[1.0] == pytest.approx(1.0)
    assert by_w[0.0] < by_w[1.0]
    assert out.best_dense_weight >= 0.5
    assert out.best_score == pytest.approx(1.0)


def test_populate_retrieval_cache_is_resume_friendly(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_dirs()

    q1_result = RetrievalResult(
        query="a",
        retriever_name="Dense",
        status="OK",
        chunks=[_chunk("c1", "a", 1.0)],
    )
    q2_result = RetrievalResult(
        query="b",
        retriever_name="Dense",
        status="OK",
        chunks=[_chunk("c2", "b", 1.0)],
    )
    dense = _MapRetriever("Dense", {"a": q1_result, "b": q2_result})

    rows = [
        {"question_id": "q1", "question": "a"},
        {"question_id": "q2", "question": "b"},
    ]

    n1 = populate_retrieval_cache(rows, paths.retrieval_dense, dense, label="dense")
    assert n1 == 2
    assert dense.calls == 2

    n2 = populate_retrieval_cache(rows, paths.retrieval_dense, dense, label="dense")
    assert n2 == 0
    assert dense.calls == 2

    rows2 = rows + [{"question_id": "q3", "question": "a"}]
    n3 = populate_retrieval_cache(rows2, paths.retrieval_dense, dense, label="dense")
    assert n3 == 1
    assert dense.calls == 3


def test_sweep_missing_oracle_scores_skips_when_branch_cache_missing(tmp_path):
    paths = _paths(tmp_path)
    paths.ensure_dirs()
    rows = [{"question_id": "q1", "question": "a", "dataset_source": "nq"}]
    n = sweep_missing_oracle_scores(
        rows,
        paths,
        weight_grid=DEFAULT_DENSE_WEIGHT_GRID,
        fusion_keep_k=5,
        oracle_metric_k=10,
        diagnostic_metric_ks=(5, 10, 20),
    )
    assert n == 0
    assert read_oracle_score_rows(paths) == []


def test_prepare_oracle_run_end_to_end(tmp_path):
    bench = _make_benchmark(
        tmp_path,
        [
            {
                "question_id": "q1",
                "question": "What is the target?",
                "dataset_source": "nq",
                "gold_support_sentences": ["the target fact."],
            },
            {
                "question_id": "q2",
                "question": "Another?",
                "dataset_source": "2wiki",
                "gold_support_sentences": ["fact one.", "fact two."],
            },
        ],
    )

    dense_map = {
        "What is the target?": RetrievalResult(
            query="What is the target?",
            retriever_name="Dense",
            status="OK",
            chunks=[_chunk("c1", "the target fact.", 1.0)],
        ),
        "Another?": RetrievalResult(
            query="Another?",
            retriever_name="Dense",
            status="OK",
            chunks=[_chunk("c2", "fact one.", 1.0)],
        ),
    }
    graph_map = {
        "What is the target?": RetrievalResult(
            query="What is the target?",
            retriever_name="Graph",
            status="NO_CONTEXT",
            chunks=[],
        ),
        "Another?": RetrievalResult(
            query="Another?",
            retriever_name="Graph",
            status="OK",
            chunks=[_chunk("c3", "fact two.", 1.0)],
        ),
    }

    paths = _paths(tmp_path)
    cfg = OracleRunConfig(
        benchmark="mix",
        split="dev",
        oracle_run_id="run1",
        benchmark_path=bench,
        retrieval_asset_dir=tmp_path / "assets",
        branch_top_k=25,
        fusion_keep_k=25,
    )

    summary = prepare_oracle_run(
        cfg,
        paths,
        dense_retriever_factory=lambda: _MapRetriever("Dense", dense_map),
        graph_retriever_factory=lambda: _MapRetriever("Graph", graph_map),
    )

    assert summary["questions_snapshot"] == 2
    assert summary["dense_cached"] == 2
    assert summary["graph_cached"] == 2
    assert summary["oracle_scored"] == 2

    # Dense cache round-trips.
    dense_cache = read_retrieval_cache(paths.retrieval_dense)
    assert set(dense_cache.keys()) == {"q1", "q2"}

    # Oracle rows exist and winners match branch story.
    score_rows = read_oracle_score_rows(paths)
    assert {r["question_id"] for r in score_rows} == {"q1", "q2"}
    q1_row = next(r for r in score_rows if r["question_id"] == "q1")
    assert q1_row["best_score"] == pytest.approx(1.0)
    # q1 only has gold in dense, so the best bin should favor dense.
    assert q1_row["best_dense_weight"] >= 0.5

    # Second call is a no-op retrieval/oracle wise (idempotent).
    summary2 = prepare_oracle_run(
        cfg,
        paths,
        dense_retriever_factory=lambda: _MapRetriever("Dense", dense_map),
        graph_retriever_factory=lambda: _MapRetriever("Graph", graph_map),
    )
    assert summary2["newly_retrieved_dense"] == 0
    assert summary2["newly_retrieved_graph"] == 0
    assert summary2["newly_scored"] == 0

    # Summary file round-trips.
    assert read_summary(paths)["oracle_scored"] == 2


def test_prepare_oracle_run_handles_retriever_error_gracefully(tmp_path):
    bench = _make_benchmark(
        tmp_path,
        [
            {
                "question_id": "q1",
                "question": "q",
                "dataset_source": "nq",
                "gold_support_sentences": ["gold"],
            }
        ],
    )

    class _BrokenRetriever(BranchRetriever):
        def retrieve(self, query: str, **_: object) -> RetrievalResult:
            raise RuntimeError("boom")

    ok_dense = _MapRetriever(
        "Dense",
        {
            "q": RetrievalResult(
                query="q",
                retriever_name="Dense",
                status="OK",
                chunks=[_chunk("c1", "gold", 1.0)],
            )
        },
    )

    paths = _paths(tmp_path)
    cfg = OracleRunConfig(
        benchmark="nq",
        split="dev",
        oracle_run_id="run1",
        benchmark_path=bench,
        retrieval_asset_dir=tmp_path / "assets",
    )

    summary = prepare_oracle_run(
        cfg,
        paths,
        dense_retriever_factory=lambda: ok_dense,
        graph_retriever_factory=lambda: _BrokenRetriever(),
    )

    assert summary["dense_cached"] == 1
    assert summary["graph_cached"] == 1  # error row still recorded
    graph_cache = read_retrieval_cache(paths.retrieval_graph)
    assert graph_cache["q1"].status == "ERROR"
    # Oracle still runs with dense only; graph contributes nothing.
    score_rows = read_oracle_score_rows(paths)
    assert score_rows[0]["graph_status"] == "ERROR"
