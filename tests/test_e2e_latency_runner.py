from __future__ import annotations

import json
from pathlib import Path

from surf_rag.evaluation.e2e_runner import e2e_prepare_and_submit, evaluate_e2e_run
from surf_rag.evaluation.run_artifacts import RunArtifactPaths
from surf_rag.retrieval.types import RetrievalResult


class _FakeRetriever:
    def __init__(self, name: str) -> None:
        self.name = name

    def retrieve(self, query: str, **kwargs: object) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            retriever_name=self.name,
            status="NO_CONTEXT",
            chunks=[],
            latency_ms={"total": 5.0, "retrieval": 4.0},
        )


def _write_benchmark(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "what?",
                "gold_answers": ["a"],
                "gold_support_sentences": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_e2e_prepare_dense_only_setup_isolation(monkeypatch, tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    out = RunArtifactPaths(run_root=tmp_path / "run")
    calls = {"dense": 0, "graph": 0}

    def _dense_factory(output_dir: str):
        calls["dense"] += 1
        return _FakeRetriever("Dense")

    def _graph_factory(output_dir: str, top_k: int = 10, *, pipeline_config=None):
        calls["graph"] += 1
        return _FakeRetriever("Graph")

    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.get_generator_prompt", lambda: "p"
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.build_dense_retriever", _dense_factory
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.build_graph_retriever", _graph_factory
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.finalize_batch_submission",
        lambda *args, **kwargs: 0,
    )

    rc = e2e_prepare_and_submit(
        bench,
        benchmark_base=tmp_path,
        benchmark_name="b",
        benchmark_id="id",
        split="test",
        run_id="r1",
        routing_policy="dense-only",
        retrieval_asset_dir=tmp_path,
        dry_run=True,
        run_paths_override=out,
    )
    assert rc == 0
    assert calls["dense"] == 1
    assert calls["graph"] == 0

    rows = (
        out.retrieval_results_jsonl().read_text(encoding="utf-8").strip().splitlines()
    )
    payload = json.loads(rows[0])
    lat = payload["latency_ms"]
    assert "dense_branch_ms" in lat
    assert "graph_branch_ms" not in lat
    assert "retrieval_stage_total_ms" in lat


def test_evaluate_e2e_run_includes_startup_latency(tmp_path: Path) -> None:
    run = RunArtifactPaths(run_root=tmp_path / "run")
    run.ensure_dirs()
    run.manifest.write_text(
        json.dumps({"e2e": {"startup_latency_ms": {"startup_total_ms": 123.0}}}),
        encoding="utf-8",
    )
    run.retrieval_results_jsonl().write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "what?",
                "retriever_name": "Dense",
                "status": "NO_CONTEXT",
                "chunks": [],
                "latency_ms": {"retrieval_stage_total_ms": 9.0, "dense_branch_ms": 9.0},
                "error": None,
                "debug_info": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    rep = evaluate_e2e_run(run_paths=run, benchmark_path=bench)
    assert rep["startup_latency_ms"]["startup_total_ms"] == 123.0
    assert rep["per_question"][0]["latency_ms"]["retrieval_stage_total_ms"] == 9.0
    assert (
        rep["overlap_breakdown"]["all"]["latency_ms"]["retrieval_stage_total"][
            "mean_ms"
        ]
        == 9.0
    )


def test_prepare_writes_pretrunc_and_generation_retrieval_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    out = RunArtifactPaths(run_root=tmp_path / "run")

    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.get_generator_prompt", lambda: "p"
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.build_dense_retriever",
        lambda output_dir: _FakeRetriever("Dense"),
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.build_graph_retriever",
        lambda output_dir, top_k=10, *, pipeline_config=None: _FakeRetriever("Graph"),
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.finalize_batch_submission",
        lambda *args, **kwargs: 0,
    )

    rc = e2e_prepare_and_submit(
        bench,
        benchmark_base=tmp_path,
        benchmark_name="b",
        benchmark_id="id",
        split="test",
        run_id="r1",
        routing_policy="dense-only",
        retrieval_asset_dir=tmp_path,
        dry_run=True,
        fusion_keep_k=1,
        run_paths_override=out,
    )
    assert rc == 0
    assert out.retrieval_results_jsonl().is_file()
    assert out.retrieval_results_pretrunc_jsonl().is_file()
    gen_rows = (
        out.retrieval_results_jsonl().read_text(encoding="utf-8").strip().splitlines()
    )
    pre_rows = (
        out.retrieval_results_pretrunc_jsonl()
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(gen_rows) == 1
    assert len(pre_rows) == 1


def test_evaluate_outputs_retrieval_blocks_and_no_fallback(tmp_path: Path) -> None:
    run = RunArtifactPaths(run_root=tmp_path / "run")
    run.ensure_dirs()
    run.manifest.write_text(
        json.dumps({"e2e": {"reranker": "none", "rerank_top_k": 10}}), encoding="utf-8"
    )
    run.retrieval_results_jsonl().write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "what?",
                "retriever_name": "Dense",
                "status": "NO_CONTEXT",
                "chunks": [],
                "latency_ms": {"retrieval_stage_total_ms": 9.0, "dense_branch_ms": 9.0},
                "error": None,
                "debug_info": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    rep = evaluate_e2e_run(run_paths=run, benchmark_path=bench)
    all_block = rep["overlap_breakdown"]["all"]
    assert all_block["retrieval_before_ce"]["status"] == "unavailable"
    assert all_block["retrieval_after_ce"]["status"] == "not_applicable"
    assert set(rep["per_question"][0].keys()) == {
        "question_id",
        "qa",
        "latency_ms",
        "retrieval_before_ce",
        "retrieval_after_ce",
    }


def test_evaluate_post_ce_k_filter_respects_rerank_top_k(tmp_path: Path) -> None:
    run = RunArtifactPaths(run_root=tmp_path / "run")
    run.ensure_dirs()
    run.manifest.write_text(
        json.dumps({"e2e": {"reranker": "cross_encoder", "rerank_top_k": 5}}),
        encoding="utf-8",
    )
    row = {
        "question_id": "q1",
        "query": "what?",
        "retriever_name": "Dense",
        "status": "OK",
        "chunks": [
            {"chunk_id": "c1", "text": "a", "score": 1.0, "rank": 0, "metadata": {}}
        ],
        "latency_ms": {"retrieval_stage_total_ms": 9.0, "dense_branch_ms": 9.0},
        "error": None,
        "debug_info": None,
    }
    run.retrieval_results_jsonl().write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    run.retrieval_results_pretrunc_jsonl().write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    rep = evaluate_e2e_run(run_paths=run, benchmark_path=bench)
    assert rep["overlap_breakdown"]["all"]["retrieval_after_ce"]["status"] == "ok"
    assert rep["overlap_breakdown"]["all"]["retrieval_after_ce"]["reported_ks"] == [5]
