from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from surf_rag.evaluation.artifact_paths import e2e_policy_run_dir


def _write_benchmark(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "what?",
                "gold_answers": ["a"],
                "gold_support_sentences": ["s1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_retrieval(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "query": "what?",
                "retriever_name": "Dense",
                "status": "OK",
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "text": "s1",
                        "score": 1.0,
                        "rank": 0,
                        "metadata": {},
                    }
                ],
                "latency_ms": {"retrieval_stage_total_ms": 3.0},
                "error": None,
                "debug_info": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_cmd_evaluate_emits_family_separated_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("DATA_BASE", raising=False)
    monkeypatch.delenv("ROUTER_BASE", raising=False)
    monkeypatch.delenv("BENCHMARK_BASE", raising=False)
    from scripts.e2e_benchmark import cmd_evaluate

    benchmark_base = tmp_path / "bench_base"
    benchmark_name = "surf"
    benchmark_id = "main"
    run_id = "r1"
    run_root = e2e_policy_run_dir(
        benchmark_base, benchmark_name, benchmark_id, "dense-only", run_id
    )
    retrieval = run_root / "retrieval" / "retrieval_results.jsonl"
    pretrunc = run_root / "retrieval" / "retrieval_results_pretrunc.jsonl"
    manifest = run_root / "manifest.json"
    _write_retrieval(retrieval)
    _write_retrieval(pretrunc)
    manifest.write_text(
        json.dumps({"e2e": {"reranker": "cross_encoder", "rerank_top_k": 5}}),
        encoding="utf-8",
    )
    bench = tmp_path / "benchmark.jsonl"
    _write_benchmark(bench)

    args = Namespace(
        benchmark_base=benchmark_base,
        benchmark_name=benchmark_name,
        benchmark_id=benchmark_id,
        benchmark_path=bench,
        run_id=run_id,
        policy="dense-only",
        split_question_ids=None,
        router_id=None,
        router_base=None,
    )
    rc = cmd_evaluate(args)
    assert rc == 0
    metrics_path = run_root / "metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "overlap_breakdown" in report
    assert "retrieval_families" not in report
    assert "overlap" not in report
    all_block = report["overlap_breakdown"]["all"]
    assert "retrieval_before_ce" in all_block
    assert "retrieval_after_ce" in all_block
