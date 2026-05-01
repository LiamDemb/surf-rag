from __future__ import annotations

import json
from pathlib import Path

from surf_rag.evaluation.e2e_runner import evaluate_e2e_run
from surf_rag.evaluation.run_artifacts import RunArtifactPaths


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


def test_metrics_schema_contains_family_status_and_reported_ks(tmp_path: Path) -> None:
    run = RunArtifactPaths(run_root=tmp_path / "run")
    run.ensure_dirs()
    run.manifest.write_text(
        json.dumps({"e2e": {"reranker": "cross_encoder", "rerank_top_k": 10}}),
        encoding="utf-8",
    )
    _write_retrieval(run.retrieval_results_jsonl())
    _write_retrieval(run.retrieval_results_pretrunc_jsonl())
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)

    report = evaluate_e2e_run(run_paths=run, benchmark_path=bench)
    assert "overlap_breakdown" in report
    assert "overlap" not in report
    assert "retrieval_families" not in report
    all_block = report["overlap_breakdown"]["all"]
    assert set(all_block.keys()) == {
        "count",
        "latency_ms",
        "qa",
        "retrieval_before_ce",
        "retrieval_after_ce",
    }
    assert all_block["retrieval_before_ce"]["status"] in {
        "ok",
        "unavailable",
        "not_applicable",
    }
    assert all_block["retrieval_after_ce"]["status"] in {
        "ok",
        "unavailable",
        "not_applicable",
    }
    assert "reported_ks" in all_block["retrieval_before_ce"]
    assert "reported_ks" in all_block["retrieval_after_ce"]
