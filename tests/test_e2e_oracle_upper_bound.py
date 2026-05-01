from __future__ import annotations

import json
from pathlib import Path

import pytest

from surf_rag.evaluation.e2e_runner import e2e_prepare_and_submit
from surf_rag.evaluation.run_artifacts import RunArtifactPaths


def _write_benchmark(path: Path, qid: str = "q1") -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": qid,
                "question": "What is alpha?",
                "gold_answers": ["alpha"],
                "gold_support_sentences": ["alpha sentence"],
                "dataset_source": "nq",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_split_ids(path: Path, qids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": [],
                "dev": [],
                "test": qids,
                "counts": {"train": 0, "dev": 0, "test": len(qids)},
            }
        ),
        encoding="utf-8",
    )


def _retrieval_row(qid: str, chunk_id: str, text: str, score: float) -> dict:
    return {
        "question_id": qid,
        "query": "What is alpha?",
        "retriever_name": "Dense",
        "status": "OK",
        "chunks": [
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": score,
                "rank": 0,
                "metadata": {},
            }
        ],
        "latency_ms": {"total": 1.0, "retrieval": 1.0},
        "error": None,
        "debug_info": None,
    }


def _write_oracle_bundle(
    router_base: Path, router_id: str, *, include_score: bool = True
) -> None:
    oracle = router_base / router_id / "oracle"
    oracle.mkdir(parents=True, exist_ok=True)
    qid = "q1"
    (oracle / "retrieval_dense.jsonl").write_text(
        json.dumps(_retrieval_row(qid, "d1", "dense chunk", 0.9)) + "\n",
        encoding="utf-8",
    )
    graph_row = _retrieval_row(qid, "g1", "graph chunk", 0.8)
    graph_row["retriever_name"] = "Graph"
    (oracle / "retrieval_graph.jsonl").write_text(
        json.dumps(graph_row) + "\n",
        encoding="utf-8",
    )
    if include_score:
        score_row = {
            "question_id": qid,
            "question": "What is alpha?",
            "dataset_source": "nq",
            "weight_grid": [0.0, 0.5, 1.0],
            "oracle_metric": "stateful_ndcg",
            "oracle_metric_k": 10,
            "scores": [
                {
                    "dense_weight": 0.0,
                    "graph_weight": 1.0,
                    "ndcg_primary": 0.2,
                    "diagnostic_ndcg": {"5": 0.2, "10": 0.2, "20": 0.2},
                    "diagnostic_hit": {"5": 1.0, "10": 1.0, "20": 1.0},
                    "diagnostic_recall": {"5": 1.0, "10": 1.0, "20": 1.0},
                    "fused_chunk_ids": ["g1", "d1"],
                },
                {
                    "dense_weight": 0.7,
                    "graph_weight": 0.3,
                    "ndcg_primary": 0.9,
                    "diagnostic_ndcg": {"5": 0.9, "10": 0.9, "20": 0.9},
                    "diagnostic_hit": {"5": 1.0, "10": 1.0, "20": 1.0},
                    "diagnostic_recall": {"5": 1.0, "10": 1.0, "20": 1.0},
                    "fused_chunk_ids": ["d1", "g1"],
                },
            ],
            "best_bin_index": 1,
            "best_dense_weight": 0.7,
            "best_score": 0.9,
            "dense_status": "OK",
            "graph_status": "OK",
        }
        (oracle / "oracle_scores.jsonl").write_text(
            json.dumps(score_row) + "\n",
            encoding="utf-8",
        )


def test_oracle_upper_bound_prepare_writes_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    router_base = tmp_path / "router"
    router_id = "rid"
    _write_oracle_bundle(router_base, router_id, include_score=True)
    split_ids = router_base / router_id / "dataset" / "split_question_ids.json"
    _write_split_ids(split_ids, ["q1"])
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    out = RunArtifactPaths(run_root=tmp_path / "run")

    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.get_generator_prompt", lambda: "p"
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
        routing_policy="oracle-upper-bound",
        retrieval_asset_dir=tmp_path,
        router_id=router_id,
        router_base=router_base,
        dry_run=True,
        run_paths_override=out,
        fusion_keep_k=1,
    )
    assert rc == 0
    pre_rows = (
        out.retrieval_results_pretrunc_jsonl()
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    gen_rows = (
        out.retrieval_results_jsonl().read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(pre_rows) == 1
    assert len(gen_rows) == 1
    pre = json.loads(pre_rows[0])
    gen = json.loads(gen_rows[0])
    assert pre["debug_info"]["routing"]["routing_policy"] == "oracle-upper-bound"
    assert pre["debug_info"]["routing"]["oracle_dense_weight"] == pytest.approx(0.7)
    assert pre["debug_info"]["routing"]["oracle_best_bin_index"] == 1
    assert gen["debug_info"]["routing"]["oracle_source_router_id"] == router_id


def test_oracle_upper_bound_prepare_fails_non_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test-only"):
        e2e_prepare_and_submit(
            tmp_path / "bench.jsonl",
            benchmark_base=tmp_path,
            benchmark_name="b",
            benchmark_id="id",
            split="dev",
            run_id="r1",
            routing_policy="oracle-upper-bound",
            retrieval_asset_dir=tmp_path,
            router_id="rid",
            dry_run=True,
        )


def test_oracle_upper_bound_prepare_strict_missing_score(
    monkeypatch, tmp_path: Path
) -> None:
    router_base = tmp_path / "router"
    router_id = "rid"
    _write_oracle_bundle(router_base, router_id, include_score=False)
    split_ids = router_base / router_id / "dataset" / "split_question_ids.json"
    _write_split_ids(split_ids, ["q1"])
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    out = RunArtifactPaths(run_root=tmp_path / "run")
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.get_generator_prompt", lambda: "p"
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.finalize_batch_submission",
        lambda *args, **kwargs: 0,
    )
    with pytest.raises(ValueError, match="oracle_scores.jsonl"):
        e2e_prepare_and_submit(
            bench,
            benchmark_base=tmp_path,
            benchmark_name="b",
            benchmark_id="id",
            split="test",
            run_id="r1",
            routing_policy="oracle-upper-bound",
            retrieval_asset_dir=tmp_path,
            router_id=router_id,
            router_base=router_base,
            dry_run=True,
            run_paths_override=out,
        )


def test_oracle_upper_bound_prepare_strict_missing_dense_cache(
    monkeypatch, tmp_path: Path
) -> None:
    router_base = tmp_path / "router"
    router_id = "rid"
    _write_oracle_bundle(router_base, router_id, include_score=True)
    (router_base / router_id / "oracle" / "retrieval_dense.jsonl").write_text(
        "", encoding="utf-8"
    )
    split_ids = router_base / router_id / "dataset" / "split_question_ids.json"
    _write_split_ids(split_ids, ["q1"])
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    out = RunArtifactPaths(run_root=tmp_path / "run")
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.get_generator_prompt", lambda: "p"
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.finalize_batch_submission",
        lambda *args, **kwargs: 0,
    )
    with pytest.raises(ValueError, match="retrieval_dense.jsonl"):
        e2e_prepare_and_submit(
            bench,
            benchmark_base=tmp_path,
            benchmark_name="b",
            benchmark_id="id",
            split="test",
            run_id="r1",
            routing_policy="oracle-upper-bound",
            retrieval_asset_dir=tmp_path,
            router_id=router_id,
            router_base=router_base,
            dry_run=True,
            run_paths_override=out,
        )


def test_oracle_upper_bound_prepare_strict_invalid_best_bin(
    monkeypatch, tmp_path: Path
) -> None:
    router_base = tmp_path / "router"
    router_id = "rid"
    _write_oracle_bundle(router_base, router_id, include_score=True)
    score_path = router_base / router_id / "oracle" / "oracle_scores.jsonl"
    row = json.loads(score_path.read_text(encoding="utf-8").strip())
    row["best_bin_index"] = 99
    score_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    split_ids = router_base / router_id / "dataset" / "split_question_ids.json"
    _write_split_ids(split_ids, ["q1"])
    bench = tmp_path / "bench.jsonl"
    _write_benchmark(bench)
    out = RunArtifactPaths(run_root=tmp_path / "run")
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.get_generator_prompt", lambda: "p"
    )
    monkeypatch.setattr(
        "surf_rag.evaluation.e2e_runner.finalize_batch_submission",
        lambda *args, **kwargs: 0,
    )
    with pytest.raises(ValueError, match="best_bin_index"):
        e2e_prepare_and_submit(
            bench,
            benchmark_base=tmp_path,
            benchmark_name="b",
            benchmark_id="id",
            split="test",
            run_id="r1",
            routing_policy="oracle-upper-bound",
            retrieval_asset_dir=tmp_path,
            router_id=router_id,
            router_base=router_base,
            dry_run=True,
            run_paths_override=out,
        )
