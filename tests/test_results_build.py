from __future__ import annotations

import json
from pathlib import Path

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.results.build import build_results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _minimal_oracle_scores() -> list[dict]:
    grid = [0.0, 0.5, 1.0]
    scores = []
    for w, v in zip(grid, [0.2, 0.5, 0.9]):
        scores.append(
            {
                "dense_weight": w,
                "oracle_objective_value": v,
                "diagnostic_ndcg": {"5": v, "10": v, "20": v},
                "diagnostic_hit": {"5": v},
                "diagnostic_recall": {"5": v},
            }
        )
    return [
        {
            "question_id": "q1",
            "dataset_source": "nq",
            "weight_grid": grid,
            "scores": scores,
        },
        {
            "question_id": "q2",
            "dataset_source": "2wiki",
            "weight_grid": grid,
            "scores": scores,
        },
    ]


def _metrics_row(qid: str, ndcg: float, correct: bool) -> dict:
    return {
        "question_id": qid,
        "retrieval_before_ce": {
            "retrieval": {
                "5": {"ndcg": ndcg, "hit": 1.0, "recall": 0.5},
                "10": {"ndcg": ndcg, "hit": 1.0, "recall": 0.5},
                "20": {"ndcg": ndcg, "hit": 1.0, "recall": 0.5},
            }
        },
        "qa_llm_judge": {"correct": correct},
    }


def test_results_build_oracle_tables_only(tmp_path: Path) -> None:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
    router = tmp_path / "router" / "rid"
    oracle_dir = router / "oracle"
    ds_dir = router / "dataset"

    _write_jsonl(
        bench / "benchmark" / "benchmark.jsonl",
        [
            {"question_id": "q1", "dataset_source": "nq"},
            {"question_id": "q2", "dataset_source": "2wiki"},
        ],
    )
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps({"train": [], "dev": [], "test": ["q1", "q2"]}),
        encoding="utf-8",
    )
    _write_jsonl(oracle_dir / "oracle_scores.jsonl", _minimal_oracle_scores())
    _write_jsonl(
        oracle_dir / "retrieval_graph.jsonl",
        [
            {
                "question_id": "q1",
                "status": "OK",
                "chunks": [{"chunk_id": "c1", "text": "t"}],
            },
            {"question_id": "q2", "status": "NO_CONTEXT", "chunks": []},
        ],
    )

    policy = "dense-only"
    run_dir = bench / "evaluations" / policy / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "per_question": [
                    _metrics_row("q1", 0.5, True),
                    _metrics_row("q2", 0.3, False),
                ]
            }
        ),
        encoding="utf-8",
    )

    cfg_path = tmp_path / "results.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_base": str(tmp_path),
                    "benchmark_base": str(tmp_path / "benchmarks"),
                    "router_base": str(tmp_path / "router"),
                    "benchmark_name": "bench",
                    "benchmark_id": "v1",
                    "router_id": "rid",
                },
                "results": {
                    "bundle_id": "test-bundle",
                    "output_root": str(tmp_path / "out"),
                    "split": "test",
                    "image_format": "png",
                    "policies": {policy: {"run_id": "run1"}},
                    "artifacts": [
                        {"id": "bench_splits", "kind": "table"},
                        {"id": "oracle_stats", "kind": "table"},
                        {
                            "id": "endpoint_pref",
                            "kind": "figure",
                            "figure": "endpoint_pref",
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(cfg_path)
    manifest = build_results(cfg, config_path=cfg_path)
    out = tmp_path / "out" / "test-bundle"
    assert (out / "tables" / "bench_splits.csv").is_file()
    assert (out / "tables" / "oracle_stats.csv").is_file()
    assert (out / "figures" / "endpoint_pref.png").is_file()
    assert (out / "manifest.json").is_file()
    assert manifest.bundle_id == "test-bundle"
