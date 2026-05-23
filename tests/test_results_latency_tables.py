"""Tests for results-build latency tables."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.results.build import build_results
from surf_rag.results.tables.latency_tables import (
    build_e2e_startup_latency,
    build_oracle_ops_summary,
    build_pipeline_latency_summary,
    build_pipeline_ops_timing,
    build_router_training_timing,
)
from surf_rag.results.bundle import ResultsBundle


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _minimal_bundle(tmp_path: Path) -> ResultsBundle:
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
    _write_jsonl(oracle_dir / "oracle_scores.jsonl", [{"question_id": "q1"}])
    _write_jsonl(oracle_dir / "retrieval_graph.jsonl", [{"question_id": "q1"}])
    (oracle_dir / "summary.json").write_text(
        json.dumps(
            {
                "router_id": "rid",
                "oracle_sweep_wall_s": 12.5,
                "questions_snapshot": 2,
                "oracle_scored": 2,
                "newly_scored": 2,
                "dense_cached": 2,
                "graph_cached": 2,
            }
        ),
        encoding="utf-8",
    )

    reg_dir = router / "models" / "rg" / "regression" / "embedding"
    reg_dir.mkdir(parents=True, exist_ok=True)
    (reg_dir / "metrics.json").write_text(
        json.dumps(
            {
                "task_type": "regression",
                "training_wall_s": 3.25,
                "best_epoch": 4,
                "architecture": "mlp-v1",
            }
        ),
        encoding="utf-8",
    )

    run_dir = bench / "evaluations" / "dense-only" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "per_question": [
                    {
                        "question_id": "q1",
                        "latency_ms": {
                            "retrieval_reported_total_ms": 100.0,
                            "retrieval_stage_total_ms": 95.0,
                            "dense_branch_ms": 90.0,
                        },
                    },
                    {
                        "question_id": "q2",
                        "latency_ms": {
                            "retrieval_reported_total_ms": 110.0,
                            "retrieval_stage_total_ms": 105.0,
                            "dense_branch_ms": 100.0,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "e2e": {
                    "startup_latency_ms": {
                        "startup_total_ms": 1000.0,
                        "startup_components": {
                            "dense_init_ms": 400.0,
                            "graph_init_ms": 500.0,
                            "router_init_ms": 100.0,
                        },
                    }
                }
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
                    "bundle_id": "lat-bundle",
                    "output_root": str(tmp_path / "out"),
                    "split": "test",
                    "image_format": "png",
                    "router": {
                        "regressor": {
                            "architecture_id": "rg",
                            "input_mode": "embedding",
                            "task_type": "regression",
                        },
                    },
                    "policies": {"dense-only": {"run_id": "run1"}},
                    "artifacts": [
                        {"id": "oracle_ops_summary", "kind": "table"},
                        {"id": "pipeline_ops_timing", "kind": "table"},
                        {"id": "router_training_timing", "kind": "table"},
                        {"id": "e2e_startup_latency", "kind": "table"},
                        {"id": "pipeline_latency_summary", "kind": "table"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(cfg_path)
    return ResultsBundle.from_config(cfg, config_path=cfg_path)


def test_latency_table_builders(tmp_path: Path) -> None:
    from surf_rag.config.schema import ResultsArtifactSpec

    bundle = _minimal_bundle(tmp_path)
    spec = ResultsArtifactSpec(id="x", kind="table")

    ops_df, _ = build_oracle_ops_summary(bundle, spec)
    assert float(ops_df.iloc[0]["oracle_sweep_wall_s"]) == 12.5

    pipe_ops_df, _ = build_pipeline_ops_timing(bundle, spec)
    stages = set(pipe_ops_df["stage"])
    assert "oracle_weight_sweep" in stages
    assert "router_train_regression" in stages

    train_df, _ = build_router_training_timing(bundle, spec)
    assert float(train_df.iloc[0]["training_wall_s"]) == 3.25

    startup_df, _ = build_e2e_startup_latency(bundle, spec)
    assert float(startup_df.iloc[0]["startup_total_ms"]) == 1000.0

    lat_df, _ = build_pipeline_latency_summary(bundle, spec)
    assert "dense-only" in set(lat_df["policy"])
    assert "retrieval_reported_total" in set(lat_df["metric"])


def test_results_build_latency_artifacts(tmp_path: Path) -> None:
    bundle = _minimal_bundle(tmp_path)
    cfg = bundle.cfg
    cfg_path = tmp_path / "results.yaml"
    manifest = build_results(cfg, config_path=cfg_path)
    out = tmp_path / "out" / "lat-bundle" / "tables"
    assert (out / "oracle_ops_summary.csv").is_file()
    assert (out / "pipeline_ops_timing.csv").is_file()
    assert (out / "router_training_timing.csv").is_file()
    assert (out / "e2e_startup_latency.csv").is_file()
    assert (out / "pipeline_latency_summary.csv").is_file()
    assert any(
        r.id == "pipeline_latency_summary" and r.status == "ok"
        for r in manifest.artifacts
    )
