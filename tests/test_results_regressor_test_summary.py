from __future__ import annotations

import json
from pathlib import Path

import yaml

import pytest

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.tables.regressor_test_summary import build_regressor_test_summary


def _write_fixture(tmp_path: Path) -> Path:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
    bench_jsonl = bench / "benchmark" / "benchmark.jsonl"
    bench_jsonl.parent.mkdir(parents=True, exist_ok=True)
    bench_jsonl.write_text(
        "\n".join(
            [
                '{"question_id":"q_nq","dataset_source":"nq"}',
                '{"question_id":"q_2wiki","dataset_source":"2wiki"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    router_base = tmp_path / "router"
    rid_dir = router_base / "rid"
    ds_dir = rid_dir / "dataset"
    oracle_dir = rid_dir / "oracle"
    ds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps({"train": [], "dev": [], "test": ["q_nq", "q_2wiki"]}),
        encoding="utf-8",
    )
    (oracle_dir / "oracle_scores.jsonl").write_text('{"question_id":"q_nq"}\n')
    (oracle_dir / "retrieval_graph.jsonl").write_text('{"question_id":"q_nq"}\n')

    paths = make_router_model_paths_for_cli(
        "rid",
        router_base=router_base,
        input_mode="embedding",
        router_architecture_id="rg-001",
        router_task_type="regression",
    )
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.checkpoint.write_text("stub", encoding="utf-8")
    pred_path = paths.predictions("test")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    grid = [0.0, 0.5, 1.0]
    pred_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "q_nq",
                        "predicted_weight": 0.8,
                        "oracle_curve": [0.1, 1.0, 0.1],
                        "weight_grid": grid,
                    }
                ),
                json.dumps(
                    {
                        "question_id": "q_2wiki",
                        "predicted_weight": 0.2,
                        "oracle_curve": [0.1, 1.0, 0.1],
                        "weight_grid": grid,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = bench / "evaluations" / "dense-only" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text('{"per_question":[]}', encoding="utf-8")

    cfg_path = tmp_path / "results.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_base": str(tmp_path),
                    "benchmark_base": str(tmp_path / "benchmarks"),
                    "router_base": str(router_base),
                    "benchmark_name": "bench",
                    "benchmark_id": "v1",
                    "router_id": "rid",
                },
                "results": {
                    "bundle_id": "t",
                    "output_root": str(tmp_path / "out"),
                    "split": "test",
                    "image_format": "png",
                    "policies": {"dense-only": {"run_id": "run1"}},
                    "router": {
                        "regressor": {
                            "architecture_id": "rg-001",
                            "input_mode": "embedding",
                            "task_type": "regression",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_regressor_test_summary_by_dataset(tmp_path: Path) -> None:
    cfg = load_pipeline_config(_write_fixture(tmp_path))
    bundle = ResultsBundle.from_config(cfg)
    df, paths = build_regressor_test_summary(bundle)

    assert list(df["dataset"]) == ["Total", "NQ", "2Wiki"]
    assert (df["mean_regret_oracle"] == 0.0).all()
    assert df.loc[df["dataset"] == "NQ", "mean_predicted_weight"].iloc[0] == 0.8
    assert df.loc[df["dataset"] == "2Wiki", "mean_predicted_weight"].iloc[0] == 0.2
    assert df.loc[df["dataset"] == "Total", "n"].iloc[0] == 2
    w_star = float(df["pooled_weight_min_mean_regret"].iloc[0])
    assert w_star == pytest.approx(0.5, abs=0.02)
    assert (df["pooled_weight_min_mean_regret"] == w_star).all()
    assert df.loc[df["dataset"] == "Total", "mean_regret_at_pooled_weight"].iloc[
        0
    ] == pytest.approx(0.0, abs=1e-9)
    assert Path(paths["csv"]).is_file()
