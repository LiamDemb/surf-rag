from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.figures._confusion_matrix import routing_regret_for_prediction
from surf_rag.results.figures.classifier_regret_confusion import (
    render_classifier_regret_confusion,
)
from surf_rag.results.figures.route_confusion import render_route_confusion
from surf_rag.viz.theme import PALETTE, apply_theme, sequential_cmap_from_palette


def _write_fixture(tmp_path: Path) -> Path:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
    bench_jsonl = bench / "benchmark" / "benchmark.jsonl"
    bench_jsonl.parent.mkdir(parents=True, exist_ok=True)
    bench_jsonl.write_text(
        '{"question_id":"q1","dataset_source":"nq"}\n',
        encoding="utf-8",
    )

    router_base = tmp_path / "router"
    rid_dir = router_base / "rid"
    ds_dir = rid_dir / "dataset"
    oracle_dir = rid_dir / "oracle"
    ds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps({"train": ["q1"], "dev": [], "test": ["q1"]}),
        encoding="utf-8",
    )
    (oracle_dir / "oracle_scores.jsonl").write_text('{"question_id":"q1"}\n')
    (oracle_dir / "retrieval_graph.jsonl").write_text('{"question_id":"q1"}\n')

    paths = make_router_model_paths_for_cli(
        "rid",
        router_base=router_base,
        input_mode="embedding",
        router_architecture_id="cls-001",
        router_task_type="classification",
    )
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.checkpoint.write_text("stub", encoding="utf-8")
    grid = [0.0, 0.5, 1.0]
    row = {
        "question_id": "q1",
        "predicted_class_id": 1,
        "target_class_id": 0,
        "oracle_curve": [0.2, 0.9, 0.3],
        "target_oracle_best_score": 0.9,
        "weight_grid": grid,
    }
    line = json.dumps(row) + "\n"
    for split in ("train", "test"):
        p = paths.predictions(split)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(line, encoding="utf-8")

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
                        "classifier": {
                            "architecture_id": "cls-001",
                            "input_mode": "embedding",
                            "task_type": "classification",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_routing_regret_dense_endpoint() -> None:
    row = {
        "oracle_curve": [0.2, 0.9, 0.3],
        "target_oracle_best_score": 0.9,
    }
    assert routing_regret_for_prediction(row, 1) == pytest.approx(0.6)
    assert routing_regret_for_prediction(row, 0) == pytest.approx(0.7)


def test_sequential_cmap_uses_palette_keys() -> None:
    apply_theme(dpi=100)
    assert "red" in PALETTE
    cmap_p = sequential_cmap_from_palette("primary")
    cmap_r = sequential_cmap_from_palette("red")
    assert not np.allclose(cmap_p(1.0)[:3], cmap_r(1.0)[:3])


def test_classifier_regret_and_route_confusion_figures(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    cfg = load_pipeline_config(_write_fixture(tmp_path))
    bundle = ResultsBundle.from_config(cfg)

    from surf_rag.config.schema import ResultsArtifactSpec

    regret_paths = render_classifier_regret_confusion(
        bundle,
        ResultsArtifactSpec(
            id="classifier_regret_confusion",
            kind="figure",
            figure="classifier_regret_confusion",
        ),
    )
    assert Path(regret_paths["image"]).is_file()

    route_paths = render_route_confusion(
        bundle,
        ResultsArtifactSpec(
            id="route_confusion", kind="figure", figure="route_confusion"
        ),
    )
    assert Path(route_paths["image"]).is_file()
