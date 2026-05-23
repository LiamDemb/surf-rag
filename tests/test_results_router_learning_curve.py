from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.figures.router_training_learning_curve import (
    render_router_training_learning_curve_results,
)
from surf_rag.viz.theme import apply_theme


def _write_training_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "loss": "regret",
                "history": [
                    {"epoch": 1, "train_loss": 0.9, "dev_loss": 1.0},
                    {"epoch": 2, "train_loss": 0.7, "dev_loss": 0.8},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fixture(tmp_path: Path) -> Path:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
    (bench / "benchmark").mkdir(parents=True, exist_ok=True)
    (bench / "benchmark" / "benchmark.jsonl").write_text(
        '{"question_id":"q1","dataset_source":"nq"}\n', encoding="utf-8"
    )
    router_base = tmp_path / "router"
    rid_dir = router_base / "rid"
    ds_dir = rid_dir / "dataset"
    oracle_dir = rid_dir / "oracle"
    ds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps({"train": [], "dev": [], "test": ["q1"]}),
        encoding="utf-8",
    )
    (oracle_dir / "oracle_scores.jsonl").write_text('{"question_id":"q1"}\n')
    (oracle_dir / "retrieval_graph.jsonl").write_text('{"question_id":"q1"}\n')

    for role, arch_id, task in (
        ("regressor", "rg-001", "regression"),
        ("classifier", "cls-001", "classification"),
    ):
        paths = make_router_model_paths_for_cli(
            "rid",
            router_base=router_base,
            input_mode="embedding",
            router_architecture_id=arch_id,
            router_task_type=task,
        )
        _write_training_history(paths.training_history)
        paths.manifest.write_text(
            json.dumps({"task_type": task, "model": {"weight_grid": [0.0, 1.0]}}),
            encoding="utf-8",
        )

    run_dir = bench / "evaluations" / "dense-only" / "run1"
    run_dir.mkdir(parents=True)
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
                    "image_format": "pdf",
                    "policies": {"dense-only": {"run_id": "run1"}},
                    "router": {
                        "regressor": {
                            "architecture_id": "rg-001",
                            "input_mode": "embedding",
                            "task_type": "regression",
                        },
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


def test_results_learning_curve_regressor_and_classifier(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    cfg = load_pipeline_config(_write_fixture(tmp_path))
    bundle = ResultsBundle.from_config(cfg)

    for role, artifact_id in (("regressor", "regressor_learning_curve"),):
        spec = ResultsArtifactSpec(
            id=artifact_id,
            kind="figure",
            figure="router_training_learning_curve",
            router_role=role,
            show_loss=True,
            show_regret=False,
            include_dev=True,
            show_plot_subtitle=False,
        )
        paths = render_router_training_learning_curve_results(bundle, spec)
        img = Path(paths["image"])
        assert img.is_file()
        assert img.name == f"{artifact_id}.pdf"

    spec_cls = ResultsArtifactSpec(
        id="classifier_learning_curve",
        kind="figure",
        figure="router_training_learning_curve",
        router_role="classifier",
        show_loss=True,
        include_dev=True,
        show_plot_subtitle=False,
    )
    paths_cls = render_router_training_learning_curve_results(bundle, spec_cls)
    assert Path(paths_cls["image"]).is_file()
