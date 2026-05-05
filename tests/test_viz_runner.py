from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from surf_rag.config.schema import (
    FiguresSection,
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.evaluation.router_model_artifacts import (
    make_router_model_paths_for_cli,
    write_json,
)
from surf_rag.viz.runner import render_figures_from_config


def _write_predictions(path: Path, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = [0.0, 1.0, 0.0]
    rows = [
        {
            "question_id": f"q{i}",
            "oracle_curve": list(curve),
            "predicted_weight": 0.5 + 0.01 * i,
            "is_valid_for_router_training": True,
        }
        for i in range(n)
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _write_training_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.9,
                "dev_loss": 1.2,
                "train_regret": 0.6,
                "dev_regret": 0.7,
            },
            {
                "epoch": 2,
                "train_loss": 0.7,
                "dev_loss": 0.9,
                "train_regret": 0.4,
                "dev_regret": 0.5,
            },
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_render_figures_from_config_runs_two_plots(tmp_path: Path) -> None:
    rb = tmp_path / "router"
    mp = make_router_model_paths_for_cli(
        "rid",
        router_base=rb,
        input_mode="both",
        router_architecture_id="arch1",
    )
    mp.ensure_dirs()
    write_json(mp.manifest, {"model": {"weight_grid": [0.0, 0.5, 1.0]}})
    _write_predictions(mp.predictions("test"), n=4)
    figures = FiguresSection(
        enabled=True,
        plots=[
            {
                "kind": "router_pred_vs_oracle",
                "split": "test",
                "filename_stem": "a",
            },
            {
                "kind": "router_pred_vs_oracle",
                "split": "test",
                "filename_stem": "b",
            },
        ],
    )
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            router_id="rid",
            router_base=str(rb),
            router_architecture_id="arch1",
            data_base=str(tmp_path),
            figures_base=str(tmp_path / "figures"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
        figures=figures,
    )
    outs = render_figures_from_config(cfg, force=True)
    assert len(outs) == 2
    assert outs[0].path_image.name.startswith("a_")
    assert outs[1].path_image.name.startswith("b_")


def test_render_figures_from_config_noop_when_disabled(tmp_path: Path) -> None:
    cfg = replace(PipelineConfig(), figures=replace(FiguresSection(), enabled=False))
    outs = render_figures_from_config(cfg)
    assert outs == []


def test_render_figures_from_config_learning_curve(tmp_path: Path) -> None:
    rb = tmp_path / "router"
    mp = make_router_model_paths_for_cli(
        "rid",
        router_base=rb,
        input_mode="both",
        router_architecture_id="arch1",
    )
    mp.ensure_dirs()
    write_json(
        mp.manifest, {"task_type": "regression", "model": {"weight_grid": [0, 1]}}
    )
    _write_training_history(mp.training_history)
    figures = FiguresSection(
        enabled=True,
        plots=[
            {
                "kind": "router_training_learning_curve",
                "filename_stem": "learning_curve",
            }
        ],
    )
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            router_id="rid",
            router_base=str(rb),
            router_architecture_id="arch1",
            data_base=str(tmp_path),
            figures_base=str(tmp_path / "figures"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
        figures=figures,
    )
    outs = render_figures_from_config(cfg, force=True)
    assert len(outs) == 1
    assert outs[0].path_image.name.startswith("learning_curve")
