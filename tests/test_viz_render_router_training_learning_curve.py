from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.router_training_learning_curve import (
    render_router_training_learning_curve,
)
from surf_rag.viz.specs import RouterTrainingLearningCurveSpec
from surf_rag.viz.theme import apply_theme


def _ctx(tmp_path: Path) -> FigureRunContext:
    rb = tmp_path / "router"
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            router_base=str(rb),
            router_id="rid",
            router_architecture_id="arch1",
            figures_base=str(tmp_path / "figures"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
    )
    return FigureRunContext.from_pipeline(cfg, force=True)


def _write_training_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "loss": "regret",
        "loss_effective": "regret",
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.9,
                "dev_loss": 1.1,
                "train_regret": 0.6,
                "dev_regret": 0.7,
            },
            {
                "epoch": 2,
                "train_loss": 0.7,
                "dev_loss": 0.85,
                "train_regret": 0.4,
                "dev_regret": 0.5,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_render_router_training_learning_curve_writes_outputs(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    mp = make_router_model_paths_for_cli(
        "rid",
        router_base=tmp_path / "router",
        input_mode="both",
        router_architecture_id="arch1",
    )
    _write_training_history(mp.training_history)
    mp.manifest.write_text(
        json.dumps({"task_type": "regression", "model": {"weight_grid": [0.0, 1.0]}})
        + "\n",
        encoding="utf-8",
    )

    spec = RouterTrainingLearningCurveSpec(kind="router_training_learning_curve")
    out = render_router_training_learning_curve(spec, ctx)
    assert out.path_image.is_file()
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["figure_kind"] == "router_training_learning_curve"
    assert meta["n_epochs"] == 2
    assert "train_loss" in meta["metrics_plotted"]
    assert "dev_loss" in meta["metrics_plotted"]


def test_render_router_training_learning_curve_no_dev_when_disabled(
    tmp_path: Path,
) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    mp = make_router_model_paths_for_cli(
        "rid",
        router_base=tmp_path / "router",
        input_mode="both",
        router_architecture_id="arch1",
    )
    _write_training_history(mp.training_history)
    mp.manifest.write_text(
        json.dumps({"task_type": "regression", "model": {"weight_grid": [0.0, 1.0]}})
        + "\n",
        encoding="utf-8",
    )
    spec = RouterTrainingLearningCurveSpec(
        kind="router_training_learning_curve",
        include_dev=False,
        show_regret=False,
        filename_stem="train_only_loss",
    )
    out = render_router_training_learning_curve(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert "train_loss" in meta["metrics_plotted"]
    assert "dev_loss" not in meta["metrics_plotted"]
