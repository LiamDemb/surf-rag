from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from dataclasses import replace

from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.router_pred_vs_oracle import render_router_pred_vs_oracle
from surf_rag.viz.specs import RouterPredVsOracleSpec
from surf_rag.viz.theme import apply_theme


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _ctx(
    tmp_path: Path, *, arch: str | None = "tower", force: bool = True
) -> FigureRunContext:
    rb = tmp_path / "router"
    make_router_model_paths_for_cli(
        "rid",
        router_base=rb,
        input_mode="both",
        router_architecture_id=arch,
    )
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            data_base=str(tmp_path),
            router_base=str(rb),
            router_id="rid",
            router_architecture_id=arch,
            figures_base=str(tmp_path / "figures"),
        ),
        router=replace(
            RouterSection(),
            train=replace(RouterTrainSection(), input_mode="both"),
        ),
        experiment_id="exp",
    )
    return FigureRunContext.from_pipeline(cfg, force=force)


def test_render_writes_png_and_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    rows = [
        {
            "question_id": f"q{i}",
            "target_oracle_best_weight": float(i) / 10.0,
            "predicted_weight": float(i) / 10.0 + 0.05,
            "is_valid_for_router_training": True,
        }
        for i in range(5)
    ]
    _write_predictions(pred, rows)
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle",
        split="test",
        filename_stem="tscatter",
    )
    out = render_router_pred_vs_oracle(spec, ctx)
    assert out.path_image.is_file()
    assert out.path_image.stat().st_size > 100
    assert out.path_meta.is_file()
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_points"] == 5
    assert meta["predictions_path"] == str(pred.resolve())
    assert "pearson_r" in meta


def test_render_pearson_matches_numpy_manual(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    y = 0.3 * x + 0.1
    rows = [
        {
            "question_id": f"q{i}",
            "target_oracle_best_weight": float(x[i]),
            "predicted_weight": float(y[i]),
            "is_valid_for_router_training": True,
        }
        for i in range(len(x))
    ]
    _write_predictions(pred, rows)
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle", split="test", filename_stem="pearson"
    )
    out = render_router_pred_vs_oracle(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    expected = float(np.corrcoef(x, y)[0, 1])
    assert meta["pearson_r"] == pytest.approx(expected, rel=1e-5)


def test_render_with_filter_invalid_only(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    rows = [
        {
            "question_id": "q0",
            "target_oracle_best_weight": 0.1,
            "predicted_weight": 0.2,
            "is_valid_for_router_training": True,
        },
        {
            "question_id": "q1",
            "target_oracle_best_weight": 0.9,
            "predicted_weight": 0.8,
            "is_valid_for_router_training": False,
        },
    ]
    _write_predictions(pred, rows)
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle",
        split="test",
        filter_invalid_only=True,
        filename_stem="filt",
    )
    out = render_router_pred_vs_oracle(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_points"] == 1


def test_render_raises_file_not_found(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    spec = RouterPredVsOracleSpec(kind="router_pred_vs_oracle", split="test")
    with pytest.raises(FileNotFoundError):
        render_router_pred_vs_oracle(spec, ctx)


def test_render_raises_empty_after_filter(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    rows = [
        {
            "question_id": "q1",
            "target_oracle_best_weight": 0.5,
            "predicted_weight": 0.5,
            "is_valid_for_router_training": False,
        },
    ]
    _write_predictions(pred, rows)
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle",
        split="test",
        filter_invalid_only=True,
    )
    with pytest.raises(ValueError, match="No rows left"):
        render_router_pred_vs_oracle(spec, ctx)


def test_render_single_row_meta_pearson_none(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    _write_predictions(
        pred,
        [
            {
                "question_id": "q1",
                "target_oracle_best_weight": 0.4,
                "predicted_weight": 0.6,
                "is_valid_for_router_training": True,
            }
        ],
    )
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle", split="test", filename_stem="one"
    )
    out = render_router_pred_vs_oracle(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_points"] == 1
    assert meta["pearson_r"] is None


def test_render_raises_file_exists_without_force(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path, force=False)
    pred = ctx.model_paths.predictions("test")
    _write_predictions(
        pred,
        [
            {
                "question_id": "q1",
                "target_oracle_best_weight": 0.2,
                "predicted_weight": 0.3,
                "is_valid_for_router_training": True,
            }
        ],
    )
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle", split="test", filename_stem="dup"
    )
    render_router_pred_vs_oracle(spec, ctx)
    with pytest.raises(FileExistsError):
        render_router_pred_vs_oracle(spec, ctx)
