from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")

import pytest
from dataclasses import replace

from surf_rag.config.schema import (
    PathsSection,
    PipelineConfig,
    RouterSection,
    RouterTrainSection,
)
from surf_rag.evaluation.router_model_artifacts import (
    make_router_model_paths_for_cli,
    write_json,
)
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.router_pred_vs_oracle import render_router_pred_vs_oracle
from surf_rag.viz.specs import RouterPredVsOracleSpec
from surf_rag.viz.theme import apply_theme

_GRID = [0.0, 0.5, 1.0]
_CURVE_PEAK_MID = [0.0, 1.0, 0.0]


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
    ctx = FigureRunContext.from_pipeline(cfg, force=force)
    ctx.model_paths.ensure_dirs()
    write_json(ctx.model_paths.manifest, {"model": {"weight_grid": _GRID}})
    return ctx


def test_render_writes_png_and_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    rows = [
        {
            "question_id": f"q{i}",
            "oracle_curve": list(_CURVE_PEAK_MID),
            "predicted_weight": 0.5 + 0.01 * i,
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
    assert "argmax_interval_distance_mae" in meta
    assert "fraction_hits_argmax_interval" in meta


def test_render_distance_metrics_near_zero_at_plateau(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    rows = [
        {
            "question_id": f"q{i}",
            "oracle_curve": list(_CURVE_PEAK_MID),
            "predicted_weight": 0.5,
            "is_valid_for_router_training": True,
        }
        for i in range(5)
    ]
    _write_predictions(pred, rows)
    spec = RouterPredVsOracleSpec(
        kind="router_pred_vs_oracle", split="test", filename_stem="plat"
    )
    out = render_router_pred_vs_oracle(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["argmax_interval_distance_mae"] == pytest.approx(0.0, abs=1e-9)
    assert meta["fraction_hits_argmax_interval"] == pytest.approx(1.0)


def test_render_with_filter_invalid_only(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    rows = [
        {
            "question_id": "q0",
            "oracle_curve": list(_CURVE_PEAK_MID),
            "predicted_weight": 0.5,
            "is_valid_for_router_training": True,
        },
        {
            "question_id": "q1",
            "oracle_curve": list(_CURVE_PEAK_MID),
            "predicted_weight": 0.5,
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
            "oracle_curve": list(_CURVE_PEAK_MID),
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


def test_render_single_row_meta(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    pred = ctx.model_paths.predictions("test")
    _write_predictions(
        pred,
        [
            {
                "question_id": "q1",
                "oracle_curve": list(_CURVE_PEAK_MID),
                "predicted_weight": 0.5,
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
    assert meta["argmax_interval_distance_mae"] == pytest.approx(0.0, abs=1e-9)


def test_render_raises_file_exists_without_force(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path, force=False)
    pred = ctx.model_paths.predictions("test")
    _write_predictions(
        pred,
        [
            {
                "question_id": "q1",
                "oracle_curve": list(_CURVE_PEAK_MID),
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
