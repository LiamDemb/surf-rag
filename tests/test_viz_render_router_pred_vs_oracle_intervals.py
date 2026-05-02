from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from surf_rag.evaluation.router_model_artifacts import (
    make_router_model_paths_for_cli,
    write_json,
)
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.router_pred_vs_oracle_intervals import (
    render_router_pred_vs_oracle_intervals,
)
from surf_rag.viz.specs import RouterPredVsOracleIntervalsSpec
from surf_rag.viz.theme import apply_theme


def _write_manifest(mp, weight_grid: list[float]) -> None:
    mp.run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        mp.manifest,
        {
            "model": {"weight_grid": weight_grid},
        },
    )


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _ctx(tmp_path: Path, *, force: bool = True) -> FigureRunContext:
    rb = tmp_path / "router"
    mp = make_router_model_paths_for_cli(
        "rid",
        router_base=rb,
        input_mode="both",
        router_architecture_id="tower",
    )
    out = tmp_path / "fig_out"
    return FigureRunContext(
        router_id="rid",
        router_architecture_id="tower",
        input_mode="both",
        router_base=rb,
        model_paths=mp,
        output_dir=out,
        experiment_id="exp",
        image_format="png",
        force=force,
    )


def test_interval_render_writes_meta_accuracy(tmp_path: Path) -> None:
    """Two queries: disjoint maxima segments; one hit, one miss → 50%."""
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    wgrid = [0.0, 0.25, 0.5, 0.75, 1.0]
    _write_manifest(ctx.model_paths, wgrid)
    # max at indices 0, 2, 4 → three thin intervals
    curve_hit = [1.0, 0.0, 1.0, 0.0, 1.0]
    curve_miss = [1.0, 0.0, 1.0, 0.0, 1.0]
    pred = ctx.model_paths.predictions("test")
    _write_predictions(
        pred,
        [
            {
                "question_id": "q_a",
                "oracle_curve": curve_hit,
                "predicted_weight": 0.5,
                "is_valid_for_router_training": True,
            },
            {
                "question_id": "q_b",
                "oracle_curve": curve_miss,
                "predicted_weight": 0.35,
                "is_valid_for_router_training": True,
            },
        ],
    )
    spec = RouterPredVsOracleIntervalsSpec(
        kind="router_pred_vs_oracle_intervals",
        split="test",
        filename_stem="iv",
    )
    out = render_router_pred_vs_oracle_intervals(spec, ctx)
    assert out.path_image.is_file()
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_points_plotted"] == 2
    assert meta["n_correct"] == 1
    assert meta["accuracy"] == pytest.approx(0.5)


def test_subsample_max_queries(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    ctx = _ctx(tmp_path)
    wgrid = np.linspace(0.0, 1.0, 5).tolist()
    _write_manifest(ctx.model_paths, wgrid)
    rows = [
        {
            "question_id": f"q{i}",
            "oracle_curve": [1.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_weight": 0.0,
            "is_valid_for_router_training": True,
        }
        for i in range(10)
    ]
    _write_predictions(ctx.model_paths.predictions("test"), rows)
    spec = RouterPredVsOracleIntervalsSpec(
        kind="router_pred_vs_oracle_intervals",
        split="test",
        max_queries=3,
        subsample_seed=7,
    )
    out = render_router_pred_vs_oracle_intervals(spec, ctx)
    meta = json.loads(out.path_meta.read_text(encoding="utf-8"))
    assert meta["n_points_plotted"] == 3
    assert meta["n_points_before_subsample"] == 10
