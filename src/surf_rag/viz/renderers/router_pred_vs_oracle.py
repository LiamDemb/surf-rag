"""Scatter: predicted dense weight vs distance to oracle argmax interval."""

from __future__ import annotations

import json
import sys
from importlib import metadata

import numpy as np

from surf_rag.evaluation.oracle_argmax_intervals import (
    DEFAULT_ARGMAX_INTERVAL_ATOL,
    DEFAULT_ARGMAX_INTERVAL_RTOL,
    dense_weight_argmax_intervals,
    distance_weight_to_argmax_intervals,
    prediction_hits_any_interval,
)
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.specs import BaseFigureSpec, RouterPredVsOracleSpec
from surf_rag.viz.sources.router_predictions import (
    load_router_prediction_rows,
    load_weight_grid_from_manifest,
    predictions_path_for,
)
from surf_rag.viz.theme import PALETTE
from surf_rag.viz.types import FigureOutput


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        out["surf_rag"] = metadata.version("surf_rag")
    except metadata.PackageNotFoundError:
        out["surf_rag"] = "unknown"
    try:
        import matplotlib as mpl

        out["matplotlib"] = mpl.__version__
    except Exception:
        out["matplotlib"] = "unknown"
    out["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return out


def render_router_pred_vs_oracle(
    spec: BaseFigureSpec, ctx: FigureRunContext
) -> FigureOutput:
    if not isinstance(spec, RouterPredVsOracleSpec):
        raise TypeError(
            "RouterPredVsOracle renderer expected RouterPredVsOracleSpec, "
            f"got {type(spec)}"
        )
    pred_path = predictions_path_for(ctx.model_paths, spec.split)
    manifest_path = ctx.model_paths.manifest
    weight_grid = load_weight_grid_from_manifest(manifest_path)
    df = load_router_prediction_rows(pred_path)
    if spec.filter_invalid_only:
        df = df.loc[df["valid"]].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"No rows left after loading/filtering predictions at {pred_path} "
            f"(filter_invalid_only={spec.filter_invalid_only})"
        )

    n_w = len(weight_grid)
    dists: list[float] = []
    hits: list[bool] = []
    x_pred: list[float] = []
    for _, row in df.iterrows():
        curve = row["oracle_curve"]
        assert isinstance(curve, list)
        if len(curve) != n_w:
            raise ValueError(
                f"oracle_curve length {len(curve)} does not match "
                f"manifest weight_grid length {n_w} "
                f"(question_id={row['question_id']!r})"
            )
        intr = dense_weight_argmax_intervals(
            curve,
            weight_grid,
            rtol=DEFAULT_ARGMAX_INTERVAL_RTOL,
            atol=DEFAULT_ARGMAX_INTERVAL_ATOL,
        )
        pred = float(row["predicted_weight"])
        d = distance_weight_to_argmax_intervals(pred, intr)
        if not np.isfinite(d):
            continue
        dists.append(float(d))
        hits.append(prediction_hits_any_interval(pred, intr))
        x_pred.append(pred)

    n = len(dists)
    if n == 0:
        raise ValueError("No finite distance values computed for scatter plot")

    x_arr = np.asarray(x_pred, dtype=np.float64)
    y_arr = np.asarray(dists, dtype=np.float64)
    mae = float(np.mean(y_arr))
    rmse = float(np.sqrt(np.mean(y_arr**2)))
    frac_hit = float(np.mean(np.asarray(hits, dtype=np.float64)))

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ext = ctx.image_format
    stem = f"{spec.filename_stem}_{spec.split}"
    path_image = (ctx.output_dir / f"{stem}.{ext}").resolve()
    path_meta = (ctx.output_dir / f"{stem}.meta.json").resolve()
    if not ctx.force and (path_image.exists() or path_meta.exists()):
        raise FileExistsError(
            f"Refusing to overwrite {path_image} / {path_meta} without force=True"
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    try:
        ax.scatter(
            x_arr,
            y_arr,
            s=spec.point_size,
            alpha=spec.alpha,
            c=PALETTE["primary"],
            edgecolors="none",
        )
        ax.set_xlim(0.0, 1.0)
        ymax = float(np.max(y_arr)) if len(y_arr) else 1.0
        ax.set_ylim(0.0, max(ymax * 1.05, 0.02))
        ax.set_xlabel("Predicted dense weight")
        ax.set_ylabel("Distance to oracle argmax interval")
        sub = (
            f"router_id={ctx.router_id}"
            + (
                f", arch_id={ctx.router_architecture_id}"
                if ctx.router_architecture_id
                else ""
            )
            + f", input_mode={ctx.input_mode}, split={spec.split}, N={n}"
        )
        ax.set_title("Router: prediction vs argmax-interval distance")
        ax.text(0.02, 0.98, sub, transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        fig.savefig(path_image, bbox_inches="tight")
    finally:
        plt.close(fig)

    meta: dict[str, object] = {
        "figure_kind": spec.kind,
        "predictions_path": str(pred_path.resolve()),
        "output_image": str(path_image),
        "router_id": ctx.router_id,
        "router_architecture_id": ctx.router_architecture_id,
        "input_mode": ctx.input_mode,
        "split": spec.split,
        "filter_invalid_only": spec.filter_invalid_only,
        "n_points": n,
        "argmax_interval_distance_mae": mae,
        "argmax_interval_distance_rmse": rmse,
        "fraction_hits_argmax_interval": frac_hit,
        "versions": _package_versions(),
    }
    path_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return FigureOutput(path_image=path_image, path_meta=path_meta)
