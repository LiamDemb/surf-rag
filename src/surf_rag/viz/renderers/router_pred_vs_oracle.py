"""Scatter: predicted dense weight vs oracle best weight on the training grid."""

from __future__ import annotations

import json
import sys
from importlib import metadata

import numpy as np

from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.specs import BaseFigureSpec, RouterPredVsOracleSpec
from surf_rag.viz.sources.router_predictions import (
    load_router_prediction_rows,
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


def _stats(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float, float | None]:
    n = int(len(x))
    if n == 0:
        return 0, float("nan"), float("nan"), None
    err = y.astype(np.float64) - x.astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if n < 2:
        return n, mae, rmse, None
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return n, mae, rmse, None
    return n, mae, rmse, r


def render_router_pred_vs_oracle(
    spec: BaseFigureSpec, ctx: FigureRunContext
) -> FigureOutput:
    if not isinstance(spec, RouterPredVsOracleSpec):
        raise TypeError(
            "RouterPredVsOracle renderer expected RouterPredVsOracleSpec, "
            f"got {type(spec)}"
        )
    pred_path = predictions_path_for(ctx.model_paths, spec.split)
    df = load_router_prediction_rows(pred_path)
    if spec.filter_invalid_only:
        df = df.loc[df["valid"]].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"No rows left after loading/filtering predictions at {pred_path} "
            f"(filter_invalid_only={spec.filter_invalid_only})"
        )
    x = df["oracle_weight"].to_numpy(dtype=np.float64)
    y = df["predicted_weight"].to_numpy(dtype=np.float64)
    n, mae, rmse, pearson = _stats(x, y)

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
            x,
            y,
            s=spec.point_size,
            alpha=spec.alpha,
            c=PALETTE["primary"],
            edgecolors="none",
        )
        lims = (0.0, 1.0)
        ax.plot(
            lims, lims, color=PALETTE["identity_line"], linestyle="--", linewidth=1.0
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Oracle best dense weight (grid argmax)")
        ax.set_ylabel("Predicted dense weight")
        sub = (
            f"router_id={ctx.router_id}"
            + (
                f", arch_id={ctx.router_architecture_id}"
                if ctx.router_architecture_id
                else ""
            )
            + f", input_mode={ctx.input_mode}, split={spec.split}, N={n}"
        )
        ax.set_title("Router: predicted vs oracle weight")
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
        "mean_abs_error": mae,
        "rmse": rmse,
        "pearson_r": pearson,
        "versions": _package_versions(),
    }
    path_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return FigureOutput(path_image=path_image, path_meta=path_meta)
