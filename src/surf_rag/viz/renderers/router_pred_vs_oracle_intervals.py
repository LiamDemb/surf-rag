"""Per-query oracle dense-weight intervals (max oracle_curve ties) vs prediction."""

from __future__ import annotations

import json
import sys
from importlib import metadata

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from surf_rag.viz.context import FigureRunContext
from surf_rag.evaluation.oracle_argmax_intervals import (
    dense_weight_argmax_intervals,
    mean_interval_midpoint,
    prediction_hits_any_interval,
)
from surf_rag.viz.specs import BaseFigureSpec, RouterPredVsOracleIntervalsSpec
from surf_rag.viz.sources.router_predictions import (
    load_router_predictions_with_curves,
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


def render_router_pred_vs_oracle_intervals(
    spec: BaseFigureSpec, ctx: FigureRunContext
) -> FigureOutput:
    if not isinstance(spec, RouterPredVsOracleIntervalsSpec):
        raise TypeError(f"Expected RouterPredVsOracleIntervalsSpec, got {type(spec)}")
    pred_path = predictions_path_for(ctx.model_paths, spec.split)
    manifest_path = ctx.model_paths.manifest
    weight_grid = load_weight_grid_from_manifest(manifest_path)
    df_raw = load_router_predictions_with_curves(pred_path)
    if spec.filter_invalid_only:
        df_raw = df_raw.loc[df_raw["valid"]].reset_index(drop=True)
    if len(df_raw) == 0:
        raise ValueError(
            f"No rows left after loading/filtering predictions at {pred_path} "
            f"(filter_invalid_only={spec.filter_invalid_only})"
        )

    records: list[dict[str, object]] = []
    for _, row in df_raw.iterrows():
        curve = row["oracle_curve"]
        assert isinstance(curve, list)
        if len(curve) != len(weight_grid):
            raise ValueError(
                f"oracle_curve length {len(curve)} does not match "
                f"manifest weight_grid length {len(weight_grid)} "
                f"(question_id={row['question_id']!r})"
            )
        intr = dense_weight_argmax_intervals(
            curve,
            weight_grid,
            rtol=spec.rtol,
            atol=spec.atol,
        )
        pred = float(row["predicted_weight"])
        hits = prediction_hits_any_interval(pred, intr)
        mid = mean_interval_midpoint(intr)
        records.append(
            {
                "question_id": row["question_id"],
                "_midpoint": mid,
                "_intervals": intr,
                "predicted_weight": pred,
                "_hits": hits,
            }
        )
    tbl = pd.DataFrame.from_records(records)
    tbl = tbl.sort_values(["_midpoint", "question_id"], kind="mergesort").reset_index(
        drop=True
    )

    n_full = len(tbl)
    if spec.max_queries is not None and n_full > spec.max_queries:
        rng = np.random.default_rng(spec.subsample_seed)
        pick = np.sort(rng.choice(n_full, size=spec.max_queries, replace=False))
        tbl = tbl.iloc[pick].reset_index(drop=True)

    n = len(tbl)
    n_correct = int(tbl["_hits"].sum())
    acc_pct = 100.0 * n_correct / n if n else 0.0

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

    interval_gray = "#bfbfbf"
    correct_c = "#2ca02c"
    wrong_c = "#d62728"

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    try:
        half = 0.5 * spec.interval_bar_width_ratio
        for i, r in tbl.iterrows():
            intrs = r["_intervals"]
            assert isinstance(intrs, list)
            for y_lo, y_hi in intrs:
                ax.add_patch(
                    Rectangle(
                        (i - half, y_lo),
                        2.0 * half,
                        y_hi - y_lo,
                        facecolor=interval_gray,
                        edgecolor="none",
                        alpha=spec.interval_alpha,
                        zorder=1,
                    )
                )
            c = correct_c if bool(r["_hits"]) else wrong_c
            ax.scatter(
                [i],
                [r["predicted_weight"]],
                s=spec.dot_size,
                c=c,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

        ax.set_xlim(-0.5, max(n - 1, 0) + 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Test queries (sorted by oracle-interval midpoint)")
        ax.set_ylabel("Dense weight")
        title = (
            f"Oracle optimal intervals vs prediction — {spec.split} — "
            f"acc={acc_pct:.1f}% ({n_correct}/{n})"
        )
        ax.set_title(title, color=PALETTE["text"])
        sub = (
            f"router_id={ctx.router_id}"
            + (
                f", arch_id={ctx.router_architecture_id}"
                if ctx.router_architecture_id
                else ""
            )
            + f", input_mode={ctx.input_mode}, N_plot={n}"
            + (f", N_full={n_full}" if n != n_full else "")
        )
        ax.text(0.01, 0.99, sub, transform=ax.transAxes, fontsize=9, va="top")
        leg_el = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=correct_c,
                markersize=8,
                label="Prediction in an optimal interval",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=wrong_c,
                markersize=8,
                label="Outside all optimal intervals",
            ),
        ]
        ax.legend(handles=leg_el, loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        fig.savefig(path_image, bbox_inches="tight")
    finally:
        plt.close(fig)

    meta: dict[str, object] = {
        "figure_kind": spec.kind,
        "predictions_path": str(pred_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "weight_grid_len": int(len(weight_grid)),
        "output_image": str(path_image),
        "router_id": ctx.router_id,
        "router_architecture_id": ctx.router_architecture_id,
        "input_mode": ctx.input_mode,
        "split": spec.split,
        "filter_invalid_only": spec.filter_invalid_only,
        "n_points_plotted": n,
        "n_points_before_subsample": n_full,
        "n_correct": n_correct,
        "accuracy": acc_pct / 100.0,
        "max_queries": spec.max_queries,
        "subsample_seed": spec.subsample_seed,
        "rtol": spec.rtol,
        "atol": spec.atol,
        "versions": _package_versions(),
    }
    path_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return FigureOutput(path_image=path_image, path_meta=path_meta)
