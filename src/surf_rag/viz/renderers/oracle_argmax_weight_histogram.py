"""Stacked histogram: every grid bin at oracle curve maximum, single vs tied peaks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from surf_rag.evaluation.oracle_argmax_intervals import argmax_plateau_bin_indices
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.renderers.benchmark_oracle_ndcg_heatmap import (
    _curve_all_zeros,
    _curve_list,
    _package_versions,
)
from surf_rag.viz.specs import BaseFigureSpec, OracleArgmaxWeightHistogramSpec
from surf_rag.viz.theme import PALETTE
from surf_rag.viz.types import FigureOutput

_COLOR_SINGLE = PALETTE["dark-blue"]
_COLOR_TIE = PALETTE["light-blue"]


def _equal_width_bin_index(w: float, n_bins: int) -> int:
    w = float(np.clip(w, 0.0, 1.0))
    i = int(np.floor(w * float(n_bins)))
    return min(i, n_bins - 1)


def render_oracle_argmax_weight_histogram(
    spec: BaseFigureSpec, ctx: FigureRunContext
) -> FigureOutput:
    if not isinstance(spec, OracleArgmaxWeightHistogramSpec):
        raise TypeError(f"Expected OracleArgmaxWeightHistogramSpec, got {type(spec)}")
    parq = ctx.router_dataset_parquet
    if not parq.is_file():
        raise FileNotFoundError(
            f"router_dataset.parquet not found: {parq} "
            "(build the router dataset for paths.router_id first)"
        )
    df = pd.read_parquet(parq)
    for col in ("question_id", "oracle_curve", "weight_grid", "split"):
        if col not in df.columns:
            raise ValueError(
                f"router_dataset.parquet missing column {col!r}; path={parq}"
            )
    if spec.split != "all":
        df = df.loc[df["split"] == spec.split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No rows after split={spec.split!r} filter in {parq}")
    wg0 = _curve_list(df.iloc[0]["weight_grid"])
    if not wg0:
        raise ValueError("weight_grid is empty in router_dataset")
    w_arr = np.asarray(wg0, dtype=np.float64)
    n_w = len(w_arr)

    use_grid_bins = spec.hist_bins <= 0
    if use_grid_bins:
        n_bins = n_w
        single = np.zeros(n_bins, dtype=np.float64)
        tie = np.zeros(n_bins, dtype=np.float64)
        x = w_arr.astype(np.float64)
        if n_w > 1:
            bar_width = float(np.min(np.diff(w_arr))) * 0.9
        else:
            bar_width = 0.08
    else:
        n_bins = int(spec.hist_bins)
        single = np.zeros(n_bins, dtype=np.float64)
        tie = np.zeros(n_bins, dtype=np.float64)
        x = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
        bar_width = (1.0 / float(n_bins)) * 0.85

    n_skipped_all_zero = 0
    n_queries_used = 0
    n_multi_argmax_queries = 0
    for _, r in df.iterrows():
        wg = np.asarray(_curve_list(r["weight_grid"]), dtype=np.float64)
        if wg.shape != w_arr.shape or not np.allclose(wg, w_arr):
            qid = str(r.get("question_id", ""))
            raise ValueError(
                f"Inconsistent weight_grid for question_id={qid!r} in {parq}"
            )
        curve = _curve_list(r["oracle_curve"])
        if len(curve) != n_w:
            raise ValueError(
                f"oracle_curve length {len(curve)} != weight grid {n_w} "
                f"for question_id={r.get('question_id')!r}"
            )
        if spec.exclude_all_zero_queries and _curve_all_zeros(curve):
            n_skipped_all_zero += 1
            continue
        c = np.asarray(curve, dtype=np.float64)
        idx = argmax_plateau_bin_indices(c, rtol=spec.rtol, atol=spec.atol)
        if idx.size == 0:
            continue
        n_queries_used += 1
        is_tie = idx.size > 1
        if is_tie:
            n_multi_argmax_queries += 1
        for j in idx:
            jj = int(j)
            if use_grid_bins:
                b = jj
            else:
                b = _equal_width_bin_index(float(w_arr[jj]), n_bins)
            if is_tie:
                tie[b] += 1.0
            else:
                single[b] += 1.0

    if n_queries_used == 0:
        raise ValueError(
            "No queries left for histogram — all rows skipped as all-zero "
            f"(excluded {n_skipped_all_zero}). Set exclude_all_zero_queries: false "
            "if you want flat curves included."
        )

    total_contributions = float(np.sum(single) + np.sum(tie))

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

    rp = ctx.resolved_paths
    fig, ax = plt.subplots(figsize=(spec.fig_width, spec.fig_height))
    try:
        ax.bar(
            x,
            single,
            width=bar_width,
            align="center",
            label="Unique maximum (one argmax bin)",
            color=_COLOR_SINGLE,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.bar(
            x,
            tie,
            width=bar_width,
            bottom=single,
            align="center",
            label="Tied maximum (counts each tied bin)",
            color=_COLOR_TIE,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlabel("Dense weight")
        ax.set_ylabel("Count (queries can add >1 if tied)")
        ax.set_title(
            "Oracle argmax weights (all bins at curve maximum)",
            color=PALETTE["text"],
        )
        ax.legend(loc="upper right", framealpha=0.92)
        ax.set_xlim(-0.02, 1.02)
        if spec.show_plot_subtitle:
            mode = (
                "per grid point" if use_grid_bins else f"{n_bins} equal bins on [0, 1]"
            )
            sub = (
                f"benchmark={rp.benchmark_name}/{rp.benchmark_id}, "
                f"router_id={ctx.router_id}, split={spec.split}, "
                f"queries={n_queries_used}, ties≥2 bins: {n_multi_argmax_queries}, "
                f"bins={mode}, total contributions={int(round(total_contributions))}"
                + (
                    f", excluded_all_zero={n_skipped_all_zero}"
                    if n_skipped_all_zero
                    else ""
                )
            )
            ax.text(0.01, 0.99, sub, transform=ax.transAxes, fontsize=9, va="top")
        fig.tight_layout()
        fig.savefig(path_image, bbox_inches="tight")
    finally:
        plt.close(fig)

    meta: dict[str, object] = {
        "figure_kind": spec.kind,
        "router_dataset_parquet": str(parq.resolve()),
        "output_image": str(path_image),
        "benchmark_name": rp.benchmark_name,
        "benchmark_id": rp.benchmark_id,
        "router_id": ctx.router_id,
        "split": spec.split,
        "n_queries": n_queries_used,
        "n_queries_excluded_all_zero": n_skipped_all_zero,
        "n_queries_multi_argmax": n_multi_argmax_queries,
        "hist_bins": 0 if use_grid_bins else n_bins,
        "hist_binning": "weight_grid" if use_grid_bins else "equal_width_0_1",
        "n_weight_grid_points": n_w,
        "total_contributions": total_contributions,
        "exclude_all_zero_queries": spec.exclude_all_zero_queries,
        "rtol": spec.rtol,
        "atol": spec.atol,
        "show_plot_subtitle": spec.show_plot_subtitle,
        "versions": _package_versions(),
    }
    path_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return FigureOutput(path_image=path_image, path_meta=path_meta)
