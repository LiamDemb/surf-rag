"""Heatmap of oracle metric vs dense weight for every query in router_dataset.parquet."""

from __future__ import annotations

import json
import sys
from importlib import metadata

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, PowerNorm

from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.specs import BaseFigureSpec, BenchmarkOracleHeatmapSpec
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


def _curve_list(cell: object) -> list[float]:
    if cell is None:
        return []
    if hasattr(cell, "tolist"):
        return [float(x) for x in cell.tolist()]
    return [float(x) for x in list(cell)]


def _oracle_peak_has_interior_dense_weight(
    curve: list[float],
    weight_grid: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """True iff max oracle score is achieved at some dense weight not ~0.0 and not ~1.0."""
    c = np.asarray(curve, dtype=np.float64)
    w = np.asarray(weight_grid, dtype=np.float64).reshape(-1)
    if c.size == 0 or c.shape != w.shape:
        return False
    peak = float(np.max(c))
    at_max = np.isclose(c, peak, rtol=rtol, atol=atol)
    if not np.any(at_max):
        return False
    w_at_max = w[at_max]
    low = np.isclose(w_at_max, 0.0, rtol=rtol, atol=atol)
    high = np.isclose(w_at_max, 1.0, rtol=rtol, atol=atol)
    return bool(np.any(~(low | high)))


def _curve_all_zeros(curve: list[float], *, eps: float = 1e-15) -> bool:
    """True iff every oracle score in the curve is (approximately) zero."""
    if not curve:
        return True
    a = np.asarray(curve, dtype=np.float64)
    return bool(np.all(np.abs(a) < eps))


def _curve_all_ones(
    curve: list[float],
    *,
    rtol: float,
    atol: float,
) -> bool:
    """True iff every oracle score is (approximately) 1.0."""
    if not curve:
        return False
    a = np.asarray(curve, dtype=np.float64)
    return bool(np.all(np.isclose(a, 1.0, rtol=rtol, atol=atol)))


def _oracle_mid_band_strictly_beats_tails(
    curve: list[float],
    weight_grid: np.ndarray,
    *,
    w_mid_lo: float = 0.2,
    w_mid_hi: float = 0.8,
) -> bool:
    """True iff max score on w in [w_mid_lo, w_mid_hi] exceeds both tail maxima.

    Tails: w < w_mid_lo and w > w_mid_hi. Empty tail → max treated as −∞.
    Requires at least one grid point in the mid band.
    """
    c = np.asarray(curve, dtype=np.float64)
    w = np.asarray(weight_grid, dtype=np.float64).reshape(-1)
    if c.size == 0 or c.shape != w.shape:
        return False
    mid = (w >= w_mid_lo) & (w <= w_mid_hi)
    if not np.any(mid):
        return False

    def _band_max(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("-inf")
        x = float(np.nanmax(c[mask]))
        return x if np.isfinite(x) else float("-inf")

    low = w < w_mid_lo
    high = w > w_mid_hi
    mid_max = _band_max(mid)
    return mid_max > _band_max(low) and mid_max > _band_max(high)


def _score_weighted_centroid_dense_weight(
    curve: list[float], weight_grid: np.ndarray
) -> float:
    """Dense-weight coordinate in [min(w), max(w)] where oracle mass concentrates.

    ``sum(score_i * w_i) / sum(score_i)`` — higher scores on low weights pull toward 0,
    high weights toward 1 (when ``weight_grid`` spans [0, 1]).
    """
    s = np.maximum(np.asarray(curve, dtype=np.float64), 0.0)
    w = np.asarray(weight_grid, dtype=np.float64).reshape(-1)
    if s.shape != w.shape:
        raise ValueError("curve and weight_grid length mismatch in centroid")
    mass = float(np.sum(s))
    if mass <= 1e-30:
        return float(w[0])
    return float(np.dot(s, w) / mass)


def _norm_and_vlims(
    mat: np.ndarray,
    *,
    color_norm: str,
    color_power_gamma: float,
) -> tuple[Normalize | LogNorm | PowerNorm, float, float | None]:
    """Return matplotlib norm, vmax reference, vmin for log norm (meta), else None."""
    vmax = float(np.max(mat))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    if color_norm == "linear":
        return Normalize(vmin=0.0, vmax=vmax), vmax, None
    if color_norm == "power":
        return (
            PowerNorm(
                gamma=color_power_gamma,
                vmin=0.0,
                vmax=vmax,
            ),
            vmax,
            None,
        )
    eps = 1e-6
    pos = mat[mat > eps]
    vmin_log = float(np.min(pos)) if pos.size > 0 else eps
    vmin_log = max(vmin_log, eps)
    vmax_log = max(vmax, vmin_log * 1.0000001)
    if vmin_log >= vmax_log:
        vmax_log = vmin_log * (1.0 + 1e-6)
        if vmax_log <= vmin_log:
            vmax_log = vmin_log * 1.01
    return LogNorm(vmin=vmin_log, vmax=vmax_log), vmax, vmin_log


def _matplotlib_colormap_copy(name: str):
    """Return a copy of a registered matplotlib colormap, or raise a clear error."""
    import matplotlib.pyplot as plt

    try:
        return plt.get_cmap(name).copy()
    except ValueError as e:
        raise ValueError(
            f"Unknown colormap {name!r}. Use a matplotlib built-in name "
            f"(e.g. Blues, RdYlGn, viridis, plasma); see matplotlib colormap reference."
        ) from e


def render_benchmark_oracle_ndcg_heatmap(
    spec: BaseFigureSpec, ctx: FigureRunContext
) -> FigureOutput:
    if not isinstance(spec, BenchmarkOracleHeatmapSpec):
        raise TypeError(f"Expected BenchmarkOracleHeatmapSpec, got {type(spec)}")
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

    rows: list[tuple[float, str, list[float]]] = []
    n_skipped_all_zero = 0
    n_skipped_all_one = 0
    n_skipped_endpoint_peak_only = 0
    n_skipped_mid_band_not_strict_best = 0
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
        if spec.exclude_all_one_queries and _curve_all_ones(
            curve, rtol=spec.rtol, atol=spec.atol
        ):
            n_skipped_all_one += 1
            continue
        if spec.interior_peak_only and not _oracle_peak_has_interior_dense_weight(
            curve, w_arr, rtol=spec.rtol, atol=spec.atol
        ):
            n_skipped_endpoint_peak_only += 1
            continue
        if (
            spec.mid_dense_band_strict_best
            and not _oracle_mid_band_strictly_beats_tails(curve, w_arr)
        ):
            n_skipped_mid_band_not_strict_best += 1
            continue
        centroid_w = _score_weighted_centroid_dense_weight(curve, w_arr)
        rows.append((centroid_w, str(r["question_id"]), curve))
    if len(rows) == 0:
        parts = [
            "No queries left to plot after filters — ",
            f"excluded_all_zero={n_skipped_all_zero}, ",
            f"excluded_all_one={n_skipped_all_one}, ",
            f"excluded_endpoint_peak_only={n_skipped_endpoint_peak_only}, ",
            f"excluded_mid_band_not_strict_best={n_skipped_mid_band_not_strict_best}. ",
        ]
        if spec.exclude_all_zero_queries and n_skipped_all_zero:
            parts.append(
                "Try exclude_all_zero_queries: false for flat all-zero columns. "
            )
        if spec.exclude_all_one_queries and n_skipped_all_one:
            parts.append(
                "Try exclude_all_one_queries: false for uniform all-one curves. "
            )
        if spec.interior_peak_only and n_skipped_endpoint_peak_only:
            parts.append("Try interior_peak_only: false or relax rtol/atol. ")
        if spec.mid_dense_band_strict_best and n_skipped_mid_band_not_strict_best:
            parts.append(
                "Try mid_dense_band_strict_best: false (needs max on w in [0.2,0.8] "
                "strictly above both tail maxima). "
            )
        raise ValueError("".join(parts))
    rows.sort(key=lambda t: (t[0], t[1]))
    n_q = len(rows)
    mat = np.zeros((n_w, n_q), dtype=np.float64)
    for j, (_, _, curve) in enumerate(rows):
        mat[:, j] = np.asarray(curve, dtype=np.float64)

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
    norm, vmax_ref, vmin_log_used = _norm_and_vlims(
        mat,
        color_norm=spec.color_norm,
        color_power_gamma=spec.color_power_gamma,
    )
    fig, ax = plt.subplots(figsize=(spec.fig_width, spec.fig_height))
    try:
        cmap = _matplotlib_colormap_copy(spec.colormap)
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            extent=(
                -0.5,
                n_q - 0.5,
                float(w_arr[0]),
                float(w_arr[-1]),
            ),
        )
        ax.set_xlabel("Queries (sorted by score-weighted centroid of dense weight)")
        ax.set_ylabel("Dense weight")
        ax.set_title(
            "Oracle scores vs dense weight",
            color=PALETTE["text"],
        )
        if spec.show_plot_subtitle:
            sub = f"benchmark={rp.benchmark_name}/{rp.benchmark_id}, " f"router_id={ctx.router_id}, split={spec.split}, N={n_q}, weights={n_w}" + (
                f", excluded_all_zero={n_skipped_all_zero}"
                if n_skipped_all_zero
                else ""
            ) + (
                f", excluded_all_one={n_skipped_all_one}" if n_skipped_all_one else ""
            ) + (
                f", excluded_endpoint_peak_only={n_skipped_endpoint_peak_only}"
                if n_skipped_endpoint_peak_only
                else ""
            ) + (
                f", excluded_mid_band_not_strict_best={n_skipped_mid_band_not_strict_best}"
                if n_skipped_mid_band_not_strict_best
                else ""
            )
            ax.text(0.01, 0.99, sub, transform=ax.transAxes, fontsize=9, va="top")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        lab = "Oracle score"
        if spec.color_norm == "log":
            lab += " (log scale)"
        elif spec.color_norm == "power":
            lab += f" (power γ={spec.color_power_gamma})"
        cbar.set_label(lab)
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
        "n_queries": n_q,
        "n_weights": n_w,
        "exclude_all_zero_queries": spec.exclude_all_zero_queries,
        "n_queries_excluded_all_zero": n_skipped_all_zero,
        "exclude_all_one_queries": spec.exclude_all_one_queries,
        "n_queries_excluded_all_one": n_skipped_all_one,
        "interior_peak_only": spec.interior_peak_only,
        "n_queries_excluded_endpoint_peak_only": n_skipped_endpoint_peak_only,
        "mid_dense_band_strict_best": spec.mid_dense_band_strict_best,
        "mid_dense_band_strict_best_w_lo": 0.2,
        "mid_dense_band_strict_best_w_hi": 0.8,
        "n_queries_excluded_mid_band_not_strict_best": n_skipped_mid_band_not_strict_best,
        "query_sort": "score_weighted_centroid_dense_weight",
        "rtol": spec.rtol,
        "atol": spec.atol,
        "color_norm": spec.color_norm,
        "vmax_used": vmax_ref,
        "vmin_log_used": vmin_log_used,
        "color_power_gamma": spec.color_power_gamma,
        "colormap": spec.colormap,
        "show_plot_subtitle": spec.show_plot_subtitle,
        "versions": _package_versions(),
    }
    path_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return FigureOutput(path_image=path_image, path_meta=path_meta)
