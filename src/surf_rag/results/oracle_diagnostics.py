"""Per-query oracle curve diagnostics and aggregates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from surf_rag.evaluation.oracle_argmax_intervals import dense_weight_argmax_intervals
from surf_rag.results.metric_fields import oracle_bin_value, oracle_curve_series
from surf_rag.router.splits import normalize_dataset_source


def _trapezoid_volume(values: list[float], grid: list[float]) -> float:
    if len(values) < 2:
        return float(values[0]) if values else 0.0
    w = np.asarray(grid, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(w)
    w = w[order]
    v = v[order]
    return float(np.trapezoid(v, w))


def _plateau_width_weight(grid: list[float], values: list[float], tau: float) -> float:
    if not values:
        return 0.0
    peak = float(np.max(values))
    tied_idx = [i for i, v in enumerate(values) if (peak - float(v)) <= tau]
    if not tied_idx:
        return 0.0
    w = [grid[i] for i in tied_idx]
    return float(max(w) - min(w))


def _oracle_optimal_weight(values: list[float], grid: list[float], tau: float) -> float:
    intervals = dense_weight_argmax_intervals(values, grid, rtol=0.0, atol=tau)
    if not intervals:
        return 0.5
    lo, hi = intervals[0]
    return float((lo + hi) / 2.0)


def per_query_diagnostics(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    k: int,
    diagnostic_ks: list[int],
    plateau_tau: float,
    qid_to_source: dict[str, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        if not qid:
            continue
        src = normalize_dataset_source(
            str(row.get("dataset_source") or qid_to_source.get(qid, ""))
        )
        vals_k, grid = oracle_curve_series(row, metric=metric, k=k)
        if not grid:
            continue
        dense_fav = float(vals_k[-1] - vals_k[0]) if vals_k else 0.0
        delta = dense_fav
        dispersion = float(max(vals_k) - min(vals_k)) if vals_k else 0.0
        peak = float(max(vals_k)) if vals_k else 0.0
        if peak <= 0.0:
            category = "zero_oracle"
        elif abs(delta) <= plateau_tau:
            category = "near_tied"
        elif delta > 0:
            category = "dense_favoured"
        else:
            category = "graph_favoured"

        rec: dict[str, Any] = {
            "question_id": qid,
            "dataset_source": src,
            "delta": delta,
            "dispersion": dispersion,
            "plateau_width": _plateau_width_weight(grid, vals_k, plateau_tau),
            "oracle_optimal_weight": _oracle_optimal_weight(vals_k, grid, plateau_tau),
            "category": category,
        }
        for dk in diagnostic_ks:
            dvals, _ = oracle_curve_series(row, metric=metric, k=int(dk))
            rec[f"volume_k{dk}"] = _trapezoid_volume(dvals, grid)
        records.append(rec)
    return pd.DataFrame.from_records(records)


def aggregate_oracle_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format summary rows for oracle_stats table."""
    if df.empty:
        return pd.DataFrame(columns=["dataset_source", "stat_name", "k", "value"])
    rows: list[dict[str, Any]] = []
    sources = ["all"] + sorted(
        s for s in df["dataset_source"].unique() if s in ("nq", "2wiki")
    )
    for src in sources:
        sub = df if src == "all" else df.loc[df["dataset_source"] == src]
        if sub.empty:
            continue
        n = len(sub)
        rows.append(
            {
                "dataset_source": src,
                "stat_name": "mean_dispersion",
                "k": "",
                "value": float(sub["dispersion"].mean()),
            }
        )
        rows.append(
            {
                "dataset_source": src,
                "stat_name": "mean_plateau_width",
                "k": "",
                "value": float(sub["plateau_width"].mean()),
            }
        )
        for col in [c for c in sub.columns if c.startswith("volume_k")]:
            k_part = col.replace("volume_k", "")
            rows.append(
                {
                    "dataset_source": src,
                    "stat_name": "mean_curve_volume",
                    "k": k_part,
                    "value": float(sub[col].mean()),
                }
            )
        for cat in (
            "dense_favoured",
            "graph_favoured",
            "near_tied",
            "zero_oracle",
        ):
            pct = 100.0 * float((sub["category"] == cat).sum()) / n
            rows.append(
                {
                    "dataset_source": src,
                    "stat_name": f"pct_{cat}",
                    "k": "",
                    "value": pct,
                }
            )
    return pd.DataFrame.from_records(rows)
