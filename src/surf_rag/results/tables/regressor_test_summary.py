"""Test-split regressor regret vs baselines; global pooled weight minimizing mean regret."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.tables.regressor_metrics import (
    SkippedArtifact,
    _curve_regret,
    collect_regressor_split_buckets,
)
from surf_rag.results.tables.writer import write_table

# Dense fusion weight grid for the scalar policy that minimizes overall mean regret.
_POOLED_WEIGHT_SEARCH_POINTS: int = 501

_DATASET_ROWS: tuple[tuple[str, str], ...] = (
    ("all", "Total"),
    ("nq", "NQ"),
    ("2wiki", "2Wiki"),
)


def _optimal_pooled_weight_min_mean_regret(
    items: list[dict[str, Any]],
) -> float:
    """Single dense weight w* in [0, 1] that minimizes mean oracle-curve regret over ``items``."""
    if not items:
        raise ValueError("items must be non-empty")
    ws = np.linspace(0.0, 1.0, _POOLED_WEIGHT_SEARCH_POINTS)
    best_w = 0.5
    best_mean = float("inf")
    for w in ws:
        m = float(
            np.mean([_curve_regret(x["curve"], x["grid"], float(w)) for x in items])
        )
        if m < best_mean:
            best_mean = m
            best_w = float(w)
    return best_w


def _mean_regret_at_weight(items: list[dict[str, Any]], w: float) -> float:
    return float(np.mean([_curve_regret(x["curve"], x["grid"], w) for x in items]))


def build_regressor_test_summary(
    bundle: ResultsBundle,
) -> tuple[pd.DataFrame, dict]:
    arch = get_router_arch(bundle, "regressor")
    if arch is None or not arch.architecture_id.strip():
        raise SkippedArtifact("results.router.regressor not configured")

    split, buckets = collect_regressor_split_buckets(bundle)
    all_items = buckets["all"]
    w_star = _optimal_pooled_weight_min_mean_regret(all_items)

    rows: list[dict] = []
    for src_key, dataset in _DATASET_ROWS:
        items = buckets.get(src_key) or []
        if not items:
            continue
        preds = [x["pred"] for x in items]
        rows.append(
            {
                "split": split,
                "dataset": dataset,
                "mean_regret_regressor": float(np.mean([x["regret"] for x in items])),
                "mean_regret_oracle": 0.0,
                "mean_regret_50_50": float(np.mean([x["fusion_50_50"] for x in items])),
                "mean_predicted_weight": float(np.mean(preds)),
                "pooled_weight_min_mean_regret": w_star,
                "mean_regret_at_pooled_weight": _mean_regret_at_weight(items, w_star),
                "n": len(items),
            }
        )
    if not rows:
        raise SkippedArtifact(f"No regressor predictions on split {split!r}")

    df = pd.DataFrame(rows)
    paths_out = write_table(
        bundle,
        "regressor_test_summary",
        df,
        {
            "architecture_id": arch.architecture_id,
            "split": split,
            "pooled_weight_min_mean_regret": w_star,
            "pooled_weight_search_points": _POOLED_WEIGHT_SEARCH_POINTS,
        },
    )
    return df, {"csv": str(paths_out[0]), "meta": str(paths_out[1])}
