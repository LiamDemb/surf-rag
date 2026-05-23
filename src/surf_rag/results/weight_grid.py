"""Resolve dense fusion weight grid for regret / interpolation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from surf_rag.evaluation.router_model_artifacts import RouterModelPaths, read_json
from surf_rag.evaluation.weight_grid import DEFAULT_DENSE_WEIGHT_GRID


def weight_grid_for_curve(
    curve: list[float],
    grid: list[float] | None,
    *,
    fallback_grid: list[float] | None = None,
) -> list[float]:
    """Return a weight grid aligned to ``curve`` length."""
    if grid:
        g = [float(x) for x in grid]
        if len(g) == len(curve):
            return g
    if fallback_grid:
        g = [float(x) for x in fallback_grid]
        if len(g) == len(curve):
            return g
    if len(curve) > 1:
        return [i / float(len(curve) - 1) for i in range(len(curve))]
    return list(DEFAULT_DENSE_WEIGHT_GRID)


def load_router_weight_grid(
    router_dataset_parquet: Path,
    model_paths: RouterModelPaths | None = None,
) -> list[float]:
    """Load canonical weight grid from dataset parquet or model manifest."""
    if router_dataset_parquet.is_file():
        df = pd.read_parquet(router_dataset_parquet, columns=["weight_grid"])
        if len(df) > 0:
            wg = df.iloc[0].get("weight_grid")
            if wg is not None and not (isinstance(wg, float) and pd.isna(wg)):
                return [float(x) for x in wg]
    if model_paths is not None and model_paths.manifest.is_file():
        try:
            manifest = read_json(model_paths.manifest)
            wg = (manifest.get("model") or {}).get("weight_grid")
            if wg:
                return [float(x) for x in wg]
        except (OSError, ValueError, TypeError):
            pass
    return [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]
