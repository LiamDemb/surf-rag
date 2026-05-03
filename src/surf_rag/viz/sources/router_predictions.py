"""Load router eval ``predictions_<split>.jsonl`` into a normalized frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from surf_rag.evaluation.router_model_artifacts import RouterModelPaths, read_json

_REQUIRED_RAW = (
    "question_id",
    "oracle_curve",
    "predicted_weight",
)


def _oracle_curve_cell_to_list(cell: object) -> list[float]:
    if cell is None:
        return []
    if isinstance(cell, float) and np.isnan(cell):
        return []
    if isinstance(cell, str):
        obj = json.loads(cell)
        return [float(x) for x in obj]
    return [float(x) for x in list(cell)]


def predictions_path_for(model_paths: RouterModelPaths, split: str) -> Path:
    """Canonical path for one split's prediction JSONL."""
    return model_paths.predictions(split)


def load_router_prediction_rows(path: Path) -> pd.DataFrame:
    """Read JSONL written by :func:`surf_rag.router.training.export_split_predictions`.

    Returns columns ``question_id``, ``oracle_curve``, ``predicted_weight``, ``valid``.
    Rows with non-finite ``predicted_weight`` are dropped.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_json(path, lines=True)
    missing = [c for c in _REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError(
            f"Predictions JSONL missing required column(s) {missing!r}; "
            f"expected at least {_REQUIRED_RAW}. Path: {path}"
        )
    if "is_valid_for_router_training" in df.columns:
        valid_s = df["is_valid_for_router_training"].fillna(False).astype(bool)
    else:
        valid_s = pd.Series(True, index=df.index, dtype=bool)

    curves = [_oracle_curve_cell_to_list(v) for v in df["oracle_curve"].tolist()]
    pred = pd.to_numeric(df["predicted_weight"], errors="coerce")
    out = pd.DataFrame(
        {
            "question_id": df["question_id"].astype(str),
            "oracle_curve": curves,
            "predicted_weight": pred,
            "valid": valid_s,
        }
    )
    ok = out["predicted_weight"].notna()
    return out.loc[ok].reset_index(drop=True)


def load_weight_grid_from_manifest(manifest_path: Path) -> np.ndarray:
    """Read ``model.weight_grid`` from a router ``manifest.json``."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Router manifest not found: {manifest_path}")
    data = read_json(manifest_path)
    wg = list((data.get("model") or {}).get("weight_grid") or [])
    if not wg:
        raise ValueError(f"manifest missing model.weight_grid: {manifest_path}")
    return np.asarray([float(x) for x in wg], dtype=np.float64)


def load_router_predictions_with_curves(path: Path) -> pd.DataFrame:
    """Alias for :func:`load_router_prediction_rows` (same schema)."""
    return load_router_prediction_rows(path)
