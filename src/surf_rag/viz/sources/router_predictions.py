"""Load router eval ``predictions_<split>.jsonl`` into a normalized frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from surf_rag.evaluation.router_model_artifacts import RouterModelPaths

_REQUIRED_RAW = (
    "question_id",
    "target_oracle_best_weight",
    "predicted_weight",
)


def predictions_path_for(model_paths: RouterModelPaths, split: str) -> Path:
    """Canonical path for one split's prediction JSONL."""
    return model_paths.predictions(split)


def load_router_prediction_rows(path: Path) -> pd.DataFrame:
    """Read JSONL written by :func:`surf_rag.router.training.export_split_predictions`.

    Returns columns ``question_id``, ``oracle_weight``, ``predicted_weight``, ``valid``
    (``valid`` is False when ``is_valid_for_router_training`` is false or missing).
    Rows with non-finite ``oracle_weight`` or ``predicted_weight`` are dropped.
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
    out = pd.DataFrame(
        {
            "question_id": df["question_id"].astype(str),
            "oracle_weight": pd.to_numeric(
                df["target_oracle_best_weight"], errors="coerce"
            ),
            "predicted_weight": pd.to_numeric(df["predicted_weight"], errors="coerce"),
            "valid": valid_s,
        }
    )
    ok = out["oracle_weight"].notna() & out["predicted_weight"].notna()
    return out.loc[ok].reset_index(drop=True)
