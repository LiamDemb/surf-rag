"""Deterministic router labels from oracle sweep curves."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _as_finite_floats(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"Non-finite value encountered: {v!r}")
        out.append(fv)
    return out


def _extract_oracle_curve(row: Dict[str, Any]) -> List[float]:
    return [float(s.get("ndcg_primary", 0.0)) for s in (row.get("scores") or [])]


def _extract_weight_grid(row: Dict[str, Any]) -> List[float]:
    return [float(w) for w in (row.get("weight_grid") or [])]


def oracle_label_from_curve(
    *,
    oracle_curve: Sequence[float],
    weight_grid: Sequence[float],
    dataset_source: str,
) -> Dict[str, Any]:
    """Build one deterministic router-label record from an oracle curve."""
    curve = _as_finite_floats(oracle_curve)
    grid = _as_finite_floats(weight_grid)
    if not curve or not grid:
        raise ValueError("oracle_curve and weight_grid must be non-empty")
    if len(curve) != len(grid):
        raise ValueError(
            f"oracle_curve length {len(curve)} != weight_grid length {len(grid)}"
        )

    arr = np.asarray(curve, dtype=np.float32)
    best_score = float(np.max(arr))
    std = float(np.std(arr))
    return {
        "weight_grid": grid,
        "oracle_curve": curve,
        "oracle_best_score": best_score,
        "oracle_curve_std": std,
        "dataset_source": str(dataset_source or "").strip(),
        "is_valid_for_router_training": bool(best_score > 0.0),
    }


def materialize_router_labels(
    oracle_score_rows: Iterable[Dict[str, Any]],
    output_path: Path,
) -> int:
    """Materialize deterministic router labels from oracle score rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in oracle_score_rows:
            qid = str(row.get("question_id", "")).strip()
            if not qid:
                continue
            label = oracle_label_from_curve(
                oracle_curve=_extract_oracle_curve(row),
                weight_grid=_extract_weight_grid(row),
                dataset_source=str(row.get("dataset_source", "")),
            )
            rec = {"question_id": qid, **label}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count
