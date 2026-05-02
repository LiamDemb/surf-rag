"""Derive dense-weight intervals where the oracle curve is tied at its maximum."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def optimal_dense_weight_intervals(
    curve: Sequence[float] | np.ndarray,
    weight_grid: Sequence[float] | np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[tuple[float, float]]:
    """Contiguous grid runs whose scores equal ``max(curve)`` → closed [y_low, y_high] intervals."""
    s = np.asarray(curve, dtype=np.float64).reshape(-1)
    w = np.asarray(weight_grid, dtype=np.float64).reshape(-1)
    if s.shape != w.shape:
        raise ValueError(f"oracle_curve length {len(s)} != weight_grid length {len(w)}")
    if len(s) == 0:
        return []
    max_s = float(np.max(s))
    tied = np.isclose(s, max_s, rtol=rtol, atol=atol)
    idx = np.flatnonzero(tied)
    if idx.size == 0:
        return []
    intervals: list[tuple[float, float]] = []
    run_lo = int(idx[0])
    run_hi = run_lo
    for k in idx[1:]:
        k = int(k)
        if k == run_hi + 1:
            run_hi = k
        else:
            intervals.append(_interval_span(w, run_lo, run_hi))
            run_lo = run_hi = k
    intervals.append(_interval_span(w, run_lo, run_hi))
    return intervals


def _interval_span(w: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    y0 = float(w[lo])
    y1 = float(w[hi])
    return (min(y0, y1), max(y0, y1))


def mean_interval_midpoint(intervals: list[tuple[float, float]]) -> float:
    """Average of interval midpoints (for sorting queries left-to-right)."""
    if not intervals:
        return 0.0
    mids = [(a + b) * 0.5 for a, b in intervals]
    return float(np.mean(mids))


def prediction_hits_any_interval(
    pred: float, intervals: list[tuple[float, float]]
) -> bool:
    """True if ``pred`` lies in any closed interval."""
    pv = float(pred)
    for y_lo, y_hi in intervals:
        if y_lo - 1e-15 <= pv <= y_hi + 1e-15:
            return True
    return False
