"""Dense-weight **argmax intervals**: oracle curve plateaus vs a discrete weight grid.

**Argmax intervals:** bins whose score equals ``max(curve)`` within ``rtol``/``atol``;
each maximal contiguous run maps to a closed interval ``[w_lo, w_hi]`` on the weight axis.
This replaces ``np.argmax(curve)`` when reasoning about optimal *weights* under ties.

Used by router metrics, visualization, and any code that must agree on tie geometry.
Defaults ``DEFAULT_ARGMAX_INTERVAL_RTOL`` / ``DEFAULT_ARGMAX_INTERVAL_ATOL`` match viz specs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_ARGMAX_INTERVAL_RTOL: float = 1e-5
DEFAULT_ARGMAX_INTERVAL_ATOL: float = 1e-8


def dense_weight_argmax_intervals(
    curve: Sequence[float] | np.ndarray,
    weight_grid: Sequence[float] | np.ndarray,
    *,
    rtol: float = DEFAULT_ARGMAX_INTERVAL_RTOL,
    atol: float = DEFAULT_ARGMAX_INTERVAL_ATOL,
) -> list[tuple[float, float]]:
    """Contiguous grid runs tied at ``max(curve)`` → closed intervals on the weight axis."""
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


def optimal_dense_weight_intervals(
    curve: Sequence[float] | np.ndarray,
    weight_grid: Sequence[float] | np.ndarray,
    *,
    rtol: float = DEFAULT_ARGMAX_INTERVAL_RTOL,
    atol: float = DEFAULT_ARGMAX_INTERVAL_ATOL,
) -> list[tuple[float, float]]:
    """Backward-compatible alias for :func:`dense_weight_argmax_intervals`."""
    return dense_weight_argmax_intervals(curve, weight_grid, rtol=rtol, atol=atol)


def argmax_plateau_bin_indices(
    curve: Sequence[float] | np.ndarray,
    *,
    rtol: float = DEFAULT_ARGMAX_INTERVAL_RTOL,
    atol: float = DEFAULT_ARGMAX_INTERVAL_ATOL,
) -> np.ndarray:
    """Indices of all bins whose score is tied at ``max(curve)`` (within rtol/atol)."""
    s = np.asarray(curve, dtype=np.float64).reshape(-1)
    if len(s) == 0:
        return np.array([], dtype=np.int64)
    peak = float(np.max(s))
    return np.flatnonzero(np.isclose(s, peak, rtol=rtol, atol=atol)).astype(np.int64)


def _distance_point_to_closed_interval(w: float, lo: float, hi: float) -> float:
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    if a <= w <= b:
        return 0.0
    if w < a:
        return float(a - w)
    return float(w - b)


def distance_weight_to_argmax_intervals(
    pred_weight: float, intervals: list[tuple[float, float]]
) -> float:
    """Minimum distance from ``pred_weight`` to the union of closed intervals."""
    if not intervals:
        return float("nan")
    w = float(pred_weight)
    return float(
        min(_distance_point_to_closed_interval(w, lo, hi) for lo, hi in intervals)
    )


def prediction_hits_any_interval(
    pred: float, intervals: list[tuple[float, float]]
) -> bool:
    """True if ``pred`` lies in any closed interval (same semantics as before)."""
    pv = float(pred)
    for y_lo, y_hi in intervals:
        if y_lo - 1e-15 <= pv <= y_hi + 1e-15:
            return True
    return False


def mean_interval_midpoint(intervals: list[tuple[float, float]]) -> float:
    """Average of interval midpoints (for sorting queries left-to-right)."""
    if not intervals:
        return 0.0
    mids = [(a + b) * 0.5 for a, b in intervals]
    return float(np.mean(mids))


def _interval_span(w: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    y0 = float(w[lo])
    y1 = float(w[hi])
    return (min(y0, y1), max(y0, y1))
