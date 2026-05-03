"""Shim: oracle argmax-interval geometry lives in ``surf_rag.evaluation``.

Import from here or from :mod:`surf_rag.evaluation.oracle_argmax_intervals` directly.
"""

from __future__ import annotations

from surf_rag.evaluation.oracle_argmax_intervals import (
    DEFAULT_ARGMAX_INTERVAL_ATOL,
    DEFAULT_ARGMAX_INTERVAL_RTOL,
    argmax_plateau_bin_indices,
    dense_weight_argmax_intervals,
    distance_weight_to_argmax_intervals,
    mean_interval_midpoint,
    optimal_dense_weight_intervals,
    prediction_hits_any_interval,
)

__all__ = [
    "DEFAULT_ARGMAX_INTERVAL_ATOL",
    "DEFAULT_ARGMAX_INTERVAL_RTOL",
    "argmax_plateau_bin_indices",
    "dense_weight_argmax_intervals",
    "distance_weight_to_argmax_intervals",
    "mean_interval_midpoint",
    "optimal_dense_weight_intervals",
    "prediction_hits_any_interval",
]
