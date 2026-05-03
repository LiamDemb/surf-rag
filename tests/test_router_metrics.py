"""Router metric aggregation."""

from __future__ import annotations

import numpy as np

from surf_rag.router.router_metrics import aggregate_router_metrics


def test_perfect_match_argmax_interval_distance() -> None:
    grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    curves = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    pred_w = np.asarray([0.5], dtype=np.float32)
    valid = np.asarray([True], dtype=bool)
    m = aggregate_router_metrics(curves, pred_w, valid, grid)
    assert m["mean_regret"] < 1e-6
    assert m["argmax_interval_distance_mae"] < 1e-5
    assert m["argmax_interval_distance_rmse"] < 1e-5


def test_normalized_regret_uses_oracle_best_score_denominator() -> None:
    # Two curves with best score=1.0 but best weight=0.0 and 1.0. If denominator
    # incorrectly uses oracle-best weight mean (0.5), normalized regret doubles.
    grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    curves = np.asarray(
        [
            [1.0, 0.0, 0.0],  # best at w=0.0, C*=1.0
            [0.0, 0.0, 1.0],  # best at w=1.0, C*=1.0
        ],
        dtype=np.float32,
    )
    pred_w = np.asarray([0.5, 0.5], dtype=np.float32)
    valid = np.asarray([True, True], dtype=bool)

    m = aggregate_router_metrics(curves, pred_w, valid, grid)
    # Interpolated score at w=0.5 is 0 for both rows, so regret is 1 each.
    # mean_regret=1.0; mean(C*)=1.0 => normalized_regret must be 1.0.
    assert m["mean_regret"] == np.float32(1.0)
    assert m["normalized_regret"] == np.float32(1.0)
    # Guard against accidental denominator regression to mean oracle-best weight (0.5),
    # which would incorrectly produce normalized_regret=2.0 here.
    assert m["normalized_regret"] != np.float32(2.0)


def test_num_rows_counts_only_valid_mask_rows() -> None:
    grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    curves = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    pred_w = np.asarray([0.5, 0.5], dtype=np.float32)
    valid = np.asarray([True, False], dtype=bool)
    m = aggregate_router_metrics(curves, pred_w, valid, grid)
    assert m["num_rows"] == 1.0


def test_plateau_two_bins_distance_zero_inside() -> None:
    grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    curves = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32)
    pred_w = np.asarray([0.7], dtype=np.float32)
    valid = np.asarray([True], dtype=bool)
    m = aggregate_router_metrics(curves, pred_w, valid, grid)
    assert m["argmax_interval_distance_mae"] < 1e-5
