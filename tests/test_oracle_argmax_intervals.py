from __future__ import annotations

import numpy as np
import pytest

from surf_rag.evaluation.oracle_argmax_intervals import (
    argmax_plateau_bin_indices,
    dense_weight_argmax_intervals,
    distance_weight_to_argmax_intervals,
    mean_interval_midpoint,
    optimal_dense_weight_intervals,
    prediction_hits_any_interval,
)


def test_single_contiguous_optimal_band() -> None:
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curve = np.array([0.1, 0.3, 0.9, 0.9, 0.2])
    intr = dense_weight_argmax_intervals(curve, weights)
    assert len(intr) == 1
    assert intr[0][0] == pytest.approx(0.5)
    assert intr[0][1] == pytest.approx(0.75)


def test_optimal_dense_weight_intervals_alias() -> None:
    weights = np.array([0.0, 0.5, 1.0])
    curve = np.array([0.0, 1.0, 0.0])
    a = optimal_dense_weight_intervals(curve, weights)
    b = dense_weight_argmax_intervals(curve, weights)
    assert a == b


def test_non_contiguous_max_ties_yield_multiple_segments() -> None:
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curve = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    intr = dense_weight_argmax_intervals(curve, weights, rtol=0.0, atol=0.0)
    assert len(intr) == 3
    assert intr[0][0] == pytest.approx(0.0) and intr[0][1] == pytest.approx(0.0)
    assert intr[1][0] == pytest.approx(0.5) and intr[1][1] == pytest.approx(0.5)
    assert intr[2][0] == pytest.approx(1.0) and intr[2][1] == pytest.approx(1.0)


def test_mean_midpoint_three_segments() -> None:
    intr = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    assert mean_interval_midpoint(intr) == pytest.approx((0 + 0.5 + 1.0) / 3)


def test_prediction_hits_boundary() -> None:
    intr = [(0.2, 0.8)]
    assert prediction_hits_any_interval(0.2, intr)
    assert prediction_hits_any_interval(0.8, intr)
    assert prediction_hits_any_interval(0.5, intr)
    assert not prediction_hits_any_interval(0.19, intr)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        dense_weight_argmax_intervals([0.0, 1.0], [0.0, 0.5, 1.0])


def test_distance_inside_interval_zero() -> None:
    intr = [(0.2, 0.4)]
    assert distance_weight_to_argmax_intervals(0.3, intr) == pytest.approx(0.0)


def test_distance_outside_interval_nearest_endpoint() -> None:
    intr = [(0.2, 0.4)]
    assert distance_weight_to_argmax_intervals(0.1, intr) == pytest.approx(0.1)
    assert distance_weight_to_argmax_intervals(0.5, intr) == pytest.approx(0.1)


def test_distance_multi_interval_minimum_gap() -> None:
    intr = [(0.0, 0.1), (0.9, 1.0)]
    assert distance_weight_to_argmax_intervals(0.5, intr) == pytest.approx(0.4)


def test_distance_empty_intervals_nan() -> None:
    assert np.isnan(distance_weight_to_argmax_intervals(0.5, []))


def test_plateau_indices_consistent_with_intervals_cover_weights() -> None:
    weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    curve = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    idx = argmax_plateau_bin_indices(curve, rtol=0.0, atol=0.0)
    intr = dense_weight_argmax_intervals(curve, weights, rtol=0.0, atol=0.0)
    assert set(idx.tolist()) == {0, 2, 4}
    for i in idx:
        wv = float(weights[int(i)])
        dist = distance_weight_to_argmax_intervals(wv, intr)
        assert dist == pytest.approx(0.0)
