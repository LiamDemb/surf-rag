"""Tests for plateau midpoint bucketing and train undersampling."""

from __future__ import annotations

import pandas as pd
import pytest

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.midpoint_balance import (
    build_train_midpoint_balance_indices,
    midpoint_to_group,
    plateau_midpoint,
)


def _grid() -> list[float]:
    return [float(x) for x in DEFAULT_DENSE_WEIGHT_GRID]


def test_plateau_single_peak_at_bin() -> None:
    g = _grid()
    c = [0.0] * len(g)
    c[3] = 1.0
    mid = plateau_midpoint(c, g, epsilon=1e-6)
    assert mid == pytest.approx(0.3)
    assert midpoint_to_group(mid) == 1


def test_plateau_flat_full_span_midpoint_half() -> None:
    g = _grid()
    c = [1.0] * len(g)
    mid = plateau_midpoint(c, g, epsilon=1e-6)
    assert mid == pytest.approx(0.5)
    assert midpoint_to_group(mid) == 2


def test_two_separated_peaks_uses_left_plateau_only() -> None:
    """First argmax wins; contiguous expansion stays on the left plateau."""
    g = _grid()
    c = [1.0, 0.5, 0.5, 1.0] + [0.5] * (len(g) - 4)
    mid = plateau_midpoint(c, g, epsilon=1e-6)
    assert mid == pytest.approx(0.0)
    assert midpoint_to_group(mid) == 0


def test_midpoint_to_group_boundaries() -> None:
    assert midpoint_to_group(0.0) == 0
    assert midpoint_to_group(0.2) == 1
    assert midpoint_to_group(0.8) == 4
    assert midpoint_to_group(1.0) == 4


def test_plateau_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same-length"):
        plateau_midpoint([1.0, 0.0], [0.0], epsilon=1e-6)


def test_build_mask_balances_to_min_per_group() -> None:
    g = _grid()

    def row_for_group(target_g: int) -> dict:
        c = [0.0] * len(g)
        if target_g == 0:
            c[0] = 1.0
        elif target_g == 1:
            c[3] = 1.0
        elif target_g == 2:
            for i in range(len(c)):
                c[i] = 1.0
        elif target_g == 3:
            c[7] = 1.0
        else:
            c[10] = 1.0
        return {"oracle_curve": c, "weight_grid": g}

    rows = []
    counts = [10, 5, 8, 5, 3]
    for gi, m in enumerate(counts):
        for _ in range(m):
            rows.append(row_for_group(gi))
    df = pd.DataFrame(rows)
    idx, stats = build_train_midpoint_balance_indices(df, epsilon=1e-6, seed=0)
    assert stats["target_per_group"] == 3
    assert stats["train_rows_before"] == sum(counts)
    assert stats["train_rows_after"] == 15
    assert stats["counts_after_per_group"] == [3, 3, 3, 3, 3]
    assert len(idx) == 15
    assert stats["dropped_per_group"] == [7, 2, 5, 2, 0]


def test_build_mask_raises_when_group_empty() -> None:
    g = _grid()
    c = [0.0] * len(g)
    c[0] = 1.0
    df = pd.DataFrame([{"oracle_curve": c, "weight_grid": g}] * 4)
    with pytest.raises(ValueError, match="every midpoint group"):
        build_train_midpoint_balance_indices(df, epsilon=1e-6, seed=0)
