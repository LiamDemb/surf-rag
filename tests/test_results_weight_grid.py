from __future__ import annotations

from surf_rag.results.weight_grid import load_router_weight_grid, weight_grid_for_curve


def test_weight_grid_for_curve_fallback_linspace() -> None:
    curve = [0.1, 0.5, 0.9]
    grid = weight_grid_for_curve(curve, [])
    assert len(grid) == 3
    assert grid[0] == 0.0
    assert grid[-1] == 1.0


def test_weight_grid_for_curve_uses_provided_grid() -> None:
    curve = [0.0, 1.0]
    grid = weight_grid_for_curve(curve, [0.0, 1.0])
    assert grid == [0.0, 1.0]
