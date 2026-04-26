"""Routing policy decisions."""

from __future__ import annotations

import numpy as np
import pytest

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.policies import (
    RoutingPolicyName,
    decide_routing,
    hard_branch_from_distribution,
)


def test_equal_and_single_branch() -> None:
    d50 = decide_routing(RoutingPolicyName.EQUAL_50_50)
    assert d50.dense_weight == 0.5 and d50.run_dense and d50.run_graph
    dd = decide_routing(RoutingPolicyName.DENSE_ONLY)
    assert dd.run_dense and not dd.run_graph
    dg = decide_routing(RoutingPolicyName.GRAPH_ONLY)
    assert dg.run_graph and not dg.run_dense


def test_learned_soft() -> None:
    grid = np.asarray(DEFAULT_DENSE_WEIGHT_GRID, dtype=np.float32)
    dist = np.zeros(11, dtype=np.float32)
    dist[7] = 1.0
    d = decide_routing(
        RoutingPolicyName.LEARNED_SOFT, predicted_dist=dist, weight_grid=grid
    )
    assert d.run_dense and d.run_graph
    assert abs(d.dense_weight - 0.7) < 0.01


def test_hard_tie_default_dense() -> None:
    grid = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    dist = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    br, reason = hard_branch_from_distribution(dist, grid, threshold=0.5)
    assert br == "dense" and "tie" in reason


def test_hard_prefers_graph() -> None:
    grid = np.asarray([0.0, 0.4, 1.0], dtype=np.float32)
    # One-hot at w=0.0 => expected weight 0.0 < 0.5 => graph branch
    dist = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d = decide_routing(
        RoutingPolicyName.LEARNED_HARD, predicted_dist=dist, weight_grid=grid
    )
    assert d.run_graph and not d.run_dense
    assert d.hard_branch == "graph"
