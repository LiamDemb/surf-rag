"""Routing policies: learned soft/hard, constant 50/50, dense-only, graph-only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from surf_rag.evaluation.weight_grid import DEFAULT_DENSE_WEIGHT_GRID


class RoutingPolicyName(str, Enum):
    LEARNED_SOFT = "learned-soft"
    LEARNED_HARD = "learned-hard"
    EQUAL_50_50 = "50-50"
    DENSE_ONLY = "dense-only"
    GRAPH_ONLY = "graph-only"


@dataclass(frozen=True)
class RoutingDecision:
    """Inputs to fusion or single-branch retrieval."""

    policy: RoutingPolicyName
    dense_weight: float
    run_dense: bool
    run_graph: bool
    predicted_distribution: Optional[np.ndarray]
    expected_dense_weight: Optional[float]
    hard_branch: Optional[Literal["dense", "graph"]]
    tie_break: Optional[str]


def _mass_balance(p: np.ndarray, grid: np.ndarray, threshold: float = 0.5) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    g = np.asarray(grid, dtype=np.float64).reshape(-1)
    return float(np.sum(p[g > threshold]) - np.sum(p[g < threshold]))


def hard_branch_from_distribution(
    dist: np.ndarray,
    weight_grid: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[Literal["dense", "graph"], str]:
    """Choose dense or graph from soft distribution over the weight grid."""
    g = np.asarray(weight_grid, dtype=np.float64).reshape(-1)
    p = np.asarray(dist, dtype=np.float64).reshape(-1)
    ev = float(np.dot(p, g))
    if ev > threshold:
        return "dense", "expected_gt_threshold"
    if ev < threshold:
        return "graph", "expected_lt_threshold"
    bal = _mass_balance(p, g, threshold)
    if bal > 0:
        return "dense", "tie_mass_dense"
    if bal < 0:
        return "graph", "tie_mass_graph"
    return "dense", "tie_default_dense"


def decide_routing(
    policy: RoutingPolicyName,
    *,
    predicted_dist: Optional[np.ndarray] = None,
    weight_grid: Optional[np.ndarray] = None,
) -> RoutingDecision:
    """Return branch flags and fusion weight (for soft fusion when both run)."""
    wg = (
        np.asarray(weight_grid, dtype=np.float32).reshape(-1)
        if weight_grid is not None
        else np.asarray(DEFAULT_DENSE_WEIGHT_GRID, dtype=np.float32)
    )
    if policy == RoutingPolicyName.EQUAL_50_50:
        return RoutingDecision(
            policy=policy,
            dense_weight=0.5,
            run_dense=True,
            run_graph=True,
            predicted_distribution=None,
            expected_dense_weight=0.5,
            hard_branch=None,
            tie_break=None,
        )
    if policy == RoutingPolicyName.DENSE_ONLY:
        return RoutingDecision(
            policy=policy,
            dense_weight=1.0,
            run_dense=True,
            run_graph=False,
            predicted_distribution=None,
            expected_dense_weight=None,
            hard_branch="dense",
            tie_break=None,
        )
    if policy == RoutingPolicyName.GRAPH_ONLY:
        return RoutingDecision(
            policy=policy,
            dense_weight=0.0,
            run_dense=False,
            run_graph=True,
            predicted_distribution=None,
            expected_dense_weight=None,
            hard_branch="graph",
            tie_break=None,
        )
    if predicted_dist is None:
        raise ValueError("learned policies require predicted_dist")
    p = np.asarray(predicted_dist, dtype=np.float64).reshape(-1)
    ev = float(np.dot(p, wg.astype(np.float64)))
    if policy == RoutingPolicyName.LEARNED_SOFT:
        return RoutingDecision(
            policy=policy,
            dense_weight=float(np.clip(ev, 0.0, 1.0)),
            run_dense=True,
            run_graph=True,
            predicted_distribution=p.astype(np.float32),
            expected_dense_weight=float(ev),
            hard_branch=None,
            tie_break=None,
        )
    if policy == RoutingPolicyName.LEARNED_HARD:
        branch, reason = hard_branch_from_distribution(p, wg)
        return RoutingDecision(
            policy=policy,
            dense_weight=float(np.clip(ev, 0.0, 1.0)),
            run_dense=branch == "dense",
            run_graph=branch == "graph",
            predicted_distribution=p.astype(np.float32),
            expected_dense_weight=float(ev),
            hard_branch=branch,
            tie_break=reason,
        )
    raise ValueError(f"unknown policy {policy}")


def decision_to_debug_info(decision: RoutingDecision) -> Dict[str, Any]:
    return {
        "routing_policy": decision.policy.value,
        "dense_weight": decision.dense_weight,
        "run_dense": decision.run_dense,
        "run_graph": decision.run_graph,
        "expected_dense_weight": decision.expected_dense_weight,
        "hard_branch": decision.hard_branch,
        "tie_break": decision.tie_break,
    }
