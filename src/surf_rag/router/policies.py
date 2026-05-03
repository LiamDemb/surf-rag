"""Routing policies: learned soft/hard/hybrid, constant 50/50, dense-only, graph-only."""

from __future__ import annotations

# Fusion band is inclusive on both ends; outer bands are strict (< low, > high).
LEARNED_HYBRID_FUSION_MIN = 0.4
LEARNED_HYBRID_FUSION_MAX = 0.6

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional


class RoutingPolicyName(str, Enum):
    LEARNED_SOFT = "learned-soft"
    LEARNED_HARD = "learned-hard"
    LEARNED_HYBRID = "learned-hybrid"
    EQUAL_50_50 = "50-50"
    DENSE_ONLY = "dense-only"
    GRAPH_ONLY = "graph-only"
    ORACLE_UPPER_BOUND = "oracle-upper-bound"


@dataclass(frozen=True)
class RoutingDecision:
    """Inputs to fusion or single-branch retrieval."""

    policy: RoutingPolicyName
    dense_weight: float
    run_dense: bool
    run_graph: bool
    predicted_weight: Optional[float]
    hard_branch: Optional[Literal["dense", "graph"]]
    tie_break: Optional[str]


def decide_routing(
    policy: RoutingPolicyName,
    *,
    predicted_weight: Optional[float] = None,
) -> RoutingDecision:
    """Return branch flags and fusion weight (for soft fusion when both run)."""
    if policy == RoutingPolicyName.EQUAL_50_50:
        return RoutingDecision(
            policy=policy,
            dense_weight=0.5,
            run_dense=True,
            run_graph=True,
            predicted_weight=0.5,
            hard_branch=None,
            tie_break=None,
        )
    if policy == RoutingPolicyName.DENSE_ONLY:
        return RoutingDecision(
            policy=policy,
            dense_weight=1.0,
            run_dense=True,
            run_graph=False,
            predicted_weight=None,
            hard_branch="dense",
            tie_break=None,
        )
    if policy == RoutingPolicyName.GRAPH_ONLY:
        return RoutingDecision(
            policy=policy,
            dense_weight=0.0,
            run_dense=False,
            run_graph=True,
            predicted_weight=None,
            hard_branch="graph",
            tie_break=None,
        )
    if policy == RoutingPolicyName.ORACLE_UPPER_BOUND:
        raise ValueError(
            "oracle-upper-bound decisions are computed from oracle_scores and must be "
            "handled in the e2e evaluation layer."
        )
    if predicted_weight is None:
        raise ValueError("learned policies require predicted_weight")
    ev = float(predicted_weight)
    clipped = float(max(0.0, min(1.0, ev)))
    if policy == RoutingPolicyName.LEARNED_SOFT:
        return RoutingDecision(
            policy=policy,
            dense_weight=clipped,
            run_dense=True,
            run_graph=True,
            predicted_weight=clipped,
            hard_branch=None,
            tie_break=None,
        )
    if policy == RoutingPolicyName.LEARNED_HARD:
        branch = "dense" if clipped >= 0.5 else "graph"
        reason = "weight_gte_0.5" if branch == "dense" else "weight_lt_0.5"
        return RoutingDecision(
            policy=policy,
            dense_weight=clipped,
            run_dense=branch == "dense",
            run_graph=branch == "graph",
            predicted_weight=clipped,
            hard_branch=branch,
            tie_break=reason,
        )
    if policy == RoutingPolicyName.LEARNED_HYBRID:
        if clipped < LEARNED_HYBRID_FUSION_MIN:
            return RoutingDecision(
                policy=policy,
                dense_weight=clipped,
                run_dense=False,
                run_graph=True,
                predicted_weight=clipped,
                hard_branch="graph",
                tie_break="weight_lt_0.4",
            )
        if clipped <= LEARNED_HYBRID_FUSION_MAX:
            return RoutingDecision(
                policy=policy,
                dense_weight=clipped,
                run_dense=True,
                run_graph=True,
                predicted_weight=clipped,
                hard_branch=None,
                tie_break="fusion_band_inclusive",
            )
        return RoutingDecision(
            policy=policy,
            dense_weight=clipped,
            run_dense=True,
            run_graph=False,
            predicted_weight=clipped,
            hard_branch="dense",
            tie_break="weight_gt_0.6",
        )
    raise ValueError(f"unknown policy {policy}")


def decision_to_debug_info(decision: RoutingDecision) -> Dict[str, Any]:
    return {
        "routing_policy": decision.policy.value,
        "dense_weight": decision.dense_weight,
        "run_dense": decision.run_dense,
        "run_graph": decision.run_graph,
        "predicted_weight": decision.predicted_weight,
        "hard_branch": decision.hard_branch,
        "tie_break": decision.tie_break,
    }
