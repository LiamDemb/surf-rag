"""Routing policies for learned-soft, hard-routing, and hybrid fallback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional


class RoutingPolicyName(str, Enum):
    LEARNED_SOFT = "learned-soft"
    HARD_ROUTING = "hard-routing"
    HYBRID = "hybrid"
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
    predicted_class_id: Optional[int]
    confidence: Optional[float]
    confidence_threshold: Optional[float]
    fallback_triggered: bool
    fallback_weight: Optional[float]
    hard_branch: Optional[Literal["dense", "graph"]]
    tie_break: Optional[str]


def decide_routing(
    policy: RoutingPolicyName,
    *,
    predicted_weight: Optional[float] = None,
    predicted_class_id: Optional[int] = None,
    predicted_class_probs: Optional[tuple[float, float]] = None,
    confidence_threshold: Optional[float] = None,
    fallback_weight: Optional[float] = None,
) -> RoutingDecision:
    """Return branch flags and fusion weight (for soft fusion when both run)."""
    if policy == RoutingPolicyName.EQUAL_50_50:
        return RoutingDecision(
            policy=policy,
            dense_weight=0.5,
            run_dense=True,
            run_graph=True,
            predicted_weight=0.5,
            predicted_class_id=None,
            confidence=None,
            confidence_threshold=None,
            fallback_triggered=False,
            fallback_weight=None,
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
            predicted_class_id=None,
            confidence=None,
            confidence_threshold=None,
            fallback_triggered=False,
            fallback_weight=None,
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
            predicted_class_id=None,
            confidence=None,
            confidence_threshold=None,
            fallback_triggered=False,
            fallback_weight=None,
            hard_branch="graph",
            tie_break=None,
        )
    if policy == RoutingPolicyName.ORACLE_UPPER_BOUND:
        raise ValueError(
            "oracle-upper-bound decisions are computed from oracle_scores and must be "
            "handled in the e2e evaluation layer."
        )
    if policy in (RoutingPolicyName.HARD_ROUTING, RoutingPolicyName.HYBRID):
        if predicted_class_id is None:
            raise ValueError(f"{policy.value} requires predicted_class_id")
        cid = int(predicted_class_id)
        conf = None
        if predicted_class_probs is not None and len(predicted_class_probs) == 2:
            conf = float(max(predicted_class_probs))
        if policy == RoutingPolicyName.HYBRID:
            th = (
                float(confidence_threshold)
                if confidence_threshold is not None
                else None
            )
            if th is None:
                raise ValueError("hybrid requires confidence_threshold")
            if conf is None:
                raise ValueError("hybrid requires predicted_class_probs for confidence")
            if conf < th:
                if fallback_weight is None:
                    raise ValueError(
                        "hybrid low-confidence path requires fallback_weight"
                    )
                fw = float(max(0.0, min(1.0, float(fallback_weight))))
                return RoutingDecision(
                    policy=policy,
                    dense_weight=fw,
                    run_dense=True,
                    run_graph=True,
                    predicted_weight=None,
                    predicted_class_id=cid,
                    confidence=conf,
                    confidence_threshold=th,
                    fallback_triggered=True,
                    fallback_weight=fw,
                    hard_branch=None,
                    tie_break="low_confidence_fallback",
                )
        if cid == 1:
            return RoutingDecision(
                policy=policy,
                dense_weight=1.0,
                run_dense=True,
                run_graph=False,
                predicted_weight=None,
                predicted_class_id=cid,
                confidence=conf,
                confidence_threshold=(
                    float(confidence_threshold)
                    if confidence_threshold is not None
                    else None
                ),
                fallback_triggered=False,
                fallback_weight=None,
                hard_branch="dense",
                tie_break="class_dense",
            )
        return RoutingDecision(
            policy=policy,
            dense_weight=0.0,
            run_dense=False,
            run_graph=True,
            predicted_weight=None,
            predicted_class_id=cid,
            confidence=conf,
            confidence_threshold=(
                float(confidence_threshold)
                if confidence_threshold is not None
                else None
            ),
            fallback_triggered=False,
            fallback_weight=None,
            hard_branch="graph",
            tie_break="class_graph",
        )
    if predicted_weight is None:
        raise ValueError("learned-soft requires predicted_weight")
    ev = float(predicted_weight)
    clipped = float(max(0.0, min(1.0, ev)))
    if policy == RoutingPolicyName.LEARNED_SOFT:
        return RoutingDecision(
            policy=policy,
            dense_weight=clipped,
            run_dense=True,
            run_graph=True,
            predicted_weight=clipped,
            predicted_class_id=None,
            confidence=None,
            confidence_threshold=None,
            fallback_triggered=False,
            fallback_weight=None,
            hard_branch=None,
            tie_break=None,
        )
    raise ValueError(f"unknown policy {policy}")


def decision_to_debug_info(decision: RoutingDecision) -> Dict[str, Any]:
    return {
        "routing_policy": decision.policy.value,
        "dense_weight": decision.dense_weight,
        "run_dense": decision.run_dense,
        "run_graph": decision.run_graph,
        "predicted_weight": decision.predicted_weight,
        "predicted_class_id": decision.predicted_class_id,
        "confidence": decision.confidence,
        "confidence_threshold": decision.confidence_threshold,
        "fallback_triggered": decision.fallback_triggered,
        "fallback_weight": decision.fallback_weight,
        "hard_branch": decision.hard_branch,
        "tie_break": decision.tie_break,
    }
