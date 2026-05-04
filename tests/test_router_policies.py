"""Routing policy decisions."""

from __future__ import annotations

from surf_rag.router.policies import (
    RoutingPolicyName,
    decide_routing,
)


def test_equal_and_single_branch() -> None:
    d50 = decide_routing(RoutingPolicyName.EQUAL_50_50)
    assert d50.dense_weight == 0.5 and d50.run_dense and d50.run_graph
    dd = decide_routing(RoutingPolicyName.DENSE_ONLY)
    assert dd.run_dense and not dd.run_graph
    dg = decide_routing(RoutingPolicyName.GRAPH_ONLY)
    assert dg.run_graph and not dg.run_dense


def test_learned_soft() -> None:
    d = decide_routing(RoutingPolicyName.LEARNED_SOFT, predicted_weight=0.7)
    assert d.run_dense and d.run_graph
    assert abs(d.dense_weight - 0.7) < 0.01


def test_hard_routing_dense() -> None:
    d = decide_routing(
        RoutingPolicyName.HARD_ROUTING,
        predicted_class_id=1,
        predicted_class_probs=(0.1, 0.9),
    )
    assert d.run_dense and not d.run_graph
    assert d.hard_branch == "dense"


def test_hard_routing_graph() -> None:
    d = decide_routing(
        RoutingPolicyName.HARD_ROUTING,
        predicted_class_id=0,
        predicted_class_probs=(0.8, 0.2),
    )
    assert d.run_graph and not d.run_dense
    assert d.hard_branch == "graph"


def test_hybrid_high_confidence_uses_classifier_graph() -> None:
    d = decide_routing(
        RoutingPolicyName.HYBRID,
        predicted_class_id=0,
        predicted_class_probs=(0.9, 0.1),
        confidence_threshold=0.7,
    )
    assert d.run_graph and not d.run_dense
    assert d.hard_branch == "graph"
    assert d.tie_break == "class_graph"


def test_hybrid_low_confidence_falls_back_to_regressor_weight() -> None:
    d = decide_routing(
        RoutingPolicyName.HYBRID,
        predicted_class_id=1,
        predicted_class_probs=(0.45, 0.55),
        confidence_threshold=0.7,
        fallback_weight=0.35,
    )
    assert d.run_dense and d.run_graph
    assert d.fallback_triggered
    assert d.tie_break == "low_confidence_fallback"
