"""Routing policy decisions."""

from __future__ import annotations

from surf_rag.router.policies import (
    LEARNED_HYBRID_FUSION_MAX,
    LEARNED_HYBRID_FUSION_MIN,
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


def test_hard_threshold_dense() -> None:
    d = decide_routing(RoutingPolicyName.LEARNED_HARD, predicted_weight=0.5)
    assert d.run_dense and not d.run_graph
    assert d.hard_branch == "dense"


def test_hard_prefers_graph() -> None:
    d = decide_routing(RoutingPolicyName.LEARNED_HARD, predicted_weight=0.4)
    assert d.run_graph and not d.run_dense
    assert d.hard_branch == "graph"


def test_learned_hybrid_graph_strict_below_band() -> None:
    d = decide_routing(
        RoutingPolicyName.LEARNED_HYBRID,
        predicted_weight=LEARNED_HYBRID_FUSION_MIN - 1e-9,
    )
    assert d.run_graph and not d.run_dense
    assert d.hard_branch == "graph"
    assert d.tie_break == "weight_lt_0.4"


def test_learned_hybrid_graph_at_low_threshold_exclusive() -> None:
    """Exactly 0.4 routes to graph-only (not fusion)."""
    d = decide_routing(
        RoutingPolicyName.LEARNED_HYBRID,
        predicted_weight=LEARNED_HYBRID_FUSION_MIN,
    )
    assert d.run_graph and not d.run_dense
    assert d.hard_branch == "graph"
    assert d.tie_break == "weight_lt_0.4"
    assert abs(d.dense_weight - LEARNED_HYBRID_FUSION_MIN) < 1e-12


def test_learned_hybrid_dense_at_high_threshold_exclusive() -> None:
    """Exactly 0.6 routes to dense-only (not fusion)."""
    d = decide_routing(
        RoutingPolicyName.LEARNED_HYBRID,
        predicted_weight=LEARNED_HYBRID_FUSION_MAX,
    )
    assert d.run_dense and not d.run_graph
    assert d.hard_branch == "dense"
    assert d.tie_break == "weight_gt_0.6"
    assert abs(d.dense_weight - LEARNED_HYBRID_FUSION_MAX) < 1e-12


def test_learned_hybrid_fusion_mid() -> None:
    d = decide_routing(RoutingPolicyName.LEARNED_HYBRID, predicted_weight=0.5)
    assert d.run_dense and d.run_graph
    assert d.tie_break == "fusion_band_open"


def test_learned_hybrid_dense_strict_above_band() -> None:
    d = decide_routing(
        RoutingPolicyName.LEARNED_HYBRID,
        predicted_weight=LEARNED_HYBRID_FUSION_MAX + 1e-9,
    )
    assert d.run_dense and not d.run_graph
    assert d.hard_branch == "dense"
    assert d.tie_break == "weight_gt_0.6"
