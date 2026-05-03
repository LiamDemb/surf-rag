from surf_rag.evaluation.e2e_policies import (
    ORACLE_UPPER_BOUND_POLICY,
    e2e_pipeline_manifest_name,
    parse_routing_policy,
)
from surf_rag.router.policies import RoutingPolicyName


def test_parse_routing_policy_aliases() -> None:
    assert parse_routing_policy("dense") == RoutingPolicyName.DENSE_ONLY.value
    assert parse_routing_policy("50_50") == RoutingPolicyName.EQUAL_50_50.value
    assert parse_routing_policy("learned-soft") == RoutingPolicyName.LEARNED_SOFT.value
    assert (
        parse_routing_policy("learned-hybrid") == RoutingPolicyName.LEARNED_HYBRID.value
    )
    assert (
        parse_routing_policy("learnedhybrid") == RoutingPolicyName.LEARNED_HYBRID.value
    )
    assert parse_routing_policy("oracle-upper-bound") == ORACLE_UPPER_BOUND_POLICY


def test_e2e_pipeline_manifest_name() -> None:
    assert (
        e2e_pipeline_manifest_name(RoutingPolicyName.DENSE_ONLY.value)
        == "routed-dense-only"
    )
