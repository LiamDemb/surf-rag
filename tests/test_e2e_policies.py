from surf_rag.evaluation.e2e_policies import (
    e2e_pipeline_manifest_name,
    parse_routing_policy,
)
from surf_rag.router.policies import RoutingPolicyName


def test_parse_routing_policy_aliases() -> None:
    assert parse_routing_policy("dense") == RoutingPolicyName.DENSE_ONLY
    assert parse_routing_policy("50_50") == RoutingPolicyName.EQUAL_50_50
    assert parse_routing_policy("learned-soft") == RoutingPolicyName.LEARNED_SOFT


def test_e2e_pipeline_manifest_name() -> None:
    assert (
        e2e_pipeline_manifest_name(RoutingPolicyName.DENSE_ONLY) == "routed-dense-only"
    )
