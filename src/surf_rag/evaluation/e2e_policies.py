"""Parse CLI / env routing policy names for e2e evaluation."""

from __future__ import annotations

from surf_rag.router.policies import RoutingPolicyName

ORACLE_UPPER_BOUND_POLICY = "oracle-upper-bound"


def parse_routing_policy(name: str) -> str:
    """Accept policy names with minor aliases."""
    s = name.strip().lower().replace("_", "-")
    if s == ORACLE_UPPER_BOUND_POLICY:
        return ORACLE_UPPER_BOUND_POLICY
    aliases = {
        "learnedsoft": RoutingPolicyName.LEARNED_SOFT,
        "hardrouting": RoutingPolicyName.HARD_ROUTING,
        "hybridrouting": RoutingPolicyName.HYBRID,
        "5050": RoutingPolicyName.EQUAL_50_50,
        "50-50": RoutingPolicyName.EQUAL_50_50,
        "equal": RoutingPolicyName.EQUAL_50_50,
        "dense": RoutingPolicyName.DENSE_ONLY,
        "graph": RoutingPolicyName.GRAPH_ONLY,
    }
    if s in aliases:
        return aliases[s].value
    for p in RoutingPolicyName:
        if p.value == s:
            return p.value
    choices = ", ".join(
        sorted([ORACLE_UPPER_BOUND_POLICY, *(p.value for p in RoutingPolicyName)])
    )
    raise ValueError(f"Unknown routing policy {name!r}; expected one of: {choices}")


def e2e_pipeline_manifest_name(policy: str) -> str:
    """Stable ``pipeline_name`` / batch ``custom_id`` segment for routed e2e runs."""
    return f"routed-{policy}"
