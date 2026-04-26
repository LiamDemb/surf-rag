"""Parse CLI / env routing policy names for e2e evaluation."""

from __future__ import annotations

from surf_rag.router.policies import RoutingPolicyName


def parse_routing_policy(name: str) -> RoutingPolicyName:
    """Accept enum values (``learned-soft``, ``dense-only``, …) with minor aliases."""
    s = name.strip().lower().replace("_", "-")
    aliases = {
        "learnedsoft": RoutingPolicyName.LEARNED_SOFT,
        "learnedhard": RoutingPolicyName.LEARNED_HARD,
        "5050": RoutingPolicyName.EQUAL_50_50,
        "50-50": RoutingPolicyName.EQUAL_50_50,
        "equal": RoutingPolicyName.EQUAL_50_50,
        "dense": RoutingPolicyName.DENSE_ONLY,
        "graph": RoutingPolicyName.GRAPH_ONLY,
    }
    if s in aliases:
        return aliases[s]
    for p in RoutingPolicyName:
        if p.value == s:
            return p
    choices = ", ".join(sorted(p.value for p in RoutingPolicyName))
    raise ValueError(f"Unknown routing policy {name!r}; expected one of: {choices}")


def e2e_pipeline_manifest_name(policy: RoutingPolicyName) -> str:
    """Stable ``pipeline_name`` / batch ``custom_id`` segment for routed e2e runs."""
    return f"routed-{policy.value}"
