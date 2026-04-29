"""Global frontier relational path enumeration for canonical graph retrieval.

Hop validity matches :func:`surf_rag.graph.graph_paths.iter_valid_rel_expansions`.
The canonical enumerator is :func:`enumerate_global_frontier_paths`.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

from surf_rag.graph.graph_grounding import stored_edge_endpoints
from surf_rag.graph.graph_paths import (
    GraphPathEnumerationDiagnostics,
    iter_valid_rel_expansions,
)
from surf_rag.graph.graph_specificity import node_specificity_score
from surf_rag.graph.graph_types import GraphHop, GraphPath


def edge_support_boost(graph: Any, hop: GraphHop) -> float:
    """Relation-level support from aggregated corpus counts on this directed edge."""
    u, v = stored_edge_endpoints(hop)
    if not graph.has_edge(u, v):
        return 1.0
    data = graph[u][v]
    counts = data.get("support_count_by_label") or {}
    cnt = int(counts.get(hop.relation, 0))
    return float(math.log1p(cnt) + 1.0)


def enumerate_global_frontier_paths(
    graph: Any,
    seed_weights: dict[str, float],
    max_hops: int,
    bidirectional: bool = True,
    global_max_paths: int = 500,
    global_max_pops: int = 50_000,
) -> tuple[list[GraphPath], GraphPathEnumerationDiagnostics]:
    """Global frontier enumeration over all seeds; prioritizes seed mass × specificity × support."""
    paths: list[GraphPath] = []
    diag = GraphPathEnumerationDiagnostics()
    diag.enumeration_backend = "global_canonical"

    heap: list[
        tuple[float, int, str, tuple[GraphHop, ...], frozenset[str], float, str, float]
    ] = []
    counter = 0

    for seed, mass in sorted(seed_weights.items()):
        if mass <= 0.0 or seed not in graph:
            continue
        diag.start_nodes_used += 1
        seed_mass_log = math.log(max(float(mass), 1e-12))
        seed_spec = float(node_specificity_score(graph, seed))
        init_pri = seed_mass_log + math.log(max(seed_spec, 1e-9))
        heapq.heappush(
            heap,
            (
                -init_pri,
                counter,
                seed,
                (),
                frozenset({seed}),
                seed_spec,
                seed,
                seed_mass_log,
            ),
        )
        counter += 1
        diag.beam_pushes += 1
        diag.beam_frontier_peak = max(diag.beam_frontier_peak, len(heap))

    while heap and len(paths) < global_max_paths:
        if diag.beam_pops >= global_max_pops:
            diag.beam_truncated = True
            break

        (
            _,
            _,
            current,
            hops_tuple,
            visited_frozen,
            cum_min_spec,
            seed_origin,
            seed_mass_log,
        ) = heapq.heappop(heap)
        diag.beam_pops += 1

        hops_list = list(hops_tuple)

        if hops_list:
            paths.append(GraphPath(start_node=seed_origin, hops=tuple(hops_list)))

        if len(hops_list) >= max_hops:
            continue

        visited_set = set(visited_frozen)
        for next_ent, new_hop in iter_valid_rel_expansions(
            graph,
            current,
            visited_set,
            bidirectional=bidirectional,
            diag=diag,
        ):
            next_spec = float(node_specificity_score(graph, next_ent))
            new_min = min(cum_min_spec, next_spec)
            eb = edge_support_boost(graph, new_hop)
            log_pri = (
                seed_mass_log + math.log(max(new_min, 1e-9)) + math.log(max(eb, 1e-9))
            )
            new_hops = hops_tuple + (new_hop,)
            new_vis = visited_frozen | {next_ent}
            heapq.heappush(
                heap,
                (
                    -log_pri,
                    counter,
                    next_ent,
                    new_hops,
                    new_vis,
                    new_min,
                    seed_origin,
                    seed_mass_log,
                ),
            )
            counter += 1
            diag.beam_pushes += 1
            diag.beam_frontier_peak = max(diag.beam_frontier_peak, len(heap))

    diag.paths_emitted = len(paths)
    return paths, diag
