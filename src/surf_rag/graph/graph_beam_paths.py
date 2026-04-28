"""Specificity-aware best-first (beam-style) relational path enumeration.

Uses the same hop validity rules as depth-first search via
:func:`surf_rag.graph.graph_paths.iter_valid_rel_expansions`, but expands the
frontier in priority order using graph specificity and relation support counts.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, List

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


def enumerate_beam_candidate_paths(
    graph: Any,
    start_nodes: set[str],
    max_hops: int,
    bidirectional: bool = True,
    max_paths_per_start: int = 50,
    beam_max_pops: int = 20_000,
) -> tuple[list[GraphPath], GraphPathEnumerationDiagnostics]:
    """Enumerate relational paths using a priority frontier (beam-style BFS).

    Paths are explored in decreasing order of ``min(specificity on path) ×
    relation support``, approximating a beam search without aggressive early
    pruning so recall stays comparable to DFS.

    ``beam_max_pops`` caps worst-case work on large hubs.
    """
    paths: list[GraphPath] = []
    diag = GraphPathEnumerationDiagnostics()
    diag.enumeration_backend = "beam"

    for node in start_nodes:
        if node not in graph:
            continue

        diag.start_nodes_used += 1
        start_node_paths: List[GraphPath] = []

        seed_spec = float(node_specificity_score(graph, node))
        heap: list[
            tuple[float, int, str, tuple[GraphHop, ...], frozenset[str], float]
        ] = []
        counter = 0
        init_pri = math.log(max(seed_spec, 1e-9))
        heapq.heappush(
            heap,
            (-init_pri, counter, node, (), frozenset({node}), seed_spec),
        )
        counter += 1
        diag.beam_pushes += 1
        diag.beam_frontier_peak = max(diag.beam_frontier_peak, len(heap))

        while heap and len(start_node_paths) < max_paths_per_start:
            if diag.beam_pops >= beam_max_pops:
                diag.beam_truncated = True
                break

            _, _, current, hops_tuple, visited_frozen, cum_min_spec = heapq.heappop(
                heap
            )
            diag.beam_pops += 1

            hops_list = list(hops_tuple)

            if hops_list:
                start_node_paths.append(
                    GraphPath(start_node=node, hops=tuple(hops_list))
                )

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
                log_pri = math.log(max(new_min, 1e-9)) + math.log(max(eb, 1e-9))

                new_hops = hops_tuple + (new_hop,)
                new_vis = visited_frozen | {next_ent}
                heapq.heappush(
                    heap,
                    (-log_pri, counter, next_ent, new_hops, new_vis, new_min),
                )
                counter += 1
                diag.beam_pushes += 1
                diag.beam_frontier_peak = max(diag.beam_frontier_peak, len(heap))

        if len(start_node_paths) >= max_paths_per_start:
            diag.starts_hit_path_budget.append(node)
        paths.extend(start_node_paths)

    diag.paths_emitted = len(paths)
    return paths, diag


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
