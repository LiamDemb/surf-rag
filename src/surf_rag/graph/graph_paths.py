from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple

from surf_rag.graph.graph_types import GraphHop, GraphPath

MAX_DEGREE = 250
MAX_APPEARANCES = 30


def count_appearances(graph, node):
    count = sum(
        1
        for _, _, d in graph.out_edges(node, data=True)
        if d.get("kind") == "appears_in"
    )
    return count


@dataclass
class GraphPathEnumerationDiagnostics:
    """Counters collected while enumerating candidate relational paths (per query)."""

    start_nodes_used: int = 0
    paths_emitted: int = 0
    starts_hit_path_budget: List[str] = field(default_factory=list)

    skip_outgoing_visited: int = 0
    skip_outgoing_degree: int = 0
    skip_outgoing_appearances: int = 0
    skip_outgoing_not_rel: int = 0
    skip_outgoing_instance_of_only: int = 0

    skip_incoming_visited: int = 0
    skip_incoming_degree: int = 0
    skip_incoming_appearances: int = 0
    skip_incoming_not_rel: int = 0
    skip_incoming_instance_of_only: int = 0

    enumeration_backend: str = "dfs"
    beam_frontier_peak: int = 0
    beam_pops: int = 0
    beam_pushes: int = 0
    beam_truncated: bool = False

    def to_json(self) -> Dict[str, Any]:
        out = {
            "start_nodes_used": self.start_nodes_used,
            "paths_emitted": self.paths_emitted,
            "starts_hit_path_budget": list(self.starts_hit_path_budget),
            "enumeration_backend": self.enumeration_backend,
            "skip_outgoing": {
                "visited": self.skip_outgoing_visited,
                "degree_gt_cap": self.skip_outgoing_degree,
                "appearances_gt_cap": self.skip_outgoing_appearances,
                "not_rel_edge": self.skip_outgoing_not_rel,
                "instance_of_only": self.skip_outgoing_instance_of_only,
            },
            "skip_incoming": {
                "visited": self.skip_incoming_visited,
                "degree_gt_cap": self.skip_incoming_degree,
                "appearances_gt_cap": self.skip_incoming_appearances,
                "not_rel_edge": self.skip_incoming_not_rel,
                "instance_of_only": self.skip_incoming_instance_of_only,
            },
        }
        out["beam"] = {
            "frontier_peak": self.beam_frontier_peak,
            "pops": self.beam_pops,
            "pushes": self.beam_pushes,
            "truncated": self.beam_truncated,
        }
        return out


def iter_valid_rel_expansions(
    graph,
    current_node: str,
    visited: set[str],
    *,
    bidirectional: bool,
    diag: GraphPathEnumerationDiagnostics | None = None,
) -> Iterator[Tuple[str, GraphHop]]:
    """Yield ``(next_entity, hop)`` for every valid one-hop relational expansion.

    Mirrors the skip rules used by :func:`surf_rag.graph.graph_beam_paths.enumerate_global_frontier_paths`
    so enumeration backends stay consistent.
    """
    d = diag if diag is not None else GraphPathEnumerationDiagnostics()

    # Explore outgoing edges
    for neighbour in graph.successors(current_node):
        if neighbour in visited:
            d.skip_outgoing_visited += 1
            continue

        if graph.degree(neighbour) > MAX_DEGREE:
            d.skip_outgoing_degree += 1
            continue
        if count_appearances(graph, neighbour) > MAX_APPEARANCES:
            d.skip_outgoing_appearances += 1
            continue

        outgoing_edge = graph[current_node][neighbour]
        if outgoing_edge.get("kind") != "rel":
            d.skip_outgoing_not_rel += 1
            continue

        preds = set(outgoing_edge["labels"])
        if preds == {"instance_of"}:
            d.skip_outgoing_instance_of_only += 1
            continue

        for pred in outgoing_edge["labels"]:
            new_hop = GraphHop(source=current_node, relation=pred, target=neighbour)
            yield neighbour, new_hop

    # Explore incoming edges (reverse hops)
    if bidirectional:
        for prev_node in graph.predecessors(current_node):
            if prev_node in visited:
                d.skip_incoming_visited += 1
                continue

            if graph.degree(prev_node) > MAX_DEGREE:
                d.skip_incoming_degree += 1
                continue
            if count_appearances(graph, prev_node) > MAX_APPEARANCES:
                d.skip_incoming_appearances += 1
                continue

            incoming_edge = graph[prev_node][current_node]
            if incoming_edge.get("kind") != "rel":
                d.skip_incoming_not_rel += 1
                continue

            preds = set(incoming_edge["labels"])
            if preds == {"instance_of"}:
                d.skip_incoming_instance_of_only += 1
                continue

            for pred in incoming_edge["labels"]:
                new_hop = GraphHop(
                    source=current_node,
                    relation=pred,
                    target=prev_node,
                    is_reverse=True,
                )
                yield prev_node, new_hop


def relation_labels_from_edge(data: dict) -> list[str]:
    labels = data.get("labels")
    if labels:
        return sorted(labels)
    label = data.get("label")
    if label:
        return [label]
    return []
