"""Path grounding for canonical graph retrieval diagnostics.

Maps each relational hop on a candidate path to chunk-level evidence using corpus
provenance on relation edges (``chunk_ids_by_label``). Used for grounded evidence
bundles and explanation traces only; final chunk ranking uses heterogeneous PPR
masses on chunk nodes, not these hop support scores.
"""

from dataclasses import dataclass
from typing import Optional

from surf_rag.graph.graph_types import GraphHop, GraphPath, GroundedHop, EvidenceBundle


def stored_edge_endpoints(hop: GraphHop):
    if getattr(hop, "is_reverse", False):
        return hop.target, hop.source
    return hop.source, hop.target


def chunk_ids_for_entity(graph, entity_node):
    if entity_node not in graph:
        return set()

    chunks = set()

    for neighbour, edge in graph[entity_node].items():
        if edge["kind"] != "appears_in":
            continue

        chunks.add(neighbour[2:])

    return chunks


def candidate_chunk_ids_for_hop(graph, hop: GraphHop):
    chunks = set()

    u, v = stored_edge_endpoints(hop)
    if not graph.has_edge(u, v):
        return None
    edge = graph[u][v]

    for chunk_id in edge["chunk_ids_by_label"].get(hop.relation, []):
        chunks.add(chunk_id)

    # No direct support — fall back to entity co-occurrence in chunks
    if not chunks:
        chunks_source = {
            neighbour[2:]
            for neighbour, data in graph[hop.source].items()
            if data.get("kind") == "appears_in"
        }

        chunks_target = {
            neighbour[2:]
            for neighbour, data in graph[hop.target].items()
            if data.get("kind") == "appears_in"
        }

        intersection = chunks_source & chunks_target
        if intersection:
            chunks.update(intersection)
        else:
            chunks.update(chunks_source | chunks_target)

    return chunks


def hop_support_score(graph, hop: GraphHop, chunk: str):
    """Score how well a chunk supports a relation hop."""
    u, v = stored_edge_endpoints(hop)
    if not graph.has_edge(u, v):
        return 0.0
    edge = graph[u][v]
    if chunk in edge["chunk_ids_by_label"][hop.relation]:
        return 1.0
    return 0.0


@dataclass(frozen=True)
class GroundPathReport:
    """Outcome of grounding one path; includes failure detail for diagnostics."""

    bundle: Optional[EvidenceBundle]
    failure_kind: Optional[str] = None  # missing_rel_edge | weak_hop_support
    failure_hop_index: Optional[int] = None
    best_support_at_failure: Optional[float] = None


def ground_path_report(
    graph,
    path: GraphPath,
    *,
    support_threshold: float = 0.5,
) -> GroundPathReport:
    """Ground each hop to chunk IDs for explanations (orthogonal to PPR chunk scores).

    Canonical retrieval ranks chunks by stationary mass on ``C:*`` after
    heterogeneous PPR. This function only builds human-readable / diagnostic
    evidence linkage: it does **not** define final retrieval scores.
    """
    grounded_hops = []
    supporting_chunk_ids = []

    for hop_index, hop in enumerate(path.hops):
        candidate_chunks = candidate_chunk_ids_for_hop(graph, hop)
        if candidate_chunks is None:
            return GroundPathReport(
                bundle=None,
                failure_kind="missing_rel_edge",
                failure_hop_index=hop_index,
                best_support_at_failure=None,
            )
        best_score, best_chunk = max(
            (
                (hop_support_score(graph, hop, chunk), chunk)
                for chunk in candidate_chunks
            ),
            key=lambda item: item[0],
        )
        if best_score < support_threshold:
            return GroundPathReport(
                bundle=None,
                failure_kind="weak_hop_support",
                failure_hop_index=hop_index,
                best_support_at_failure=float(best_score),
            )
        grounded_hop = GroundedHop(
            hop=hop, chunk_id=best_chunk, support_score=best_score
        )
        grounded_hops.append(grounded_hop)
        supporting_chunk_ids.append(best_chunk)

    return GroundPathReport(
        bundle=EvidenceBundle(
            path=path,
            grounded_hops=grounded_hops,
            supporting_chunk_ids=supporting_chunk_ids,
        ),
        failure_kind=None,
        failure_hop_index=None,
        best_support_at_failure=None,
    )


def ground_path(
    graph, path: GraphPath, support_threshold: float = 0.5
) -> EvidenceBundle:
    """Convert a graph path into a grounded evidence bundle."""
    rep = ground_path_report(graph, path, support_threshold=support_threshold)
    return rep.bundle
