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

    # No direct support found - might be rudundent if we don't have lexical scoring
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


def ground_path(graph, path: GraphPath) -> EvidenceBundle:
    """Convert a graph path into a grounded evidence bundle"""
    grounded_hops = []
    supporting_chunk_ids = []

    for hop in path.hops:
        # Find best chunk score
        candidate_chunks = candidate_chunk_ids_for_hop(graph, hop)
        if candidate_chunks is None:
            return None
        best_score, best_chunk = max(
            (
                (hop_support_score(graph, hop, chunk), chunk)
                for chunk in candidate_chunks
            ),
            key=lambda item: item[0],
        )
        # If best chunk score is lower than the threshold -> Path fails
        if best_score < 0.5:
            return None
        else:
            grounded_hop = GroundedHop(
                hop=hop, chunk_id=best_chunk, support_score=best_score
            )
            grounded_hops.append(grounded_hop)
            supporting_chunk_ids.append(best_chunk)

    return EvidenceBundle(
        path=path,
        grounded_hops=grounded_hops,
        supporting_chunk_ids=supporting_chunk_ids,
    )
