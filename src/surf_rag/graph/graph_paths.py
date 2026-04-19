from typing import List

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


def enumerate_candidate_paths(
    graph,
    start_nodes: set[str],
    max_hops: int,
    bidirectional: bool = True,
    max_paths_per_start: int = 50,
) -> list[GraphPath]:
    paths = []
    for node in start_nodes:
        if node not in graph:
            continue

        start_node_paths = []

        def dfs(current_node: str, current_hops: List[GraphHop], visited: set):
            if len(start_node_paths) >= max_paths_per_start:
                return

            if current_hops:
                start_node_paths.append(
                    GraphPath(start_node=node, hops=tuple(current_hops))
                )

            # Stop at max_hops
            if len(current_hops) >= max_hops:
                return

            # Explore outgoing edges
            for neighbour in graph.successors(current_node):
                if neighbour in visited:
                    continue

                if graph.degree(neighbour) > MAX_DEGREE:
                    continue
                if count_appearances(graph, neighbour) > MAX_APPEARANCES:
                    continue

                outgoing_edge = graph[current_node][neighbour]
                if outgoing_edge.get("kind") != "rel":
                    continue

                preds = set()
                labels = outgoing_edge["labels"]
                for pred in labels:
                    preds.add(pred)
                if preds == {"instance_of"}:
                    continue

                for pred in labels:
                    new_hop = GraphHop(
                        source=current_node, relation=pred, target=neighbour
                    )
                    dfs(neighbour, current_hops + [new_hop], visited | {neighbour})

            # Explore incoming edges
            if bidirectional:
                for prev_node in graph.predecessors(current_node):
                    if prev_node in visited:
                        continue

                    if graph.degree(prev_node) > MAX_DEGREE:
                        continue
                    if count_appearances(graph, neighbour) > MAX_APPEARANCES:
                        continue

                    incoming_edge = graph[prev_node][current_node]
                    if incoming_edge.get("kind") != "rel":
                        continue

                    preds = set()
                    labels = incoming_edge["labels"]
                    for pred in labels:
                        preds.add(pred)
                    if preds == {"instance_of"}:
                        continue

                    for pred in labels:
                        new_hop = GraphHop(
                            source=current_node,
                            relation=pred,
                            target=prev_node,
                            is_reverse=True,
                        )
                        dfs(
                            prev_node,
                            current_hops + [new_hop],
                            visited | {prev_node},
                        )

        dfs(node, [], {node})
        paths.extend(start_node_paths)

    return paths


def relation_labels_from_edge(data: dict) -> list[str]:
    labels = data.get("labels")
    if labels:
        return sorted(labels)
    label = data.get("label")
    if label:
        return [label]
    return []
