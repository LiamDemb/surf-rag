from __future__ import annotations

from surf_rag.core.build_graph import build_graph
from surf_rag.graph.graph_grounding import (
    candidate_chunk_ids_for_hop,
    chunk_ids_for_entity,
    ground_path,
)
from surf_rag.graph.graph_types import GraphHop, GraphPath


def test_chunk_ids_for_entity_reads_appears_in_provenance():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {"entities": [{"norm": "a"}], "relations": []},
            },
            {
                "chunk_id": "c2",
                "metadata": {"entities": [{"norm": "a"}], "relations": []},
            },
        ]
    )

    assert chunk_ids_for_entity(graph, "E:a") == {"c1", "c2"}


def test_candidate_chunk_ids_for_hop_prefers_exact_relation_provenance():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "film"}, {"norm": "person"}],
                    "relations": [
                        {
                            "subj_norm": "film",
                            "pred": "directed_by",
                            "obj_norm": "person",
                        }
                    ],
                },
            },
            {
                "chunk_id": "c2",
                "metadata": {
                    "entities": [{"norm": "film"}, {"norm": "person"}],
                    "relations": [
                        {
                            "subj_norm": "film",
                            "pred": "written_by",
                            "obj_norm": "person",
                        }
                    ],
                },
            },
        ]
    )

    hop = GraphHop(source="E:film", relation="directed_by", target="E:person")
    assert candidate_chunk_ids_for_hop(graph, hop) == {"c1"}


def test_candidate_chunk_ids_for_hop_handles_reverse_hop_against_stored_edge():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "film"}, {"norm": "director"}],
                    "relations": [
                        {
                            "subj_norm": "film",
                            "pred": "directed_by",
                            "obj_norm": "director",
                        }
                    ],
                },
            }
        ]
    )

    hop = GraphHop(
        source="E:director", relation="directed_by", target="E:film", is_reverse=True
    )
    assert candidate_chunk_ids_for_hop(graph, hop) == {"c1"}


def test_ground_path_returns_bundle_when_all_hops_are_grounded():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "film"}, {"norm": "director"}],
                    "relations": [
                        {
                            "subj_norm": "film",
                            "pred": "directed_by",
                            "obj_norm": "director",
                        }
                    ],
                },
            },
            {
                "chunk_id": "c2",
                "metadata": {
                    "entities": [{"norm": "director"}, {"norm": "date_1948"}],
                    "relations": [
                        {
                            "subj_norm": "director",
                            "pred": "died_on",
                            "obj_norm": "date_1948",
                        }
                    ],
                },
            },
        ]
    )
    path = GraphPath(
        start_node="E:film",
        hops=(
            GraphHop(source="E:film", relation="directed_by", target="E:director"),
            GraphHop(source="E:director", relation="died_on", target="E:date_1948"),
        ),
    )

    bundle = ground_path(path=path, graph=graph)

    assert bundle is not None
    assert bundle.path == path
    assert len(bundle.grounded_hops) == 2
    assert set(bundle.supporting_chunk_ids) == {"c1", "c2"}


def test_ground_path_returns_none_if_any_hop_cannot_be_grounded():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "film"}, {"norm": "director"}],
                    "relations": [
                        {
                            "subj_norm": "film",
                            "pred": "directed_by",
                            "obj_norm": "director",
                        }
                    ],
                },
            }
        ]
    )
    path = GraphPath(
        start_node="E:film",
        hops=(
            GraphHop(source="E:film", relation="directed_by", target="E:director"),
            GraphHop(source="E:director", relation="died_on", target="E:date_1948"),
        ),
    )

    bundle = ground_path(path=path, graph=graph)
    assert bundle is None
