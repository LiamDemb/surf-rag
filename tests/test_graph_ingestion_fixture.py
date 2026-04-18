"""Golden fixture tests for the updated graph ingestion schema."""

from __future__ import annotations

from surf_rag.core.build_graph import build_graph



def test_build_graph_creates_expected_nodes_and_edges():
    chunks = [
        {
            "chunk_id": "c100",
            "metadata": {
                "entities": [
                    {"norm": "a", "type": "ORG"},
                    {"norm": "b", "type": "ORG"},
                ],
                "relations": [
                    {"subj_norm": "a", "pred": "caused", "obj_norm": "b"},
                ],
            },
        },
        {
            "chunk_id": "c200",
            "metadata": {
                "entities": [{"norm": "c", "type": "ORG"}],
                "relations": [],
            },
        },
    ]

    graph = build_graph(chunks)

    assert graph.nodes["E:a"]["kind"] == "entity"
    assert graph.nodes["E:b"]["kind"] == "entity"
    assert graph.nodes["C:c100"]["kind"] == "chunk"
    assert graph.nodes["C:c200"]["kind"] == "chunk"

    edge = graph["E:a"]["E:b"]
    assert edge["kind"] == "rel"
    assert edge["label"] == "caused"
    assert set(edge["labels"]) == {"caused"}
    assert set(edge["chunk_ids_by_label"]["caused"]) == {"c100"}
    assert edge["support_count_by_label"]["caused"] == 1

    assert graph["E:a"]["C:c100"]["kind"] == "appears_in"
    assert graph["E:b"]["C:c100"]["kind"] == "appears_in"



def test_build_graph_accumulates_relation_provenance_across_chunks():
    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [{"norm": "a"}, {"norm": "b"}],
                "relations": [{"subj_norm": "a", "pred": "caused", "obj_norm": "b"}],
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "entities": [{"norm": "a"}, {"norm": "b"}],
                "relations": [{"subj_norm": "a", "pred": "caused", "obj_norm": "b"}],
            },
        },
    ]

    graph = build_graph(chunks)
    edge = graph["E:a"]["E:b"]

    assert set(edge["labels"]) == {"caused"}
    assert set(edge["chunk_ids_by_label"]["caused"]) == {"c1", "c2"}
    assert edge["support_count_by_label"]["caused"] == 2



def test_build_graph_preserves_multiple_predicates_between_same_nodes():
    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [{"norm": "film"}, {"norm": "person"}],
                "relations": [
                    {"subj_norm": "film", "pred": "directed_by", "obj_norm": "person"}
                ],
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "entities": [{"norm": "film"}, {"norm": "person"}],
                "relations": [
                    {"subj_norm": "film", "pred": "written_by", "obj_norm": "person"}
                ],
            },
        },
    ]

    graph = build_graph(chunks)
    edge = graph["E:film"]["E:person"]

    assert edge["kind"] == "rel"
    assert set(edge["labels"]) == {"directed_by", "written_by"}
    assert set(edge["chunk_ids_by_label"]["directed_by"]) == {"c1"}
    assert set(edge["chunk_ids_by_label"]["written_by"]) == {"c2"}
