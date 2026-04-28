from __future__ import annotations

from surf_rag.core.build_graph import build_graph
from surf_rag.graph.graph_paths import (
    enumerate_candidate_paths,
    relation_labels_from_edge,
)
from surf_rag.graph.graph_types import GraphHop


def _sig(path) -> tuple:
    return tuple(
        (hop.source, hop.relation, bool(getattr(hop, "is_reverse", False)), hop.target)
        for hop in path.hops
    )


def test_relation_labels_from_edge_uses_labels_then_falls_back_to_label():
    assert sorted(relation_labels_from_edge({"labels": {"a", "b"}, "label": "a"})) == [
        "a",
        "b",
    ]
    assert relation_labels_from_edge({"label": "only_one"}) == ["only_one"]
    assert relation_labels_from_edge({}) == []


def test_enumerate_candidate_paths_skips_appears_in_and_returns_rel_paths_only():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "a"}, {"norm": "b"}],
                    "relations": [
                        {"subj_norm": "a", "pred": "causes", "obj_norm": "b"}
                    ],
                },
            }
        ]
    )

    paths, diag = enumerate_candidate_paths(
        graph=graph,
        start_nodes={"E:a"},
        max_hops=1,
        bidirectional=False,
        max_paths_per_start=50,
    )

    assert {_sig(p) for p in paths} == {
        (("E:a", "causes", False, "E:b"),),
    }
    assert diag.paths_emitted == len(paths)


def test_enumerate_candidate_paths_respects_bidirectionality():
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

    paths, _diag = enumerate_candidate_paths(
        graph=graph,
        start_nodes={"E:director"},
        max_hops=1,
        bidirectional=True,
        max_paths_per_start=50,
    )

    assert {_sig(p) for p in paths} == {
        (("E:director", "directed_by", True, "E:film"),),
    }


def test_enumerate_candidate_paths_expands_multiple_labels_on_same_edge():
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
                        },
                        {
                            "subj_norm": "film",
                            "pred": "written_by",
                            "obj_norm": "person",
                        },
                    ],
                },
            }
        ]
    )

    paths, _diag = enumerate_candidate_paths(
        graph=graph,
        start_nodes={"E:film"},
        max_hops=1,
        bidirectional=False,
        max_paths_per_start=50,
    )

    assert {_sig(p) for p in paths} == {
        (("E:film", "directed_by", False, "E:person"),),
        (("E:film", "written_by", False, "E:person"),),
    }


def test_enumerate_candidate_paths_respects_max_hops_and_deduplicates():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "a"}, {"norm": "b"}],
                    "relations": [{"subj_norm": "a", "pred": "r1", "obj_norm": "b"}],
                },
            },
            {
                "chunk_id": "c2",
                "metadata": {
                    "entities": [{"norm": "b"}, {"norm": "c"}],
                    "relations": [{"subj_norm": "b", "pred": "r2", "obj_norm": "c"}],
                },
            },
        ]
    )

    paths_1, _d1 = enumerate_candidate_paths(
        graph=graph,
        start_nodes={"E:a"},
        max_hops=1,
        bidirectional=False,
        max_paths_per_start=50,
    )
    assert {_sig(p) for p in paths_1} == {
        (("E:a", "r1", False, "E:b"),),
    }

    paths_2, _d2 = enumerate_candidate_paths(
        graph=graph,
        start_nodes={"E:a"},
        max_hops=2,
        bidirectional=False,
        max_paths_per_start=50,
    )
    assert {_sig(p) for p in paths_2} == {
        (("E:a", "r1", False, "E:b"),),
        (("E:a", "r1", False, "E:b"), ("E:b", "r2", False, "E:c")),
    }
