"""Unit tests for canonical GraphRAG: semantic seeds, heterogeneous PPR, frontier."""

from __future__ import annotations

import numpy as np

from surf_rag.core.build_graph import build_graph
from surf_rag.core.scoring_config import ScoringConfig
from surf_rag.entity_matching.types import PhraseSource, SeedCandidate
from surf_rag.graph.graph_beam_paths import enumerate_global_frontier_paths
from surf_rag.graph.graph_paths import relation_labels_from_edge
from surf_rag.graph.graph_scoring import (
    canonical_ppr_rank_chunks,
    _build_heterogeneous_transition_matrix,
)
from surf_rag.graph.graph_seeds import compute_restart_distribution_canonical
from surf_rag.graph.graph_specificity import (
    label_shape_score,
    normalize_restart_masses,
    seed_restart_mass_for_nodes,
)
from surf_rag.graph.graph_types import GraphPath


class _DictEmbedder:
    """Deterministic embed_query(text) → fixed vector for tests."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self._mapping = mapping

    def embed_query(self, text: str) -> np.ndarray:
        return self._mapping[text]


def test_label_shape_score_penalizes_digit_heavy_labels():
    assert label_shape_score("chapter 7 title 42") < label_shape_score("john smith")


def test_restart_masses_normalize_to_simplex():
    g = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "lonely_entity"}],
                    "relations": [],
                },
            }
        ]
    )
    raw = seed_restart_mass_for_nodes(g, {"E:lonely_entity"}, ["lonely_entity"], [])
    norm = normalize_restart_masses(raw)
    assert abs(sum(norm.values()) - 1.0) < 1e-6


def test_heterogeneous_transition_matrix_rows_sum_to_one():
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "a"}, {"norm": "b"}],
                    "relations": [{"subj_norm": "a", "pred": "links", "obj_norm": "b"}],
                },
            }
        ]
    )
    ordered_nodes = ["E:a", "E:b", "C:c1"]
    idx_map = {n: i for i, n in enumerate(ordered_nodes)}
    cfg = ScoringConfig(
        graph_entity_chunk_edge_weight=0.5,
        graph_transition_mode="uniform",
    )
    p_mat = _build_heterogeneous_transition_matrix(graph, ordered_nodes, idx_map, cfg)
    assert np.allclose(p_mat.sum(axis=1), 1.0)


def test_heterogeneous_entity_only_scope_is_row_stochastic():
    """Entity subgraph scope (no chunk nodes listed) still yields stochastic rows."""
    graph = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "a"}, {"norm": "b"}],
                    "relations": [
                        {"subj_norm": "a", "pred": "relates", "obj_norm": "b"}
                    ],
                },
            }
        ]
    )
    ordered_nodes = ["E:a", "E:b"]
    idx_map = {n: i for i, n in enumerate(ordered_nodes)}
    cfg = ScoringConfig(graph_transition_mode="uniform")
    p_mat = _build_heterogeneous_transition_matrix(graph, ordered_nodes, idx_map, cfg)
    assert np.allclose(p_mat.sum(axis=1), 1.0)


def test_canonical_ppr_prefers_chunk_linked_to_high_mass_entity():
    chunks = [
        {
            "chunk_id": "high",
            "metadata": {
                "entities": [{"norm": "hub"}, {"norm": "leaf"}],
                "relations": [
                    {"subj_norm": "hub", "pred": "links", "obj_norm": "leaf"}
                ],
            },
        },
        {
            "chunk_id": "noise",
            "metadata": {
                "entities": [{"norm": "other"}],
                "relations": [],
            },
        },
    ]
    graph = build_graph(chunks)
    paths = [
        GraphPath(
            start_node="E:hub",
            hops=(),
        )
    ]
    cfg = ScoringConfig()
    scores, _pi_dict, extra = canonical_ppr_rank_chunks(
        graph,
        paths,
        {"E:hub"},
        {"E:hub": 1.0},
        config=cfg,
    )
    assert scores.get("high", 0.0) >= scores.get("noise", 0.0)
    assert extra["mode"] == "canonical_heterogeneous_ppr"
    assert extra["ppr"]["transition"] == "heterogeneous_entity_chunk"
    assert extra["ppr"]["iterations"] >= 1
    assert "chunk_ppr_mass" in extra["chunk_scoring"]


def test_relation_labels_helper_stable():
    assert relation_labels_from_edge({"labels": {"x"}, "label": "y"}) == ["x"]


def test_restart_distribution_semantic_softmax_idf_sums_to_one():
    g = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "rare"}, {"norm": "freq"}],
                    "relations": [],
                },
            }
        ]
    )
    seeds = [
        SeedCandidate(
            canonical_norm="rare",
            matched_text="rare",
            start=0,
            end=4,
            span_token_count=1,
            df=1,
            source=PhraseSource.CANONICAL,
            match_key="rare",
            node_id="E:rare",
            graph_present=True,
        ),
        SeedCandidate(
            canonical_norm="freq",
            matched_text="freq",
            start=0,
            end=4,
            span_token_count=1,
            df=500,
            source=PhraseSource.CANONICAL,
            match_key="freq",
            node_id="E:freq",
            graph_present=True,
        ),
    ]
    emb = _DictEmbedder(
        {
            "q": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "rare": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "freq": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        }
    )
    masses, diag = compute_restart_distribution_canonical(
        g,
        "q",
        seeds,
        ["rare", "freq"],
        emb,
        softmax_temperature=1.0,
    )
    assert abs(sum(masses.values()) - 1.0) < 1e-5
    assert masses["E:rare"] > masses["E:freq"]
    assert diag["mode"] == "canonical_semantic_softmax_idf"


def test_low_temperature_near_one_hot():
    g = build_graph(
        [
            {
                "chunk_id": "c1",
                "metadata": {
                    "entities": [{"norm": "win"}, {"norm": "lose"}],
                    "relations": [],
                },
            }
        ]
    )
    seeds = [
        SeedCandidate(
            canonical_norm="win",
            matched_text="win",
            start=0,
            end=3,
            span_token_count=1,
            df=10,
            source=PhraseSource.CANONICAL,
            match_key="win",
            node_id="E:win",
            graph_present=True,
        ),
        SeedCandidate(
            canonical_norm="lose",
            matched_text="lose",
            start=0,
            end=4,
            span_token_count=1,
            df=10,
            source=PhraseSource.CANONICAL,
            match_key="lose",
            node_id="E:lose",
            graph_present=True,
        ),
    ]
    emb = _DictEmbedder(
        {
            "subtle": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "win": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "lose": np.asarray([0.99, 0.1414, 0.0], dtype=np.float32),
        }
    )
    masses, _diag = compute_restart_distribution_canonical(
        g,
        "subtle",
        seeds,
        ["win", "lose"],
        emb,
        softmax_temperature=1e-6,
    )
    assert masses["E:win"] > 0.99


def test_global_frontier_canonical_emits_paths():
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
    paths, diag = enumerate_global_frontier_paths(
        graph=graph,
        seed_weights={"E:a": 0.9, "E:b": 0.1},
        max_hops=1,
        bidirectional=False,
        global_max_paths=20,
        global_max_pops=500,
    )
    assert diag.enumeration_backend == "global_canonical"
    assert len(paths) >= 1


def test_graph_retriever_canonical_returns_chunks():
    from surf_rag.strategies.graph import GraphRetriever

    from tests._graph_test_utils import (
        CorpusStub,
        GraphStoreStub,
        StaticExtractor,
        strategy_embedder,
        toy_chunks,
    )

    graph = build_graph(toy_chunks())
    retriever = GraphRetriever(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        scoring_config=ScoringConfig(),
        top_k=3,
        max_hops=2,
    )
    result = retriever.retrieve("How is A related to B?")
    assert result.status == "OK"
    assert len(result.chunks) > 0
    gd = result.debug_info.get("graph_diagnostics") if result.debug_info else None
    assert gd is not None
    assert gd["retriever_config"]["graph_retrieval_mode"] == "canonical_ppr"
    assert gd["ppr"]["transition"] == "heterogeneous_entity_chunk"
    assert "chunk_projection" in gd
    cs = gd["chunk_projection"].get("chunk_scoring") or {}
    assert "chunk_ppr_mass" in cs
