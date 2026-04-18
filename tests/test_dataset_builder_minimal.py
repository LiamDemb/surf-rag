"""Minimal tests for dataset builder with stubbed embedder, probe, and qfeat."""

import pytest

pytest.skip(
    "Router dataset builder (build_router_dataset_rows) is not part of this package yet.",
    allow_module_level=True,
)


def _stub_embedder():
    """Returns an embedder that returns a fixed 4-dim vector."""
    class Stub:
        def embed_query(self, text: str):
            return [0.1, 0.2, 0.3, 0.4]
    return Stub()


def _stub_probe():
    """Returns a probe that returns fixed signals."""
    class Stub:
        def run(self, query: str):
            return {
                "probe_scores": [0.9, 0.8, 0.7],
                "probe_max_score": 0.9,
                "probe_min_score": 0.7,
                "probe_score_sd": 0.1,
                "probe_skewness": 0.0,
                "probe_semantic_dispersion": 0.2,
                "probe_entropy": 1.0,
                "probe_gini": 0.05,
                "probe_mass_k_80": 2.0,
                "probe_mass_k_90": 2.0,
                "probe_mass_k_95": 3.0,
                "probe_top1_top2_gap": 0.1,
                "probe_top1_top2_ratio": 1.05,
            }
    return Stub()


def _stub_qfeat():
    """Returns a qfeat callable that returns fixed features."""
    def fn(query: str):
        return {
            "entity_count": 1,
            "syntactic_depth": 3,
            "query_length_tokens": 5,
            "relational_keyword_flag": 0,
        }
    return fn


@pytest.fixture
def oracle_raw_scores_fixture():
    """Tiny in-memory oracle_raw_scores fixture."""
    return [
        {
            "question_id": "q1",
            "question": "What is the capital of France?",
            "gold_answers": ["Paris"],
            "pred_dense": "Paris",
            "pred_graph": "Paris",
            "dataset_source": "nq",
        },
        {
            "question_id": "q2",
            "question": "Compare X and Y.",
            "gold_answers": ["A", "B"],
            "pred_dense": "A",
            "pred_graph": "B",
            "dataset_source": "2wiki",
        },
    ]


def test_output_schema(oracle_raw_scores_fixture):
    """Verify output row has required schema."""
    rows = list(build_router_dataset_rows(
        oracle_raw_scores_fixture,
        embedder=_stub_embedder(),
        probe=_stub_probe(),
        compute_qfeat=_stub_qfeat(),
        delta=0.05,
    ))
    assert len(rows) == 2

    required = {
        "question_id", "question", "split", "dataset_source",
        "question_embedding", "gold_answers", "gold_label",
        "f1_dense", "f1_graph", "em_dense", "em_graph",
        "pred_dense", "pred_graph",
        "probe_scores", "probe_max_score", "probe_min_score",
        "probe_score_sd", "probe_skewness", "probe_semantic_dispersion",
        "probe_entropy", "probe_gini", "probe_mass_k_80", "probe_mass_k_90",
        "probe_mass_k_95", "probe_top1_top2_gap", "probe_top1_top2_ratio",
        "entity_count", "syntactic_depth", "query_length_tokens",
        "relational_keyword_flag",
    }
    for row in rows:
        for key in required:
            assert key in row, f"Missing key: {key}"


def test_dense_graph_pairing_preserved(oracle_raw_scores_fixture):
    """Verify Dense/Graph preds and scores are preserved."""
    rows = list(build_router_dataset_rows(
        oracle_raw_scores_fixture,
        embedder=_stub_embedder(),
        probe=_stub_probe(),
        compute_qfeat=_stub_qfeat(),
        delta=0.05,
    ))
    r1, r2 = rows[0], rows[1]
    # q1: both predict Paris -> gold_label Dense (tie)
    assert r1["pred_dense"] == "Paris"
    assert r1["pred_graph"] == "Paris"
    assert r1["gold_label"] == "Dense"
    assert r1["f1_dense"] == 1.0
    assert r1["f1_graph"] == 1.0

    # q2: pred_dense=A, pred_graph=B; golds A,B. Both get F1 1.0 -> tie -> Dense
    assert r2["pred_dense"] == "A"
    assert r2["pred_graph"] == "B"
    assert r2["gold_label"] == "Dense"


def test_probe_and_qfeat_in_output(oracle_raw_scores_fixture):
    """Verify probe and Q-feat values appear in output."""
    rows = list(build_router_dataset_rows(
        oracle_raw_scores_fixture,
        embedder=_stub_embedder(),
        probe=_stub_probe(),
        compute_qfeat=_stub_qfeat(),
        delta=0.05,
    ))
    row = rows[0]
    assert row["probe_scores"] == [0.9, 0.8, 0.7]
    assert row["probe_max_score"] == 0.9
    assert row["entity_count"] == 1
    assert row["syntactic_depth"] == 3
    assert row["query_length_tokens"] == 5
