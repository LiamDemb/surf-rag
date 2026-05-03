"""Tests for stateful, task-conditioned retrieval metrics."""

from __future__ import annotations

import math
from typing import Iterable

import pytest

from surf_rag.evaluation.retrieval_metrics import (
    DEFAULT_NDCG_KS,
    PRIMARY_NDCG_K,
    RankedMetricSuite,
    compute_metric_suite,
    dcg_at_k,
    hit_at_k,
    ideal_dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    score_retrieval_result,
    stateful_relevances,
)
from surf_rag.retrieval.types import RetrievalResult, RetrievedChunk


def _chunk(chunk_id: str, text: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, score=float(score), rank=0, metadata={}
    )


def _ranked(chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    """Return a stable ordering that bypasses RetrievalResult sort rules."""
    return list(chunks)


def test_default_ks_include_primary_k():
    assert PRIMARY_NDCG_K == 10
    assert PRIMARY_NDCG_K in DEFAULT_NDCG_KS
    assert DEFAULT_NDCG_KS == (5, 10, 20)


def test_dcg_at_k_basic():
    rels = [1, 0, 1]
    expected = 1 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    assert dcg_at_k(rels, k=3) == pytest.approx(expected)


def test_dcg_at_k_truncates_to_k():
    rels = [1, 1, 1, 1]
    assert dcg_at_k(rels, k=2) == pytest.approx(1 / math.log2(2) + 1 / math.log2(3))


def test_ideal_dcg_or_mode_is_one_when_gold_present():
    assert ideal_dcg_at_k(["gold"], k=10, dataset_source="nq") == pytest.approx(1.0)


def test_ideal_dcg_or_mode_is_zero_without_gold():
    assert ideal_dcg_at_k([], k=10, dataset_source="nq") == 0.0


def test_ideal_dcg_and_mode_sums_top_m():
    idcg = ideal_dcg_at_k(["a", "b", "c"], k=10, dataset_source="2wiki")
    expected = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert idcg == pytest.approx(expected)


def test_ideal_dcg_and_mode_caps_at_k():
    idcg = ideal_dcg_at_k(["a", "b", "c"], k=1, dataset_source="2wiki")
    assert idcg == pytest.approx(1 / math.log2(2))


def test_nq_or_credits_only_first_matching_chunk():
    """NQ: only the first chunk containing a gold sentence is relevant."""
    chunks = _ranked(
        [
            _chunk("c1", "The capital of France is Paris."),
            _chunk("c2", "Paris is the capital of France."),  # duplicate info
            _chunk("c3", "Totally unrelated content."),
        ]
    )
    gold = ["The capital of France is Paris."]
    rels = stateful_relevances(chunks, gold, dataset_source="nq")
    assert rels == [1, 0, 0]


def test_nq_or_ndcg_is_one_when_gold_at_rank_one():
    chunks = _ranked(
        [
            _chunk("c1", "The capital of France is Paris."),
            _chunk("c2", "Other."),
        ]
    )
    gold = ["The capital of France is Paris."]
    assert ndcg_at_k(chunks, gold, k=10, dataset_source="nq") == pytest.approx(1.0)


def test_nq_or_ndcg_decays_with_rank():
    chunks = _ranked(
        [
            _chunk("c1", "Other."),
            _chunk("c2", "The capital of France is Paris."),
        ]
    )
    gold = ["The capital of France is Paris."]
    # DCG = 1/log2(3); IDCG = 1
    expected = 1.0 / math.log2(3)
    assert ndcg_at_k(chunks, gold, k=10, dataset_source="nq") == pytest.approx(expected)


def test_2wiki_and_credits_each_new_gold_once():
    chunks = _ranked(
        [
            _chunk("c1", "Alpha fact here."),
            _chunk("c2", "Alpha fact again."),  # already credited -> 0
            _chunk("c3", "Beta fact goes."),
            _chunk("c4", "Gamma fact ends."),
        ]
    )
    gold = ["Alpha fact here.", "Beta fact goes.", "Gamma fact ends."]
    rels = stateful_relevances(chunks, gold, dataset_source="2wiki")
    assert rels == [1, 0, 1, 1]


def test_2wiki_and_ndcg_perfect_when_all_gold_at_top():
    golds = ["g1 one", "g2 two", "g3 three"]
    chunks = _ranked(
        [
            _chunk("a", "g1 one"),
            _chunk("b", "g2 two"),
            _chunk("c", "g3 three"),
            _chunk("d", "noise"),
        ]
    )
    assert ndcg_at_k(chunks, golds, k=10, dataset_source="2wiki") == pytest.approx(1.0)


def test_2wiki_and_recall_counts_distinct_credited_sentences():
    golds = ["a fact", "b fact", "c fact"]
    chunks = _ranked(
        [
            _chunk("c1", "a fact"),
            _chunk("c2", "b fact"),
            _chunk("c3", "not relevant"),
        ]
    )
    assert recall_at_k(chunks, golds, k=3, dataset_source="2wiki") == pytest.approx(
        2 / 3
    )


def test_nq_recall_is_one_when_gold_present_else_zero():
    golds = ["target sentence"]
    chunks = _ranked([_chunk("c1", "target sentence")])
    assert recall_at_k(chunks, golds, k=3, dataset_source="nq") == 1.0
    chunks = _ranked([_chunk("c1", "none")])
    assert recall_at_k(chunks, golds, k=3, dataset_source="nq") == 0.0


def test_hit_at_k_returns_binary():
    golds = ["x fact"]
    chunks = _ranked([_chunk("c1", "noise"), _chunk("c2", "x fact")])
    assert hit_at_k(chunks, golds, k=1, dataset_source="nq") == 0.0
    assert hit_at_k(chunks, golds, k=2, dataset_source="nq") == 1.0


def test_unknown_dataset_source_defaults_to_and_behavior():
    golds = ["alpha", "beta"]
    chunks = _ranked(
        [_chunk("c1", "alpha"), _chunk("c2", "alpha"), _chunk("c3", "beta")]
    )
    rels = stateful_relevances(chunks, golds, dataset_source="strange")
    assert rels == [1, 0, 1]


def test_empty_gold_or_chunks_returns_zero():
    assert ndcg_at_k([], ["g"], k=5, dataset_source="nq") == 0.0
    assert ndcg_at_k([_chunk("c", "g")], [], k=5, dataset_source="nq") == 0.0


def test_normalization_handles_case_and_whitespace():
    """Matching uses ``contains_normalized`` (normalize + whitespace-insensitive fallback)."""
    golds = ["The capital of France is Paris."]
    chunks = _ranked(
        [
            _chunk(
                "c1",
                "Article fragment:   THE  CAPITAL of FRANCE is   Paris. More.",
            )
        ]
    )
    assert ndcg_at_k(chunks, golds, k=5, dataset_source="nq") == pytest.approx(1.0)


def test_compact_fallback_matches_hyphen_spacing_variant():
    """Whitespace-insensitive fallback credits gold when only intra-token spacing differs."""
    gold = ["computer- supported cooperative work ( CSCW)."]
    chunks = _ranked(
        [
            _chunk(
                "c1",
                "Irene Greif is an American computer scientist and a founder of the "
                "field of computer-supported cooperative work (CSCW).",
            )
        ]
    )
    assert ndcg_at_k(chunks, gold, k=5, dataset_source="nq") == pytest.approx(1.0)


def test_2wiki_and_mode_uses_compact_fallback_for_both_facts():
    golds = [
        "Alpha fact here.",
        "Beta fact goes.",
    ]
    chunks = _ranked(
        [
            _chunk("c1", "Alpha fact  here."),  # extra space
            _chunk("c2", "Beta  fact goes."),  # extra space
        ]
    )
    assert ndcg_at_k(chunks, golds, k=5, dataset_source="2wiki") == pytest.approx(1.0)


def test_compute_metric_suite_shape():
    golds = ["x"]
    chunks = _ranked([_chunk("c1", "x"), _chunk("c2", "y")])
    suite = compute_metric_suite(chunks, golds, dataset_source="nq", ks=(5, 10, 20))
    assert [m.k for m in suite] == [5, 10, 20]
    assert all(isinstance(m, RankedMetricSuite) for m in suite)
    assert all(0.0 <= m.ndcg <= 1.0 for m in suite)
    # Each suite should round-trip cleanly to JSON.
    json_rows = [m.to_json() for m in suite]
    assert set(json_rows[0].keys()) == {"k", "ndcg", "hit", "recall"}


def test_score_retrieval_result_handles_non_ok_result():
    no_ctx = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="NO_CONTEXT",
        chunks=[],
    )
    suite = score_retrieval_result(no_ctx, ["g"], dataset_source="nq", ks=(5,))
    assert suite[0].ndcg == 0.0
    assert suite[0].hit == 0.0


def test_score_retrieval_result_handles_ok_result():
    res = RetrievalResult(
        query="q",
        retriever_name="Fused",
        status="OK",
        chunks=[_chunk("c1", "target"), _chunk("c2", "other", score=0.5)],
    )
    suite = score_retrieval_result(res, ["target"], dataset_source="nq", ks=(5,))
    assert suite[0].ndcg == pytest.approx(1.0)


def test_score_retrieval_result_returns_5_10_20_with_single_chunk():
    res = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="OK",
        chunks=[_chunk("c1", "target", score=1.0)],
    )
    suite = score_retrieval_result(
        res, ["target"], dataset_source="nq", ks=DEFAULT_NDCG_KS
    )
    assert [s.k for s in suite] == [5, 10, 20]
    assert all(0.0 <= s.ndcg <= 1.0 for s in suite)


def test_non_ok_results_emit_zeroed_5_10_20():
    res = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="ERROR",
        chunks=[],
        error="x",
    )
    suite = score_retrieval_result(
        res, ["target"], dataset_source="nq", ks=DEFAULT_NDCG_KS
    )
    assert [s.k for s in suite] == [5, 10, 20]
    assert all(s.ndcg == 0.0 and s.hit == 0.0 and s.recall == 0.0 for s in suite)
