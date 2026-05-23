from __future__ import annotations

from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    oracle_bin_value,
)


def test_oracle_bin_value_ndcg_at_k() -> None:
    bin_score = {
        "diagnostic_ndcg": {"5": 0.42, "10": 0.5},
        "oracle_objective_value": 0.4,
    }
    assert oracle_bin_value(bin_score, metric="stateful_ndcg", k=5) == 0.42


def test_e2e_retrieval_value() -> None:
    row = {
        "retrieval_before_ce": {
            "retrieval": {"5": {"ndcg": 0.8, "hit": 1.0, "recall": 0.5}}
        }
    }
    assert e2e_retrieval_value(row, metric="ndcg", k=5) == 0.8
