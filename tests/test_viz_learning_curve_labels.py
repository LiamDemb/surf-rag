from __future__ import annotations

import json
from pathlib import Path

from surf_rag.viz.learning_curve_labels import (
    format_learning_curve_ylabel,
    format_metric_at_k,
    parse_router_id_objective,
    resolve_oracle_objective,
    resolve_training_loss_id,
)


def test_format_metric_at_k() -> None:
    assert format_metric_at_k("stateful_ndcg", 10) == "NDCG@10"
    assert format_metric_at_k("recall", 10) == "Recall@10"
    assert format_metric_at_k("hit", 5) == "Hit@5"


def test_format_learning_curve_ylabel_regression() -> None:
    assert (
        format_learning_curve_ylabel(
            "regret",
            task_type="regression",
            oracle_metric="stateful_ndcg",
            oracle_metric_k=10,
        )
        == "NDCG@10 Regret Loss"
    )


def test_format_learning_curve_ylabel_classification() -> None:
    assert (
        format_learning_curve_ylabel(
            "cross_entropy",
            task_type="classification",
            oracle_metric="stateful_ndcg",
            oracle_metric_k=10,
        )
        == "Cross Entropy Loss"
    )


def test_parse_router_id_objective() -> None:
    assert parse_router_id_objective("ndcg@10") == ("stateful_ndcg", 10)
    assert parse_router_id_objective("recall@10") == ("recall", 10)
    assert parse_router_id_objective("main-v01") is None


def test_resolve_oracle_objective_prefers_summary(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"oracle_metric": "recall", "oracle_metric_k": 10}),
        encoding="utf-8",
    )
    metric, k = resolve_oracle_objective(
        router_id="ndcg@10",
        config_metric="stateful_ndcg",
        config_k=10,
        oracle_summary_path=summary,
    )
    assert metric == "recall"
    assert k == 10


def test_resolve_oracle_objective_falls_back_to_router_id() -> None:
    metric, k = resolve_oracle_objective(
        router_id="recall@10",
        config_metric="stateful_ndcg",
        config_k=5,
        oracle_summary_path=None,
    )
    assert metric == "recall"
    assert k == 10


def test_resolve_training_loss_id_prefers_history() -> None:
    loss = resolve_training_loss_id(
        config_loss="regret",
        history_payload={"loss_effective": "hinge_squared_regret"},
        manifest_path=None,
    )
    assert loss == "hinge_squared_regret"
