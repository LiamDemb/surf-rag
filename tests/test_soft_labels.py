"""Tests for deterministic oracle-curve label materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surf_rag.router.soft_labels import (
    materialize_router_labels,
    oracle_label_from_curve,
)


def test_oracle_label_from_curve_basics() -> None:
    curve = [0.1, 0.3, 0.2]
    grid = [0.0, 0.5, 1.0]
    rec = oracle_label_from_curve(
        oracle_curve=curve, weight_grid=grid, dataset_source="nq"
    )
    assert rec["oracle_best_score"] == pytest.approx(0.3)
    assert rec["dataset_source"] == "nq"
    assert rec["is_valid_for_router_training"] is True


def test_oracle_label_zero_best_marked_invalid() -> None:
    rec = oracle_label_from_curve(
        oracle_curve=[0.0, 0.0, 0.0],
        weight_grid=[0.0, 0.5, 1.0],
        dataset_source="2wiki",
    )
    assert rec["is_valid_for_router_training"] is False


def test_oracle_label_rejects_bad_lengths() -> None:
    with pytest.raises(ValueError):
        oracle_label_from_curve(
            oracle_curve=[0.1, 0.2], weight_grid=[0.0, 0.5, 1.0], dataset_source="nq"
        )


def _oracle_row(qid: str, scores, grid) -> dict:
    return {
        "question_id": qid,
        "dataset_source": "nq",
        "weight_grid": list(grid),
        "scores": [
            {
                "dense_weight": float(w),
                "graph_weight": 1.0 - float(w),
                "ndcg_primary": float(s),
            }
            for w, s in zip(grid, scores)
        ],
    }


def test_materialize_router_labels_writes_expected_jsonl(tmp_path: Path):
    grid = [0.0, 0.5, 1.0]
    rows = [
        _oracle_row("q1", [0.0, 0.2, 1.0], grid),
        _oracle_row("q2", [1.0, 0.5, 0.0], grid),
    ]
    out = tmp_path / "labels" / "router_labels.jsonl"
    n = materialize_router_labels(rows, output_path=out)
    assert n == 2

    records = [
        json.loads(line) for line in out.read_text().splitlines() if line.strip()
    ]
    assert [r["question_id"] for r in records] == ["q1", "q2"]
    for rec in records:
        assert len(rec["oracle_curve"]) == 3
        assert rec["oracle_best_score"] == pytest.approx(max(rec["oracle_curve"]))
        assert "oracle_best_weight" not in rec
        assert "oracle_best_index" not in rec


def test_materialize_router_labels_uses_recall_objective(tmp_path: Path) -> None:
    grid = [0.0, 0.5, 1.0]
    rows = [
        {
            "question_id": "q1",
            "dataset_source": "nq",
            "weight_grid": grid,
            "scores": [
                {
                    "dense_weight": 0.0,
                    "graph_weight": 1.0,
                    "ndcg_primary": 0.9,
                    "diagnostic_recall": {"20": 0.1},
                },
                {
                    "dense_weight": 0.5,
                    "graph_weight": 0.5,
                    "ndcg_primary": 0.5,
                    "diagnostic_recall": {"20": 0.8},
                },
                {
                    "dense_weight": 1.0,
                    "graph_weight": 0.0,
                    "ndcg_primary": 0.2,
                    "diagnostic_recall": {"20": 1.0},
                },
            ],
        }
    ]
    out = tmp_path / "labels" / "router_labels.jsonl"
    n = materialize_router_labels(
        rows,
        output_path=out,
        oracle_metric="recall",
        oracle_metric_k=20,
    )
    assert n == 1
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["oracle_curve"] == pytest.approx([0.1, 0.8, 1.0])
    assert rec["oracle_best_score"] == pytest.approx(1.0)


def test_materialize_router_labels_prefers_oracle_objective_value(
    tmp_path: Path,
) -> None:
    grid = [0.0, 1.0]
    rows = [
        {
            "question_id": "q1",
            "dataset_source": "nq",
            "weight_grid": grid,
            "scores": [
                {
                    "dense_weight": 0.0,
                    "graph_weight": 1.0,
                    "ndcg_primary": 0.9,
                    "oracle_objective_value": 0.2,
                },
                {
                    "dense_weight": 1.0,
                    "graph_weight": 0.0,
                    "ndcg_primary": 0.1,
                    "oracle_objective_value": 0.7,
                },
            ],
        }
    ]
    out = tmp_path / "labels" / "router_labels.jsonl"
    n = materialize_router_labels(
        rows,
        output_path=out,
        oracle_metric="recall",
        oracle_metric_k=20,
    )
    assert n == 1
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["oracle_curve"] == pytest.approx([0.2, 0.7])


def test_materialize_router_labels_binary_tie_break_dense_on_equal(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "question_id": "q1",
            "dataset_source": "nq",
            "weight_grid": [0.0, 0.5, 1.0],
            "scores": [
                {"dense_weight": 0.0, "oracle_objective_value": 0.9},
                {"dense_weight": 0.5, "oracle_objective_value": 0.9},
                {"dense_weight": 1.0, "oracle_objective_value": 0.9},
            ],
        }
    ]
    out = tmp_path / "labels" / "router_labels.jsonl"
    n = materialize_router_labels(rows, output_path=out)
    assert n == 1
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["oracle_binary_class"] == "dense"
    assert rec["oracle_binary_class_id"] == 1


def test_materialize_router_labels_binary_prefers_dense_over_graph(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "question_id": "q1",
            "dataset_source": "nq",
            "weight_grid": [0.0, 0.5, 1.0],
            "scores": [
                {"dense_weight": 0.0, "oracle_objective_value": 0.8},
                {"dense_weight": 0.5, "oracle_objective_value": 0.3},
                {"dense_weight": 1.0, "oracle_objective_value": 0.8},
            ],
        }
    ]
    out = tmp_path / "labels" / "router_labels.jsonl"
    n = materialize_router_labels(rows, output_path=out)
    assert n == 1
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["oracle_binary_class"] == "dense"
    assert rec["oracle_binary_class_id"] == 1
