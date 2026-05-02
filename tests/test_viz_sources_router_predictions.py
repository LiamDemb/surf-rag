from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from surf_rag.evaluation.router_model_artifacts import RouterModelPaths
from surf_rag.viz.sources.router_predictions import (
    load_router_prediction_rows,
    predictions_path_for,
)


def test_predictions_path_for() -> None:
    mp = RouterModelPaths(run_root=Path("/tmp/r"))
    assert predictions_path_for(mp, "test") == mp.run_root / "predictions_test.jsonl"


def test_load_router_prediction_rows_renames_columns(tmp_path: Path) -> None:
    p = tmp_path / "predictions_test.jsonl"
    row = {
        "question_id": "q1",
        "target_oracle_best_weight": 0.3,
        "predicted_weight": 0.4,
        "is_valid_for_router_training": True,
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    df = load_router_prediction_rows(p)
    assert list(df.columns) == [
        "question_id",
        "oracle_weight",
        "predicted_weight",
        "valid",
    ]
    assert df.iloc[0]["oracle_weight"] == pytest.approx(0.3)
    assert df.iloc[0]["predicted_weight"] == pytest.approx(0.4)
    assert bool(df.iloc[0]["valid"])


def test_load_router_prediction_rows_missing_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predicted_weight"):
        load_router_prediction_rows(p)


def test_load_router_prediction_rows_drops_non_finite(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    rows = [
        {
            "question_id": "q1",
            "target_oracle_best_weight": 0.1,
            "predicted_weight": 0.2,
        },
        {
            "question_id": "q2",
            "target_oracle_best_weight": None,
            "predicted_weight": 0.3,
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    df = load_router_prediction_rows(p)
    assert len(df) == 1
    assert df.iloc[0]["question_id"] == "q1"


def test_load_router_prediction_rows_valid_defaults_true(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    p.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "target_oracle_best_weight": 0.5,
                "predicted_weight": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    df = load_router_prediction_rows(p)
    assert bool(df.iloc[0]["valid"])


def test_load_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_router_prediction_rows(tmp_path / "missing.jsonl")
