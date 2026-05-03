from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from surf_rag.evaluation.router_model_artifacts import RouterModelPaths
from surf_rag.viz.sources.router_predictions import (
    load_router_prediction_rows,
    load_router_predictions_with_curves,
    load_weight_grid_from_manifest,
    predictions_path_for,
)


def test_predictions_path_for() -> None:
    mp = RouterModelPaths(run_root=Path("/tmp/r"))
    assert predictions_path_for(mp, "test") == mp.run_root / "predictions_test.jsonl"


def test_load_router_prediction_rows_returns_curve_columns(tmp_path: Path) -> None:
    p = tmp_path / "predictions_test.jsonl"
    row = {
        "question_id": "q1",
        "oracle_curve": [0.0, 1.0, 0.0],
        "predicted_weight": 0.4,
        "is_valid_for_router_training": True,
    }
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    df = load_router_prediction_rows(p)
    assert list(df.columns) == [
        "question_id",
        "oracle_curve",
        "predicted_weight",
        "valid",
    ]
    assert df.iloc[0]["oracle_curve"] == [0.0, 1.0, 0.0]
    assert df.iloc[0]["predicted_weight"] == pytest.approx(0.4)
    assert bool(df.iloc[0]["valid"])


def test_load_router_prediction_rows_missing_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="oracle_curve"):
        load_router_prediction_rows(p)


def test_load_router_prediction_rows_drops_non_finite_pred(tmp_path: Path) -> None:
    p = tmp_path / "p.jsonl"
    curve = [0.0, 1.0, 0.0]
    rows = [
        {
            "question_id": "q1",
            "oracle_curve": curve,
            "predicted_weight": 0.2,
        },
        {
            "question_id": "q2",
            "oracle_curve": curve,
            "predicted_weight": None,
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
                "oracle_curve": [0.0, 1.0, 0.0],
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


def test_load_weight_grid_from_manifest(tmp_path: Path) -> None:
    m = tmp_path / "manifest.json"
    m.write_text(
        json.dumps({"model": {"weight_grid": [0.0, 0.5, 1.0]}}),
        encoding="utf-8",
    )
    g = load_weight_grid_from_manifest(m)
    assert list(g) == pytest.approx([0.0, 0.5, 1.0])


def test_load_router_predictions_with_curves(tmp_path: Path) -> None:
    p = tmp_path / "predictions.jsonl"
    rows_curve = [
        {
            "question_id": "q1",
            "oracle_curve": [0.1, 0.2],
            "predicted_weight": 0.35,
            "is_valid_for_router_training": True,
        },
    ]
    p.write_text(
        "\n".join(json.dumps(r) for r in rows_curve) + "\n",
        encoding="utf-8",
    )
    df = load_router_predictions_with_curves(p)
    assert df.iloc[0]["oracle_curve"] == [0.1, 0.2]
    assert df.iloc[0]["predicted_weight"] == pytest.approx(0.35)
