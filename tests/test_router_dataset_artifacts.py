"""Tests for router dataset artifact paths and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.evaluation.router_dataset_artifacts import (
    RouterDatasetPaths,
    build_router_dataset_root,
    build_split_question_ids_dict,
    make_router_dataset_paths_for_cli,
    read_jsonl_dict,
    write_router_dataset_manifest,
    write_split_question_ids,
)


def test_default_router_base(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_router_base() == Path("data/router")


def test_run_root_layout() -> None:
    p = make_router_dataset_paths_for_cli("ds1", router_base=Path("/base"))
    assert p.run_root == Path("/base") / "ds1" / "dataset"
    assert p.router_dataset_parquet == p.run_root / "router_dataset.parquet"
    assert p.split_question_ids == p.run_root / "split_question_ids.json"


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    paths = RouterDatasetPaths(run_root=build_router_dataset_root(tmp_path, "d1"))
    write_router_dataset_manifest(
        paths,
        router_id="d1",
        source_benchmark_name="b",
        source_benchmark_id="v01",
        benchmark_path="/x/b.jsonl",
        retrieval_asset_dir="/x/corpus",
        oracle_run_root="/base/d1/oracle",
        router_labels_path="/base/d1/oracle/router_labels.jsonl",
        feature_set_version="1",
        embedding_model="m",
        split_seed=42,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
    )
    m = json.loads(paths.manifest.read_text())
    assert m["router_id"] == "d1"
    assert m["oracle"]["run_root"] == "/base/d1/oracle"
    assert m["oracle"]["router_labels"] == "/base/d1/oracle/router_labels.jsonl"


def test_build_split_question_ids_dict_matches_df(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "question_id": ["a", "b", "c", "d"],
            "split": ["train", "train", "dev", "test"],
        }
    )
    d = build_split_question_ids_dict(
        df,
        router_id="r1",
        source_benchmark_name="mix",
        source_benchmark_id="v01",
        split_seed=7,
    )
    assert d["counts"] == {"train": 2, "dev": 1, "test": 1}
    assert set(d["train"]) == {"a", "b"}
    assert d["canonical_question_hash_available"] is False

    out = Path(tmp_path) / "sqi.json"
    write_split_question_ids(out, d)
    assert json.loads(out.read_text())["router_id"] == "r1"


def test_read_jsonl_dict(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"question_id": "a", "k": 1})
        + "\n"
        + json.dumps({"question_id": "b", "k": 2})
        + "\n",
        encoding="utf-8",
    )
    d = read_jsonl_dict(p, "question_id")
    assert d["a"]["k"] == 1
    assert d["b"]["k"] == 2
