"""Tests for router dataset artifact paths and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surf_rag.evaluation.router_dataset_artifacts import (
    RouterDatasetPaths,
    build_router_dataset_root,
    default_router_dataset_base,
    make_router_dataset_paths_for_cli,
    read_jsonl_dict,
    write_router_dataset_manifest,
)


def test_default_router_base(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert default_router_dataset_base() == Path("data/router")


def test_run_root_layout() -> None:
    p = make_router_dataset_paths_for_cli("mix", "ds1", router_base=Path("/base"))
    assert p.run_root == Path("/base") / "mix" / "ds1"
    assert p.router_dataset_parquet == p.run_root / "router_dataset.parquet"


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    paths = RouterDatasetPaths(run_root=build_router_dataset_root(tmp_path, "b", "d1"))
    write_router_dataset_manifest(
        paths,
        dataset_id="d1",
        benchmark="b",
        benchmark_path="/x/b.jsonl",
        oracle_base="/o",
        oracle_benchmark="b",
        oracle_split="dev",
        oracle_run_id="r1",
        oracle_run_root="/o/b/dev/r1",
        labels_selected_path="/o/b/dev/r1/labels/selected.jsonl",
        selected_beta=2.0,
        feature_set_version="1",
        embedding_model="m",
        split_seed=42,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
    )
    m = json.loads(paths.manifest.read_text())
    assert m["dataset_id"] == "d1"
    assert m["oracle"]["oracle_run_id"] == "r1"
    assert m["oracle"]["selected_beta"] == 2.0


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
