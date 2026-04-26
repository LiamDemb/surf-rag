"""Tests for trained router model bundle paths."""

from __future__ import annotations

import json
from pathlib import Path

from surf_rag.evaluation.router_model_artifacts import (
    RouterModelPaths,
    build_router_model_root,
    make_router_model_paths_for_cli,
    read_router_model_manifest,
    update_router_model_manifest,
    write_router_model_manifest,
)


def test_model_root() -> None:
    assert build_router_model_root(Path("data/router"), "v01") == Path(
        "data/router/v01/model"
    )


def test_paths() -> None:
    p = make_router_model_paths_for_cli("r1", router_base=Path("/b"))
    assert p.run_root == Path("/b/r1/model")
    assert p.checkpoint == Path("/b/r1/model/model.pt")
    assert p.predictions("test") == Path("/b/r1/model/predictions_test.jsonl")


def test_manifest_roundtrip(tmp_path: Path) -> None:
    paths = RouterModelPaths(run_root=build_router_model_root(tmp_path, "m1"))
    paths.ensure_dirs()
    write_router_model_manifest(
        paths,
        router_id="m1",
        dataset_manifest_path="/x/dataset/manifest.json",
        model_config={"hidden_dim": 32},
        training_config={"epochs": 10},
        feature_set_version="1",
        embedding_model="m",
        weight_grid=[0.0, 0.5, 1.0],
    )
    m = read_router_model_manifest(paths)
    assert m["model"]["weight_grid"] == [0.0, 0.5, 1.0]
    assert m["artifacts"]["checkpoint"] == "model.pt"
    update_router_model_manifest(paths, {"note": "x"})
    m2 = read_router_model_manifest(paths)
    assert m2.get("note") == "x"
    assert "updated_at" in m2
