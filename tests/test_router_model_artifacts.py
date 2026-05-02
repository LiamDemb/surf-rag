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
        "data/router/v01/model/both"
    )
    assert build_router_model_root(
        Path("data/router"), "v01", "both", "mlp-v1-default"
    ) == Path("data/router/v01/models/mlp-v1-default/both")
    assert build_router_model_root(
        Path("data/router"), "v01", "query-features"
    ) == Path("data/router/v01/model/query-features")


def test_paths() -> None:
    p = make_router_model_paths_for_cli("r1", router_base=Path("/b"))
    assert p.run_root == Path("/b/r1/model/both")
    assert p.checkpoint == Path("/b/r1/model/both/model.pt")
    assert p.predictions("test") == Path("/b/r1/model/both/predictions_test.jsonl")
    p2 = make_router_model_paths_for_cli(
        "r1", router_base=Path("/b"), router_architecture_id="logreg-v1-baseline"
    )
    assert p2.run_root == Path("/b/r1/models/logreg-v1-baseline/both")
    assert p2.checkpoint == Path("/b/r1/models/logreg-v1-baseline/both/model.pt")


def test_manifest_roundtrip(tmp_path: Path) -> None:
    paths = RouterModelPaths(run_root=build_router_model_root(tmp_path, "m1", "both"))
    paths.ensure_dirs()
    write_router_model_manifest(
        paths,
        router_id="m1",
        router_architecture_id="mlp-v1-default",
        input_mode="both",
        architecture_name="mlp-v1",
        architecture_kwargs={"hidden_dim": 32},
        dataset_manifest_path="/x/dataset/manifest.json",
        model_config={"hidden_dim": 32},
        training_config={"epochs": 10},
        feature_set_version="1",
        embedding_model="m",
        weight_grid=[0.0, 0.5, 1.0],
    )
    m = read_router_model_manifest(paths)
    assert m["model_id"] == "m1:mlp-v1-default:both"
    assert m["router_architecture_id"] == "mlp-v1-default"
    assert m["model"]["input_mode"] == "both"
    assert m["model"]["architecture_name"] == "mlp-v1"
    assert m["model"]["architecture_kwargs"] == {"hidden_dim": 32}
    assert m["model"]["active_inputs"] == [
        "query_embedding",
        "feature_vector_norm",
    ]
    assert m["model"]["weight_grid"] == [0.0, 0.5, 1.0]
    assert m["artifacts"]["checkpoint"] == "model.pt"
    update_router_model_manifest(paths, {"note": "x"})
    m2 = read_router_model_manifest(paths)
    assert m2.get("note") == "x"
    assert "updated_at" in m2
