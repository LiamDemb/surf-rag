"""Tests for canonical artifact path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from surf_rag.evaluation.artifact_paths import (
    benchmark_bundle_dir,
    default_data_base,
    default_router_base,
    evaluation_policy_dir,
    evaluations_root,
    hard_router_policy_id,
    trained_router_policy_id,
    router_dataset_dir,
    router_model_dir,
    router_oracle_dir,
)


def test_benchmark_bundle_and_eval_paths() -> None:
    b = default_data_base() / "mix" / "v01"
    assert benchmark_bundle_dir(Path("data"), "mix", "v01") == b
    assert evaluations_root(Path("data"), "mix", "v01") == b / "evaluations"
    assert evaluation_policy_dir(Path("data"), "mix", "v01", "50-50") == b / "evaluations" / "50-50"


def test_router_subdirs() -> None:
    rb = Path("data") / "router"
    assert router_oracle_dir(rb, "v01") == rb / "v01" / "oracle"
    assert router_dataset_dir(rb, "v01") == rb / "v01" / "dataset"
    assert router_model_dir(rb, "v01") == rb / "v01" / "model"


def test_policy_id_helpers() -> None:
    assert trained_router_policy_id("v01") == "trained-router-v01"
    assert hard_router_policy_id("v01") == "hard-router-v01"


def test_default_bases_respect_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_BASE", "/d")
    assert default_data_base() == Path("/d")
    monkeypatch.setenv("ROUTER_BASE", "/r")
    assert default_router_base() == Path("/r")
