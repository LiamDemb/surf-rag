"""Tests for canonical artifact path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from surf_rag.evaluation.artifact_paths import (
    benchmark_bundle_dir,
    benchmark_jsonl_path,
    default_benchmark_base,
    default_data_base,
    default_router_base,
    evaluation_policy_dir,
    evaluations_root,
    hard_router_policy_id,
    trained_router_policy_id,
    router_dataset_dir,
    router_model_architecture_dir,
    router_model_dir,
    router_models_dir,
    router_oracle_dir,
    safe_benchmark_bundle_subpath,
)


def test_benchmark_bundle_and_eval_paths() -> None:
    base = Path("data") / "benchmarks"
    b = base / "mix" / "v01"
    assert benchmark_bundle_dir(base, "mix", "v01") == b
    assert (
        benchmark_jsonl_path(base, "mix", "v01") == b / "benchmark" / "benchmark.jsonl"
    )
    assert evaluations_root(base, "mix", "v01") == b / "evaluations"
    assert (
        evaluation_policy_dir(base, "mix", "v01", "50-50")
        == b / "evaluations" / "50-50"
    )


def test_router_subdirs() -> None:
    rb = Path("data") / "router"
    assert router_oracle_dir(rb, "v01") == rb / "v01" / "oracle"
    assert router_dataset_dir(rb, "v01") == rb / "v01" / "dataset"
    assert router_model_dir(rb, "v01") == rb / "v01" / "model"
    assert router_models_dir(rb, "v01") == rb / "v01" / "models"
    assert (
        router_model_architecture_dir(rb, "v01", "mlp-v1-a")
        == rb / "v01" / "models" / "mlp-v1-a"
    )


def test_policy_id_helpers() -> None:
    assert trained_router_policy_id("v01") == "trained-router-v01"
    assert hard_router_policy_id("v01") == "hard-router-v01"


def test_default_bases_respect_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_BASE", "/d")
    assert default_data_base() == Path("/d")
    monkeypatch.delenv("BENCHMARK_BASE", raising=False)
    assert default_benchmark_base() == Path("/d") / "benchmarks"
    monkeypatch.setenv("BENCHMARK_BASE", "/b")
    assert default_benchmark_base() == Path("/b")
    monkeypatch.setenv("ROUTER_BASE", "/r")
    assert default_router_base() == Path("/r")


def test_safe_benchmark_bundle_subpath_sanitizes() -> None:
    assert safe_benchmark_bundle_subpath("graph rag tuning!") == "graph-rag-tuning"
    assert safe_benchmark_bundle_subpath("already-safe_name.1") == "already-safe_name.1"
    assert safe_benchmark_bundle_subpath("   ") == "graph-rag-tuning"
