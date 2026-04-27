"""Tests for validate-oracle / validate-router CLI helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from surf_rag.config.validate_prereqs import (
    validate_oracle,
    validate_router_dataset,
)


def test_validate_oracle_fails_without_benchmark(tmp_path: Path) -> None:
    cfg = {
        "schema_version": "surf-rag/pipeline/v1",
        "paths": {
            "data_base": str(tmp_path / "data"),
            "benchmark_base": str(tmp_path / "b"),
            "router_base": str(tmp_path / "r"),
            "benchmark_name": "bn",
            "benchmark_id": "bid",
            "router_id": "rid",
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert validate_oracle(p) != 0


def test_validate_oracle_ok_with_minimal_tree(tmp_path: Path) -> None:
    bundle = tmp_path / "benchmarks" / "bn" / "bid"
    bench = bundle / "benchmark"
    corp = bundle / "corpus"
    bench.mkdir(parents=True)
    corp.mkdir(parents=True)
    (bench / "benchmark.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = {
        "schema_version": "surf-rag/pipeline/v1",
        "paths": {
            "data_base": str(tmp_path),
            "benchmark_base": str(tmp_path / "benchmarks"),
            "router_base": str(tmp_path / "router"),
            "benchmark_name": "bn",
            "benchmark_id": "bid",
            "router_id": "rid",
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert validate_oracle(p) == 0


def test_validate_router_dataset_requires_selected(tmp_path: Path) -> None:
    bundle = tmp_path / "benchmarks" / "bn" / "bid"
    (bundle / "benchmark").mkdir(parents=True)
    (bundle / "corpus").mkdir(parents=True)
    (bundle / "benchmark" / "benchmark.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = {
        "schema_version": "surf-rag/pipeline/v1",
        "paths": {
            "data_base": str(tmp_path),
            "benchmark_base": str(tmp_path / "benchmarks"),
            "router_base": str(tmp_path / "router"),
            "benchmark_name": "bn",
            "benchmark_id": "bid",
            "router_id": "rid",
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert validate_router_dataset(p) != 0
