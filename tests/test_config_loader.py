"""Tests for surf_rag.config loader and path resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from dataclasses import replace

from surf_rag.config.loader import (
    config_to_resolved_dict,
    load_pipeline_config,
    pipeline_config_from_dict,
    resolve_paths,
    validate_e2e_config,
)
from surf_rag.config.schema import PathsSection, PipelineConfig
from surf_rag.config.env import apply_pipeline_env_from_config


def test_yaml_unquoted_ids_coerce_to_str_for_paths_and_env() -> None:
    """YAML parses ``benchmark_id: 100`` as int; loader and env must treat as str."""
    cfg = pipeline_config_from_dict(
        {
            "paths": {
                "benchmark_name": "hotpotqa",
                "benchmark_id": 100,
                "router_id": 42,
                "router_architecture_id": 7,
            }
        }
    )
    assert cfg.paths.benchmark_id == "100"
    assert cfg.paths.router_id == "42"
    assert cfg.paths.router_architecture_id == "7"
    apply_pipeline_env_from_config(cfg)
    assert os.environ["BENCHMARK_ID"] == "100"
    assert os.environ["ROUTER_ID"] == "42"
    assert os.environ["ROUTER_ARCHITECTURE_ID"] == "7"


def test_load_example_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "development" / "graph-rag-tuning.yaml"
    cfg = load_pipeline_config(cfg_path)
    assert cfg.schema_version == "surf-rag/pipeline/v1"
    assert cfg.paths.benchmark_name == "surf-bench"
    assert cfg.paths.benchmark_id == "development"
    assert cfg.paths.router_id == ""
    assert cfg.graph_retrieval_sweep.use_router_overlap_splits is False
    rp = resolve_paths(cfg)
    assert rp.benchmark_path.name == "benchmark.jsonl"
    assert rp.corpus_dir == rp.bundle / "corpus"
    d = config_to_resolved_dict(cfg, rp)
    assert "resolved_paths" in d
    assert d["resolved_paths"]["benchmark_path"] == str(rp.benchmark_path)
    assert d["resolved_paths"]["figures_base"] == str(rp.figures_base)
    assert rp.figures_base == (tmp_path / "figures").resolve()


def test_figures_base_filesystem_root_raises() -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(PathsSection(), figures_base="/"),
    )
    with pytest.raises(ValueError, match="cannot be '/'"):
        resolve_paths(cfg)


def test_pipeline_config_from_dict_empty() -> None:
    cfg = pipeline_config_from_dict({})
    assert isinstance(cfg, PipelineConfig)
    assert cfg.seed == 42
    assert cfg.graph_retrieval_sweep.use_router_overlap_splits is False


def test_e2e_fusion_keep_inherits_oracle_when_unset() -> None:
    cfg = pipeline_config_from_dict(
        {
            "oracle": {"fusion_keep_k": 42},
            "e2e": {"policy": "dense-only"},
        }
    )
    assert cfg.e2e.fusion_keep_k == 42


def test_e2e_branch_top_k_inherits_oracle_when_unset() -> None:
    cfg = pipeline_config_from_dict(
        {
            "oracle": {"branch_top_k": 15},
            "e2e": {"policy": "dense-only"},
        }
    )
    assert cfg.e2e.branch_top_k == 15


def test_router_train_midpoint_balance_fields_roundtrip() -> None:
    cfg = pipeline_config_from_dict(
        {
            "router": {
                "train": {
                    "midpoint_balance_masking": True,
                    "midpoint_balance_epsilon": 1e-4,
                }
            },
        }
    )
    assert cfg.router.train.midpoint_balance_masking is True
    assert cfg.router.train.midpoint_balance_epsilon == pytest.approx(1e-4)


def test_validate_e2e_learned_requires_router_id() -> None:
    cfg = pipeline_config_from_dict(
        {
            "paths": {"router_id": ""},
            "e2e": {"policy": "learned-soft"},
        }
    )
    with pytest.raises(ValueError, match="router_id"):
        validate_e2e_config(cfg)


def test_validate_e2e_learned_hybrid_requires_router_id() -> None:
    cfg = pipeline_config_from_dict(
        {
            "paths": {"router_id": ""},
            "e2e": {"policy": "learned-hybrid"},
        }
    )
    with pytest.raises(ValueError, match="router_id"):
        validate_e2e_config(cfg)


def test_validate_e2e_oracle_upper_bound_requires_router_id() -> None:
    cfg = pipeline_config_from_dict(
        {
            "paths": {"router_id": ""},
            "e2e": {"policy": "oracle-upper-bound"},
        }
    )
    with pytest.raises(ValueError, match="router_id"):
        validate_e2e_config(cfg)


def test_minimal_yaml_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "paths": {
                    "benchmark_name": "b",
                    "benchmark_id": "v1",
                    "router_id": "r1",
                    "router_architecture_id": "mlp-v1-default",
                },
                "router": {
                    "train": {
                        "architecture": "logreg-v1",
                        "architecture_kwargs": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(p)
    rp = resolve_paths(cfg)
    assert rp.benchmark_name == "b"
    assert rp.router_id == "r1"
    assert rp.router_architecture_id == "mlp-v1-default"
    assert cfg.router.train.architecture == "logreg-v1"
    assert cfg.router.train.loss == "regret"
    assert cfg.router.train.loss_kwargs == {}


def test_graph_retrieval_sweep_bool_parsed() -> None:
    cfg = pipeline_config_from_dict(
        {
            "graph_retrieval_sweep": {
                "use_router_overlap_splits": True,
            }
        }
    )
    assert cfg.graph_retrieval_sweep.use_router_overlap_splits is True
