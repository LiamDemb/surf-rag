"""Tests for surf_rag.config loader and path resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from surf_rag.config.loader import (
    config_to_resolved_dict,
    load_pipeline_config,
    pipeline_config_from_dict,
    resolve_paths,
    validate_e2e_config,
)
from surf_rag.config.schema import PipelineConfig
from surf_rag.config.env import apply_pipeline_env_from_config


def test_yaml_unquoted_ids_coerce_to_str_for_paths_and_env() -> None:
    """YAML parses ``benchmark_id: 100`` as int; loader and env must treat as str."""
    cfg = pipeline_config_from_dict(
        {
            "paths": {
                "benchmark_name": "hotpotqa",
                "benchmark_id": 100,
                "router_id": 42,
            }
        }
    )
    assert cfg.paths.benchmark_id == "100"
    assert cfg.paths.router_id == "42"
    apply_pipeline_env_from_config(cfg)
    assert os.environ["BENCHMARK_ID"] == "100"
    assert os.environ["ROUTER_ID"] == "42"


def test_load_example_pipeline(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "dev" / "examples" / "surf-bench-200-pipeline.yaml"
    cfg = load_pipeline_config(cfg_path)
    assert cfg.schema_version == "surf-rag/pipeline/v1"
    assert cfg.paths.benchmark_name == "surf-bench"
    assert cfg.paths.benchmark_id == "200-test"
    assert cfg.paths.router_id == "4000-test"
    rp = resolve_paths(cfg)
    assert rp.benchmark_path.name == "benchmark.jsonl"
    assert (rp.bundle / "corpus").samefile(rp.corpus_dir)
    d = config_to_resolved_dict(cfg, rp)
    assert "resolved_paths" in d
    assert d["resolved_paths"]["benchmark_path"] == str(rp.benchmark_path)


def test_pipeline_config_from_dict_empty() -> None:
    cfg = pipeline_config_from_dict({})
    assert isinstance(cfg, PipelineConfig)
    assert cfg.seed == 42


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


def test_validate_e2e_learned_requires_router_id() -> None:
    cfg = pipeline_config_from_dict(
        {
            "paths": {"router_id": ""},
            "e2e": {"policy": "learned-soft"},
        }
    )
    with pytest.raises(ValueError, match="router_id"):
        validate_e2e_config(cfg)


def test_e2e_sentence_rerank_fields_roundtrip() -> None:
    cfg = pipeline_config_from_dict(
        {
            "e2e": {
                "sentence_rerank_enabled": True,
                "sentence_rerank_top_k": 15,
                "sentence_rerank_max_sentences": 32,
                "sentence_rerank_max_words": 900,
                "sentence_rerank_include_title": False,
                "sentence_rerank_prompt_style": "inline",
            }
        }
    )
    assert cfg.e2e.sentence_rerank_enabled is True
    assert cfg.e2e.sentence_rerank_top_k == 15
    assert cfg.e2e.sentence_rerank_max_sentences == 32
    assert cfg.e2e.sentence_rerank_max_words == 900
    assert cfg.e2e.sentence_rerank_include_title is False
    assert cfg.e2e.sentence_rerank_prompt_style == "inline"


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
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(p)
    rp = resolve_paths(cfg)
    assert rp.benchmark_name == "b"
    assert rp.router_id == "r1"
