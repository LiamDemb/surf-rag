"""Tests for merge helpers (CLI override vs config)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from surf_rag.config.loader import (
    load_pipeline_config,
    pipeline_config_from_dict,
    resolve_paths,
)
from surf_rag.config.merge import (
    merge_e2e_common_args,
    merge_e2e_prepare_args,
    merge_ingest_args,
    merge_sweep_beta_args,
)
from surf_rag.config.schema import PipelineConfig


def _cfg() -> PipelineConfig:
    return load_pipeline_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "dev"
        / "examples"
        / "surf-bench-200-pipeline.yaml"
    )


def test_merge_ingest_yaml_null_clears_nq() -> None:
    """Null/omitted in YAML is normalized to None; merge must not keep argparse defaults."""
    cfg = pipeline_config_from_dict(
        {
            "paths": {
                "benchmark_name": "bn",
                "benchmark_id": "bid",
                "router_id": "r",
            },
            "raw_sources": {
                "nq_path": None,
                "wiki2_path": "data/w.jsonl",
            },
        }
    )
    assert cfg.raw_sources.nq_path is None
    args = Namespace()
    args.nq = "/pretend/env/default"
    args.wiki2 = None
    args.hotpotqa = None
    args.output_dir = None
    args.nq_version = None
    args.wiki2_version = None
    args.hotpotqa_version = None
    merge_ingest_args(args, cfg, argv=["ingest", "--config", "c.yaml"])
    assert args.nq is None
    assert args.wiki2 == "data/w.jsonl"


def test_merge_ingest_fills_from_config() -> None:
    cfg = _cfg()
    args = Namespace()
    args.nq = None
    args.wiki2 = None
    args.hotpotqa = None
    args.output_dir = None
    args.nq_version = None
    args.wiki2_version = None
    args.hotpotqa_version = None
    merge_ingest_args(args, cfg, argv=["prog", "--config", "x.yaml"])
    rp = resolve_paths(cfg)
    assert Path(args.output_dir) == rp.benchmark_dir
    assert args.nq == cfg.raw_sources.nq_path
    assert args.hotpotqa == cfg.raw_sources.hotpotqa_path


def test_merge_ingest_hotpotqa_path_from_yaml() -> None:
    cfg = pipeline_config_from_dict(
        {
            "paths": {
                "benchmark_name": "bn",
                "benchmark_id": "bid",
                "router_id": "r",
            },
            "raw_sources": {
                "hotpotqa_path": "data/raw/hotpot.jsonl",
                "hotpotqa_version": "v9",
            },
        }
    )
    args = Namespace()
    args.nq = None
    args.wiki2 = None
    args.hotpotqa = None
    args.output_dir = None
    args.nq_version = None
    args.wiki2_version = None
    args.hotpotqa_version = None
    merge_ingest_args(args, cfg, argv=["ingest", "--config", "c.yaml"])
    assert args.hotpotqa == "data/raw/hotpot.jsonl"
    assert args.hotpotqa_version == "v9"


def test_merge_sweep_betas_from_config_when_flag_absent() -> None:
    cfg = _cfg()
    args = Namespace()
    args.router_id = None
    args.min_entropy_nats = None
    args.betas = None
    args.router_base = None
    merge_sweep_beta_args(args, cfg, argv=["prog", "--config", "x.yaml"])
    assert args.betas == cfg.oracle.betas


def test_e2e_prepare_cli_run_id_overrides_config() -> None:
    cfg = _cfg()
    cfg.e2e.run_id = "from-yaml"
    args = Namespace()
    args.benchmark_base = None
    args.benchmark_name = None
    args.benchmark_id = None
    args.benchmark_path = None
    args.split = None
    args.run_id = "from-cli"
    args.policy = None
    args.retrieval_asset_dir = None
    args.router_id = None
    args.router_base = None
    args.fusion_keep_k = 25
    args.reranker = "none"
    args.rerank_top_k = 10
    args.cross_encoder_model = None
    args.limit = None
    args.completion_window = None
    args.include_graph_provenance = False
    args.dry_run = False
    args.router_device = "cpu"
    args.router_input_mode = "both"
    args.router_inference_batch_size = 32
    args.only_question_id = []
    merge_e2e_prepare_args(
        args,
        cfg,
        argv=[
            "prog",
            "prepare",
            "--config",
            "x.yaml",
            "--run-id",
            "from-cli",
        ],
    )
    assert args.run_id == "from-cli"


def test_e2e_common_fills_benchmark_path() -> None:
    cfg = _cfg()
    args = Namespace()
    for a in (
        "benchmark_base",
        "benchmark_name",
        "benchmark_id",
        "benchmark_path",
        "split",
        "run_id",
        "policy",
        "retrieval_asset_dir",
    ):
        setattr(args, a, None)
    merge_e2e_common_args(args, cfg, argv=["prog", "--config", "x.yaml"])
    rp = resolve_paths(cfg)
    assert args.benchmark_path == rp.benchmark_path
