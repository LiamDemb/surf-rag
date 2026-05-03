"""Evaluate a trained router checkpoint; reads ``.../model/<input_mode>/model.pt``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.merge import merge_router_evaluate_args

import pandas as pd

from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)
from surf_rag.evaluation.router_model_artifacts import (
    make_router_model_paths_for_cli,
    read_json,
    write_json,
    write_predictions_jsonl,
)
from surf_rag.router.inference import load_router_checkpoint
from surf_rag.router.model import parse_router_input_mode
from surf_rag.router.training import (
    _eval_splits,
    _weight_grid_from_df,
    export_split_predictions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RouterMLP checkpoint.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (router.train section).",
    )
    p.add_argument("--router-id", default=None)
    p.add_argument("--router-architecture-id", default=None)
    p.add_argument("--router-base", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--input-mode",
        default=None,
        help="both | query-features | embedding (default: ROUTER_INPUT_MODE or both)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    load_app_env()
    load_dotenv()
    import logging

    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)
    if args.config:
        pcfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(pcfg)
        merge_router_evaluate_args(args, pcfg)
    if not args.router_id:
        log.error("Provide --config or --router-id.")
        return 2

    ds_paths = make_router_dataset_paths_for_cli(
        args.router_id, router_base=args.router_base
    )
    raw_mode = (args.input_mode or "").strip() or os.getenv("ROUTER_INPUT_MODE", "both")
    input_mode = parse_router_input_mode(str(raw_mode))
    m_paths = make_router_model_paths_for_cli(
        args.router_id,
        router_base=args.router_base,
        input_mode=input_mode,
        router_architecture_id=args.router_architecture_id,
    )
    if not ds_paths.router_dataset_parquet.is_file():
        log.error("Missing %s", ds_paths.router_dataset_parquet)
        return 1
    if not m_paths.checkpoint.is_file():
        log.error("Missing %s", m_paths.checkpoint)
        return 1

    device = args.device or os.getenv("ROUTER_TRAIN_DEVICE", "cpu")
    loaded = load_router_checkpoint(
        m_paths.checkpoint, device=device, manifest_path=m_paths.manifest
    )
    model = loaded.model
    mcfg = loaded.config

    df = pd.read_parquet(ds_paths.router_dataset_parquet)
    wg = _weight_grid_from_df(df)
    metrics = _eval_splits(model, df, wg, device, mcfg, loaded.architecture)

    # Do not clobber training-only keys (loss, midpoint_balance_masking report,
    # best_epoch, …). ``router-eval`` is often run after ``router-train`` and must
    # refresh ``splits`` while preserving artifacts written by train_router.py.
    existing: Dict[str, Any] = {}
    if m_paths.metrics.is_file():
        try:
            existing = read_json(m_paths.metrics)
            if not isinstance(existing, dict):
                existing = {}
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}

    payload = {
        "router_id": args.router_id,
        "router_architecture_id": args.router_architecture_id,
        "architecture": loaded.architecture,
        "input_mode": input_mode,
        "splits": metrics,
        "router_quality_filtering": dict(metrics.get("router_quality_filtering") or {}),
    }
    write_json(m_paths.metrics, {**existing, **payload})

    for split in ("train", "dev", "test"):
        rows = export_split_predictions(model, df, split, device, wg)
        if rows:
            write_predictions_jsonl(m_paths.predictions(split), rows)
    log.info("Wrote %s", m_paths.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
