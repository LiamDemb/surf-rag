"""Materialize deterministic router labels from an oracle run."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.config.merge import merge_create_soft_labels_args
from surf_rag.config.resolved import write_resolved_config_yaml

from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    build_oracle_run_root,
    default_oracle_base,
    read_manifest,
    read_oracle_score_rows,
    update_manifest,
)
from surf_rag.evaluation.retrieval_metrics import PRIMARY_NDCG_K
from surf_rag.router.soft_labels import materialize_router_labels

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize deterministic router labels from an oracle run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (paths.router_id).",
    )
    parser.add_argument(
        "--router-id", default=None, help="Router bundle id (oracle directory key)."
    )
    parser.add_argument(
        "--router-base",
        type=Path,
        default=None,
        help="Override router bundle root (falls back to $ROUTER_BASE, then $DATA_BASE/router).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> OracleRunPaths:
    base = args.router_base if args.router_base else default_oracle_base()
    return OracleRunPaths(run_root=build_oracle_run_root(base, args.router_id))


def main() -> int:
    load_app_env()
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    args._pipeline_cfg = None
    if args.config:
        pcfg = load_pipeline_config(args.config.resolve())
        args._pipeline_cfg = pcfg
        apply_pipeline_env_from_config(pcfg)
        merge_create_soft_labels_args(args, pcfg)
    if not args.router_id:
        logger.error("Provide --config or --router-id.")
        return 2

    paths = _resolve_paths(args)
    if not paths.oracle_scores.is_file():
        logger.error("Missing %s. Run prepare_oracle_run first.", paths.oracle_scores)
        return 1

    paths.ensure_dirs()
    rows = read_oracle_score_rows(paths)
    if not rows:
        logger.error("No oracle score rows found in %s.", paths.oracle_scores)
        return 1

    manifest = read_manifest(paths) if paths.manifest.is_file() else {}
    oracle_metric = str(manifest.get("oracle_metric", "stateful_ndcg"))
    oracle_metric_k = int(manifest.get("oracle_metric_k", PRIMARY_NDCG_K))
    n = materialize_router_labels(
        rows,
        output_path=paths.router_labels,
        oracle_metric=oracle_metric,
        oracle_metric_k=oracle_metric_k,
    )
    logger.info("wrote %d labels to %s", n, paths.router_labels)

    update_manifest(
        paths,
        {
            "router_labels": {
                "count": int(n),
                "file": paths.router_labels.name,
            }
        },
    )
    if getattr(args, "_pipeline_cfg", None) is not None:
        write_resolved_config_yaml(
            paths.run_root / "resolved_config.yaml",
            args._pipeline_cfg,
            resolve_paths(args._pipeline_cfg),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
