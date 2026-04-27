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
from surf_rag.config.merge import merge_sweep_beta_args
from surf_rag.config.resolved import write_resolved_config_yaml

from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    build_oracle_run_root,
    default_oracle_base,
    read_oracle_score_rows,
    update_manifest,
    write_jsonl,
)
from surf_rag.router.soft_labels import BetaSweepStats, sweep_beta

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep beta values and report aggregate label stats."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (oracle.betas, paths.router_id).",
    )
    parser.add_argument(
        "--router-id", default=None, help="Router bundle id (oracle directory key)."
    )
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=None,
        help="Beta values to try (e.g. 0.5 1.0 2.0 5.0).",
    )
    parser.add_argument(
        "--min-entropy-nats",
        type=float,
        default=0.1,
        help=(
            "Floor for the recommended-beta heuristic; we pick the largest "
            "beta whose mean entropy is still above this floor."
        ),
    )
    parser.add_argument(
        "--router-base",
        type=Path,
        default=None,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> OracleRunPaths:
    base = args.router_base if args.router_base else default_oracle_base()
    return OracleRunPaths(run_root=build_oracle_run_root(base, args.router_id))


def _recommend_beta(
    stats: List[BetaSweepStats],
    min_entropy_nats: float,
) -> BetaSweepStats:
    """Pick the largest beta whose mean_entropy is still >= the floor.

    If no beta clears the floor, fall back to the beta with the lowest
    entropy (i.e. the sharpest labels available).
    """
    by_beta_sorted = sorted(stats, key=lambda s: s.beta)
    eligible = [s for s in by_beta_sorted if s.mean_entropy >= min_entropy_nats]
    if eligible:
        return eligible[-1]
    return min(stats, key=lambda s: s.mean_entropy)


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
        merge_sweep_beta_args(args, pcfg)
    if not args.router_id or not args.betas:
        logger.error("Provide --config or both --router-id and --betas.")
        return 2

    paths = _resolve_paths(args)
    if not paths.oracle_scores.is_file():
        logger.error("Missing %s. Run prepare_oracle_run first.", paths.oracle_scores)
        return 1

    rows = read_oracle_score_rows(paths)
    if not rows:
        logger.error("oracle_scores.jsonl is empty at %s.", paths.oracle_scores)
        return 1

    stats = sweep_beta(rows, args.betas)
    write_jsonl(paths.beta_sweep, (s.to_json() for s in stats))

    if getattr(args, "_pipeline_cfg", None) is not None:
        write_resolved_config_yaml(
            paths.run_root / "resolved_config.yaml",
            args._pipeline_cfg,
            resolve_paths(args._pipeline_cfg),
        )

    recommended = _recommend_beta(stats, args.min_entropy_nats)
    payload = {
        "recommended_beta": recommended.beta,
        "min_entropy_nats": args.min_entropy_nats,
        "rationale": (
            "largest beta with mean_entropy >= min_entropy_nats; "
            "fallback: beta with min mean_entropy."
        ),
        "stats": [s.to_json() for s in stats],
    }
    paths.recommended_beta.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("beta sweep -> %s", paths.beta_sweep)
    logger.info(
        "recommended beta=%s (mean_entropy=%.4f)",
        recommended.beta,
        recommended.mean_entropy,
    )

    update_manifest(
        paths,
        {
            "beta_sweep": {
                "betas": [float(b) for b in args.betas],
                "min_entropy_nats": float(args.min_entropy_nats),
                "recommended_beta": float(recommended.beta),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
