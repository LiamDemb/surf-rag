"""Sweep beta values against an oracle run and write beta_sweep diagnostics.

Given an oracle run's ``oracle_scores.jsonl``, this computes per-question
soft labels for each candidate beta and aggregates entropy / expected
weight / argmax weight statistics into ``beta_sweep.jsonl``. A
recommended beta is selected heuristically (lowest entropy above a
configurable floor) and written to ``recommended_beta.json``. You are
always free to override the recommendation by passing
``--selected-beta`` to :mod:`scripts.create_soft_labels`.

Example:

    python -m scripts.sweep_beta \\
        --benchmark mix --split dev --oracle-run-id run1 \\
        --betas 0.5 1.0 2.0 5.0 10.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

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
    parser = argparse.ArgumentParser(description="Sweep beta values and report aggregate label stats.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--oracle-run-id", required=True)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        required=True,
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
        "--oracle-base",
        type=Path,
        default=None,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> OracleRunPaths:
    base = args.oracle_base if args.oracle_base else default_oracle_base()
    return OracleRunPaths(
        run_root=build_oracle_run_root(
            base, args.benchmark, args.split, args.oracle_run_id
        )
    )


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
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

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
    logger.info("recommended beta=%s (mean_entropy=%.4f)", recommended.beta, recommended.mean_entropy)

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
