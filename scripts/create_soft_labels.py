"""Materialize soft labels from an oracle run's saved score vectors.

Inputs:
  - An oracle run root created by :mod:`scripts.prepare_oracle_run`.
  - One or more ``--beta`` values.

Outputs (under ``<oracle_run_root>/labels/``):
  - ``beta_<value>.jsonl`` for each ``--beta``.
  - ``selected.jsonl`` as a symlink-like canonical pointer (copy of the
    ``--selected-beta`` value, or the first if not specified).

Changing ``beta`` never retriggers retrieval or the metric sweep - it
only re-reads ``oracle_scores.jsonl`` and writes new label files.

Example:

    python -m scripts.create_soft_labels \\
        --benchmark mix --split dev --oracle-run-id run1 \\
        --beta 1.0 --beta 2.0 --beta 5.0 --selected-beta 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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
)
from surf_rag.router.soft_labels import materialize_soft_labels

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize soft labels from an oracle run."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--oracle-run-id", required=True)
    parser.add_argument(
        "--beta",
        action="append",
        type=float,
        required=True,
        help="Beta value to materialize (repeatable).",
    )
    parser.add_argument(
        "--selected-beta",
        type=float,
        default=None,
        help="Beta whose label file becomes labels/selected.jsonl. Defaults to the first --beta.",
    )
    parser.add_argument(
        "--oracle-base",
        type=Path,
        default=None,
        help="Override oracle run root base (falls back to $ORACLE_BASE, then data/oracle).",
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


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    paths = _resolve_paths(args)
    if not paths.oracle_scores.is_file():
        logger.error("Missing %s. Run prepare_oracle_run first.", paths.oracle_scores)
        return 1

    paths.ensure_dirs()
    rows = read_oracle_score_rows(paths)
    if not rows:
        logger.error("No oracle score rows found in %s.", paths.oracle_scores)
        return 1

    written: List[dict] = []
    for beta in args.beta:
        out = paths.labels_for_beta(beta)
        n = materialize_soft_labels(rows, beta=beta, output_path=out)
        logger.info("beta=%s -> wrote %d labels to %s", beta, n, out)
        written.append({"beta": float(beta), "path": out.name, "count": n})

    selected_beta = (
        args.selected_beta if args.selected_beta is not None else args.beta[0]
    )
    selected_src = paths.labels_for_beta(selected_beta)
    if not selected_src.is_file():
        logger.error(
            "--selected-beta %s did not produce a label file (check --beta).",
            selected_beta,
        )
        return 2

    shutil.copyfile(selected_src, paths.labels_selected)
    logger.info(
        "labels/selected.jsonl <- %s (beta=%s)", selected_src.name, selected_beta
    )

    update_manifest(
        paths,
        {
            "soft_labels": {
                "written": written,
                "selected_beta": float(selected_beta),
                "selected_file": paths.labels_selected.name,
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
