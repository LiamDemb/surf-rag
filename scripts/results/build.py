#!/usr/bin/env python3
"""Build dissertation results tables (CSV) and figures from frozen artefacts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.config.env import apply_pipeline_env_from_config, load_app_env
from surf_rag.config.loader import load_pipeline_config
from surf_rag.results.build import build_results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build results bundle (CSV tables + themed figures)."
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pipeline YAML with paths and results section.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ID",
        help="Build only these artifact ids (repeatable).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    load_app_env()
    load_dotenv()
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    cfg_path = args.config.resolve()
    cfg = load_pipeline_config(cfg_path)
    apply_pipeline_env_from_config(cfg)

    only_ids = frozenset(args.only) if args.only else None
    try:
        build_results(cfg, config_path=cfg_path, only_ids=only_ids)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
