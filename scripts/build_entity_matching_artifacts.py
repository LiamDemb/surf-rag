#!/usr/bin/env python3
"""Precompute lexicon phrase matcher artifacts for an existing corpus directory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from surf_rag.config.env import apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.entity_matching.artifacts import build_entity_matching_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config; sets corpus_dir from paths unless overridden.",
    )
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory containing alias_map.json and entity_lexicon.parquet",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if entity_matching_manifest.json exists.",
    )
    args = ap.parse_args()
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        if args.corpus_dir is None:
            args.corpus_dir = resolve_paths(cfg).corpus_dir
        if cfg.entity_matching.force and "--force" not in sys.argv:
            args.force = True
    if args.corpus_dir is None:
        ap.error("Pass --corpus-dir or --config with paths.corpus implied.")
    try:
        path = build_entity_matching_artifacts(
            args.corpus_dir.resolve(), force=args.force
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    logger.info("Done: %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
