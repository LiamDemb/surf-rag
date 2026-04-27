#!/usr/bin/env python3
"""Precompute lexicon phrase matcher artifacts for an existing corpus directory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from surf_rag.entity_matching.artifacts import build_entity_matching_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing alias_map.json and entity_lexicon.parquet",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if entity_matching_manifest.json exists.",
    )
    args = ap.parse_args()
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
