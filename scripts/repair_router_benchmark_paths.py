"""Rewrite router oracle/dataset manifests to canonical benchmark bundle paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.evaluation.repair_router_benchmark_paths import (
    repair_router_bundle_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Point router oracle/dataset manifests at BENCHMARK_BASE/<name>/<id>."
    )
    p.add_argument("--router-id", required=True)
    p.add_argument("--benchmark-name", required=True)
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--router-base", type=Path, default=None)
    p.add_argument("--benchmark-base", type=Path, default=None)
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write files (default is dry-run).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    import logging

    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    summary = repair_router_bundle_metadata(
        args.router_id,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        router_base=args.router_base,
        benchmark_base=args.benchmark_base,
        dry_run=not args.apply,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
