from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.merge import merge_filter_benchmark_args

from surf_rag.benchmark.corpus_filter import (
    filter_benchmark_rows,
    iter_jsonl,
    write_jsonl,
)

logger = logging.getLogger(__name__)


def _default_backup_path(benchmark_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return benchmark_path.with_name(
        f"{benchmark_path.stem}.backup.{stamp}{benchmark_path.suffix}"
    )


def main() -> int:
    load_app_env()
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Filter benchmark.jsonl by checking gold_support_sentences against corpus.jsonl."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (see configs/templates/pipeline.yaml)",
    )
    parser.add_argument(
        "--benchmark",
        default=os.getenv("BENCHMARK_PATH", "data/processed/benchmark.jsonl"),
        help="Path to benchmark JSONL to filter in-place.",
    )
    parser.add_argument(
        "--corpus",
        default=os.getenv("CORPUS_PATH", "data/processed/corpus.jsonl"),
        help="Path to corpus JSONL used for support sentence matching.",
    )
    parser.add_argument(
        "--backup",
        default=None,
        help="Optional explicit backup output path for the original benchmark.",
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        merge_filter_benchmark_args(args, cfg)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    benchmark_path = Path(args.benchmark)
    corpus_path = Path(args.corpus)

    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    backup_path = (
        Path(args.backup) if args.backup else _default_backup_path(benchmark_path)
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(benchmark_path, backup_path)
    logger.info("Backed up original benchmark to %s", backup_path)

    benchmark_rows = list(iter_jsonl(benchmark_path))
    corpus_rows = list(iter_jsonl(corpus_path))
    kept_rows, stats = filter_benchmark_rows(benchmark_rows, corpus_rows)
    write_jsonl(benchmark_path, kept_rows)

    logger.info(
        "Filtered benchmark: total=%d kept=%d dropped=%d",
        stats.total,
        stats.kept,
        stats.dropped,
    )
    if stats.dropped_by_source:
        for source in sorted(stats.dropped_by_source):
            logger.info(
                "Dropped %d row(s) from source '%s'",
                stats.dropped_by_source[source],
                source,
            )
    logger.info("Wrote filtered benchmark to %s", benchmark_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
