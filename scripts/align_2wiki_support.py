"""Align 2Wiki gold support sentences to cleaned Wikipedia sentences (title-localized).

Runs after ``fetch-wikipedia-articles`` and before ``build-corpus``. Candidate
sentences are built from DocStore HTML using the same clean+chunk path as
corpus generation. Rewrites ``benchmark.jsonl`` in place after a timestamped
backup; can drop rows whose gold support cannot be fully resolved.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.merge import merge_align_2wiki_args

from surf_rag.benchmark.corpus_filter import iter_jsonl
from surf_rag.benchmark.align_2wiki import run_2wiki_support_alignment
from surf_rag.benchmark.pipeline_audit import (
    resolve_pipeline_run_id,
    write_pipeline_step_report,
)

logger = logging.getLogger(__name__)


def _default_backup_path(benchmark_path: Path) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return benchmark_path.with_name(
        f"{benchmark_path.stem}.backup.{stamp}{benchmark_path.suffix}"
    )


def _default_report_path(benchmark_path: Path) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return benchmark_path.with_name(
        f"{benchmark_path.stem}.support_alignment.{stamp}.md"
    )


def main() -> int:
    load_app_env()
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Align 2Wiki gold support sentences to DocStore-backed article text by Wikipedia title."
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
        help="Path to benchmark JSONL (rewritten in place).",
    )
    parser.add_argument(
        "--docstore",
        default=os.getenv("DOCSTORE_PATH", "data/processed/docstore.sqlite"),
        help="DocStore SQLite path (preferred source for candidate sentences).",
    )
    parser.add_argument(
        "--corpus",
        default=os.getenv("CORPUS_PATH") or None,
        help="Optional legacy: use corpus.jsonl chunk text instead of DocStore.",
    )
    parser.add_argument(
        "--backup",
        default=None,
        help="Optional explicit backup path for the original benchmark.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional explicit path for the markdown replacement report.",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"),
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--tau-sem",
        type=float,
        default=float(os.getenv("ALIGN_TAU_SEM", "0.9")),
        help="Minimum semantic cosine (normalized embeddings).",
    )
    parser.add_argument(
        "--tau-lex",
        type=float,
        default=float(os.getenv("ALIGN_TAU_LEX", "0.3")),
        help="Minimum ROUGE-L F1 vs original sentence.",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help=(
            "Write markdown for every 2Wiki support line. If a line was not replaced "
            "and was not found exactly, include the nearest candidate "
            "sentence with semantic cosine and ROUGE-L."
        ),
    )
    parser.add_argument(
        "--keep-unresolved",
        action="store_true",
        help="Keep 2Wiki rows even when some gold support lines cannot be aligned (default: drop them).",
    )
    parser.add_argument(
        "--chunk-min-tokens",
        type=int,
        default=int(os.getenv("CHUNK_MIN_TOKENS", "500")),
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=int(os.getenv("CHUNK_MAX_TOKENS", "800")),
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
    )
    parser.add_argument(
        "--pipeline-run-id",
        default=os.getenv("PIPELINE_RUN_ID"),
        help="Optional shared run id for cross-step benchmark count reporting.",
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        merge_align_2wiki_args(args, cfg)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    benchmark_path = Path(args.benchmark)
    if args.corpus and Path(args.corpus).is_file():
        corpus_path = Path(args.corpus)
        docstore_path = None
    else:
        corpus_path = None
        docstore_path = Path(args.docstore) if args.docstore else None
    backup_path = (
        Path(args.backup) if args.backup else _default_backup_path(benchmark_path)
    )
    report_path = (
        Path(args.report) if args.report else _default_report_path(benchmark_path)
    )

    before_count = (
        len(list(iter_jsonl(benchmark_path))) if benchmark_path.is_file() else 0
    )
    stats = run_2wiki_support_alignment(
        benchmark_path,
        backup_path=backup_path,
        report_path=report_path,
        docstore_path=docstore_path,
        corpus_path=corpus_path,
        model_name=args.model_name,
        tau_sem=args.tau_sem,
        tau_lex=args.tau_lex,
        full_report=args.full_report,
        drop_unresolved=not args.keep_unresolved,
        chunk_min_tokens=args.chunk_min_tokens,
        chunk_max_tokens=args.chunk_max_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
    )
    after_count = len(list(iter_jsonl(benchmark_path)))
    run_id = resolve_pipeline_run_id(args.pipeline_run_id)
    counts_report_path = write_pipeline_step_report(
        benchmark_path=benchmark_path,
        step_name="align_2wiki_support",
        before=before_count,
        after=after_count,
        run_id=run_id,
        details={
            "total_rows": stats["total_rows"],
            "two_wiki_rows": stats["two_wiki_rows"],
            "two_wiki_kept": stats["two_wiki_kept"],
            "two_wiki_dropped": stats["two_wiki_dropped"],
            "facts_replaced": stats["facts_replaced"],
            "skipped_no_provenance": stats["skipped_no_provenance"],
            "drop_unresolved": not args.keep_unresolved,
        },
    )
    logger.info("Wrote pipeline counts report: %s", counts_report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
