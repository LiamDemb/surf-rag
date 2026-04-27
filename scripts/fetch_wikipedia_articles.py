"""Warm DocStore with Wikipedia HTML for all benchmark-backed 2Wiki (and NQ) samples."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.merge import merge_fetch_wikipedia_args
from tqdm.auto import tqdm

from surf_rag.core.benchmark_samples import (
    build_samples,
    load_benchmark_by_source,
    resolve_raw_paths_for_benchmark_sources,
)
from surf_rag.core.corpus_acquisition import (
    Budgets,
    ingest_2wiki,
    ingest_nq,
    supporting_titles_from_2wiki_sample,
)
from surf_rag.core.docstore import DocStore
from surf_rag.core.wikipedia_client import WikipediaClient

logger = logging.getLogger(__name__)


def main() -> int:
    load_app_env()
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Wikipedia HTML for benchmark questions into DocStore (idempotent; "
            "cached titles are skipped)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (see configs/templates/pipeline.yaml)",
    )
    parser.add_argument("--benchmark", default=os.getenv("BENCHMARK_PATH"))
    parser.add_argument("--nq", default=os.getenv("NQ_PATH"))
    parser.add_argument("--2wiki", dest="wiki2", default=os.getenv("2WIKI_PATH"))
    parser.add_argument(
        "--output-dir", default=os.getenv("OUTPUT_DIR", "data/processed")
    )
    parser.add_argument(
        "--docstore",
        default=os.getenv("DOCSTORE_PATH", "data/processed/docstore.sqlite"),
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path for fetch summary JSON (default: output-dir/wikipedia_fetch_summary.json).",
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        merge_fetch_wikipedia_args(args, cfg)

    if not args.benchmark or not str(args.benchmark).strip():
        raise ValueError("BENCHMARK_PATH is required (--benchmark or env).")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_by_source = load_benchmark_by_source(args.benchmark)
    sources_in_benchmark = set(benchmark_by_source.keys())
    paths_by_source, missing = resolve_raw_paths_for_benchmark_sources(
        sources_in_benchmark,
        nq_path=args.nq,
        wiki2_path=args.wiki2,
    )
    if missing:
        raise ValueError(
            f"Benchmark contains sources {sorted(sources_in_benchmark)} but paths are missing "
            f"for: {', '.join(missing)}."
        )

    samples = build_samples(benchmark_by_source, paths_by_source)
    budgets = Budgets()
    docstore = DocStore(args.docstore)
    wiki = WikipediaClient()
    if wiki.oauth2_authenticated:
        logger.info("Wikipedia Action API uses OAuth2 (authenticated rate limits)")
    elif "2wiki" in paths_by_source:
        logger.warning(
            "WIKIMEDIA_OAUTH2_ACCESS_TOKEN is not set; unauthenticated limits may cause 429s"
        )

    stats = {
        "questions_total": len(samples),
        "questions_2wiki": sum(1 for s in samples if s["source"] == "2wiki"),
        "questions_nq": sum(1 for s in samples if s["source"] == "nq"),
        "title_cache_hits": 0,
        "title_fetches": 0,
    }

    for sample in tqdm(samples, desc="Fetch Wikipedia", unit="question"):
        if sample["source"] == "2wiki":
            titles = supporting_titles_from_2wiki_sample(sample)
            cached_before = [docstore.get(f"title:{t}") is not None for t in titles]
            ingest_2wiki(sample, budgets, docstore, wiki)
            for was_cached in cached_before:
                if was_cached:
                    stats["title_cache_hits"] += 1
                else:
                    stats["title_fetches"] += 1
        else:
            ingest_nq(sample, budgets, docstore, wiki)

    docstore.close()

    summary_path = (
        Path(args.summary)
        if args.summary
        else output_dir / "wikipedia_fetch_summary.json"
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(Path(args.benchmark).resolve()),
        "docstore": str(Path(args.docstore).resolve()),
        **stats,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote fetch summary: %s", summary_path)
    logger.info(
        "Done: %d questions, %d title cache hits, %d title API fetches",
        stats["questions_total"],
        stats["title_cache_hits"],
        stats["title_fetches"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
