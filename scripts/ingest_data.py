from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.merge import merge_ingest_args
from surf_rag.benchmark.pipeline_audit import (
    resolve_pipeline_run_id,
    write_pipeline_step_report,
)
from surf_rag.core.loaders import load_2wiki, load_hotpotqa, load_nq
from surf_rag.core.schemas import (
    BenchmarkItem,
    parse_benchmark_support_fields,
    sha256_text,
)

logger = logging.getLogger(__name__)


def normalize_question(text: str) -> str:
    return " ".join(text.lower().strip().split())


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def validate_outputs(benchmark: List[BenchmarkItem]) -> None:
    if not benchmark:
        raise ValueError("Benchmark is empty.")

    sources = {item.dataset_source for item in benchmark}
    if not sources:
        raise ValueError("Benchmark has no dataset sources.")

    question_ids = set()
    question_texts = {}
    for item in benchmark:
        if not item.question_id or not item.question:
            raise ValueError("Benchmark contains empty question or question_id.")
        if not item.gold_answers:
            raise ValueError("Benchmark contains missing gold_answers.")
        if item.question_id in question_ids:
            raise ValueError("Duplicate question_id found in benchmark.")
        question_ids.add(item.question_id)

        normalized = normalize_question(item.question)
        normalized_hash = sha256_text(normalized)
        if normalized_hash in question_texts:
            pass  # duplicate normalized text is allowed across unassigned items
        else:
            question_texts[normalized_hash] = True


def _load_existing_benchmark(path: Path) -> List[BenchmarkItem]:
    """Load an existing benchmark.jsonl, returning a list of BenchmarkItems."""
    items: List[BenchmarkItem] = []
    if not path.is_file():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sentences, titles, sent_ids = parse_benchmark_support_fields(row)
            items.append(
                BenchmarkItem(
                    question_id=row["question_id"],
                    question=row["question"],
                    gold_answers=row["gold_answers"],
                    dataset_source=row["dataset_source"],
                    gold_support_sentences=sentences,
                    gold_support_titles=titles,
                    gold_support_sent_ids=sent_ids,
                    dataset_version=row.get("dataset_version"),
                )
            )
    return items


def main() -> int:
    load_app_env()
    load_dotenv()
    parser = argparse.ArgumentParser(description="Phase 1 ingestion pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (see configs/templates/pipeline.yaml)",
    )
    parser.add_argument(
        "--nq",
        default=os.getenv("NQ_PATH"),
        help="Path to NQ JSON/JSONL.",
    )
    parser.add_argument(
        "--2wiki",
        dest="wiki2",
        default=os.getenv("2WIKI_PATH"),
        help="Path to 2WikiMultiHopQA JSON/JSONL.",
    )
    parser.add_argument(
        "--hotpotqa",
        default=os.getenv("HOTPOTQA_PATH"),
        help="Path to HotPotQA JSON/JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory for processed artifacts.",
    )
    parser.add_argument(
        "--pipeline-run-id",
        default=os.getenv("PIPELINE_RUN_ID"),
        help="Optional shared run id for cross-step benchmark count reporting.",
    )
    parser.add_argument("--nq-version", default=os.getenv("NQ_VERSION"))
    parser.add_argument(
        "--2wiki-version", dest="wiki2_version", default=os.getenv("2WIKI_VERSION")
    )
    parser.add_argument(
        "--hotpotqa-version",
        default=os.getenv("HOTPOTQA_VERSION"),
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_pipeline_config(args.config.resolve())
        apply_pipeline_env_from_config(cfg)
        merge_ingest_args(args, cfg)

    dataset_paths = [
        args.nq,
        args.wiki2,
        getattr(args, "hotpotqa", None),
    ]
    if not any(p and str(p).strip() for p in dataset_paths):
        raise ValueError(
            "Provide at least one dataset path via CLI or env. "
            "Example: --2wiki data/raw/2wikimultihop_50.jsonl or --hotpotqa data/raw/hotpotqa.jsonl"
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load existing benchmark (if any) to preserve prior questions
    output_dir = Path(args.output_dir)
    benchmark_path = output_dir / "benchmark.jsonl"
    existing_items = _load_existing_benchmark(benchmark_path)
    seen_ids: set = {item.question_id for item in existing_items}
    if existing_items:
        logger.info(
            "Found existing benchmark with %d questions – will skip duplicates.",
            len(existing_items),
        )

    # Load new items from source datasets, filtering out seen ids
    new_by_source: Dict[str, List[BenchmarkItem]] = defaultdict(list)

    if args.nq and str(args.nq).strip():
        for item in load_nq(args.nq, dataset_version=args.nq_version):
            if item.question_id not in seen_ids:
                new_by_source[item.dataset_source].append(item)
        logger.info("Loaded NQ from %s", args.nq)

    if args.wiki2 and str(args.wiki2).strip():
        for item in load_2wiki(args.wiki2, dataset_version=args.wiki2_version):
            if item.question_id not in seen_ids:
                new_by_source[item.dataset_source].append(item)
        logger.info("Loaded 2WikiMultiHopQA from %s", args.wiki2)

    hotpot_path = getattr(args, "hotpotqa", None)
    hotpot_ver = getattr(args, "hotpotqa_version", None)
    if hotpot_path and str(hotpot_path).strip():
        for item in load_hotpotqa(hotpot_path, dataset_version=hotpot_ver):
            if item.question_id not in seen_ids:
                new_by_source[item.dataset_source].append(item)
        logger.info("Loaded HotPotQA from %s", hotpot_path)

    novel_count = sum(len(v) for v in new_by_source.values())
    logger.info("Novel questions to add: %d", novel_count)

    if novel_count == 0:
        logger.info("No new questions to ingest. Benchmark unchanged.")
        return 0

    sources_loaded = sorted(new_by_source.keys())
    logger.info("New datasets included: %s", ", ".join(sources_loaded))

    # New items (train/dev/test split is applied later in build_router_dataset)
    new_items: List[BenchmarkItem] = []
    for source, items in new_by_source.items():
        for item in items:
            if not item.gold_support_sentences:
                raise ValueError(
                    "Newly ingested benchmark item is missing gold_support_sentences."
                )
            new_items.append(
                BenchmarkItem(
                    question_id=item.question_id,
                    question=item.question,
                    gold_answers=item.gold_answers,
                    dataset_source=item.dataset_source,
                    gold_support_sentences=item.gold_support_sentences,
                    gold_support_titles=item.gold_support_titles,
                    gold_support_sent_ids=item.gold_support_sent_ids,
                    dataset_version=item.dataset_version,
                )
            )
        logger.info("Queued %d new items from source '%s'.", len(items), source)

    # Combine existing + new items
    combined = existing_items + new_items
    validate_outputs(combined)

    write_jsonl(benchmark_path, (item.to_json() for item in combined))

    logger.info(
        "Benchmark size: %d (existing=%d, added=%d)",
        len(combined),
        len(existing_items),
        len(new_items),
    )
    logger.info("Wrote %s", benchmark_path)
    run_id = resolve_pipeline_run_id(args.pipeline_run_id)
    report_path = write_pipeline_step_report(
        benchmark_path=benchmark_path,
        step_name="ingest",
        before=len(existing_items),
        after=len(combined),
        run_id=run_id,
        details={
            "existing": len(existing_items),
            "added": len(new_items),
            "novel_questions": novel_count,
            "new_by_source": {k: len(v) for k, v in sorted(new_by_source.items())},
        },
    )
    logger.info("Wrote pipeline counts report: %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
