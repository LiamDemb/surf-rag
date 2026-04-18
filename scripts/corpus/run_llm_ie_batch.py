"""Orchestrate LLM information extraction batch: submit -> wait -> collect -> replace corpus.

Resume-safe: skips submit if batch files exist, skips collect if output exists.
Waits for all shards by default. Use --no-wait to exit after submit.

Migration: checks batch_state_ie.json first, then batch_state_ie.json.
Checks corpus_llm_ie.jsonl first, then corpus_llm_ie.jsonl.

Usage:
    poetry run python scripts/corpus/run_llm_ie_batch.py --corpus data/processed/corpus.jsonl
    poetry run python scripts/corpus/run_llm_ie_batch.py --corpus data/processed/corpus.jsonl --no-wait
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 600
DEFAULT_TIMEOUT_SECONDS = 48 * 3600

STATE_FILENAMES = ("batch_state_ie.json", "batch_state_ie.json")
OUTPUT_FILENAMES = ("corpus_llm_ie.jsonl", "corpus_llm_ie.jsonl")
SUBMIT_SCRIPT = "submit_llm_ie_batch.py"
COLLECT_SCRIPT = "collect_llm_ie_batch.py"


def _run_script(name: str, *args: str) -> int:
    script_dir = Path(__file__).resolve().parent
    script = script_dir / name
    cmd = [sys.executable, str(script), *args]
    return subprocess.run(cmd).returncode


def _wait_for_batch(
    client: OpenAI, batch_id: str, poll_seconds: int, timeout_seconds: int
) -> bool:
    start = time.monotonic()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", "unknown")
        if status == "completed":
            logger.info("Batch %s completed.", batch_id)
            return True
        if status in ("failed", "cancelled", "expired"):
            logger.error("Batch %s ended with status: %s", batch_id, status)
            return False
        elapsed = time.monotonic() - start
        if elapsed >= timeout_seconds:
            logger.error("Batch %s timed out.", batch_id)
            return False
        next_poll = min(poll_seconds, max(1, int(timeout_seconds - elapsed)))
        logger.info(
            "Batch %s status=%s (elapsed %.1f min). Polling in %d s...",
            batch_id,
            status,
            elapsed / 60,
            next_poll,
        )
        time.sleep(next_poll)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate LLM IE extraction batch.",
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for batches; exit after submit.",
    )
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wiki_titles_path = output_dir / "wiki_titles.jsonl"
    if not wiki_titles_path.is_file():
        logger.info("Building wiki_titles.jsonl from corpus...")
        if (
            _run_script(
                "build_wiki_title_seed.py",
                "--corpus",
                str(corpus_path),
                "--output-dir",
                str(output_dir),
            )
            != 0
        ):
            logger.error("Failed to build wiki_titles.jsonl.")
            return 1

    state_path = None
    for name in STATE_FILENAMES:
        p = output_dir / name
        if p.is_file():
            state_path = p
            break
    if not state_path or not state_path.is_file():
        logger.info("Submitting IE batch...")
        if (
            _run_script(
                SUBMIT_SCRIPT,
                "--corpus",
                str(corpus_path),
                "--output-dir",
                str(output_dir),
            )
            != 0
        ):
            logger.error("Submit failed.")
            return 1
        state_path = output_dir / STATE_FILENAMES[0]
        if not state_path.is_file():
            state_path = output_dir / STATE_FILENAMES[1]

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    shards = state.get("shards") or []
    if not shards:
        logger.info(
            "No shards to process (all chunks already extracted). Corpus is up to date."
        )
        return 0

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not args.no_wait:
        all_ok = True
        for shard in shards:
            batch_id = shard.get("batch_id")
            if not batch_id:
                continue
            b = client.batches.retrieve(batch_id)
            if b.status != "completed":
                if not _wait_for_batch(
                    client, batch_id, args.poll_seconds, args.timeout_seconds
                ):
                    all_ok = False
        if not all_ok:
            logger.error("One or more batches failed or timed out.")
            return 1

    output_path = None
    for name in OUTPUT_FILENAMES:
        p = output_dir / name
        if p.is_file():
            output_path = p
            break
    if output_path is None:
        output_path = output_dir / OUTPUT_FILENAMES[0]
        logger.info("Collecting batch results...")
        if (
            _run_script(
                COLLECT_SCRIPT,
                "--state",
                str(state_path),
                "--corpus",
                str(corpus_path),
                "--output",
                str(output_path),
                "--output-dir",
                str(output_dir),
            )
            != 0
        ):
            logger.error("Collect failed.")
            return 1
    else:
        logger.info("Output already exists at %s. Skipping collect.", output_path)

    backup_path = corpus_path.parent / (corpus_path.stem + ".bak.jsonl")
    if corpus_path.exists():
        corpus_path.rename(backup_path)
    output_path.rename(corpus_path)
    if backup_path.exists():
        backup_path.unlink()
    logger.info("Replaced %s with IE-enriched corpus.", corpus_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
