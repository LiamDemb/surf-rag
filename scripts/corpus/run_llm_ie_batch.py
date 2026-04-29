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
DEFAULT_MAX_ATTEMPTS = 3

STATE_FILENAMES = ("batch_state_ie.json", "batch_state_ie.json")
OUTPUT_FILENAMES = ("corpus_llm_ie.jsonl", "corpus_llm_ie.jsonl")
SUBMIT_SCRIPT = "submit_llm_ie_batch.py"
COLLECT_SCRIPT = "collect_llm_ie_batch.py"
RETRY_REPORT_FILENAME = "ie_retry_report.json"
FINAL_FAILURES_FILENAME = "ie_failures_final.json"


def _run_script(name: str, *args: str) -> int:
    script_dir = Path(__file__).resolve().parent
    script = script_dir / name
    cmd = [sys.executable, str(script), *args]
    return subprocess.run(cmd).returncode


def _write_chunk_ids_file(path: Path, chunk_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(f"{cid}\n")


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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum IE retry attempts before failing.",
    )
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

    max_attempts = max(1, int(args.max_attempts))
    pending_ids_path = output_dir / "ie_pending_ids.txt"
    retry_report_path = output_dir / RETRY_REPORT_FILENAME
    final_failures_path = output_dir / FINAL_FAILURES_FILENAME
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    unresolved_ids: list[str] | None = None
    for attempt in range(1, max_attempts + 1):
        submit_args = [
            "--corpus",
            str(corpus_path),
            "--output-dir",
            str(output_dir),
            "--attempt",
            str(attempt),
        ]
        if unresolved_ids is not None:
            if not unresolved_ids:
                logger.info(
                    "No unresolved chunk IDs remain after attempt %d.", attempt - 1
                )
                return 0
            _write_chunk_ids_file(pending_ids_path, unresolved_ids)
            submit_args.extend(["--only-chunk-ids", str(pending_ids_path)])

        logger.info("Submitting IE batch attempt %d/%d...", attempt, max_attempts)
        if _run_script(SUBMIT_SCRIPT, *submit_args) != 0:
            logger.error("Submit failed on attempt %d.", attempt)
            return 1

        state_path = None
        for name in STATE_FILENAMES:
            p = output_dir / name
            if p.is_file():
                state_path = p
                break
        if not state_path or not state_path.is_file():
            logger.error("State file not found after submit on attempt %d.", attempt)
            return 1

        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        shards = state.get("shards") or []
        if not shards:
            logger.info(
                "No shards submitted on attempt %d. Corpus is up to date.", attempt
            )
            return 0

        if args.no_wait:
            logger.info("--no-wait enabled. Submitted attempt %d and exiting.", attempt)
            return 0

        for shard in shards:
            batch_id = shard.get("batch_id")
            if not batch_id:
                continue
            b = client.batches.retrieve(batch_id)
            if b.status != "completed":
                _wait_for_batch(
                    client, batch_id, args.poll_seconds, args.timeout_seconds
                )

        attempt_output_path = output_dir / f"corpus_llm_ie_attempt_{attempt:02d}.jsonl"
        logger.info("Collecting batch results for attempt %d...", attempt)
        if (
            _run_script(
                COLLECT_SCRIPT,
                "--state",
                str(state_path),
                "--corpus",
                str(corpus_path),
                "--output",
                str(attempt_output_path),
                "--output-dir",
                str(output_dir),
                "--retry-report",
                str(retry_report_path),
            )
            != 0
        ):
            logger.error("Collect failed on attempt %d.", attempt)
            return 1

        backup_path = corpus_path.parent / (corpus_path.stem + ".bak.jsonl")
        if corpus_path.exists():
            corpus_path.rename(backup_path)
        attempt_output_path.rename(corpus_path)
        if backup_path.exists():
            backup_path.unlink()
        logger.info("Merged attempt %d results into %s", attempt, corpus_path)

        if not retry_report_path.is_file():
            logger.error("Retry report not found after collect: %s", retry_report_path)
            return 1
        report = json.loads(retry_report_path.read_text(encoding="utf-8"))
        unresolved_ids = [
            str(cid).strip()
            for cid in (report.get("unresolved_ids") or [])
            if str(cid).strip()
        ]
        if not unresolved_ids:
            logger.info("All IE chunk requests resolved by attempt %d.", attempt)
            return 0
        logger.warning(
            "Attempt %d completed with %d unresolved chunk(s).",
            attempt,
            len(unresolved_ids),
        )

    payload = {
        "max_attempts": max_attempts,
        "remaining_unresolved_ids": unresolved_ids or [],
    }
    final_failures_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.error(
        "IE extraction exhausted retry budget (%d attempts). Final failures: %s",
        max_attempts,
        final_failures_path,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
