"""Submit OpenAI Batch API job(s) for LLM information extraction (entities + triples).

Supports sharding: when request count or file size exceeds limits, creates
multiple batch jobs. State file tracks all shards for collection.

Usage:
    poetry run python scripts/corpus/submit_llm_ie_batch.py --corpus data/processed/corpus.jsonl
    poetry run python scripts/corpus/submit_llm_ie_batch.py --corpus data/processed/corpus.jsonl --limit 50

Requires: OPENAI_API_KEY, wiki_titles.jsonl, alias_map.json (for FlashText matcher).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from surf_rag.core.canonical_clean import normalize_text_for_extraction
from surf_rag.core.llm_ie import (
    build_ie_batch_line,
    build_ie_chat_request,
)
from surf_rag.core.wiki_title_matcher import build_wiki_title_matcher, load_wiki_titles
from surf_rag.core.openai_batch_limits import batch_limit_requests
from surf_rag.core.prompts import get_ie_extraction_prompt

load_dotenv()

logger = logging.getLogger(__name__)

BATCH_LIMIT_BYTES = 200 * 1024 * 1024  # 200 MB
SHARD_PREFIX = "batch_input_ie"
STATE_FILENAME = "batch_state_ie.json"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit OpenAI Batch API job(s) for LLM IE extraction (with sharding).",
    )
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl.")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory for batch files and state.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max chunks to include."
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Batch completion window (default: 24h).",
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
        logger.error(
            "wiki_titles.jsonl not found at %s. Run build_wiki_title_seed first.",
            wiki_titles_path,
        )
        return 1

    titles = load_wiki_titles(wiki_titles_path)
    alias_map: dict = {}
    alias_path = output_dir / "alias_map.json"
    if alias_path.is_file():
        with alias_path.open("r", encoding="utf-8") as f:
            alias_map = json.load(f)
    try:
        matcher = build_wiki_title_matcher(titles, alias_map=alias_map)
    except ImportError as e:
        logger.error("FlashText required for wiki title matching: %s", e)
        return 1

    prompt_template = get_ie_extraction_prompt()

    all_chunks = list(_iter_jsonl(corpus_path))
    if args.limit is not None:
        all_chunks = all_chunks[: args.limit]
    if not all_chunks:
        logger.error("No chunks to process.")
        return 1

    # Filter out already-extracted chunks (ie_extracted flag from build_corpus)
    chunks = [
        c for c in all_chunks if not c.get("metadata", {}).get("ie_extracted", False)
    ]
    skipped = len(all_chunks) - len(chunks)
    if skipped:
        logger.info(
            "Skipping %d already-extracted chunks (%d remaining for batch).",
            skipped,
            len(chunks),
        )
    if not chunks:
        logger.info("All chunks already extracted. Nothing to submit.")
        # Write empty state so downstream scripts can detect no work was needed
        state = {"shards": [], "corpus_path": str(corpus_path), "total_requests": 0}
        state_path = output_dir / STATE_FILENAME
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        return 0

    limit_requests = batch_limit_requests()
    shards: list[dict] = []
    shard_idx = 0
    shard_count = 0
    shard_bytes = 0
    shard_file = output_dir / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"

    def _flush_shard(batch_out):
        nonlocal shard_idx, shard_count, shard_bytes, shard_file
        if shard_count == 0:
            return output_dir / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
        batch_out.close()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("Uploading shard %d (%d requests)...", shard_idx, shard_count)
        with shard_file.open("rb") as r:
            uploaded = client.files.create(file=r, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window=args.completion_window,
            metadata={
                "description": "LLM IE extraction",
                "corpus": str(corpus_path),
                "shard": str(shard_idx),
            },
        )
        shards.append(
            {
                "batch_id": batch.id,
                "input_path": str(shard_file),
                "request_count": shard_count,
            }
        )
        logger.info("Shard %d batch created: %s", shard_idx, batch.id)
        shard_idx += 1
        shard_count = 0
        shard_bytes = 0
        shard_file = output_dir / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
        return shard_file

    batch_out = shard_file.open("w", encoding="utf-8")
    try:
        for chunk in chunks:
            chunk_id = (
                chunk.get("chunk_id")
                or f"chunk_{len(shards) * limit_requests + shard_count}"
            )
            text_raw = chunk.get("text") or ""
            text_for_extraction = normalize_text_for_extraction(text_raw)
            title = chunk.get("title") or "N/A"

            seed_titles = matcher.find_titles_in_text(
                text_for_extraction, max_results=50
            )
            seed_str = (
                "\n".join(f"- {s}" for s in seed_titles)
                if seed_titles
                else "(none detected)"
            )

            prompt = prompt_template.format(
                title=title,
                text=text_for_extraction,
                seed_titles_in_chunk=seed_str,
            )

            body = build_ie_chat_request(prompt)
            batch_line = build_ie_batch_line(custom_id=chunk_id, body=body)
            line_json = json.dumps(batch_line, ensure_ascii=False) + "\n"
            line_bytes = len(line_json.encode("utf-8"))

            if shard_count > 0 and (
                shard_count >= limit_requests
                or shard_bytes + line_bytes > BATCH_LIMIT_BYTES
            ):
                next_file = _flush_shard(batch_out)
                batch_out = next_file.open("w", encoding="utf-8")

            batch_out.write(line_json)
            shard_count += 1
            shard_bytes += line_bytes
    finally:
        if shard_count > 0:
            _flush_shard(batch_out)
        else:
            batch_out.close()

    state = {
        "shards": shards,
        "corpus_path": str(corpus_path),
        "total_requests": sum(s["request_count"] for s in shards),
    }
    state_path = output_dir / STATE_FILENAME
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    logger.info(
        "Submitted %d shard(s), %d total requests. State: %s",
        len(shards),
        state["total_requests"],
        state_path,
    )
    for s in shards:
        print(s["batch_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
