"""Collect OpenAI Batch API results for LLM IE extraction and merge into corpus.

Downloads output from one or more batch shards, parses entities+triples,
and writes corpus_llm_ie.jsonl with metadata.entities and metadata.relations
populated. Run after all shards have completed.

Usage:
    poetry run python scripts/corpus/collect_llm_ie_batch.py --state data/processed/batch_state_ie.json
    poetry run python scripts/corpus/collect_llm_ie_batch.py --state data/processed/batch_state_ie.json --corpus data/processed/corpus.jsonl

Migration: --state also accepts batch_state_ie.json from prior runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from surf_rag.core.llm_ie import (
    _post_process_ie,
    parse_ie_batch_output_line,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "corpus_llm_ie.jsonl"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect LLM IE batch results and merge into corpus.",
    )
    parser.add_argument(
        "--state",
        required=True,
        help="Path to batch_state_ie.json (or batch_state_ie.json) from submit script.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Path to corpus.jsonl (default: from state).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output path (default: <output-dir>/{DEFAULT_OUTPUT_FILENAME}).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    state_path = Path(args.state)
    if not state_path.is_file():
        logger.error("State file not found: %s", state_path)
        return 1

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    shards = state.get("shards") or []
    if not shards:
        logger.error("No shards in state file.")
        return 1

    corpus_path = Path(args.corpus or state.get("corpus_path", ""))
    if not corpus_path.is_file():
        logger.error("Corpus file not found: %s", corpus_path)
        return 1

    output_dir = Path(args.output_dir)
    output_path = (
        Path(args.output) if args.output else output_dir / DEFAULT_OUTPUT_FILENAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)

    raw_by_id: dict[str, tuple[list, list]] = {}
    completed = 0
    failed = 0

    for shard in shards:
        batch_id = shard.get("batch_id")
        if not batch_id:
            continue
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            logger.warning(
                "Shard batch %s not completed (status=%s). Skipping.",
                batch_id,
                batch.status,
            )
            continue

        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file", None
        )
        if not output_file_id:
            logger.warning("Shard batch %s has no output file.", batch_id)
            continue

        logger.info("Downloading shard output for %s...", batch_id)
        content = client.files.content(output_file_id)
        text = content.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        for line in text.strip().split("\n"):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                failed += 1
                raw_by_id[obj.get("custom_id", "")] = ([], [])
                continue
            completed += 1
            cid, raw_entities, raw_triples = parse_ie_batch_output_line(obj)
            raw_by_id[cid] = (raw_entities, raw_triples)

        error_file_id = getattr(batch, "error_file_id", None) or getattr(
            batch, "error_file", None
        )
        if error_file_id:
            try:
                err_content = client.files.content(error_file_id)
                err_text = err_content.read()
                if isinstance(err_text, bytes):
                    err_text = err_text.decode("utf-8")
                for line in err_text.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    failed += 1
                    raw_by_id[obj.get("custom_id", "")] = ([], [])
            except Exception as e:
                logger.warning("Could not fetch error file for %s: %s", batch_id, e)

    chunk_list = list(_iter_jsonl(corpus_path))
    empty_entities = 0
    empty_relations = 0
    total_entities = 0
    total_relations = 0

    logger.info("Merging entities and relations into corpus...")
    with output_path.open("w", encoding="utf-8") as out:
        for chunk in chunk_list:
            chunk_id = chunk.get("chunk_id") or ""
            chunk = dict(chunk)
            meta = chunk.setdefault("metadata", {})

            if chunk_id in raw_by_id:
                # This chunk was in the current batch – apply new extraction
                raw_entities, raw_triples = raw_by_id[chunk_id]
                text = chunk.get("text") or ""
                entities, relations = _post_process_ie(
                    raw_entities, raw_triples, text, chunk_id
                )
                meta["entities"] = entities
                meta["relations"] = relations
            elif meta.get("ie_extracted"):
                # Already extracted in a prior run – keep existing data
                entities = meta.get("entities", [])
                relations = meta.get("relations", [])
            else:
                # Not in batch and not cached – leave empty
                entities = []
                relations = []
                meta["entities"] = entities
                meta["relations"] = relations

            meta["ie_extracted"] = True

            if not entities:
                empty_entities += 1
            if not relations:
                empty_relations += 1
            total_entities += len(entities)
            total_relations += len(relations)

            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("Wrote %s", output_path)
    logger.info(
        "Summary: completed=%d failed=%d; chunks with 0 entities=%d, 0 relations=%d; total entities=%d, relations=%d",
        completed,
        failed,
        empty_entities,
        empty_relations,
        total_entities,
        total_relations,
    )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
