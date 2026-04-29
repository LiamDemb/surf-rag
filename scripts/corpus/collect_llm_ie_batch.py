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
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from surf_rag.core.llm_ie import (
    _post_process_ie,
    parse_ie_batch_output_line,
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILENAME = "corpus_llm_ie.jsonl"
RETRY_REPORT_FILENAME = "ie_retry_report.json"
IE_STATUS_SUCCESS = "success"
IE_STATUS_FAILED = "failed"
IE_STATUS_PENDING = "pending"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_ie_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    entities = meta.get("entities", [])
    relations = meta.get("relations", [])
    has_extraction = bool(entities) or bool(relations)
    status = str(meta.get("ie_status", "")).strip().lower()
    if status not in (IE_STATUS_SUCCESS, IE_STATUS_FAILED, IE_STATUS_PENDING):
        if has_extraction or bool(meta.get("ie_extracted", False)):
            status = IE_STATUS_SUCCESS
        else:
            status = IE_STATUS_PENDING
    attempts = int(meta.get("ie_attempts", 0) or 0)
    if attempts < 0:
        attempts = 0
    last_error = meta.get("ie_last_error")
    if status == IE_STATUS_SUCCESS:
        last_error = None
    meta["ie_status"] = status
    meta["ie_attempts"] = attempts
    meta["ie_last_error"] = None if last_error is None else str(last_error)
    meta["ie_extracted"] = status == IE_STATUS_SUCCESS
    return meta


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
    parser.add_argument(
        "--retry-report",
        default=None,
        help=f"Retry report path (default: <output-dir>/{RETRY_REPORT_FILENAME}).",
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
    retry_report_path = (
        Path(args.retry_report)
        if args.retry_report
        else output_dir / RETRY_REPORT_FILENAME
    )
    retry_report_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)

    raw_by_id: dict[str, tuple[list, list]] = {}
    errors_by_id: dict[str, str] = {}
    failure_reason_counts: dict[str, int] = {}
    completed = 0
    failed = 0
    submitted_chunk_ids: set[str] = set(state.get("submitted_chunk_ids") or [])
    attempt = int(state.get("attempt", 1) or 1)

    for shard in shards:
        batch_id = shard.get("batch_id")
        if not batch_id:
            continue
        shard_ids = {
            str(x).strip() for x in (shard.get("chunk_ids") or []) if str(x).strip()
        }
        submitted_chunk_ids.update(shard_ids)
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            logger.warning(
                "Shard batch %s not completed (status=%s). Skipping.",
                batch_id,
                batch.status,
            )
            for cid in shard_ids:
                errors_by_id[cid] = f"batch_not_completed:{batch.status}"
                failure_reason_counts["batch_not_completed"] = (
                    failure_reason_counts.get("batch_not_completed", 0) + 1
                )
            continue

        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file", None
        )
        if not output_file_id:
            logger.warning("Shard batch %s has no output file.", batch_id)
            for cid in shard_ids:
                errors_by_id[cid] = "missing_batch_output_file"
                failure_reason_counts["missing_batch_output_file"] = (
                    failure_reason_counts.get("missing_batch_output_file", 0) + 1
                )
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
                cid = str(obj.get("custom_id", "")).strip()
                raw_by_id[cid] = ([], [])
                err = obj.get("error")
                errors_by_id[cid] = json.dumps(err, ensure_ascii=False)
                failure_reason_counts["api_error"] = (
                    failure_reason_counts.get("api_error", 0) + 1
                )
                continue
            completed += 1
            cid, raw_entities, raw_triples, parse_error = parse_ie_batch_output_line(
                obj
            )
            raw_by_id[cid] = (raw_entities, raw_triples)
            if parse_error:
                errors_by_id[cid] = f"parse_error:{parse_error}"
                failure_reason_counts["parse_error"] = (
                    failure_reason_counts.get("parse_error", 0) + 1
                )
            else:
                errors_by_id.pop(cid, None)

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
                    cid = str(obj.get("custom_id", "")).strip()
                    raw_by_id[cid] = ([], [])
                    errors_by_id[cid] = json.dumps(obj.get("error"), ensure_ascii=False)
                    failure_reason_counts["api_error"] = (
                        failure_reason_counts.get("api_error", 0) + 1
                    )
            except Exception as e:
                logger.warning("Could not fetch error file for %s: %s", batch_id, e)

    for cid in submitted_chunk_ids:
        if cid not in raw_by_id and cid not in errors_by_id:
            errors_by_id[cid] = "missing_batch_output"
            failure_reason_counts["missing_batch_output"] = (
                failure_reason_counts.get("missing_batch_output", 0) + 1
            )

    success_ids_attempt = {
        cid
        for cid, (ents, rels) in raw_by_id.items()
        if cid in submitted_chunk_ids and (ents or rels) and cid not in errors_by_id
    }
    failed_ids_attempt = {
        cid
        for cid in submitted_chunk_ids
        if cid in errors_by_id or cid not in success_ids_attempt
    }

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
            meta = _normalize_ie_metadata(meta)
            if chunk_id in submitted_chunk_ids:
                meta["ie_attempts"] = int(meta.get("ie_attempts", 0)) + 1

            if chunk_id in raw_by_id:
                # This chunk was in the current batch – apply new extraction
                raw_entities, raw_triples = raw_by_id[chunk_id]
                text = chunk.get("text") or ""
                entities, relations = _post_process_ie(
                    raw_entities, raw_triples, text, chunk_id
                )
                meta["entities"] = entities
                meta["relations"] = relations
                if chunk_id in errors_by_id:
                    meta["ie_status"] = IE_STATUS_FAILED
                    meta["ie_last_error"] = errors_by_id[chunk_id]
                else:
                    meta["ie_status"] = IE_STATUS_SUCCESS
                    meta["ie_last_error"] = None
            elif meta.get("ie_status") == IE_STATUS_SUCCESS:
                # Already extracted in a prior run – keep existing data
                entities = meta.get("entities", [])
                relations = meta.get("relations", [])
            else:
                # Not newly extracted this attempt: preserve existing values.
                entities = meta.get("entities", [])
                relations = meta.get("relations", [])
                meta.setdefault("entities", entities)
                meta.setdefault("relations", relations)
                if chunk_id in errors_by_id:
                    meta["ie_status"] = IE_STATUS_FAILED
                    meta["ie_last_error"] = errors_by_id[chunk_id]
                elif meta.get("ie_status") != IE_STATUS_SUCCESS:
                    meta["ie_status"] = IE_STATUS_PENDING
                    meta["ie_last_error"] = meta.get("ie_last_error")

            meta["ie_extracted"] = meta.get("ie_status") == IE_STATUS_SUCCESS

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
    unresolved_ids = []
    for chunk in _iter_jsonl(output_path):
        cid = str(chunk.get("chunk_id", "")).strip()
        if not cid:
            continue
        meta = _normalize_ie_metadata(dict(chunk.get("metadata", {})))
        if meta.get("ie_status") != IE_STATUS_SUCCESS:
            unresolved_ids.append(cid)
    retry_report = {
        "attempt": attempt,
        "submitted_count": len(submitted_chunk_ids),
        "success_ids": sorted(success_ids_attempt),
        "failed_ids": sorted(failed_ids_attempt),
        "unresolved_ids": sorted(set(unresolved_ids)),
        "failure_reasons": failure_reason_counts,
    }
    retry_report_path.write_text(
        json.dumps(retry_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote retry report: %s", retry_report_path)
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
