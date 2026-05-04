"""Shard and submit OpenAI Batch jobs for chat completions (shared by E2E and audit)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from openai import OpenAI

from surf_rag.core.openai_batch_limits import batch_limit_requests
from surf_rag.generation.batch import build_batch_line

logger = logging.getLogger(__name__)

BATCH_LIMIT_BYTES = 200 * 1024 * 1024


def upload_sharded_chat_completion_batches(
    records: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    work_dir: Path,
    shard_filename_prefix: str,
    completion_window: str,
    dry_run: bool,
    batch_metadata: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Write shards under ``work_dir``, upload each, return ``shards`` list for state JSON.

    Each record is ``(custom_id, chat_completion_body)``. Uses same size/request limits
    as generation batch orchestrator.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    limit_requests = batch_limit_requests()
    shards: List[Dict[str, Any]] = []
    shard_idx = 0
    shard_count = 0
    shard_bytes = 0
    shard_file = work_dir / f"{shard_filename_prefix}_{shard_idx:03d}.jsonl"
    batch_out = shard_file.open("w", encoding="utf-8")

    def flush_shard() -> None:
        nonlocal shard_idx, shard_count, shard_bytes, shard_file, batch_out
        if shard_count == 0:
            return
        batch_out.close()
        if dry_run:
            logger.info(
                "[dry-run] Would upload shard %d (%d requests) at %s",
                shard_idx,
                shard_count,
                shard_file,
            )
            shards.append(
                {
                    "batch_id": f"dry-run-{shard_idx}",
                    "input_path": str(shard_file),
                    "request_count": shard_count,
                }
            )
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Uploading shard %d (%d requests)...", shard_idx, shard_count)
            with shard_file.open("rb") as r:
                uploaded = client.files.create(file=r, purpose="batch")
            meta = dict(batch_metadata)
            meta["shard"] = str(shard_idx)
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
                metadata={k: str(v)[:512] for k, v in meta.items()},
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
        shard_file = work_dir / f"{shard_filename_prefix}_{shard_idx:03d}.jsonl"
        batch_out = shard_file.open("w", encoding="utf-8")

    for custom_id, body in records:
        line_json = (
            json.dumps(build_batch_line(custom_id, body), ensure_ascii=False) + "\n"
        )
        line_bytes = len(line_json.encode("utf-8"))
        if shard_count > 0 and (
            shard_count >= limit_requests
            or shard_bytes + line_bytes > BATCH_LIMIT_BYTES
        ):
            flush_shard()
        batch_out.write(line_json)
        shard_count += 1
        shard_bytes += line_bytes

    if shard_count > 0:
        flush_shard()
    else:
        batch_out.close()
        if shard_file.is_file() and shard_file.stat().st_size == 0:
            shard_file.unlink(missing_ok=True)

    return shards
