"""Submit, poll and collect OpenAI batch jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from surf_rag.core.openai_batch_limits import batch_limit_requests
from surf_rag.generation.batch import parse_generation_output


# Match default orchestrator / OpenAI docs
DEFAULT_SHARD_BYTES_LIMIT = 200 * 1024 * 1024


class OpenAIBatchManager:
    """Create batches from JSONL files, retrieve outputs, demultiplex by custom_id."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for batch operations")
        self._client = OpenAI(api_key=key)

    def upload_input_file(self, jsonl_path: Path) -> str:
        with jsonl_path.open("rb") as f:
            uploaded = self._client.files.create(file=f, purpose="batch")
        return uploaded.id

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        batch = self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata or {},
        )
        return batch.id

    def get_batch(self, batch_id: str) -> Any:
        return self._client.batches.retrieve(batch_id)

    def download_output_text(self, batch_id: str) -> str:
        batch = self._client.batches.retrieve(batch_id)
        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} not completed (status={batch.status})"
            )
        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file", None
        )
        if not output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output file")
        content = self._client.files.content(output_file_id)
        raw = content.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def parse_output_answers(self, output_text: str) -> Dict[str, str]:
        """Map custom_id -> answer text."""
        answers: Dict[str, str] = {}
        for line in output_text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid, ans = parse_generation_output(obj)
            if cid:
                answers[cid] = ans
        return answers


def shard_jsonl_file(
    input_jsonl: Path,
    output_dir: Path,
    prefix: str = "batch_shard",
    max_requests: int | None = None,
    max_bytes: int = DEFAULT_SHARD_BYTES_LIMIT,
) -> List[Path]:
    """Split a large JSONL into shard files under byte/request limits."""
    max_requests = max_requests if max_requests is not None else batch_limit_requests()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Path] = []
    shard_idx = 0
    buf: List[str] = []
    buf_bytes = 0
    count = 0

    def flush() -> None:
        nonlocal shard_idx, buf, buf_bytes, count
        if not buf:
            return
        p = output_dir / f"{prefix}_{shard_idx:03d}.jsonl"
        p.write_text("".join(buf), encoding="utf-8")
        shards.append(p)
        shard_idx += 1
        buf = []
        buf_bytes = 0
        count = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            line_bytes = len(line.encode("utf-8"))
            if buf and (
                count >= max_requests or buf_bytes + line_bytes > max_bytes
            ):
                flush()
            buf.append(line)
            buf_bytes += line_bytes
            count += 1
    flush()
    return shards
