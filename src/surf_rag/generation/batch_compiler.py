"""Compile batch input JSONL from (custom_id, chat completion body) records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List


@dataclass(frozen=True)
class BatchRequestRecord:
    """One row to be written as OpenAI Batch JSONL."""

    custom_id: str
    body: Dict[str, Any]


def iter_batch_jsonl_lines(records: List[BatchRequestRecord]) -> Iterator[str]:
    from surf_rag.generation.batch import build_batch_line

    for rec in records:
        line_obj = build_batch_line(rec.custom_id, rec.body)
        yield json.dumps(line_obj, ensure_ascii=False) + "\n"


def write_batch_jsonl(records: List[BatchRequestRecord], path: Path) -> None:
    """Write all records to a single JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in iter_batch_jsonl_lines(records):
            f.write(line)
