from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import pandas as pd

_CORPUS_META_KEYS = ("title", "doc_id", "source", "url", "section_path")


def metadata_from_corpus_record(record: dict[str, Any] | None) -> dict[str, Any]:
    """Pick stable fields from a corpus.jsonl row for ``RetrievedChunk.metadata``."""
    if not record:
        return {}
    out: dict[str, Any] = {}
    for k in _CORPUS_META_KEYS:
        if k in record and record[k] is not None:
            out[k] = record[k]
    return out


class RowIdToChunkId(Protocol):
    """Row ID in FAISS index -> chunk ID in corpus."""

    def row_to_chunk(self, row_id: int) -> Optional[str]: ...


class ChunkIdToText(Protocol):
    """Chunk ID in corpus -> text."""

    def get_text(self, chunk_id: str) -> Optional[str]: ...


@dataclass
class VectorMetaMapper:
    """Loads vector_meta.parquet into memory and provides row_id -> chunk_id mapping."""

    parquet_path: str
    row_col: str = "row_id"
    chunk_col: str = "chunk_id"

    def __post_init__(self) -> None:
        df = pd.read_parquet(self.parquet_path, columns=[self.row_col, self.chunk_col])

        # Store dict: row_id -> chunk_id
        self._map: Dict[int, str] = dict(
            zip(df[self.row_col].astype(int), df[self.chunk_col].astype(str))
        )

    def row_to_chunk(self, row_id: int) -> Optional[str]:
        return self._map.get(row_id)


@dataclass
class JsonCorpusLoader:
    """Loads corpus.jsonl: chunk text plus optional row fields for retrieval metadata."""

    jsonl_path: str
    chunk_id_col: str = "chunk_id"
    text_key: str = "text"

    def __post_init__(self) -> None:
        self.text_by_id: Dict[str, str] = {}
        self._row_by_id: Dict[str, dict[str, Any]] = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                cid = str(obj[self.chunk_id_col])
                txt = str(obj[self.text_key])
                self.text_by_id[cid] = txt
                self._row_by_id[cid] = obj

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self.text_by_id.get(chunk_id)

    def get_record(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Full JSON object for a chunk id (last row wins if duplicates)."""
        return self._row_by_id.get(chunk_id)
