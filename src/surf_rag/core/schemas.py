from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Block:
    text: str
    section_path: List[str]
    block_type: str


@dataclass(frozen=True)
class StructuredDoc:
    doc_id: str
    title: Optional[str]
    url: Optional[str]
    blocks: List[Block]
    anchors: Dict[str, List[str]]
    source: str
    dataset_origin: str
    page_id: Optional[str] = None
    revision_id: Optional[str] = None


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    source: str
    title: Optional[str]
    url: Optional[str]
    text: str
    section_path: List[str]
    char_span_in_doc: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "text": self.text,
            "section_path": list(self.section_path),
            "char_span_in_doc": list(self.char_span_in_doc),
            "metadata": dict(self.metadata),
        }


def sha256_text(value: str) -> str:
    normalized = value.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: str
    question: str
    gold_answers: List[str]
    dataset_source: str
    gold_support_sentences: List[str] = field(default_factory=list)
    dataset_version: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question_id": self.question_id,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "gold_support_sentences": list(self.gold_support_sentences),
            "dataset_source": self.dataset_source,
        }
        if self.dataset_version:
            payload["dataset_version"] = self.dataset_version
        return payload
