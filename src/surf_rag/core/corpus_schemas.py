from __future__ import annotations

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
