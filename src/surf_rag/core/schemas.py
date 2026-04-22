from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


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


def parse_benchmark_support_fields(
    row: Mapping[str, Any],
) -> Tuple[List[str], List[str], List[int]]:
    """
    Gold support lines: parallel sentences, Wikipedia titles, and sentence ids.

    ``sent_id`` is the dataset-local index for 2Wiki; use ``-1`` when not
    applicable (e.g. NQ). Legacy rows may store 2Wiki provenance only under
    ``two_wiki_support_facts``; that shape is merged into the parallel lists.
    """
    sentences = list(row.get("gold_support_sentences") or [])
    n = len(sentences)
    titles_raw = row.get("gold_support_titles")
    ids_raw = row.get("gold_support_sent_ids")
    legacy = row.get("two_wiki_support_facts")

    if isinstance(titles_raw, list) and len(titles_raw) == n:
        titles = [str(x or "").strip() for x in titles_raw]
    elif isinstance(legacy, list) and len(legacy) == n:
        titles = []
        for item in legacy:
            if isinstance(item, dict):
                titles.append(str(item.get("title", "") or "").strip())
            else:
                titles.append("")
    else:
        titles = [""] * n

    if isinstance(ids_raw, list) and len(ids_raw) == n:
        sent_ids: List[int] = []
        for x in ids_raw:
            try:
                sent_ids.append(int(x))
            except (TypeError, ValueError):
                sent_ids.append(-1)
    elif isinstance(legacy, list) and len(legacy) == n:
        sent_ids = []
        for item in legacy:
            if isinstance(item, dict):
                sid = item.get("sent_id")
                try:
                    sent_ids.append(int(sid) if sid is not None else -1)
                except (TypeError, ValueError):
                    sent_ids.append(-1)
            else:
                sent_ids.append(-1)
    else:
        sent_ids = [-1] * n

    return sentences, titles, sent_ids


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: str
    question: str
    gold_answers: List[str]
    dataset_source: str
    gold_support_sentences: List[str] = field(default_factory=list)
    """Parallel to ``gold_support_sentences``: source article title per line."""
    gold_support_titles: List[str] = field(default_factory=list)
    """Parallel: 2Wiki ``sent_id``, or ``-1`` when not used (e.g. NQ)."""
    gold_support_sent_ids: List[int] = field(default_factory=list)
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
        n = len(self.gold_support_sentences)
        if n and len(self.gold_support_titles) == n:
            payload["gold_support_titles"] = list(self.gold_support_titles)
        if n and len(self.gold_support_sent_ids) == n:
            payload["gold_support_sent_ids"] = list(self.gold_support_sent_ids)
        return payload
