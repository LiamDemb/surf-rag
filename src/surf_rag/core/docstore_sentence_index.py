"""Build title → sentence lists from DocStore-backed HTML using the same clean+chunk path as corpus build."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

from surf_rag.benchmark.support_alignment import build_title_to_candidate_sentences
from surf_rag.core.canonical_clean import clean_html_to_structured_doc
from surf_rag.core.chunking import chunk_blocks
from surf_rag.core.corpus_acquisition import RawDoc, raw_doc_from_docrecord
from surf_rag.core.docstore import DocStore


def corpus_chunk_shaped_rows_from_raw_doc(
    doc: RawDoc,
    *,
    chunk_min_tokens: int,
    chunk_max_tokens: int,
    chunk_overlap_tokens: int,
) -> List[dict]:
    """Produce ``{title, text}`` dicts matching :func:`build_title_to_candidate_sentences` input shape."""
    if not doc.html:
        return []
    structured = clean_html_to_structured_doc(
        html=doc.html,
        doc_id=doc.doc_key,
        title=doc.title,
        url=doc.url,
        anchors=doc.anchors,
        source=doc.source,
        dataset_origin=doc.dataset_origin,
        page_id=doc.page_id,
        revision_id=doc.revision_id,
    )
    rows: List[dict] = []
    for piece in chunk_blocks(
        structured.blocks,
        min_tokens=chunk_min_tokens,
        max_tokens=chunk_max_tokens,
        overlap_tokens=chunk_overlap_tokens,
    ):
        rows.append({"title": doc.title, "text": piece.text})
    return rows


def collect_raw_docs_for_titles(
    titles: Iterable[str], docstore: DocStore
) -> List[RawDoc]:
    """Load cached Wikipedia pages from DocStore (``title:{title}`` keys). Skips missing or empty HTML."""
    seen: set[str] = set()
    out: List[RawDoc] = []
    for t in titles:
        title = str(t).strip()
        if not title or title in seen:
            continue
        seen.add(title)
        rec = docstore.get(f"title:{title}")
        if rec is not None and rec.html:
            out.append(raw_doc_from_docrecord(rec))
    return out


def build_title_to_candidate_sentences_from_docstore(
    docstore: DocStore,
    titles: Iterable[str],
    *,
    chunk_min_tokens: int = 500,
    chunk_max_tokens: int = 800,
    chunk_overlap_tokens: int = 100,
) -> Dict[str, List[str]]:
    """Sentence index for alignment: same cleaning + chunking + sentencizing as corpus chunks."""
    docs = collect_raw_docs_for_titles(titles, docstore)
    return build_title_to_candidate_sentences_from_raw_docs(
        docs,
        chunk_min_tokens=chunk_min_tokens,
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )


def build_title_to_candidate_sentences_from_raw_docs(
    docs: Iterable[RawDoc],
    *,
    chunk_min_tokens: int = 500,
    chunk_max_tokens: int = 800,
    chunk_overlap_tokens: int = 100,
) -> Dict[str, List[str]]:
    all_rows: List[Mapping[str, str]] = []
    for doc in docs:
        all_rows.extend(
            corpus_chunk_shaped_rows_from_raw_doc(
                doc,
                chunk_min_tokens=chunk_min_tokens,
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        )
    return build_title_to_candidate_sentences(all_rows)
