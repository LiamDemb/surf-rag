from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set

from .docstore import DocRecord, DocStore
from .wikipedia_client import WikipediaClient
from .wikidata_client import WikidataClient
from .schemas import sha256_text


@dataclass(frozen=True)
class Budgets:
    max_pages_per_question: int = 12
    max_hops: int = 2
    max_list_pages: int = 2
    max_country_pages: int = 1
    max_context_qids: int = 8
    max_outgoing: int = 8


@dataclass(frozen=True)
class RawDoc:
    doc_key: str
    title: Optional[str]
    url: Optional[str]
    html: Optional[str]
    anchors: dict
    source: str
    dataset_origin: str
    page_id: Optional[str] = None
    revision_id: Optional[str] = None


def _cached_wiki_page(
    title: str,
    source: str,
    dataset_origin: str,
    docstore: DocStore,
    wiki: WikipediaClient,
) -> RawDoc:
    cache_key = f"title:{title}"

    def _fetch() -> DocRecord:
        page = wiki.fetch_html(title)
        return DocRecord(
            title=page.title,
            page_id=page.page_id,
            revision_id=page.revision_id,
            url=page.url,
            html=page.html,
            cleaned_text=None,
            anchors={"outgoing_titles": page.outgoing_titles, "incoming_stub": []},
            source=source,
            dataset_origin=dataset_origin,
        )

    record = docstore.get_or_fetch(cache_key, _fetch)
    doc_key = record.page_id or sha256_text(record.title)
    return RawDoc(
        doc_key=doc_key,
        title=record.title,
        url=record.url,
        html=record.html,
        anchors=record.anchors,
        source=source,
        dataset_origin=dataset_origin,
        page_id=record.page_id,
        revision_id=record.revision_id,
    )


def _dedupe_docs(docs: Iterable[RawDoc]) -> List[RawDoc]:
    seen = set()
    unique: List[RawDoc] = []
    for doc in docs:
        if doc.doc_key in seen:
            continue
        seen.add(doc.doc_key)
        unique.append(doc)
    return unique


def ingest_2wiki(
    sample: dict,
    budgets: Budgets,
    docstore: DocStore,
    wiki: WikipediaClient,
) -> List[RawDoc]:
    """Fetch only the supporting Wikipedia pages for a 2WikiMultiHopQA question."""
    source = "2wiki"
    dataset_origin = "2wiki"
    supporting_facts = sample.get("supporting_facts") or {}
    titles = list(
        dict.fromkeys(
            str(t).strip() for t in supporting_facts["title"] if str(t).strip()
        )
    )

    docs = [
        _cached_wiki_page(title, source, dataset_origin, docstore, wiki)
        for title in titles
    ]
    return _dedupe_docs(docs)


def ingest_nq(
    sample: dict,
    budgets: Budgets,
    docstore: DocStore,
    wiki: WikipediaClient,
) -> List[RawDoc]:
    source = "nq"
    dataset_origin = "nq"
    document = (
        sample.get("document", {}) if isinstance(sample.get("document"), dict) else {}
    )
    html = document.get("html") or sample.get("document_html")
    title = document.get("title") or sample.get("document_title") or sample.get("title")
    url = document.get("url")
    doc_key = sha256_text(html or title or sample.get("id", "nq"))
    docs = [
        RawDoc(
            doc_key=doc_key,
            title=title,
            url=url,
            html=html,
            anchors={"outgoing_titles": [], "incoming_stub": []},
            source=source,
            dataset_origin=dataset_origin,
            page_id=None,
            revision_id=None,
        )
    ]

    return _dedupe_docs(docs)
