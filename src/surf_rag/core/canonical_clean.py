from __future__ import annotations

import re
from typing import List

from bs4 import BeautifulSoup

from .corpus_schemas import Block, StructuredDoc


def normalize_text_for_extraction(text: str) -> str:
    """
    Collapse whitespace for NLP extraction.
    Stored chunk text remains unchanged; use this only when calling extractors.
    """
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    # Strip common Wikipedia bracket artifacts (IPA remnants, listen links)
    s = re.sub(r"\s*\[[\s\dːˈˌˌ]+\]\s*", " ", s)
    s = re.sub(r"\s*\(listen\)\s*", " ", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


DROP_TAGS = ["script", "style", "nav", "footer", "aside"]
CITATION_SELECTORS = [
    "sup.reference",
    "sup.mw-ref",
    "span.mw-ref",
    "a.mw-ref",
    "ol.references",
    "div.reflist",
    "div#mw-references-wrap",
]
# TOC, navboxes, metadata, edit links, category links, pronunciation/IPA
NOISE_SELECTORS = [
    "#toc",
    ".toc",
    ".mw-toc",
    ".vector-toc",
    ".navbox",
    ".vertical-navbox",
    ".navbox-inner",
    ".metadata",
    ".ambox",
    ".mbox-small",
    ".hatnote",
    ".dablink",
    "#catlinks",
    ".mw-editsection",
    ".IPA",
    ".ext-phonos",
    "span.IPA",
    "sup.IPA",
]
# IPA/pronunciation links (Help:IPA, etc.)
NOISE_LINK_SELECTORS = [
    'a[title^="Help:IPA"]',
    'a[title^="Help:Pronunciation"]',
]


def _update_section_path(path: List[str], heading: str, level: str) -> List[str]:
    try:
        depth = int(level.replace("h", ""))
    except ValueError:
        depth = 2
    new_path = path[: max(1, depth - 1)]
    new_path.append(heading)
    return new_path


def clean_html_to_structured_doc(
    html: str,
    doc_id: str,
    title: str | None,
    url: str | None,
    anchors: dict,
    source: str,
    dataset_origin: str,
    page_id: str | None = None,
    revision_id: str | None = None,
) -> StructuredDoc:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in DROP_TAGS:
        for node in soup.find_all(tag):
            node.decompose()
    for selector in CITATION_SELECTORS:
        for node in soup.select(selector):
            node.decompose()
    for selector in NOISE_SELECTORS + NOISE_LINK_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    blocks: List[Block] = []
    current_path = ["Lead"]

    # Restrict to main article content; fallback to body if not found
    root = soup.find("div", class_="mw-parser-output")
    if root is None:
        root = soup.body if soup.body else soup
    for table in list(root.find_all("table")):
        table.decompose()
    for node in root.descendants:
        if not getattr(node, "name", None):
            continue
        if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            current_path = _update_section_path(
                current_path, node.get_text(" ", strip=True), node.name
            )
            text = node.get_text(" ", strip=True)
            if text:
                blocks.append(
                    Block(
                        text=text,
                        section_path=list(current_path),
                        block_type="paragraph",
                    )
                )
        elif node.name == "p":
            text = node.get_text(" ", strip=True)
            if text:
                blocks.append(
                    Block(
                        text=text,
                        section_path=list(current_path),
                        block_type="paragraph",
                    )
                )
        elif node.name in ["ul", "ol"]:
            items = [
                li.get_text(" ", strip=True)
                for li in node.find_all("li", recursive=False)
            ]
            if items:
                text = "\n".join(f"- {item}" for item in items)
                blocks.append(
                    Block(text=text, section_path=list(current_path), block_type="list")
                )

    return StructuredDoc(
        doc_id=doc_id,
        title=title,
        url=url,
        blocks=blocks,
        anchors=anchors or {"outgoing_titles": [], "incoming_stub": []},
        source=source,
        dataset_origin=dataset_origin,
        page_id=page_id,
        revision_id=revision_id,
    )
