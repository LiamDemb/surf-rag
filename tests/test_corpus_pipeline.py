import json

import pytest

from surf_rag.core.build_graph import build_graph
from surf_rag.core.canonical_clean import clean_html_to_structured_doc
from surf_rag.core.chunking import chunk_blocks
from surf_rag.core.corpus_schemas import Block
from surf_rag.core.docstore import DocRecord, DocStore
from surf_rag.core.alias_map import normalize_alias_map
from surf_rag.core.enrich_entities import norm_entity
from surf_rag.core.entity_lexicon import build_entity_lexicon


def test_clean_html_to_structured_doc_parses_blocks():
    html = """
    <html><body>
      <h1>Title</h1>
      <p>First paragraph.</p>
      <ul><li>Item A</li><li>Item B</li></ul>
      <table><tr><th>Year</th><th>Event</th></tr><tr><td>1999</td><td>Test</td></tr></table>
    </body></html>
    """
    doc = clean_html_to_structured_doc(
        html=html,
        doc_id="doc1",
        title="Title",
        url=None,
        anchors={"outgoing_titles": [], "incoming_stub": []},
        source="wiki",
        dataset_origin="wiki",
    )
    assert len(doc.blocks) >= 3
    assert any(block.block_type == "paragraph" for block in doc.blocks)
    assert any(block.block_type == "list" for block in doc.blocks)
    assert all(block.block_type != "table" for block in doc.blocks)
    assert not any("1999" in block.text for block in doc.blocks)


def test_clean_html_removes_citation_markup_but_keeps_plain_brackets():
    html = """
    <html><body>
      <p>Population rose in 2010<sup class="reference"><a>[12]</a></sup>.</p>
      <p>Model [3] is referenced in plain text and should remain.</p>
      <ol class="references"><li id="cite_note-12">Citation text</li></ol>
    </body></html>
    """
    doc = clean_html_to_structured_doc(
        html=html,
        doc_id="doc-footnote",
        title="Footnotes",
        url=None,
        anchors={"outgoing_titles": [], "incoming_stub": []},
        source="wiki",
        dataset_origin="wiki",
    )
    all_text = "\n".join(block.text for block in doc.blocks)
    assert "[12]" not in all_text
    assert "Model [3]" in all_text


def test_norm_entity_alias():
    alias_map = normalize_alias_map({"u.s.": "united states"})
    assert norm_entity("U.S.", alias_map) == "united states"


def test_chunk_blocks_bounds():
    blocks = [
        Block(text="word " * 300, section_path=["Lead"], block_type="paragraph"),
        Block(text="word " * 300, section_path=["Lead"], block_type="paragraph"),
    ]
    chunks = chunk_blocks(
        blocks,
        min_tokens=200,
        max_tokens=500,
        overlap_tokens=50,
    )
    assert len(chunks) >= 1
    assert all(chunk.token_count >= 200 for chunk in chunks)
    assert all(chunk.token_count <= 500 for chunk in chunks)


def test_chunk_blocks_respects_max_tokens():
    """No chunk exceeds max_tokens (tiktoken cl100k_base)."""
    blocks = [
        Block(
            text="The quick brown fox " * 100,
            section_path=["Lead"],
            block_type="paragraph",
        ),
    ]
    max_tokens = 200
    chunks = chunk_blocks(
        blocks,
        min_tokens=50,
        max_tokens=max_tokens,
        overlap_tokens=20,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert (
            chunk.token_count <= max_tokens
        ), f"chunk has {chunk.token_count} tokens, max={max_tokens}"


def test_chunk_blocks_long_block_splitting():
    """Single block exceeding max_tokens is split into multiple chunks."""
    long_text = "The history of computing spans many decades. " * 80
    blocks = [Block(text=long_text, section_path=["History"], block_type="paragraph")]
    max_tokens = 200
    chunks = chunk_blocks(
        blocks,
        min_tokens=50,
        max_tokens=max_tokens,
        overlap_tokens=30,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= max_tokens


def test_chunk_blocks_hard_caps_max_even_below_min():
    """
    Regression: previously we could exceed max_tokens if the buffer hadn't yet reached
    min_tokens (because we only flushed once buf_tokens >= min_tokens).
    """
    blocks = [
        Block(text="word " * 600, section_path=["Lead"], block_type="paragraph"),
        Block(text="word " * 250, section_path=["Lead"], block_type="paragraph"),
    ]
    max_tokens = 800
    chunks = chunk_blocks(
        blocks,
        min_tokens=700,
        max_tokens=max_tokens,
        overlap_tokens=50,
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert (
            chunk.token_count <= max_tokens
        ), f"chunk has {chunk.token_count} tokens, max={max_tokens}"


def test_build_graph_nodes():
    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {"entities": [{"norm": "united states", "type": "GPE"}]},
        }
    ]
    graph = build_graph(chunks)
    assert "C:c1" in graph.nodes
    assert "E:united states" in graph.nodes


def test_build_graph_relations():
    chunks = [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [{"norm": "steve jobs", "type": "PERSON"}],
                "relations": [
                    {
                        "subj_norm": "steve jobs",
                        "pred": "founded",
                        "obj_norm": "apple",
                    }
                ],
            },
        }
    ]
    graph = build_graph(chunks)
    assert graph.has_edge("E:steve jobs", "E:apple")
    assert graph.edges["E:steve jobs", "E:apple"]["label"] == "founded"


def test_docstore_cache(tmp_path):
    path = tmp_path / "docstore.sqlite"
    store = DocStore(path.as_posix())
    called = {"count": 0}

    def _fetch():
        called["count"] += 1
        return DocRecord(
            title="Test",
            page_id="1",
            revision_id="10",
            url=None,
            html="<p>Test</p>",
            cleaned_text="Test",
            anchors={},
            source="wiki",
            dataset_origin="wiki",
        )

    record_1 = store.get_or_fetch("title:Test", _fetch)
    record_2 = store.get_or_fetch("title:Test", _fetch)
    assert record_1.title == "Test"
    assert record_2.title == "Test"
    assert called["count"] == 1
    store.close()


def test_build_entity_lexicon():
    chunks = [
        {
            "metadata": {
                "entities": [
                    {"norm": "united states", "surface": "United States", "qid": "Q30"}
                ]
            }
        }
    ]
    df = build_entity_lexicon(chunks)
    assert not df.empty
    assert df.iloc[0]["norm"] == "united states"
