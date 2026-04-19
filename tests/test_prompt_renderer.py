"""PromptRenderer parity with retrieval chunks."""

import os

from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def test_graph_paths_omitted_by_default_even_when_metadata_present():
    """Default prompt is dense-like: chunk text only (no Path headers)."""
    r = RetrievalResult(
        query="q",
        retriever_name="Graph",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="Hello world.",
                score=1.0,
                rank=0,
                metadata={
                    "branch": "graph",
                    "graph_path_lines": ["Path: A -[r]-> B"],
                },
            )
        ],
    )
    renderer = PromptRenderer(
        base_prompt="{context}\n{question}", include_graph_provenance=False
    )
    msgs = renderer.to_messages("What?", r)
    user = msgs[1]["content"]
    assert "Hello world." in user
    assert "Path:" not in user
    assert "[" not in user or "Path:" not in user


def test_renderer_formats_graph_path_lines_when_enabled():
    r = RetrievalResult(
        query="q",
        retriever_name="Graph",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="Hello world.",
                score=1.0,
                rank=0,
                metadata={
                    "branch": "graph",
                    "graph_path_lines": ["Path: A -[r]-> B"],
                },
            )
        ],
    )
    renderer = PromptRenderer(
        base_prompt="{context}\n{question}", include_graph_provenance=True
    )
    msgs = renderer.to_messages("What?", r)
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "[Path: A -[r]-> B]" in user
    assert "Hello world." in user


def test_include_graph_paths_respects_env(monkeypatch):
    monkeypatch.setenv("INCLUDE_GRAPH_PATHS_IN_PROMPT", "1")
    r = RetrievalResult(
        query="q",
        retriever_name="Graph",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="Hi.",
                score=1.0,
                rank=0,
                metadata={
                    "branch": "graph",
                    "graph_path_lines": ["Path: X"],
                },
            )
        ],
    )
    renderer = PromptRenderer(base_prompt="{context}\n{question}")
    assert renderer.include_graph_provenance is True
    user = renderer.to_messages("?", r)[1]["content"]
    assert "Path:" in user


def test_dense_chunk_has_no_graph_header():
    r = RetrievalResult(
        query="q",
        retriever_name="Dense",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="Plain text.",
                score=0.9,
                rank=0,
                metadata={"branch": "dense"},
            )
        ],
    )
    renderer = PromptRenderer(base_prompt="{context}\n{question}")
    msgs = renderer.to_messages("Q?", r)
    user = msgs[1]["content"]
    assert "Plain text." in user
    assert "Path:" not in user
