"""Prompt rendering for sentence-window retrieval units."""

from __future__ import annotations

from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def test_sentence_window_headers_in_context() -> None:
    ch = RetrievedChunk(
        chunk_id="doc1::sw0",
        text="The capital is Paris.",
        score=2.0,
        rank=0,
        metadata={
            "context_unit": "sentence_window",
            "title": "France",
            "original_chunk_rank": 3,
            "window_rerank_score": 1.2345,
            "original_chunk_id": "doc1",
            "prompt_w_label": "W1",
            "prompt_include_title": True,
        },
    )
    rr = RetrievalResult(
        query="q",
        retriever_name="Fused+sw",
        status="OK",
        chunks=[ch],
    )
    r = PromptRenderer(base_prompt="X\n{context}\nY\n{question}\n")
    msgs = r.to_messages("What is the capital?", rr)
    user = msgs[1]["content"]
    assert "[W1 | chunk_rank=3 | title=France | window_score=1.2345 |" in user
    assert "source_chunk_id=doc1" in user
    assert "The capital is Paris." in user


def test_generator_sentence_windows_has_placeholders_and_core_rules() -> None:
    from pathlib import Path

    p = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "generator_sentence_windows.txt"
    )
    t = p.read_text(encoding="utf-8")
    assert "{context}" in t and "{question}" in t
    assert "shortest exact span" in t
    assert "only" in t.lower() and "reasoning" in t


def test_plain_chunk_unchanged() -> None:
    ch = RetrievedChunk(
        chunk_id="c1",
        text="Plain body only.",
        score=1.0,
        rank=0,
        metadata={},
    )
    rr = RetrievalResult(
        query="q",
        retriever_name="Fused",
        status="OK",
        chunks=[ch],
    )
    r = PromptRenderer(base_prompt="{context}\n{question}\n")
    user = r.to_messages("Q?", rr)[1]["content"]
    assert "[W1" not in user
    assert "Plain body only." in user
