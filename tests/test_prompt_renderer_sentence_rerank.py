"""PromptRenderer with sentence shortlist evidence."""

from __future__ import annotations

from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.reranking.sentence_reranker import PROMPT_EVIDENCE_SENTENCE_SHORTLIST
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def test_structured_sentence_shortlist_renders_xml() -> None:
    rr = RetrievalResult(
        query="q",
        retriever_name="Fused+rerank+sentence_rerank",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1#s0",
                text="Alpha.",
                score=1.5,
                rank=0,
                metadata={
                    "parent_chunk_id": "c1",
                    "parent_chunk_rank": 0,
                    "title": 'Title "X"',
                },
            )
        ],
        latency_ms={},
        debug_info={"prompt_evidence": PROMPT_EVIDENCE_SENTENCE_SHORTLIST},
    )
    r = PromptRenderer(
        base_prompt="{context}\n{question}",
        include_graph_provenance=False,
        sentence_rerank_prompt_style="structured",
    )
    user = r.to_messages("What?", rr)[1]["content"]
    assert "<evidence_shortlist>" in user
    assert "</evidence_shortlist>" in user
    assert "<S " in user and "Alpha." in user
    assert "c1" in user
    assert "&quot;" in user or "Title" in user


def test_inline_style_joins_sentences() -> None:
    rr = RetrievalResult(
        query="q",
        retriever_name="X",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1#s0",
                text="One.",
                score=1.0,
                rank=0,
                metadata={"parent_chunk_id": "c1"},
            )
        ],
        latency_ms={},
        debug_info={"prompt_evidence": PROMPT_EVIDENCE_SENTENCE_SHORTLIST},
    )
    r = PromptRenderer(
        base_prompt="{context}\n{question}",
        sentence_rerank_prompt_style="inline",
    )
    user = r.to_messages("Q?", rr)[1]["content"]
    assert "[c1]" in user
    assert "One." in user
