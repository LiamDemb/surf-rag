"""Batch helpers: bodies match renderer output shape."""

from surf_rag.generation.batch import build_batch_line, build_completion_body
from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def test_build_completion_body_from_messages_roundtrip():
    retrieval = RetrievalResult(
        query="What is X?",
        retriever_name="Dense",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1", text="Evidence.", score=1.0, rank=0, metadata={}
            )
        ],
    )
    renderer = PromptRenderer()
    messages = renderer.to_messages("What is X?", retrieval)
    body = build_completion_body(
        messages, model_id="gpt-4o-mini", temperature=0.0, max_tokens=128
    )
    assert body["model"] == "gpt-4o-mini"
    assert body["messages"] == messages
    assert body["temperature"] == 0.0


def test_build_batch_line_shape():
    body = {"model": "m", "messages": [], "temperature": 0, "max_tokens": 1}
    line = build_batch_line("run::nq::train::dense::q1", body)
    assert line["custom_id"] == "run::nq::train::dense::q1"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"] == body
