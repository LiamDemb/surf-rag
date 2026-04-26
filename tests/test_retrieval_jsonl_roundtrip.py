from surf_rag.evaluation.retrieval_jsonl import (
    dict_to_retrieval_result,
    retrieval_result_to_dict,
)
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def test_retrieval_result_json_roundtrip() -> None:
    r = RetrievalResult(
        query="q1",
        retriever_name="fused",
        status="OK",
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                text="hello",
                score=0.9,
                rank=0,
                metadata={"branch": "dense"},
            )
        ],
        latency_ms={"total_ms": 1.0},
        error=None,
        debug_info={"routing": {"policy": "dense-only"}},
    )
    d = retrieval_result_to_dict(r, "qid-1")
    assert d["question_id"] == "qid-1"
    r2 = dict_to_retrieval_result(d)
    assert r2.query == r.query
    assert len(r2.chunks) == 1
    assert r2.chunks[0].chunk_id == "c1"
    assert r2.debug_info == r.debug_info
