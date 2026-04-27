"""E2E prepare: retrieval metrics from fusion, prompt evidence from sentence rerank."""

from __future__ import annotations

import json
from pathlib import Path

from surf_rag.evaluation import e2e_runner
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


class _FakePipeline:
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401, ANN002, ANN003
        _ = args, kwargs

    def run(self, query: str, policy, **kwargs) -> RetrievalResult:  # noqa: ANN001
        _ = policy, kwargs
        return RetrievalResult(
            query=query,
            retriever_name="Fused",
            status="OK",
            chunks=[
                RetrievedChunk(
                    chunk_id="c_fused_1",
                    text="Chunk one. Another sentence.",
                    score=9.0,
                    rank=0,
                    metadata={"branch": "dense", "title": "DocA"},
                ),
                RetrievedChunk(
                    chunk_id="c_fused_2",
                    text="Chunk two.",
                    score=8.0,
                    rank=1,
                    metadata={"branch": "graph", "title": "DocB"},
                ),
            ],
            latency_ms={},
        )


class _FakeChunkReranker:
    def rerank(self, query: str, result: RetrievalResult, top_k: int) -> RetrievalResult:
        _ = query
        out = list(result.chunks[:top_k])
        for ch in out:
            meta = dict(ch.metadata or {})
            meta["rerank_score"] = float(ch.score)
            ch.metadata = meta
        return RetrievalResult(
            query=result.query,
            retriever_name=f"{result.retriever_name}+rerank",
            status="OK",
            chunks=out,
            latency_ms=dict(result.latency_ms),
        )


def test_prepare_writes_retrieval_before_sentence_rerank(tmp_path: Path, monkeypatch) -> None:
    bench = tmp_path / "benchmark.jsonl"
    bench.write_text(
        json.dumps({"question_id": "q1", "question": "Where?"}) + "\n",
        encoding="utf-8",
    )
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(e2e_runner, "get_generator_prompt", lambda: "{context}\n{question}")
    monkeypatch.setattr(e2e_runner, "build_dense_retriever", lambda _x: object())
    monkeypatch.setattr(e2e_runner, "build_graph_retriever", lambda _x: object())
    monkeypatch.setattr(e2e_runner, "RoutedFusionPipeline", _FakePipeline)
    monkeypatch.setattr(e2e_runner, "build_reranker", lambda *_a, **_k: _FakeChunkReranker())

    def _fake_sentence(query: str, chunk_ranked: RetrievalResult, **kwargs) -> RetrievalResult:
        _ = query, kwargs
        return RetrievalResult(
            query=chunk_ranked.query,
            retriever_name=f"{chunk_ranked.retriever_name}+sentence_rerank",
            status="OK",
            chunks=[
                RetrievedChunk(
                    chunk_id="c_fused_1#s0",
                    text="Chunk one.",
                    score=11.0,
                    rank=0,
                    metadata={
                        "parent_chunk_id": "c_fused_1",
                        "parent_chunk_rank": 0,
                        "title": "DocA",
                    },
                )
            ],
            latency_ms={},
            debug_info={"prompt_evidence": "sentence_shortlist"},
        )

    monkeypatch.setattr(e2e_runner, "apply_sentence_rerank", _fake_sentence)
    monkeypatch.setattr(
        e2e_runner,
        "finalize_batch_submission",
        lambda records, *_a, **_k: 0 if len(records) == 1 else 1,
    )

    code = e2e_runner.e2e_prepare_and_submit(
        bench,
        benchmark_base=tmp_path,
        benchmark_name="b",
        benchmark_id="id",
        split="test",
        run_id="run-1",
        routing_policy="dense-only",
        retrieval_asset_dir=asset_dir,
        reranker_kind="cross_encoder",
        rerank_top_k=1,
        dry_run=True,
        sentence_rerank=True,
        sentence_rerank_top_k=5,
    )
    assert code == 0

    paths = e2e_runner.make_e2e_run_paths(
        benchmark_base=tmp_path,
        benchmark_name="b",
        benchmark_id="id",
        policy=e2e_runner.parse_routing_policy("dense-only"),
        run_id="run-1",
    )
    retrieval_row = json.loads(paths.retrieval_results_jsonl().read_text(encoding="utf-8").strip())
    evidence_row = json.loads(paths.prompt_evidence_jsonl().read_text(encoding="utf-8").strip())

    assert retrieval_row["chunks"][0]["chunk_id"] == "c_fused_1"
    assert evidence_row["chunks"][0]["chunk_id"] == "c_fused_1#s0"
