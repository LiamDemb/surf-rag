"""Tests for RRF vs learned-soft retrieval comparison."""

from __future__ import annotations

import json
from pathlib import Path

from surf_rag.config.schema import PipelineConfig, ResultsPolicyEntry, ResultsSection
from surf_rag.evaluation.policy_retrieval_compare import (
    extract_rrf_win_rows,
    resolve_policy_run_ids,
    retrieval_ndcg_at_k,
)
from surf_rag.evaluation.retrieval_jsonl import write_retrieval_line
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult


def _write_retrieval(path: Path, qid: str, chunk_specs: list[tuple[str, str]]) -> None:
    chunks = [
        RetrievedChunk(
            chunk_id=cid,
            text=text,
            score=1.0 - i * 0.1,
            rank=i + 1,
            metadata={},
        )
        for i, (cid, text) in enumerate(chunk_specs)
    ]
    result = RetrievalResult(
        query="What is X?",
        retriever_name="test",
        status="OK",
        chunks=chunks,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        write_retrieval_line(fp, result, qid)


def test_resolve_policy_run_ids_from_results() -> None:
    cfg = PipelineConfig(
        results=ResultsSection(
            policies={
                "rrf": ResultsPolicyEntry(run_id="run-rrf"),
                "learned-soft": ResultsPolicyEntry(run_id="run-ls"),
            }
        )
    )
    assert resolve_policy_run_ids(cfg) == ("run-rrf", "run-ls")


def test_resolve_policy_run_ids_shared_e2e() -> None:
    from dataclasses import replace

    from surf_rag.config.schema import E2ESection

    cfg = replace(PipelineConfig(), e2e=E2ESection(run_id="shared-1"))
    assert resolve_policy_run_ids(cfg) == ("shared-1", "shared-1")


def test_extract_rrf_win_rows(tmp_path: Path) -> None:
    bench = {
        "q1": {
            "question_id": "q1",
            "question": "Who?",
            "dataset_source": "nq",
            "gold_support_sentences": ["gold sentence one here"],
            "gold_answers": ["answer"],
        }
    }
    rrf_path = tmp_path / "rrf.jsonl"
    ls_path = tmp_path / "ls.jsonl"
    gold = "gold sentence one here"
    _write_retrieval(
        rrf_path,
        "q1",
        [("gold-chunk", gold), ("other-a", "irrelevant filler text")],
    )
    _write_retrieval(
        ls_path,
        "q1",
        [("other-b", "no match here"), ("other-c", "also unrelated")],
    )

    from surf_rag.evaluation.discrepancy_debug import load_retrieval_by_qid

    wins, counts = extract_rrf_win_rows(
        bench_by_qid=bench,
        retr_rrf=load_retrieval_by_qid(rrf_path),
        retr_ls=load_retrieval_by_qid(ls_path),
        restrict_qids=None,
        metric_k=10,
        epsilon_ndcg=1e-9,
        top_k_chunks=10,
        chunk_preview_chars=20,
    )
    assert counts.evaluated_question_ids == 1
    assert counts.rrf_wins == 1
    assert len(wins) == 1
    assert wins[0]["question_id"] == "q1"
    assert wins[0]["delta_ndcg"] > 0
    assert wins[0]["rrf"]["top_chunks"][0]["chunk_id"] == "gold-chunk"
    assert len(wins[0]["rrf"]["top_chunks"][0]["text_preview"]) <= 21


def test_retrieval_ndcg_zero_when_no_gold_match() -> None:
    sample = {
        "gold_support_sentences": ["unique gold xyz"],
        "dataset_source": "nq",
    }
    row = {
        "query": "q",
        "status": "OK",
        "chunks": [
            {
                "chunk_id": "c1",
                "text": "unrelated",
                "score": 1.0,
                "rank": 1,
                "metadata": {},
            }
        ],
    }
    ndcg, status, _ = retrieval_ndcg_at_k(sample, row, question_id="q1", k=10)
    assert status == "OK"
    assert ndcg == 0.0
