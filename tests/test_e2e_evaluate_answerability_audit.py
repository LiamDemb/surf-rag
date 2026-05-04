from __future__ import annotations

import json
from pathlib import Path

from surf_rag.evaluation.answerability_types import (
    audit_entries_from_verdicts,
    build_balance_mask,
    build_manifest_document,
    build_mask_document,
    write_json,
)
from surf_rag.evaluation.answerability_layout import (
    answerability_manifest_path,
    answerability_mask_path,
    answerability_verdicts_path,
)
from surf_rag.evaluation.e2e_runner import evaluate_e2e_run
from surf_rag.evaluation.run_artifacts import RunArtifactPaths


def _write_bench(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_retrieval(path: Path, qid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "question_id": qid,
                "query": "q",
                "retriever_name": "Dense",
                "status": "OK",
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "text": "gold",
                        "score": 1.0,
                        "rank": 0,
                        "metadata": {},
                    }
                ],
                "latency_ms": {"retrieval_stage_total_ms": 1.0},
                "error": None,
                "debug_info": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_answers(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, question, ans in rows:
            f.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "question": question,
                        "gold_answers": ["x"],
                        "answer": ans,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_evaluate_without_audit_unchanged_keys(tmp_path: Path) -> None:
    bench = tmp_path / "b" / "benchmark" / "benchmark.jsonl"
    _write_bench(
        bench,
        [
            {
                "question_id": "q1",
                "question": "a?",
                "gold_answers": ["a"],
                "gold_support_sentences": ["gold"],
                "dataset_source": "nq",
            }
        ],
    )
    run = RunArtifactPaths(run_root=tmp_path / "run")
    run.ensure_dirs()
    run.manifest.write_text(
        json.dumps({"e2e": {"reranker": "none", "rerank_top_k": 10}}),
        encoding="utf-8",
    )
    _write_retrieval(run.retrieval_results_jsonl(), "q1")
    _write_retrieval(run.retrieval_results_pretrunc_jsonl(), "q1")
    _write_answers(run.generation_answers_jsonl(), [("q1", "a?", "a")])

    rep = evaluate_e2e_run(
        run_paths=run, benchmark_path=bench, apply_answerability_audit=False
    )
    assert "answerability_audit" not in rep
    assert "overlap_breakdown_primary" not in rep
    assert "audit" not in rep["per_question"][0]


def test_evaluate_with_audit_per_question_and_manifest(tmp_path: Path) -> None:
    bench = tmp_path / "bench" / "v1" / "benchmark" / "benchmark.jsonl"
    rows = [
        {
            "question_id": "a",
            "question": "1?",
            "gold_answers": ["1"],
            "gold_support_sentences": ["g"],
            "dataset_source": "nq",
        },
        {
            "question_id": "b",
            "question": "2?",
            "gold_answers": ["2"],
            "gold_support_sentences": ["g"],
            "dataset_source": "wiki",
        },
        {
            "question_id": "c",
            "question": "3?",
            "gold_answers": ["3"],
            "gold_support_sentences": ["g"],
            "dataset_source": "nq",
        },
    ]
    _write_bench(bench, rows)
    verdicts = [
        {"question_id": "a", "answerable": True, "dataset_source": "nq"},
        {"question_id": "b", "answerable": True, "dataset_source": "wiki"},
        {"question_id": "c", "answerable": False, "dataset_source": "nq"},
    ]
    vp = answerability_verdicts_path(bench)
    vp.parent.mkdir(parents=True, exist_ok=True)
    with vp.open("w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    audit_ent = audit_entries_from_verdicts(verdicts)
    bal_ent = build_balance_mask(verdicts, seed=0, policy="equal_per_source_min")
    mask_doc = build_mask_document(audit_entries=audit_ent, balance_entries=bal_ent)
    write_json(answerability_mask_path(bench), mask_doc)
    manifest = build_manifest_document(
        benchmark_path=bench,
        audit_model="m",
        prompt_id="p",
        verdict_rows=verdicts,
        mask_entries=mask_doc["entries"],
        balance_enabled=True,
        balance_policy="equal_per_source_min",
        balance_seed=0,
    )
    write_json(answerability_manifest_path(bench), manifest)

    run = RunArtifactPaths(run_root=tmp_path / "run2")
    run.ensure_dirs()
    run.manifest.write_text(
        json.dumps({"e2e": {"reranker": "none", "rerank_top_k": 10}}),
        encoding="utf-8",
    )
    for qid in ("a", "b", "c"):
        _write_retrieval(run.retrieval_results_jsonl(), qid)
        # append - need multiple lines in one file
    rpath = run.retrieval_results_jsonl()
    rpath.parent.mkdir(parents=True, exist_ok=True)
    with rpath.open("w", encoding="utf-8") as rf:
        for qid in ("a", "b", "c"):
            rf.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "query": "q",
                        "retriever_name": "Dense",
                        "status": "OK",
                        "chunks": [
                            {
                                "chunk_id": "c1",
                                "text": "gold",
                                "score": 1.0,
                                "rank": 0,
                                "metadata": {},
                            }
                        ],
                        "latency_ms": {"retrieval_stage_total_ms": 1.0},
                        "error": None,
                        "debug_info": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with run.retrieval_results_pretrunc_jsonl().open("w", encoding="utf-8") as rf:
        for qid in ("a", "b", "c"):
            rf.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "query": "q",
                        "retriever_name": "Dense",
                        "status": "OK",
                        "chunks": [
                            {
                                "chunk_id": "c1",
                                "text": "gold",
                                "score": 1.0,
                                "rank": 0,
                                "metadata": {},
                            }
                        ],
                        "latency_ms": {"retrieval_stage_total_ms": 1.0},
                        "error": None,
                        "debug_info": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    _write_answers(
        run.generation_answers_jsonl(),
        [("a", "1?", "1"), ("b", "2?", "2"), ("c", "3?", "wrong")],
    )

    rep = evaluate_e2e_run(
        run_paths=run, benchmark_path=bench, apply_answerability_audit=True
    )
    assert "answerability_audit" not in rep
    assert "overlap_breakdown_primary" not in rep
    assert "overlap_breakdown_audit_excluded" not in rep
    assert rep["overlap_breakdown"]["all"]["count"] == 3
    by_q = {r["question_id"]: r for r in rep["per_question"]}
    assert by_q["c"]["audit"]["answerable"] is False
    assert by_q["c"]["audit"]["in_primary_eval"] is False
    assert by_q["a"]["audit"]["in_primary_eval"] is True
