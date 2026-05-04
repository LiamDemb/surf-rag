from __future__ import annotations

from pathlib import Path

from surf_rag.evaluation.answerability_types import (
    audit_entries_from_verdicts,
    build_manifest_document,
    build_mask_document,
)


def test_audit_entries_from_verdicts_dedupes() -> None:
    rows = [
        {"question_id": "q1", "answerable": False, "dataset_source": "nq"},
        {"question_id": "q1", "answerable": False, "dataset_source": "nq"},
        {"question_id": "q2", "answerable": True, "dataset_source": "nq"},
    ]
    ent = audit_entries_from_verdicts(rows)
    assert ent == [{"question_id": "q1", "reason": "audit"}]


def test_build_manifest_document_keys_and_counts(tmp_path: Path) -> None:
    bench = tmp_path / "benchmark" / "benchmark.jsonl"
    verdicts = [
        {"question_id": "a", "answerable": True, "dataset_source": "nq"},
        {"question_id": "b", "answerable": True, "dataset_source": "wiki"},
        {"question_id": "c", "answerable": False, "dataset_source": "nq"},
    ]
    audit = audit_entries_from_verdicts(verdicts)
    balance = [{"question_id": "b", "reason": "balance"}]
    mask = build_mask_document(audit_entries=audit, balance_entries=balance)["entries"]
    manifest = build_manifest_document(
        benchmark_path=bench,
        audit_model="gpt-4o-mini",
        prompt_id="p1",
        verdict_rows=verdicts,
        mask_entries=mask,
        balance_enabled=True,
        balance_policy="equal_per_source_min",
        balance_seed=99,
    )
    assert manifest["schema_version"] == 1
    assert manifest["audit_model"] == "gpt-4o-mini"
    assert manifest["totals"]["questions_in_verdicts"] == 3
    assert manifest["excluded"]["n_audit"] == 1
    assert manifest["excluded"]["n_balance"] == 1
    assert manifest["primary_eval"]["n_total"] == 1
    assert manifest["primary_eval"]["by_source"] == {"nq": 1}
    assert manifest["balance_removals_by_source"] == {"wiki": 1}
