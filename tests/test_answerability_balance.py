from __future__ import annotations

from surf_rag.evaluation.answerability_types import build_balance_mask


def test_build_balance_mask_equal_per_source_min() -> None:
    verdicts = [
        {"question_id": "a1", "answerable": True, "dataset_source": "nq"},
        {"question_id": "a2", "answerable": True, "dataset_source": "nq"},
        {"question_id": "b1", "answerable": True, "dataset_source": "wiki"},
        {"question_id": "u1", "answerable": False, "dataset_source": "nq"},
    ]
    bal = build_balance_mask(verdicts, seed=42, policy="equal_per_source_min")
    removed = {e["question_id"] for e in bal}
    assert len(removed) == 1
    assert removed <= {"a1", "a2"}
    assert all(e["reason"] == "balance" for e in bal)


def test_build_balance_mask_reproducible_seed() -> None:
    verdicts = [
        {"question_id": f"n{i}", "answerable": True, "dataset_source": "nq"}
        for i in range(5)
    ] + [
        {"question_id": f"w{i}", "answerable": True, "dataset_source": "wiki"}
        for i in range(3)
    ]
    a = build_balance_mask(verdicts, seed=123, policy="equal_per_source_min")
    b = build_balance_mask(verdicts, seed=123, policy="equal_per_source_min")
    assert a == b
    assert len(a) == 2


def test_build_balance_mask_already_balanced() -> None:
    verdicts = [
        {"question_id": "x", "answerable": True, "dataset_source": "nq"},
        {"question_id": "y", "answerable": True, "dataset_source": "wiki"},
    ]
    assert build_balance_mask(verdicts, seed=0, policy="equal_per_source_min") == []
