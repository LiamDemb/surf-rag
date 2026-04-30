"""Tests for stratified split assignment."""

from __future__ import annotations

from surf_rag.router.splits import assign_splits_stratified


def test_assign_splits_produces_all_splits() -> None:
    rows = []
    for i in range(30):
        rows.append(
            {
                "question_id": f"q{i}",
                "oracle_best_weight": float(f"{(i % 11) / 10:.1f}"),
                "oracle_curve_std": 0.1 * i,
            }
        )
    m = assign_splits_stratified(
        rows, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=0
    )
    assert set(m.values()) <= {"train", "dev", "test"}
    assert len(m) == 30
    assert all(v in ("train", "dev", "test") for v in m.values())
    assert (
        list(m.values()).count("train")
        + list(m.values()).count("dev")
        + list(m.values()).count("test")
        == 30
    )
