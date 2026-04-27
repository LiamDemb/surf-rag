"""Preflight evidence audit for sentence-window batches."""

from __future__ import annotations

from surf_rag.generation.evidence_audit import (
    audit_sentence_window_batch,
    audit_user_prompt,
)


def test_detects_full_chunk_style_separators() -> None:
    # Simulates five dense sections without W headers (cross_encoder mistake).
    huge = "word " * 400
    body = "\n\n---\n\n".join([huge] * 5)
    u = (
        "Use only evidence.\n\nEvidence (selected windows):\n"
        + body
        + "\n\nQuestion: q?\n"
    )
    s = audit_user_prompt(u, expect_sentence_windows=True)
    assert s["suspicious"] is True
    assert s["w_headers"] == 0
    b = audit_sentence_window_batch([u])
    assert b.level in ("fail", "warn")


def test_ok_sentence_window_shape() -> None:
    # Eight compact windows; typical sentence-window run shape.
    lines: list[str] = ["Evidence (selected windows):\n"]
    for i in range(8):
        lines.append(
            f"[W{i + 1} | chunk_rank={i} | title=T | window_score=0.1 | "
            f"source_chunk_id=chunk{i}]\nshort text here {i}.\n"
        )
    u = "".join(lines) + "\n\nQuestion: q?\n"
    b = audit_sentence_window_batch([u, u])
    assert b.mean_w_headers >= 8.0
    assert b.level == "ok"


def test_megawindow_block_triggers_fail() -> None:
    huge = "word " * 300
    u = (
        "Evidence (selected windows):\n"
        f"[W1 | chunk_rank=0 | title=T | window_score=0.1 | source_chunk_id=a]\n{huge}\n\n"
        "Question: q?\n"
    )
    b = audit_sentence_window_batch([u], max_window_block_words=220)
    assert b.level == "fail"
    assert "megawindow" in str(b.suspicious_flags).lower() or any(
        "megawindow" in f for f in b.suspicious_flags
    )
