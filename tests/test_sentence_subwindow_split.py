"""Subwindow splitting for oversized sentence-window spans."""

from __future__ import annotations

from surf_rag.reranking.sentence_windows import _refine_span_to_subspans, _word_count


def test_refine_splits_huge_whole_chunk_span() -> None:
    text = " ".join([f"w{i}" for i in range(400)])
    spans = _refine_span_to_subspans(text, 0, len(text), max_w=50)
    assert len(spans) >= 2
    for s0, s1 in spans:
        assert _word_count(text[s0:s1]) <= 55


def test_refine_keeps_small_span() -> None:
    text = "One two three four five."
    spans = _refine_span_to_subspans(text, 0, len(text), max_w=50)
    assert spans == [(0, len(text))]
