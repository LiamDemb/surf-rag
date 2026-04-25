"""Tests for evaluation-time router overlap labelling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surf_rag.evaluation.router_overlap import (
    RouterSplitSets,
    overlap_category_counts,
    router_overlap_split,
)


def test_router_overlap_split() -> None:
    s = RouterSplitSets(
        train={"a", "b"},
        dev={"c"},
        test={"d"},
    )
    assert router_overlap_split("a", s) == "train"
    assert router_overlap_split("c", s) == "dev"
    assert router_overlap_split("d", s) == "test"
    assert router_overlap_split("x", s) == "unseen"


def test_overlap_category_counts() -> None:
    s = RouterSplitSets(train={"a"}, dev=set(), test=set())
    c = overlap_category_counts(["a", "a", "z", None], s)
    assert c["train"] == 2
    assert c["unseen"] == 1
    assert c["dev"] == 0
    assert c["test"] == 0


def test_router_split_sets_from_json_path(tmp_path: Path) -> None:
    p = tmp_path / "split_question_ids.json"
    p.write_text(
        json.dumps(
            {
                "train": ["t1", "t2"],
                "dev": ["d1"],
                "test": ["e1", "e2", "e3"],
            }
        ),
        encoding="utf-8",
    )
    s = RouterSplitSets.from_json_path(p)
    assert s.train == {"t1", "t2"}
    assert s.test == {"e1", "e2", "e3"}
