"""Classify an evaluated question with respect to router training splits. group metrics by train/dev/test/unseen overlap."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Set

log = logging.getLogger(__name__)

RouterOverlapSplit: Final = Literal["train", "dev", "test", "unseen"]


@dataclass(frozen=True, slots=True)
class RouterSplitSets:
    """Sets of question IDs from a router training split file."""

    train: set[str] = field(default_factory=set)
    dev: set[str] = field(default_factory=set)
    test: set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RouterSplitSets":
        g = d.get
        tr = g("train", []) or []
        dv = g("dev", []) or []
        te = g("test", []) or []
        return cls(
            train=set(str(x) for x in tr),
            dev=set(str(x) for x in dv),
            test=set(str(x) for x in te),
        )

    @classmethod
    def from_json_path(cls, path: Path) -> "RouterSplitSets":
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_default_env(cls) -> "RouterSplitSets | None":
        path = _split_path_from_env()
        if not path or not path.is_file():
            return None
        return cls.from_json_path(path)

    @classmethod
    def from_path_or_default(cls, path: Path | None) -> "RouterSplitSets | None":
        p = path or _split_path_from_env()
        if p is None or not p.is_file():
            return None
        return cls.from_json_path(p)


def _split_path_from_env() -> Path | None:
    p = os.getenv("ROUTER_SPLIT_QUESTION_IDS")
    if not p or not p.strip():
        return None
    return Path(p)


def router_overlap_split(
    question_id: str,
    split_sets: RouterSplitSets,
) -> RouterOverlapSplit:
    """Map ``question_id`` to train/dev/test, else ``unseen``."""
    q = str(question_id)
    if q in split_sets.train:
        return "train"
    if q in split_sets.dev:
        return "dev"
    if q in split_sets.test:
        return "test"
    return "unseen"


def overlap_category_counts(
    question_ids: list[str | None] | set[str] | set[str | None],
    split_sets: RouterSplitSets,
) -> dict[RouterOverlapSplit, int]:
    """Count how many IDs fall in each category."""
    if isinstance(question_ids, set):
        items = (str(x) for x in question_ids if x is not None)
    else:
        items = (str(x) for x in question_ids if x is not None)
    counts: dict[RouterOverlapSplit, int] = {
        "train": 0,
        "dev": 0,
        "test": 0,
        "unseen": 0,
    }
    for q in items:
        counts[router_overlap_split(q, split_sets)] += 1
    return counts
