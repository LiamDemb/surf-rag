"""Deterministic train/dev/test assignment stratified by benchmark ``dataset_source``."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def normalize_dataset_source(dataset_source: str) -> str:
    """Strip whitespace; empty sources bucket to ``\"unknown\"``."""
    s = str(dataset_source or "").strip()
    return s if s else "unknown"


def stratum_key_dataset_source(dataset_source: str) -> str:
    """Split stratum label (one stratum per normalized dataset source)."""
    return normalize_dataset_source(dataset_source)


def _split_sizes(
    n: int, train_r: float, dev_r: float, test_r: float
) -> Tuple[int, int, int]:
    """Allocate counts with **largest remainder** (avoids `round(0.5)->0` emptying small strata)."""
    s = float(train_r) + float(dev_r) + float(test_r)
    if s <= 0:
        raise ValueError("split ratios must sum to a positive value")
    w = (train_r / s, dev_r / s, test_r / s)
    exact = [n * x for x in w]
    floors = [int(x) for x in exact]
    rem = n - sum(floors)
    if rem < 0:
        floors[0] += rem
        return floors[0], floors[1], floors[2]
    idx_order = sorted(range(3), key=lambda i: (exact[i] - floors[i], i), reverse=True)
    for j in range(rem):
        floors[idx_order[j]] += 1
    return floors[0], floors[1], floors[2]


def assign_splits_stratified(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    """Return ``question_id -> split`` where split is train|dev|test.

    Each row must provide ``question_id`` and ``dataset_source`` (typically from
    the benchmark row). Splits are stratified per distinct source (e.g. NQ vs
    2WikiMultiHopQA) so each source gets approximately the same train/dev/test mix.
    """
    by_stratum: Dict[str, List[str]] = {}
    for r in rows:
        qid = str(r.get("question_id", "")).strip()
        if not qid:
            continue
        try:
            ds = stratum_key_dataset_source(str(r.get("dataset_source", "")))
        except (TypeError, ValueError):
            ds = "unknown"
        by_stratum.setdefault(ds, []).append(qid)

    out: Dict[str, str] = {}
    rng = random.Random(int(seed))
    for stratum, qids in by_stratum.items():
        uq = list(dict.fromkeys(qids))  # preserve order, dedupe
        rng.shuffle(uq)
        n = len(uq)
        n_tr, n_dev, n_te = _split_sizes(n, train_ratio, dev_ratio, test_ratio)
        for i, q in enumerate(uq):
            if i < n_tr:
                out[q] = "train"
            elif i < n_tr + n_dev:
                out[q] = "dev"
            else:
                out[q] = "test"
    return out


def split_summary(
    qid_to_split: Dict[str, str],
    label_rows: Sequence[Any],
) -> Dict[str, Any]:
    by_split = {"train": 0, "dev": 0, "test": 0, "other": 0}
    for sp in qid_to_split.values():
        if sp in by_split:
            by_split[sp] += 1
        else:
            by_split["other"] += 1
    return {
        "total_assigned": len(qid_to_split),
        "by_split": by_split,
        "label_rows": len(list(label_rows)),
    }
