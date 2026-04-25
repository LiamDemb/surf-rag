"""Deterministic train/dev/test assignment with soft-label-aware stratification."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple


def entropy_bucket(
    entropy: float,
    q1: float,
    q2: float,
) -> str:
    """Tertile bucket labels: low / mid / high."""
    e = float(entropy)
    if e <= q1:
        return "low"
    if e <= q2:
        return "mid"
    return "high"


def _quantiles(values: Sequence[float]) -> Tuple[float, float]:
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return xs[0], xs[0]

    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo] + (idx - lo) * (xs[hi] - xs[lo])

    return q(1.0 / 3.0), q(2.0 / 3.0)


def stratum_key(argmax_weight: float, ent: float, q1: float, q2: float) -> str:
    w = float(argmax_weight)
    # One decimal for grid-aligned weights
    wk = f"{w:.1f}"
    b = entropy_bucket(ent, q1, q2)
    return f"{wk}__{b}"


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
    label_rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    """Return ``question_id -> split`` where split is train|dev|test.

    Uses ``argmax_weight`` + tertile ``entropy`` buckets for strata.
    Rows missing ``question_id`` or target fields are assigned ``train`` (defensive).
    """
    entropies = [float(r["entropy"]) for r in label_rows if "entropy" in r]
    q1, q2 = _quantiles(entropies)
    by_stratum: Dict[str, List[str]] = {}
    for r in label_rows:
        qid = str(r.get("question_id", "")).strip()
        if not qid:
            continue
        try:
            aw = float(r.get("argmax_weight", 0.0))
            ent = float(r.get("entropy", 0.0))
        except (TypeError, ValueError):
            by_stratum.setdefault("unknown__mid", []).append(qid)
            continue
        sk = stratum_key(aw, ent, q1, q2)
        by_stratum.setdefault(sk, []).append(qid)

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
    label_rows: Sequence[Dict[str, Any]],
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
