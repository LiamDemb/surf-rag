"""Answer-level exact match and token F1 (max over gold strings)."""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Sequence


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles, remove punctuation (HotpotQA-style simplification)."""
    t = (text or "").lower()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())
    return t.strip()


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    """1.0 if normalized prediction matches any normalized gold."""
    pn = normalize_answer(prediction)
    if not pn:
        return 0.0
    for g in golds:
        if normalize_answer(str(g)) == pn:
            return 1.0
    return 0.0


def _f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counter: dict[str, int] = {}
    for t in gold_tokens:
        gold_counter[t] = gold_counter.get(t, 0) + 1
    tp = 0
    pred_counter: dict[str, int] = {}
    for t in pred_tokens:
        pred_counter[t] = pred_counter.get(t, 0) + 1
    for t, pc in pred_counter.items():
        gc = gold_counter.get(t, 0)
        tp += min(pc, gc)
    prec = tp / max(len(pred_tokens), 1)
    rec = tp / max(len(gold_tokens), 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def f1_score(prediction: str, gold: str) -> float:
    pt = normalize_answer(prediction).split()
    gt = normalize_answer(gold).split()
    return _f1(pt, gt)


def max_f1_over_golds(prediction: str, golds: Iterable[str]) -> float:
    """Best token F1 vs any single gold reference."""
    best = 0.0
    for g in golds:
        best = max(best, f1_score(prediction, str(g)))
    return best
