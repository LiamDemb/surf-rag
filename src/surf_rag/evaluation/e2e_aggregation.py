"""Aggregate retrieval + QA metrics by router overlap split (train/dev/test/unseen)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from surf_rag.evaluation.qa_metrics import exact_match, max_f1_over_golds
from surf_rag.evaluation.retrieval_metrics import (
    DEFAULT_NDCG_KS,
    RankedMetricSuite,
    score_retrieval_result,
)
from surf_rag.evaluation.router_overlap import (
    RouterOverlapSplit,
    RouterSplitSets,
    router_overlap_split,
)
from surf_rag.retrieval.types import RetrievalResult

OVERLAP_KEYS: tuple[RouterOverlapSplit | str, ...] = (
    "all",
    "train",
    "dev",
    "test",
    "unseen",
)


@dataclass
class PerQuestionEval:
    question_id: str
    retrieval_suites: List[RankedMetricSuite]
    em: float
    f1: float


def _empty_retrieval_means(ks: Sequence[int]) -> dict[str, dict[str, float]]:
    return {str(k): {"ndcg": 0.0, "hit": 0.0, "recall": 0.0} for k in ks}


def _mean_retrieval(
    rows: Sequence[PerQuestionEval], ks: Sequence[int]
) -> dict[str, dict[str, float]]:
    if not rows:
        return _empty_retrieval_means(ks)
    out: dict[str, dict[str, float]] = {}
    for k in ks:
        ndcg = hit = rec = 0.0
        for r in rows:
            for suite in r.retrieval_suites:
                if suite.k == k:
                    ndcg += suite.ndcg
                    hit += suite.hit
                    rec += suite.recall
                    break
        n = float(len(rows))
        out[str(k)] = {"ndcg": ndcg / n, "hit": hit / n, "recall": rec / n}
    return out


def _mean_qa(rows: Sequence[PerQuestionEval]) -> dict[str, float]:
    if not rows:
        return {"em": 0.0, "f1": 0.0}
    n = float(len(rows))
    return {
        "em": sum(r.em for r in rows) / n,
        "f1": sum(r.f1 for r in rows) / n,
    }


def bucket_rows(
    rows: Sequence[PerQuestionEval],
    split_sets: Optional[RouterSplitSets],
) -> dict[str, List[PerQuestionEval]]:
    """Partition rows into overlap buckets; ``all`` is the full list."""
    buckets: dict[str, List[PerQuestionEval]] = {k: [] for k in OVERLAP_KEYS}
    buckets["all"] = list(rows)
    if split_sets is None:
        return buckets
    for r in rows:
        cat = router_overlap_split(r.question_id, split_sets)
        buckets[cat].append(r)
    return buckets


def aggregate_per_question(
    question_id: str,
    *,
    result: RetrievalResult,
    gold_support_sentences: Sequence[str],
    dataset_source: Optional[str],
    gold_answers: Sequence[str],
    prediction: str,
    ks: Iterable[int] = DEFAULT_NDCG_KS,
) -> PerQuestionEval:
    """Single-row retrieval suites + QA scores (EM / max-F1 over gold answers)."""
    k_list = [int(x) for x in ks]
    suites = score_retrieval_result(
        result,
        gold_support_sentences,
        dataset_source=dataset_source,
        ks=k_list,
    )
    golds = [str(g) for g in gold_answers if str(g).strip()]
    pred = prediction or ""
    em = exact_match(pred, golds) if golds else 0.0
    f1 = max_f1_over_golds(pred, golds) if golds else 0.0
    return PerQuestionEval(
        question_id=str(question_id),
        retrieval_suites=suites,
        em=em,
        f1=f1,
    )


def aggregate_e2e_report(
    rows: Sequence[PerQuestionEval],
    *,
    split_sets: Optional[RouterSplitSets],
    ks: Iterable[int] = DEFAULT_NDCG_KS,
) -> dict[str, Any]:
    """Nested JSON-serializable report: each overlap key → counts, retrieval@k, QA."""
    k_list = [int(x) for x in ks]
    buckets = bucket_rows(rows, split_sets)
    report: dict[str, Any] = {}
    for key in OVERLAP_KEYS:
        bucket = buckets.get(key, [])
        report[key] = {
            "count": len(bucket),
            "retrieval_at_k": _mean_retrieval(bucket, k_list),
            "qa": _mean_qa(bucket),
        }
    return report


def load_benchmark_index(path: Path | str) -> dict[str, dict]:
    """question_id -> raw JSONL row."""
    p = Path(path)
    by_id: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "") or "").strip()
            if qid:
                by_id[qid] = row
    return by_id
