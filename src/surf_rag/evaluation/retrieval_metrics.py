from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from surf_rag.benchmark.corpus_filter import contains_normalized, normalize_for_matching
from surf_rag.retrieval.types import RetrievalResult, RetrievedChunk

DEFAULT_NDCG_KS: Tuple[int, ...] = (5, 10, 20)
PRIMARY_NDCG_K: int = 10


NQ_SOURCES = {"nq", "naturalquestions", "natural_questions"}
AND_SOURCES = {"2wiki", "2wikimultihop", "2wikimultihopqa", "hotpotqa", "musique"}


def _task_mode(dataset_source: Optional[str]) -> str:
    """Return 'or' for NQ-like single-fact extraction, else 'and'."""
    if not dataset_source:
        return "and"
    key = str(dataset_source).strip().lower()
    if key in NQ_SOURCES:
        return "or"
    return "and"


def _deduped_gold_pairs(
    gold_support_sentences: Sequence[str],
) -> List[Tuple[str, str]]:
    """(normalized_key, raw_sentence) pairs, deduped by key, order preserved."""
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for s in gold_support_sentences:
        raw = str(s or "")
        n = normalize_for_matching(raw)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append((n, raw))
    return out


def _normalize_gold(gold_support_sentences: Sequence[str]) -> List[str]:
    """Normalize and dedupe gold support sentences (preserve order)."""
    return [n for n, _ in _deduped_gold_pairs(gold_support_sentences)]


def stateful_relevances(
    chunks: Sequence[RetrievedChunk],
    gold_support_sentences: Sequence[str],
    *,
    dataset_source: Optional[str],
) -> List[int]:
    """Compute stateful relevance (0 or 1) per ranked chunk."""
    gold_pairs = _deduped_gold_pairs(gold_support_sentences)
    if not chunks or not gold_pairs:
        return [0] * len(chunks)

    mode = _task_mode(dataset_source)

    relevances: List[int] = []
    credited_golds: set[str] = set()
    nq_already_credited = False

    for ch in chunks:
        chunk_raw = ch.text or ""
        if not chunk_raw.strip():
            relevances.append(0)
            continue

        if mode == "or":
            if nq_already_credited:
                relevances.append(0)
                continue
            matched = any(
                contains_normalized(chunk_raw, raw) for _, raw in gold_pairs
            )
            if matched:
                relevances.append(1)
                nq_already_credited = True
            else:
                relevances.append(0)
            continue

        # 2wiki-style AND: each new gold sentence can be credited once.
        newly_matched: Optional[str] = None
        for norm_key, raw in gold_pairs:
            if norm_key in credited_golds:
                continue
            if contains_normalized(chunk_raw, raw):
                newly_matched = norm_key
                break
        if newly_matched is not None:
            credited_golds.add(newly_matched)
            relevances.append(1)
        else:
            relevances.append(0)

    return relevances


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """DCG@k with standard logarithmic decay ``rel / log2(rank + 1)``."""
    if k <= 0:
        return 0.0
    total = 0.0
    limit = min(k, len(relevances))
    for i in range(limit):
        rel = float(relevances[i])
        if rel == 0.0:
            continue
        total += rel / math.log2(i + 2)
    return total


def ideal_dcg_at_k(
    gold_support_sentences: Sequence[str],
    k: int,
    *,
    dataset_source: Optional[str],
) -> float:
    """Task-conditioned IDCG@k.

    - OR mode (NQ): 1.0 (one perfect chunk at rank 1).
    - AND mode (2wiki): ``sum_{j=1..min(m, k)} 1 / log2(j + 1)`` where
      ``m`` is the number of distinct gold sentences.
    """
    if k <= 0:
        return 0.0
    mode = _task_mode(dataset_source)
    if mode == "or":
        golds = _normalize_gold(gold_support_sentences)
        if not golds:
            return 0.0
        return 1.0
    m = len(_normalize_gold(gold_support_sentences))
    limit = min(m, k)
    total = 0.0
    for j in range(1, limit + 1):
        total += 1.0 / math.log2(j + 1)
    return total


def ndcg_at_k(
    chunks: Sequence[RetrievedChunk],
    gold_support_sentences: Sequence[str],
    k: int,
    *,
    dataset_source: Optional[str],
) -> float:
    """Stateful, task-conditioned NDCG@k in ``[0.0, 1.0]``."""
    idcg = ideal_dcg_at_k(gold_support_sentences, k=k, dataset_source=dataset_source)
    if idcg == 0.0:
        return 0.0
    rels = stateful_relevances(
        chunks,
        gold_support_sentences,
        dataset_source=dataset_source,
    )
    dcg = dcg_at_k(rels, k=k)
    return dcg / idcg


def hit_at_k(
    chunks: Sequence[RetrievedChunk],
    gold_support_sentences: Sequence[str],
    k: int,
    *,
    dataset_source: Optional[str],
) -> float:
    """Binary: any gold support sentence appears in the top-k chunks."""
    if k <= 0:
        return 0.0
    rels = stateful_relevances(
        chunks, gold_support_sentences, dataset_source=dataset_source
    )
    return 1.0 if any(r > 0 for r in rels[:k]) else 0.0


def recall_at_k(
    chunks: Sequence[RetrievedChunk],
    gold_support_sentences: Sequence[str],
    k: int,
    *,
    dataset_source: Optional[str],
) -> float:
    """Fraction of distinct gold sentences credited in the top-k chunks.

    For OR-mode (NQ), denominator is ``1`` (one "correct" fact needed).
    For AND-mode (2wiki), denominator is the number of distinct gold
    sentences.
    """
    if k <= 0:
        return 0.0
    golds = _normalize_gold(gold_support_sentences)
    if not golds:
        return 0.0
    mode = _task_mode(dataset_source)
    denom = 1 if mode == "or" else len(golds)
    rels = stateful_relevances(
        chunks, gold_support_sentences, dataset_source=dataset_source
    )
    credited = sum(rels[:k])
    if denom == 0:
        return 0.0
    return min(1.0, credited / denom)


@dataclass(frozen=True)
class RankedMetricSuite:
    """Bundle of retrieval metrics for one ranked list at one ``k``."""

    k: int
    ndcg: float
    hit: float
    recall: float

    def to_json(self) -> Dict[str, float]:
        return {
            "k": self.k,
            "ndcg": self.ndcg,
            "hit": self.hit,
            "recall": self.recall,
        }


def compute_metric_suite(
    chunks: Sequence[RetrievedChunk],
    gold_support_sentences: Sequence[str],
    *,
    dataset_source: Optional[str],
    ks: Iterable[int] = DEFAULT_NDCG_KS,
) -> List[RankedMetricSuite]:
    """Compute NDCG/Hit/Recall at every ``k`` in ``ks``."""
    out: List[RankedMetricSuite] = []
    for k in ks:
        out.append(
            RankedMetricSuite(
                k=int(k),
                ndcg=ndcg_at_k(
                    chunks,
                    gold_support_sentences,
                    k=int(k),
                    dataset_source=dataset_source,
                ),
                hit=hit_at_k(
                    chunks,
                    gold_support_sentences,
                    k=int(k),
                    dataset_source=dataset_source,
                ),
                recall=recall_at_k(
                    chunks,
                    gold_support_sentences,
                    k=int(k),
                    dataset_source=dataset_source,
                ),
            )
        )
    return out


def score_retrieval_result(
    result: RetrievalResult,
    gold_support_sentences: Sequence[str],
    *,
    dataset_source: Optional[str],
    ks: Iterable[int] = DEFAULT_NDCG_KS,
) -> List[RankedMetricSuite]:
    """Convenience: compute metrics directly from a :class:`RetrievalResult`."""
    if result.status != "OK" or not result.chunks:
        return [RankedMetricSuite(k=int(k), ndcg=0.0, hit=0.0, recall=0.0) for k in ks]
    return compute_metric_suite(
        result.chunks,
        gold_support_sentences,
        dataset_source=dataset_source,
        ks=ks,
    )
