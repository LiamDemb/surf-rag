from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

EPS = 1e-12


def _ensure_finite(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            fv = 0.0
        out.append(fv)
    return out


def beta_scaled_softmax(scores: Sequence[float], beta: float) -> List[float]:
    """Numerically-stable ``softmax(beta * scores)`` over a 1D sequence.

    - ``beta < 0`` raises :class:`ValueError`.
    - ``beta == 0`` returns a uniform distribution.
    - Empty ``scores`` returns an empty list.
    - If every score is equal, the result is uniform regardless of ``beta``.
    """
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta!r}")
    xs = _ensure_finite(scores)
    n = len(xs)
    if n == 0:
        return []
    if beta == 0.0:
        return [1.0 / n for _ in xs]

    scaled = [beta * x for x in xs]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / n for _ in xs]
    return [e / total for e in exps]


def entropy(distribution: Sequence[float]) -> float:
    """Shannon entropy of a probability distribution, in nats.

    A uniform distribution of length ``n`` has entropy ``ln(n)``.
    A degenerate one-hot distribution has entropy ``0``.
    """
    total = 0.0
    for p in distribution:
        p = float(p)
        if p <= 0.0:
            continue
        total -= p * math.log(p + EPS) if p > 0 else 0.0
    # Simpler form without epsilon for numerical clarity:
    total = 0.0
    for p in distribution:
        p = float(p)
        if p <= 0.0:
            continue
        total -= p * math.log(p)
    return total


def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
) -> float:
    """KL(p || q) in nats, skipping bins where ``p_i == 0``.

    Used for diagnostic comparison between soft labels at different betas
    and (later) for validating router predictions against targets. The
    actual training-time loss uses torch's KL, but this provides the
    reference computation for tests and beta-sweep reports.
    """
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    total = 0.0
    for pi, qi in zip(p, q):
        pi = float(pi)
        qi = float(qi)
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return math.inf
        total += pi * math.log(pi / qi)
    return total


def expected_weight(
    distribution: Sequence[float],
    weight_grid: Sequence[float],
) -> float:
    """Expected value ``sum(p_i * w_i)`` over the fixed grid."""
    if len(distribution) != len(weight_grid):
        raise ValueError("distribution and weight_grid must have the same length")
    return float(sum(float(p) * float(w) for p, w in zip(distribution, weight_grid)))


def soft_label_from_scores(
    scores: Sequence[float],
    weight_grid: Sequence[float],
    beta: float,
) -> Dict[str, Any]:
    """Build one soft-label record from a raw per-bin score vector.

    Returned dict (JSONL-friendly):
        {
            "beta": float,
            "weight_grid": [...],
            "scores": [...],
            "distribution": [...],
            "expected_weight": float,
            "argmax_index": int,
            "argmax_weight": float,
            "entropy": float,
        }
    """
    if len(scores) != len(weight_grid):
        raise ValueError("scores and weight_grid must have the same length")
    distribution = beta_scaled_softmax(scores, beta)
    # Tie-break prefers dense_weight nearest 0.5
    best_idx = (
        max(
            range(len(scores)),
            key=lambda i: (scores[i], -abs(float(weight_grid[i]) - 0.5)),
        )
        if scores
        else 0
    )
    return {
        "beta": float(beta),
        "weight_grid": [float(w) for w in weight_grid],
        "scores": [float(s) for s in scores],
        "distribution": distribution,
        "expected_weight": expected_weight(distribution, weight_grid),
        "argmax_index": int(best_idx),
        "argmax_weight": float(weight_grid[best_idx]) if weight_grid else 0.0,
        "entropy": entropy(distribution),
    }


def _extract_score_vector(row: Dict[str, Any]) -> List[float]:
    return [float(s.get("ndcg_primary", 0.0)) for s in row.get("scores", [])]


def _weight_grid_from_row(row: Dict[str, Any]) -> List[float]:
    return [float(w) for w in row.get("weight_grid", [])]


def materialize_soft_labels(
    oracle_score_rows: Iterable[Dict[str, Any]],
    beta: float,
    output_path: Path,
) -> int:
    """Read oracle score rows and write per-question soft labels to JSONL.

    The output row shape is::

        {
            "question_id": "...",
            "dataset_source": "...",
            "beta": beta,
            "weight_grid": [...],
            "scores": [...],
            "distribution": [...],
            "expected_weight": float,
            "argmax_index": int,
            "argmax_weight": float,
            "entropy": float,
        }
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in oracle_score_rows:
            qid = str(row.get("question_id", "")).strip()
            if not qid:
                continue
            scores = _extract_score_vector(row)
            grid = _weight_grid_from_row(row)
            label = soft_label_from_scores(scores, grid, beta=beta)
            record = {
                "question_id": qid,
                "dataset_source": row.get("dataset_source", ""),
                **label,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


@dataclass(frozen=True)
class BetaSweepStats:
    """Per-beta aggregate stats used to pick a recommended beta.

    ``mean_entropy`` is the mean Shannon entropy of the resulting soft
    labels across questions (in nats). Large entropy = flat labels, small
    entropy = peaked labels. ``mean_expected_weight`` is useful as a
    sanity check across beta values.
    """

    beta: float
    num_questions: int
    mean_entropy: float
    mean_expected_weight: float
    mean_argmax_weight: float

    def to_json(self) -> Dict[str, float]:
        return {
            "beta": float(self.beta),
            "num_questions": int(self.num_questions),
            "mean_entropy": float(self.mean_entropy),
            "mean_expected_weight": float(self.mean_expected_weight),
            "mean_argmax_weight": float(self.mean_argmax_weight),
        }


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def sweep_beta(
    oracle_score_rows: Sequence[Dict[str, Any]],
    betas: Sequence[float],
) -> List[BetaSweepStats]:
    """Run the diagnostic sweep used by ``sweep-beta`` to pick beta.

    For each beta: materialize soft labels in-memory and aggregate
    entropy / expected-weight statistics so they can be inspected before
    committing to one beta for the canonical ``labels/selected.jsonl``.
    """
    stats: List[BetaSweepStats] = []
    for beta in betas:
        entropies: List[float] = []
        expected_ws: List[float] = []
        argmax_ws: List[float] = []
        for row in oracle_score_rows:
            scores = _extract_score_vector(row)
            grid = _weight_grid_from_row(row)
            label = soft_label_from_scores(scores, grid, beta=beta)
            entropies.append(float(label["entropy"]))
            expected_ws.append(float(label["expected_weight"]))
            argmax_ws.append(float(label["argmax_weight"]))
        stats.append(
            BetaSweepStats(
                beta=float(beta),
                num_questions=len(oracle_score_rows),
                mean_entropy=_mean(entropies),
                mean_expected_weight=_mean(expected_ws),
                mean_argmax_weight=_mean(argmax_ws),
            )
        )
    return stats
