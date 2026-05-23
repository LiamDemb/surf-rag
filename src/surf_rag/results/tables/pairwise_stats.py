"""Shared paired-comparison helpers for results tables."""

from __future__ import annotations

import math

import numpy as np

from scipy.stats import wilcoxon

from surf_rag.results.bundle import ResultsBundle

SLICE_SOURCES: tuple[str, ...] = ("all", "nq", "2wiki")


def qids_for_source(qids: list[str], bundle: ResultsBundle, source: str) -> list[str]:
    if source == "all":
        return qids
    return [q for q in qids if bundle.qid_to_source.get(q, "") == source]


def pairwise_ndcg_outcomes(
    baseline_ndcg: np.ndarray,
    other_ndcg: np.ndarray,
) -> dict[str, int | float]:
    """Per-query NDCG win/loss/tie counts from the baseline's perspective."""
    tie = np.isclose(baseline_ndcg, other_ndcg)
    win = (baseline_ndcg > other_ndcg) & ~tie
    loss = (baseline_ndcg < other_ndcg) & ~tie
    n = int(baseline_ndcg.shape[0])
    n_win = int(win.sum())
    n_loss = int(loss.sum())
    n_tie = int(tie.sum())
    if n == 0:
        return {
            "n_ndcg_win": 0,
            "n_ndcg_loss": 0,
            "n_ndcg_tie": 0,
            "pct_ndcg_win": float("nan"),
            "pct_ndcg_loss": float("nan"),
            "pct_ndcg_tie": float("nan"),
        }
    return {
        "n_ndcg_win": n_win,
        "n_ndcg_loss": n_loss,
        "n_ndcg_tie": n_tie,
        "pct_ndcg_win": 100.0 * n_win / n,
        "pct_ndcg_loss": 100.0 * n_loss / n,
        "pct_ndcg_tie": 100.0 * n_tie / n,
    }


def perfect_recall_coverage(
    baseline_recall: np.ndarray,
    other_recall: np.ndarray,
) -> dict[str, int]:
    """Exclusive perfect Recall@k sets between baseline and comparison."""
    b_perfect = np.isclose(baseline_recall, 1.0)
    o_perfect = np.isclose(other_recall, 1.0)
    baseline_only = b_perfect & ~o_perfect
    comparison_only = o_perfect & ~b_perfect
    both = b_perfect & o_perfect
    neither = ~b_perfect & ~o_perfect
    return {
        "n_recall_perfect_baseline_only": int(baseline_only.sum()),
        "n_recall_perfect_comparison_only": int(comparison_only.sum()),
        "n_recall_perfect_both": int(both.sum()),
        "n_recall_perfect_neither": int(neither.sum()),
    }


def wilcoxon_mean_difference(
    baseline: np.ndarray,
    other: np.ndarray,
) -> tuple[float, float | str, float | str, int]:
    """Return (mean_diff, statistic, p_value, n); empty strings if n < 2."""
    n = int(baseline.shape[0])
    mean_diff = float(np.mean(baseline) - np.mean(other)) if n else float("nan")
    if n < 2:
        return mean_diff, "", "", n
    try:
        stat, p_value = wilcoxon(baseline, other)
        stat_f = float(stat)
        p_f = float(p_value)
        if math.isnan(stat_f):
            stat_f = 0.0
        if math.isnan(p_f):
            p_f = 1.0
        return mean_diff, stat_f, p_f, n
    except ValueError:
        return mean_diff, 0.0, 1.0, n
