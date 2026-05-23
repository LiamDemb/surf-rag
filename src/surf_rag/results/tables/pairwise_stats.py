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
