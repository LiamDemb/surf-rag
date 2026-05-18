"""Paired Wilcoxon signed-rank tests between retrieval policies (per-query NDCG)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.writer import write_table

_DEFAULT_BASELINE = "learned-soft"
_DEFAULT_COMPARISONS: tuple[str, ...] = ("rrf", "hard-routing", "50-50")
_SOURCES: tuple[str, ...] = ("all", "nq", "2wiki")


def _per_question_scores(
    bundle: ResultsBundle,
    policy: str,
    *,
    metric: str,
    k: int,
) -> dict[str, float]:
    if policy not in bundle.policies:
        raise FileNotFoundError(
            f"Policy {policy!r} not listed in results.policies; "
            "add it before building pairwise_wilcoxon."
        )
    per_q = (
        load_policy_metrics(bundle.policies[policy].metrics_path).get("per_question")
        or []
    )
    out: dict[str, float] = {}
    for row in per_q:
        qid = str(row.get("question_id", "")).strip()
        if qid and qid in bundle.split_qids:
            out[qid] = e2e_retrieval_value(row, metric=metric, k=k)
    return out


def _qids_for_source(qids: list[str], bundle: ResultsBundle, source: str) -> list[str]:
    if source == "all":
        return qids
    return [q for q in qids if bundle.qid_to_source.get(q, "") == source]


def _wilcoxon_row(
    *,
    baseline_policy: str,
    comparison_policy: str,
    dataset_source: str,
    metric: str,
    k: int,
    baseline: np.ndarray,
    other: np.ndarray,
) -> dict[str, object]:
    n = int(baseline.shape[0])
    mean_diff = float(np.mean(baseline) - np.mean(other)) if n else float("nan")
    if n < 2:
        return {
            "baseline_policy": baseline_policy,
            "comparison_policy": comparison_policy,
            "dataset_source": dataset_source,
            "metric": metric,
            "k": k,
            "n": n,
            "mean_difference": mean_diff,
            "wilcoxon_statistic": "",
            "p_value": "",
        }
    try:
        stat, p_value = wilcoxon(baseline, other)
        stat_f = float(stat)
        p_f = float(p_value)
        if math.isnan(stat_f):
            stat_f = 0.0
        if math.isnan(p_f):
            p_f = 1.0
    except ValueError:
        stat_f = 0.0
        p_f = 1.0
    return {
        "baseline_policy": baseline_policy,
        "comparison_policy": comparison_policy,
        "dataset_source": dataset_source,
        "metric": metric,
        "k": k,
        "n": n,
        "mean_difference": mean_diff,
        "wilcoxon_statistic": stat_f,
        "p_value": p_f,
    }


def build_pairwise_wilcoxon(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
) -> tuple[pd.DataFrame, dict]:
    metric, k = resolve_retrieval_metric_k(spec, bundle)
    baseline_policy = (spec.y_policy or _DEFAULT_BASELINE).strip()
    comparisons = _DEFAULT_COMPARISONS

    baseline_scores = _per_question_scores(bundle, baseline_policy, metric=metric, k=k)
    rows: list[dict[str, object]] = []

    for comparison_policy in comparisons:
        if comparison_policy not in bundle.policies:
            bundle.warnings.append(
                f"pairwise_wilcoxon: skipping {comparison_policy!r} "
                "(not in results.policies)"
            )
            continue
        other_scores = _per_question_scores(
            bundle, comparison_policy, metric=metric, k=k
        )
        common = sorted(set(baseline_scores) & set(other_scores))
        if not common:
            bundle.warnings.append(
                f"pairwise_wilcoxon: no overlapping questions for "
                f"{baseline_policy!r} vs {comparison_policy!r}"
            )
            continue
        for source in _SOURCES:
            qids = _qids_for_source(common, bundle, source)
            if not qids:
                continue
            b = np.array([baseline_scores[q] for q in qids], dtype=np.float64)
            o = np.array([other_scores[q] for q in qids], dtype=np.float64)
            rows.append(
                _wilcoxon_row(
                    baseline_policy=baseline_policy,
                    comparison_policy=comparison_policy,
                    dataset_source=source,
                    metric=metric,
                    k=k,
                    baseline=b,
                    other=o,
                )
            )

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "pairwise_wilcoxon"
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {
            "baseline_policy": baseline_policy,
            "comparisons": list(comparisons),
            "metric": metric,
            "k": k,
            "test": "wilcoxon_signed_rank",
            "mean_difference": "mean(baseline) - mean(comparison)",
        },
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
