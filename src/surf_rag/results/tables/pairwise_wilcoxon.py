"""Paired Wilcoxon signed-rank tests between retrieval policies (per-query NDCG)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.pairwise_stats import (
    SLICE_SOURCES,
    qids_for_source,
    wilcoxon_mean_difference,
)
from surf_rag.results.tables.writer import write_table

_DEFAULT_BASELINE = "learned-soft"
_DEFAULT_COMPARISONS: tuple[str, ...] = ("rrf", "hard-routing", "50-50")


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
        for source in SLICE_SOURCES:
            qids = qids_for_source(common, bundle, source)
            if not qids:
                continue
            b = np.array([baseline_scores[q] for q in qids], dtype=np.float64)
            o = np.array([other_scores[q] for q in qids], dtype=np.float64)
            mean_diff, stat, p_val, n = wilcoxon_mean_difference(b, o)
            rows.append(
                {
                    "baseline_policy": baseline_policy,
                    "comparison_policy": comparison_policy,
                    "dataset_source": source,
                    "metric": metric,
                    "k": k,
                    "n": n,
                    "mean_difference": mean_diff,
                    "wilcoxon_statistic": stat,
                    "p_value": p_val,
                }
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
