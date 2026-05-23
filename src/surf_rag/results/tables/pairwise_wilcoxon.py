"""Paired Wilcoxon signed-rank tests between retrieval policies (per-query NDCG)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_hit_recall_ndcg,
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.pairwise_stats import (
    SLICE_SOURCES,
    pairwise_ndcg_outcomes,
    perfect_recall_coverage,
    qids_for_source,
    wilcoxon_mean_difference,
)
from surf_rag.results.tables.writer import write_table

_DEFAULT_BASELINE = "learned-soft"
_DEFAULT_COMPARISONS: tuple[str, ...] = ("rrf", "hard-routing", "50-50")


def _per_question_metrics(
    bundle: ResultsBundle,
    policy: str,
    *,
    wilcoxon_metric: str,
    k: int,
) -> dict[str, dict[str, float]]:
    if policy not in bundle.policies:
        raise FileNotFoundError(
            f"Policy {policy!r} not listed in results.policies; "
            "add it before building pairwise_wilcoxon."
        )
    per_q = (
        load_policy_metrics(bundle.policies[policy].metrics_path).get("per_question")
        or []
    )
    out: dict[str, dict[str, float]] = {}
    for row in per_q:
        qid = str(row.get("question_id", "")).strip()
        if not qid or qid not in bundle.split_qids:
            continue
        _hit, recall, ndcg = e2e_hit_recall_ndcg(row, k=k)
        out[qid] = {
            "wilcoxon": e2e_retrieval_value(row, metric=wilcoxon_metric, k=k),
            "ndcg": ndcg,
            "recall": recall,
        }
    return out


def build_pairwise_wilcoxon(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
) -> tuple[pd.DataFrame, dict]:
    metric, k = resolve_retrieval_metric_k(spec, bundle)
    baseline_policy = (spec.y_policy or _DEFAULT_BASELINE).strip()
    comparisons = _DEFAULT_COMPARISONS

    baseline_metrics = _per_question_metrics(
        bundle, baseline_policy, wilcoxon_metric=metric, k=k
    )
    rows: list[dict[str, object]] = []

    for comparison_policy in comparisons:
        if comparison_policy not in bundle.policies:
            bundle.warnings.append(
                f"pairwise_wilcoxon: skipping {comparison_policy!r} "
                "(not in results.policies)"
            )
            continue
        other_metrics = _per_question_metrics(
            bundle, comparison_policy, wilcoxon_metric=metric, k=k
        )
        common = sorted(set(baseline_metrics) & set(other_metrics))
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
            b_wilcoxon = np.array(
                [baseline_metrics[q]["wilcoxon"] for q in qids], dtype=np.float64
            )
            o_wilcoxon = np.array(
                [other_metrics[q]["wilcoxon"] for q in qids], dtype=np.float64
            )
            b_ndcg = np.array(
                [baseline_metrics[q]["ndcg"] for q in qids], dtype=np.float64
            )
            o_ndcg = np.array(
                [other_metrics[q]["ndcg"] for q in qids], dtype=np.float64
            )
            b_recall = np.array(
                [baseline_metrics[q]["recall"] for q in qids], dtype=np.float64
            )
            o_recall = np.array(
                [other_metrics[q]["recall"] for q in qids], dtype=np.float64
            )
            mean_diff, stat, p_val, n = wilcoxon_mean_difference(b_wilcoxon, o_wilcoxon)
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
                    **pairwise_ndcg_outcomes(b_ndcg, o_ndcg),
                    **perfect_recall_coverage(b_recall, o_recall),
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
            "ndcg_win_loss_tie": "per-query NDCG@k: baseline vs comparison (np.isclose ties)",
            "n_recall_perfect_baseline_only": (
                "recall@k == 1 for baseline only (exclusive coverage)"
            ),
            "n_recall_perfect_comparison_only": (
                "recall@k == 1 for comparison only (exclusive coverage)"
            ),
        },
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
