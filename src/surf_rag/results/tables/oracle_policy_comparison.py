"""Paired comparison of oracle-upper-bound vs oracle-classification retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import e2e_hit_recall_ndcg, resolve_retrieval_ks
from surf_rag.results.tables.pairwise_stats import (
    SLICE_SOURCES,
    qids_for_source,
    wilcoxon_mean_difference,
)
from surf_rag.results.tables.writer import write_table

ORACLE_UPPER_BOUND = "oracle-upper-bound"
ORACLE_CLASSIFICATION = "oracle-classification"
_RETRIEVAL_METRICS: tuple[str, ...] = ("ndcg", "hit", "recall")


def _per_question_metrics(
    bundle: ResultsBundle,
    policy: str,
    *,
    k: int,
) -> dict[str, dict[str, float]]:
    if policy not in bundle.policies:
        raise FileNotFoundError(
            f"Policy {policy!r} not in results.policies "
            f"(required for oracle_policy_comparison)."
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
        hit, recall, ndcg = e2e_hit_recall_ndcg(row, k=k)
        out[qid] = {"hit": hit, "recall": recall, "ndcg": ndcg}
    return out


def build_oracle_policy_comparison(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
) -> tuple[pd.DataFrame, dict]:
    if ORACLE_UPPER_BOUND not in bundle.policies:
        raise FileNotFoundError(f"{ORACLE_UPPER_BOUND!r} missing from results.policies")
    if ORACLE_CLASSIFICATION not in bundle.policies:
        raise FileNotFoundError(
            f"{ORACLE_CLASSIFICATION!r} missing from results.policies"
        )

    ks = resolve_retrieval_ks(spec, bundle)
    rows: list[dict[str, object]] = []

    for k in ks:
        upper = _per_question_metrics(bundle, ORACLE_UPPER_BOUND, k=k)
        cls = _per_question_metrics(bundle, ORACLE_CLASSIFICATION, k=k)
        common = sorted(set(upper) & set(cls))
        if not common:
            bundle.warnings.append(
                "oracle_policy_comparison: no overlapping test questions "
                f"between {ORACLE_UPPER_BOUND!r} and {ORACLE_CLASSIFICATION!r}"
            )
            continue
        for metric in _RETRIEVAL_METRICS:
            for source in SLICE_SOURCES:
                qids = qids_for_source(common, bundle, source)
                if not qids:
                    continue
                u = np.array([upper[q][metric] for q in qids], dtype=np.float64)
                c = np.array([cls[q][metric] for q in qids], dtype=np.float64)
                mean_diff, stat, p_val, n = wilcoxon_mean_difference(u, c)
                rows.append(
                    {
                        "dataset_source": source,
                        "metric": metric,
                        "k": k,
                        "oracle_upper_bound_mean": float(np.mean(u)),
                        "oracle_classification_mean": float(np.mean(c)),
                        "mean_difference": mean_diff,
                        "wilcoxon_statistic": stat,
                        "p_value": p_val,
                        "n": n,
                    }
                )

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "oracle_policy_comparison"
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {
            "oracle_upper_bound_policy": ORACLE_UPPER_BOUND,
            "oracle_classification_policy": ORACLE_CLASSIFICATION,
            "ks": ks,
            "metrics": list(_RETRIEVAL_METRICS),
            "test": "wilcoxon_signed_rank",
            "mean_difference": (
                f"mean({ORACLE_UPPER_BOUND}) - mean({ORACLE_CLASSIFICATION})"
            ),
        },
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
