"""All-pipeline retrieval metrics table."""

from __future__ import annotations

import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_hit_recall_ndcg,
    resolve_retrieval_ks,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.writer import write_table

DEFAULT_EXCLUDE: list[str] = []


def _metric_column_name(metric: str) -> str:
    m = metric.strip().lower()
    if m in ("stateful_ndcg", "ndcg"):
        return "ndcg"
    if m in ("hit", "recall"):
        return m
    raise ValueError(f"Unsupported retrieval metric: {metric!r}")


def build_pipeline_retrieval(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
) -> tuple[pd.DataFrame, dict]:
    if spec.exclude_policies is not None:
        exclude = set(spec.exclude_policies)
    else:
        exclude = set(DEFAULT_EXCLUDE)
    policies = policy_list(bundle, exclude=list(exclude))
    ks = resolve_retrieval_ks(spec, bundle)
    narrow_metric = spec.metric
    col_name = _metric_column_name(narrow_metric) if narrow_metric else None
    rows: list[dict] = []

    for policy in policies:
        metrics = load_policy_metrics(bundle.policies[policy].metrics_path)
        per_q = metrics.get("per_question") or []
        for k in ks:
            sums: dict[str, dict[str, float]] = {
                s: {"hit": 0, "recall": 0, "ndcg": 0, "n": 0}
                for s in ("all", "nq", "2wiki")
            }
            for row in per_q:
                qid = str(row.get("question_id", "")).strip()
                if qid not in bundle.split_qids:
                    continue
                hit, recall, ndcg = e2e_hit_recall_ndcg(row, k=k)
                src = bundle.qid_to_source.get(qid, "unknown")
                sums["all"]["hit"] += hit
                sums["all"]["recall"] += recall
                sums["all"]["ndcg"] += ndcg
                sums["all"]["n"] += 1
                if src in sums:
                    sums[src]["hit"] += hit
                    sums[src]["recall"] += recall
                    sums[src]["ndcg"] += ndcg
                    sums[src]["n"] += 1
            for src, acc in sums.items():
                n = acc["n"]
                if n == 0:
                    continue
                if col_name is not None:
                    rows.append(
                        {
                            "policy": policy,
                            "dataset_source": src,
                            "k": k,
                            col_name: acc[col_name] / n,
                            "n": int(n),
                        }
                    )
                else:
                    rows.append(
                        {
                            "policy": policy,
                            "dataset_source": src,
                            "k": k,
                            "hit": acc["hit"] / n,
                            "recall": acc["recall"] / n,
                            "ndcg": acc["ndcg"] / n,
                            "n": int(n),
                        }
                    )

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "pipeline_retrieval"
    headline_metric, headline_k = resolve_retrieval_metric_k(spec, bundle)
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {
            "exclude_policies": list(exclude),
            "metric": narrow_metric or "",
            "headline_metric": headline_metric,
            "headline_k": headline_k,
            "ks": ks,
            "mode": "narrow" if col_name else "wide",
        },
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
