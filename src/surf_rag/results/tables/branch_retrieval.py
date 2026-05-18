"""Standalone dense/graph branch retrieval + NO_CONTEXT rates."""

from __future__ import annotations

import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.loaders import load_graph_no_context_rates, load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_hit_recall_ndcg,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.writer import write_table


def build_branch_retrieval(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    branch_policies = {"dense-only": "dense", "graph-only": "graph"}
    metric, k = resolve_retrieval_metric_k(spec, bundle)

    for policy, branch in branch_policies.items():
        if policy not in bundle.policies:
            continue
        metrics = load_policy_metrics(bundle.policies[policy].metrics_path)
        per_q = metrics.get("per_question") or []
        sums: dict[str, dict[str, float]] = {
            "all": {"hit": 0, "recall": 0, "ndcg": 0, "n": 0},
            "nq": {"hit": 0, "recall": 0, "ndcg": 0, "n": 0},
            "2wiki": {"hit": 0, "recall": 0, "ndcg": 0, "n": 0},
        }
        for row in per_q:
            qid = str(row.get("question_id", "")).strip()
            if qid not in bundle.split_qids:
                continue
            hit, recall, ndcg = e2e_hit_recall_ndcg(row, k=k)
            src = bundle.qid_to_source.get(qid, "unknown")
            for key in ("all",):
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
            rows.append(
                {
                    "branch": branch,
                    "dataset_source": src,
                    "hit": acc["hit"] / n,
                    "recall": acc["recall"] / n,
                    "ndcg": acc["ndcg"] / n,
                    "no_context_rate": "",
                    "n": int(n),
                }
            )

    nc_rates = load_graph_no_context_rates(
        bundle.retrieval_graph_path, bundle.qid_to_source
    )
    for src, info in nc_rates.items():
        rows.append(
            {
                "branch": "graph",
                "dataset_source": src,
                "hit": "",
                "recall": "",
                "ndcg": "",
                "no_context_rate": info["no_context_rate"],
                "n": int(info["n"]),
            }
        )

    df = pd.DataFrame(rows)
    paths = write_table(bundle, "branch_retrieval", df, {"metric": metric, "k": k})
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
