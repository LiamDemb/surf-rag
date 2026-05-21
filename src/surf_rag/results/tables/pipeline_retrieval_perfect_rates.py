"""Per-pipeline share of questions with perfect NDCG@k and Recall@k."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_hit_recall_ndcg,
    resolve_retrieval_metric_k,
)
from surf_rag.results.tables.writer import write_table


def _is_perfect(score: float) -> bool:
    return bool(np.isclose(float(score), 1.0))


def _qa_correct(row: dict) -> bool | None:
    judge = row.get("qa_llm_judge")
    if not isinstance(judge, dict) or "correct" not in judge:
        return None
    return bool(judge["correct"])


def _mean_accuracy(values: list[bool]) -> float:
    if not values:
        return float("nan")
    return float(sum(1 for v in values if v) / len(values))


def build_pipeline_retrieval_perfect_rates(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
) -> tuple[pd.DataFrame, dict]:
    _, default_k = resolve_retrieval_metric_k(spec, bundle)
    k = int(spec.k if spec.k is not None else default_k)
    policies = policy_list(bundle)
    rows: list[dict] = []

    for policy in policies:
        metrics = load_policy_metrics(bundle.policies[policy].metrics_path)
        per_q = metrics.get("per_question") or []
        counts: dict[str, dict] = {
            s: {
                "n": 0,
                "perfect_ndcg": 0,
                "perfect_recall": 0,
                "qa_perfect_ndcg": [],
                "qa_perfect_recall": [],
            }
            for s in ("all", "nq", "2wiki")
        }
        for row in per_q:
            qid = str(row.get("question_id", "")).strip()
            if qid not in bundle.split_qids:
                continue
            _hit, recall, ndcg = e2e_hit_recall_ndcg(row, k=k)
            src = bundle.qid_to_source.get(qid, "unknown")
            perfect_ndcg = _is_perfect(ndcg)
            perfect_recall = _is_perfect(recall)
            qa = _qa_correct(row)

            for key in ("all",):
                bucket = counts["all"]
                bucket["n"] += 1
                if perfect_ndcg:
                    bucket["perfect_ndcg"] += 1
                    if qa is not None:
                        bucket["qa_perfect_ndcg"].append(qa)
                if perfect_recall:
                    bucket["perfect_recall"] += 1
                    if qa is not None:
                        bucket["qa_perfect_recall"].append(qa)
            if src in counts:
                bucket = counts[src]
                bucket["n"] += 1
                if perfect_ndcg:
                    bucket["perfect_ndcg"] += 1
                    if qa is not None:
                        bucket["qa_perfect_ndcg"].append(qa)
                if perfect_recall:
                    bucket["perfect_recall"] += 1
                    if qa is not None:
                        bucket["qa_perfect_recall"].append(qa)

        for src, acc in counts.items():
            n = acc["n"]
            if n == 0:
                continue
            qa_ndcg = _mean_accuracy(acc["qa_perfect_ndcg"])
            qa_recall = _mean_accuracy(acc["qa_perfect_recall"])
            rows.append(
                {
                    "policy": policy,
                    "dataset_source": src,
                    "k": k,
                    "n": int(n),
                    "n_perfect_ndcg": int(acc["perfect_ndcg"]),
                    "pct_perfect_ndcg": 100.0 * acc["perfect_ndcg"] / n,
                    "n_qa_perfect_ndcg": len(acc["qa_perfect_ndcg"]),
                    "qa_accuracy_perfect_ndcg": qa_ndcg,
                    "pct_qa_accuracy_perfect_ndcg": (
                        float("nan") if np.isnan(qa_ndcg) else 100.0 * qa_ndcg
                    ),
                    "n_perfect_recall": int(acc["perfect_recall"]),
                    "pct_perfect_recall": 100.0 * acc["perfect_recall"] / n,
                    "n_qa_perfect_recall": len(acc["qa_perfect_recall"]),
                    "qa_accuracy_perfect_recall": qa_recall,
                    "pct_qa_accuracy_perfect_recall": (
                        float("nan") if np.isnan(qa_recall) else 100.0 * qa_recall
                    ),
                }
            )

    df = pd.DataFrame(rows)
    artifact_id = spec.id or "pipeline_retrieval_perfect_rates"
    paths = write_table(
        bundle,
        artifact_id,
        df,
        {
            "k": k,
            "perfect_threshold": 1.0,
            "qa_metric": "qa_llm_judge.correct",
        },
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
