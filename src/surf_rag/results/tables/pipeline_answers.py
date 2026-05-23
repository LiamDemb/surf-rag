"""LLM-judged answer accuracy across subsets."""

from __future__ import annotations

import pandas as pd

from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.loaders import load_answerability, load_policy_metrics
from surf_rag.results.tables.writer import write_table


def build_pipeline_answers(bundle: ResultsBundle) -> tuple[pd.DataFrame, dict]:
    answerable = load_answerability(bundle.answerability_path)
    rows: list[dict] = []

    for policy in policy_list(bundle):
        metrics = load_policy_metrics(bundle.policies[policy].metrics_path)
        per_q = metrics.get("per_question") or []
        buckets: dict[tuple[str, str], list[bool]] = {
            (subset, src): []
            for subset in ("all", "answerable", "unanswerable")
            for src in ("all", "nq", "2wiki")
        }
        for row in per_q:
            qid = str(row.get("question_id", "")).strip()
            if qid not in bundle.split_qids:
                continue
            judge = row.get("qa_llm_judge")
            if not isinstance(judge, dict) or "correct" not in judge:
                continue
            correct = bool(judge["correct"])
            src = bundle.qid_to_source.get(qid, "unknown")
            is_ans = answerable.get(qid, False)
            subsets = ["all"]
            if is_ans:
                subsets.append("answerable")
            else:
                subsets.append("unanswerable")
            for subset in subsets:
                buckets[(subset, "all")].append(correct)
                if src in ("nq", "2wiki"):
                    buckets[(subset, src)].append(correct)

        for (subset, src), vals in sorted(buckets.items()):
            if not vals:
                continue
            acc = sum(1 for v in vals if v) / len(vals)
            rows.append(
                {
                    "subset": subset,
                    "policy": policy,
                    "dataset_source": src,
                    "accuracy": acc,
                    "n": len(vals),
                }
            )

    df = pd.DataFrame(rows)
    paths = write_table(
        bundle,
        "pipeline_answers",
        df,
        {},
    )
    return df, {"csv": str(paths[0]), "meta": str(paths[1])}
