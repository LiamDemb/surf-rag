#!/usr/bin/env python3
"""Compile a summary of an end-to-end run's test set performance.

.. deprecated::
    Use ``make results-build`` with ``pipeline_retrieval`` and ``pipeline_answers``
    artefacts in ``configs/results/*.yaml``.
"""

import argparse
import warnings
import json
import logging
from pathlib import Path
from collections import defaultdict

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.latency_metrics import reported_latency_ms_from_question_row
from surf_rag.evaluation.oracle_artifacts import (
    make_run_paths_for_cli,
    read_oracle_score_rows,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    warnings.warn(
        "compile_e2e_results is deprecated; use: make results-build "
        "RESULTS_CONFIG=configs/results/example.yaml",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config yaml (e.g. configs/router-rg.yaml)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="The run_id to search for under evaluations/ (e.g. e2e-rg-001)",
    )
    parser.add_argument(
        "--split-file",
        required=False,
        help="Optional override path to a split_question_ids.json file to use for filtering seen questions.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))

    benchmark_base = Path(cfg.paths.benchmark_base)
    benchmark_name = cfg.paths.benchmark_name
    benchmark_id = cfg.paths.benchmark_id
    router_base = Path(cfg.paths.router_base)
    router_id = cfg.paths.router_id

    router_arch_id = getattr(cfg.paths, "router_architecture_id", None)
    safe_arch_id = router_arch_id.split("/")[-1] if router_arch_id else "unknown_arch"

    # Find the run_id under evaluations
    evals_dir = benchmark_base / benchmark_name / benchmark_id / "evaluations"

    # Load splits
    test_qids = set()
    seen_qids = set()

    split_file_path = Path(args.split_file) if args.split_file else None

    if not split_file_path:
        ds_paths = make_router_dataset_paths_for_cli(router_id, router_base=router_base)
        split_file_path = ds_paths.split_question_ids

    if split_file_path and split_file_path.is_file():
        splits = json.loads(split_file_path.read_text(encoding="utf-8"))
        test_qids = set(splits.get("test", []))
        seen_qids = set(splits.get("train", [])) | set(splits.get("dev", []))
        logging.info(f"Loaded dataset splits from {split_file_path}")
    else:
        logging.warning(
            f"Could not find split_question_ids at {split_file_path}. Considering all questions as 'test' or 'unseen'."
        )

    # Get dataset source from benchmark file
    benchmark_file = (
        benchmark_base / benchmark_name / benchmark_id / "benchmark" / "benchmark.jsonl"
    )
    qid_to_source = {}
    benchmark_qids = set()
    if benchmark_file.is_file():
        with benchmark_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                benchmark_qids.add(qid)
                qid_to_source[qid] = row.get("dataset_source", "unknown")
    else:
        logging.warning(f"Could not find benchmark file at {benchmark_file}")

    policies_results = {}

    answerable_qids = set()
    audit_file = (
        benchmark_base
        / benchmark_name
        / benchmark_id
        / "audit"
        / "answerability"
        / "verdicts.jsonl"
    )
    if audit_file.is_file():
        with audit_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("answerable"):
                    answerable_qids.add(row["question_id"])

    if evals_dir.is_dir():
        for policy_dir in evals_dir.iterdir():
            if not policy_dir.is_dir():
                continue

            policy_name = policy_dir.name
            metrics_file = policy_dir / args.run_id / "metrics.json"

            if not metrics_file.is_file():
                continue

            logging.info(f"Found metrics for policy '{policy_name}' at {metrics_file}")

            # Load metrics
            with metrics_file.open("r", encoding="utf-8") as f:
                run_metrics = json.load(f)

            per_question = run_metrics.get("per_question", [])

            # Aggregate metrics
            metrics_sum = defaultdict(lambda: defaultdict(float))
            metrics_count = defaultdict(lambda: defaultdict(int))

            for item in per_question:
                qid = item.get("question_id")

                # Only include questions that are part of the benchmark
                if benchmark_qids and qid not in benchmark_qids:
                    continue

                # Exclude seen/train/dev questions (so we only measure test and unseen questions)
                if seen_qids and qid in seen_qids:
                    continue

                source = qid_to_source.get(qid, "unknown")
                groups = ["overall", source]

                # Extract metrics
                qa_em = item.get("qa", {}).get("em", 0.0)
                qa_f1 = item.get("qa", {}).get("f1", 0.0)

                retrieval_5 = (
                    item.get("retrieval_before_ce", {})
                    .get("retrieval", {})
                    .get("5", {})
                )
                retrieval_10 = (
                    item.get("retrieval_before_ce", {})
                    .get("retrieval", {})
                    .get("10", {})
                )
                retrieval_20 = (
                    item.get("retrieval_before_ce", {})
                    .get("retrieval", {})
                    .get("20", {})
                )

                ndcg_5 = retrieval_5.get("ndcg", 0.0)
                hit_5 = retrieval_5.get("hit", 0.0)
                recall_5 = retrieval_5.get("recall", 0.0)

                ndcg_10 = retrieval_10.get("ndcg", 0.0)
                hit_10 = retrieval_10.get("hit", 0.0)
                recall_10 = retrieval_10.get("recall", 0.0)

                ndcg_20 = retrieval_20.get("ndcg", 0.0)
                hit_20 = retrieval_20.get("hit", 0.0)
                recall_20 = retrieval_20.get("recall", 0.0)

                judge_correct = (
                    1.0 if item.get("qa_llm_judge", {}).get("correct") else 0.0
                )
                is_answerable = qid in answerable_qids

                lat_rep = reported_latency_ms_from_question_row(item.get("latency_ms"))

                for g in groups:
                    metrics_sum[g]["qa_em"] += qa_em
                    metrics_count[g]["qa_em"] += 1

                    metrics_sum[g]["qa_f1"] += qa_f1
                    metrics_count[g]["qa_f1"] += 1

                    metrics_sum[g]["ndcg_5"] += ndcg_5
                    metrics_count[g]["ndcg_5"] += 1

                    metrics_sum[g]["hit_5"] += hit_5
                    metrics_count[g]["hit_5"] += 1

                    metrics_sum[g]["recall_5"] += recall_5
                    metrics_count[g]["recall_5"] += 1

                    metrics_sum[g]["ndcg_10"] += ndcg_10
                    metrics_count[g]["ndcg_10"] += 1

                    metrics_sum[g]["hit_10"] += hit_10
                    metrics_count[g]["hit_10"] += 1

                    metrics_sum[g]["recall_10"] += recall_10
                    metrics_count[g]["recall_10"] += 1

                    metrics_sum[g]["ndcg_20"] += ndcg_20
                    metrics_count[g]["ndcg_20"] += 1

                    metrics_sum[g]["hit_20"] += hit_20
                    metrics_count[g]["hit_20"] += 1

                    metrics_sum[g]["recall_20"] += recall_20
                    metrics_count[g]["recall_20"] += 1

                    metrics_sum[g]["judge_correct"] += judge_correct
                    metrics_count[g]["judge_correct"] += 1

                    if is_answerable:
                        metrics_sum[g]["judge_correct_answerable"] += judge_correct
                        metrics_count[g]["judge_correct_answerable"] += 1

                    if lat_rep is not None:
                        metrics_sum[g]["latency_retrieval_reported_ms"] += lat_rep
                        metrics_count[g]["latency_retrieval_reported_ms"] += 1

            results = {}
            for g, m_sums in metrics_sum.items():
                results[g] = {}
                for m_name, val in m_sums.items():
                    if m_name == "latency_retrieval_reported_ms":
                        nlat = metrics_count[g].get(m_name, 0)
                        if nlat > 0:
                            results[g]["latency_retrieval_reported_mean_ms"] = (
                                val / nlat
                            )
                        continue
                    count = metrics_count[g][m_name]
                    results[g][m_name] = val / count if count > 0 else 0.0
                results[g]["count"] = metrics_count[g]["qa_em"]  # general count

                # Add count_answerable explicitly
                if "judge_correct_answerable" in metrics_count[g]:
                    results[g]["count_answerable"] = metrics_count[g][
                        "judge_correct_answerable"
                    ]

            policies_results[policy_name] = results

    if not policies_results:
        logging.error(
            f"Could not find run_id '{args.run_id}' under any policy in {evals_dir}"
        )
        return

    # Output to router/models/id/benchmarkname_benchid_results.json
    out_dir = router_base / router_id / "models" / safe_arch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{benchmark_name}_{benchmark_id}_results.json"

    final_output = {
        "run_id": args.run_id,
        "benchmark_name": benchmark_name,
        "benchmark_id": benchmark_id,
        "splits_evaluated": ["test", "unseen"],
        "policies": policies_results,
    }

    out_path = out_dir / out_filename
    out_path.write_text(json.dumps(final_output, indent=2) + "\n", encoding="utf-8")
    logging.info(f"Wrote summary insights to {out_path}")


if __name__ == "__main__":
    main()
