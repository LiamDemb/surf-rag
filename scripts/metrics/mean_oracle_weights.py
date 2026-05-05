#!/usr/bin/env python3
"""Compute the mean oracle weights for a given benchmark, separated by split."""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

from surf_rag.config.loader import load_pipeline_config
from surf_rag.evaluation.oracle_artifacts import (
    make_run_paths_for_cli,
    read_oracle_score_rows,
)
from surf_rag.evaluation.oracle_argmax_intervals import argmax_plateau_bin_indices
from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config yaml (e.g. configs/router-rg.yaml)",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))

    benchmark_base = Path(cfg.paths.benchmark_base)
    benchmark_name = cfg.paths.benchmark_name
    benchmark_id = cfg.paths.benchmark_id
    router_base = Path(cfg.paths.router_base)
    router_id = cfg.paths.router_id

    # Output directory
    metrics_dir = benchmark_base / benchmark_name / benchmark_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "mean_oracle_weights.json"

    # Load splits
    ds_paths = make_router_dataset_paths_for_cli(router_id, router_base=router_base)

    qid_to_split = {}
    if ds_paths.split_question_ids.is_file():
        splits = json.loads(ds_paths.split_question_ids.read_text(encoding="utf-8"))
        for split_name in ["train", "dev", "test"]:
            for qid in splits.get(split_name, []):
                qid_to_split[qid] = split_name
    else:
        logging.warning(
            f"Could not find {ds_paths.split_question_ids}. Split breakdown will not be available."
        )

    # Load oracle scores
    oracle_paths = make_run_paths_for_cli(router_id, router_base=router_base)
    if not oracle_paths.oracle_scores.is_file():
        logging.error(f"Oracle scores not found: {oracle_paths.oracle_scores}")
        return

    rows = read_oracle_score_rows(oracle_paths)

    # Compute centroids
    split_sums = defaultdict(float)
    split_counts = defaultdict(int)
    split_weights = defaultdict(list)

    for row in rows:
        qid = row["question_id"]
        split = qid_to_split.get(qid, "unknown")

        dataset_source = row.get("dataset_source", "unknown")

        weight_grid = np.array(row["weight_grid"], dtype=float)
        scores = np.array(
            [s["oracle_objective_value"] for s in row["scores"]], dtype=float
        )

        idx = argmax_plateau_bin_indices(scores)
        if len(idx) == 0:
            continue

        tied_weights = weight_grid[idx]
        centroid = float(np.mean(tied_weights))

        split_sums[split] += centroid
        split_counts[split] += 1
        split_weights[split].append(centroid)

        split_sums["overall"] += centroid
        split_counts["overall"] += 1
        split_weights["overall"].append(centroid)

        split_sums[dataset_source] += centroid
        split_counts[dataset_source] += 1
        split_weights[dataset_source].append(centroid)

    # Aggregate
    results = {}
    for split in split_counts:
        mean_weight = (
            split_sums[split] / split_counts[split] if split_counts[split] > 0 else 0.0
        )
        std_weight = (
            float(np.std(split_weights[split]))
            if len(split_weights[split]) > 0
            else 0.0
        )
        results[split] = {
            "mean": mean_weight,
            "std": std_weight,
            "count": split_counts[split],
        }

    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info(f"Wrote mean oracle weights to {out_path}")


if __name__ == "__main__":
    main()
