#!/usr/bin/env python3
"""Compute the histogram of router predictions for a given benchmark."""

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
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_bins_dict(preds):
    counts, edges = np.histogram(preds, bins=10, range=(0.0, 1.0))
    bins_dict = {}
    for i in range(len(counts)):
        # Format the bin label
        start = edges[i]
        end = edges[i + 1]
        if i == len(counts) - 1:
            key = f"[{start:.1f}, {end:.1f}]"
        else:
            key = f"[{start:.1f}, {end:.1f})"
        bins_dict[key] = int(counts[i])
    return bins_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config yaml (e.g. configs/router-rg.yaml)",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))

    router_base = Path(cfg.paths.router_base)
    router_id = cfg.paths.router_id

    router_arch_id = getattr(cfg.paths, "router_architecture_id", None)
    safe_arch_id = router_arch_id.split("/")[-1] if router_arch_id else "unknown_arch"

    # Output directory
    metrics_dir = router_base / router_id / "models" / safe_arch_id / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_filename = "router_predictions_bins.json"
    out_path = metrics_dir / out_filename

    # Get dataset source from oracle scores
    oracle_paths = make_run_paths_for_cli(router_id, router_base=router_base)
    qid_to_source = {}
    if oracle_paths.oracle_scores.is_file():
        for row in read_oracle_score_rows(oracle_paths):
            qid = row["question_id"]
            qid_to_source[qid] = row.get("dataset_source", "unknown")
    else:
        logging.warning(
            "Oracle scores not found. Dataset source breakdown will not be accurate."
        )

    # Get predictions
    input_mode = getattr(cfg.router.train, "input_mode", "both")
    task_type = getattr(cfg.router.train, "task_type", "regression")

    model_paths = make_router_model_paths_for_cli(
        router_id=router_id,
        router_base=router_base,
        input_mode=input_mode,
        router_architecture_id=router_arch_id,
        router_task_type=task_type,
    )

    split_preds = defaultdict(list)

    for split in ["train", "dev", "test"]:
        pred_path = model_paths.predictions(split)
        if not pred_path.is_file() and router_arch_id and "/" in router_arch_id:
            # Try basename fallback
            fallback_arch = router_arch_id.split("/")[-1]
            fallback_paths = make_router_model_paths_for_cli(
                router_id=router_id,
                router_base=router_base,
                input_mode=input_mode,
                router_architecture_id=fallback_arch,
                router_task_type=task_type,
            )
            pred_path = fallback_paths.predictions(split)

        if not pred_path.is_file():
            logging.warning(f"Prediction file not found for split {split}: {pred_path}")
            continue

        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = row["question_id"]
                pred = float(row["predicted_weight"])
                source = qid_to_source.get(qid, "unknown")

                split_preds[split].append(pred)
                split_preds["overall"].append(pred)
                split_preds[source].append(pred)

    results = {}
    for key, preds in split_preds.items():
        results[key] = get_bins_dict(preds)

    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logging.info(f"Wrote router prediction bins to {out_path}")


if __name__ == "__main__":
    main()
