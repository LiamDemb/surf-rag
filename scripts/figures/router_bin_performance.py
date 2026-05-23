#!/usr/bin/env python3
"""Generate a grouped bar chart of router prediction performance by predicted weight bin for NQ and 2WIKI."""

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
from surf_rag.viz.theme import apply_theme

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

    # Apply figure theme
    t = cfg.figures.theme
    overrides = dict(t.overrides or {})
    if t.font_size is not None:
        overrides["font.size"] = float(t.font_size)
    apply_theme(name=t.name, dpi=t.dpi, overrides=overrides or None, backend=t.backend)

    # Import pyplot AFTER apply_theme so the backend is set properly
    import matplotlib.pyplot as plt
    from surf_rag.viz.theme import PALETTE

    router_base = Path(cfg.paths.router_base)
    router_id = cfg.paths.router_id

    router_arch_id = getattr(cfg.paths, "router_architecture_id", None)
    safe_arch_id = router_arch_id.split("/")[-1] if router_arch_id else "unknown_arch"

    # Output directory
    figures_dir = router_base / router_id / "models" / safe_arch_id / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ext = "png"
    out_filename = f"router_bin_performance.{ext}"
    out_path = figures_dir / out_filename

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

    n_bins = 10

    # Store actual scores for each bin
    nq_bin_scores = [[] for _ in range(n_bins)]
    wiki_bin_scores = [[] for _ in range(n_bins)]

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
                pred = float(
                    row.get("predicted_weight", row.get("predicted_prob", 0.0))
                )
                source = qid_to_source.get(qid, "unknown")

                # Calculate actual score achieved by this prediction
                curve = row.get("oracle_curve", [])
                actual_score = 0.0
                if len(curve) >= 11:
                    idx = int(round(pred * 10))
                    idx = max(0, min(10, idx))
                    actual_score = curve[idx]
                else:
                    actual_score = row.get("target_oracle_best_score", 0.0)

                # Determine bin index
                bin_idx = int(pred * n_bins)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                elif bin_idx < 0:
                    bin_idx = 0

                if source == "nq":
                    nq_bin_scores[bin_idx].append(actual_score)
                elif source == "2wiki":
                    wiki_bin_scores[bin_idx].append(actual_score)

    # Compute means
    nq_means = []
    wiki_means = []
    for i in range(n_bins):
        nq_means.append(np.mean(nq_bin_scores[i]) if nq_bin_scores[i] else 0.0)
        wiki_means.append(np.mean(wiki_bin_scores[i]) if wiki_bin_scores[i] else 0.0)

    # X-axis locations
    x = np.arange(n_bins)
    bar_width = 0.35

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars side by side
    ax.bar(
        x - bar_width / 2,
        nq_means,
        width=bar_width,
        label="NQ",
        color=PALETTE.get("primary", "#2C7FB8"),
        edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2,
        wiki_means,
        width=bar_width,
        label="2WIKI",
        color=PALETTE.get("dark-blue", "#0F008A"),
        edgecolor="white",
    )

    # Formatting
    ax.set_ylabel("Mean Achieved Score")
    ax.set_xlabel("Predicted Dense Weight")
    ax.set_title(
        "Router Performance by Prediction Bin", color=PALETTE.get("text", "#1A1A1A")
    )

    # Set x-ticks to be the center of the bins and label them with the ranges
    bin_labels = []
    for i in range(n_bins):
        start = i / n_bins
        end = (i + 1) / n_bins
        if i == n_bins - 1:
            bin_labels.append(f"[{start:.1f}, {end:.1f}]")
        else:
            bin_labels.append(f"[{start:.1f}, {end:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    except RuntimeError:
        out_path = out_path.with_suffix(".png")
        fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Wrote grouped bar chart to {out_path}")


if __name__ == "__main__":
    main()
