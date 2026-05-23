#!/usr/bin/env python3
"""Generate a grouped bar chart of router predictions for NQ and 2WIKI.

.. deprecated:: Use ``weight_dists`` via ``make results-build``.
"""

import argparse
import warnings
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
    warnings.warn(
        "router_prediction_distribution is deprecated; use make results-build",
        DeprecationWarning,
        stacklevel=1,
    )
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
    ext = "pgf"
    # ext = "png"
    out_filename = f"router_prediction_distribution.{ext}"
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

    nq_preds = []
    wiki_preds = []

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

                if source == "nq":
                    nq_preds.append(pred)
                elif source == "2wiki":
                    wiki_preds.append(pred)

    if not nq_preds and not wiki_preds:
        logging.error("No predictions found for either 'nq' or '2wiki'.")
        return

    n_bins = 10
    nq_counts, edges = np.histogram(nq_preds, bins=n_bins, range=(0.0, 1.0))
    wiki_counts, _ = np.histogram(wiki_preds, bins=n_bins, range=(0.0, 1.0))

    # X-axis locations
    x = np.arange(n_bins)
    bar_width = 0.35

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars side by side
    ax.bar(
        x - bar_width / 2,
        nq_counts,
        width=bar_width,
        label="NQ",
        color=PALETTE.get("primary", "#2C7FB8"),
        edgecolor="white",
    )
    ax.bar(
        x + bar_width / 2,
        wiki_counts,
        width=bar_width,
        label="2WIKI",
        color=PALETTE.get("dark-blue", "#0F008A"),
        edgecolor="white",
    )

    # Formatting
    ax.set_ylabel("Count")
    ax.set_xlabel("Predicted Dense Weight")
    ax.set_title(
        "Router Prediction Distribution by Source", color=PALETTE.get("text", "#1A1A1A")
    )

    # Set x-ticks to be the center of the bins and label them with the ranges
    bin_labels = []
    for i in range(n_bins):
        start = edges[i]
        end = edges[i + 1]
        if i == n_bins - 1:
            bin_labels.append(f"[{start:.1f}, {end:.1f}]")
        else:
            bin_labels.append(f"[{start:.1f}, {end:.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right")

    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Wrote grouped bar chart to {out_path}")


if __name__ == "__main__":
    main()
