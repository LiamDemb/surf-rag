#!/usr/bin/env python3
"""Generate classifier evaluation figures: prediction histogram, confusion matrix, and regret matrix.

.. deprecated:: Use ``route_confusion`` via ``make results-build``.
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


def plot_heatmap(
    ax, data, title, xlabel, ylabel, xticklabels, yticklabels, cmap="Blues", fmt=".0f"
):
    im = ax.imshow(data, cmap=cmap)

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(xticklabels)), labels=xticklabels)
    ax.set_yticks(np.arange(len(yticklabels)), labels=yticklabels)

    # Loop over data dimensions and create text annotations
    for i in range(len(yticklabels)):
        for j in range(len(xticklabels)):
            val = data[i, j]
            color = "white" if im.norm(val) > 0.5 else "black"
            text_str = f"{val:{fmt}}" if isinstance(fmt, str) else fmt(val)
            ax.text(j, i, text_str, ha="center", va="center", color=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def main():
    warnings.warn(
        "classifier_evaluation is deprecated; use make results-build",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline config yaml (e.g. configs/router-cls.yaml)",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(Path(args.config))

    # Apply figure theme
    t = cfg.figures.theme
    overrides = dict(t.overrides or {})
    if t.font_size is not None:
        overrides["font.size"] = float(t.font_size)
    apply_theme(name=t.name, dpi=t.dpi, overrides=overrides or None, backend=t.backend)

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

    input_mode = getattr(cfg.router.train, "input_mode", "both")
    task_type = getattr(cfg.router.train, "task_type", "classification")

    model_paths = make_router_model_paths_for_cli(
        router_id=router_id,
        router_base=router_base,
        input_mode=input_mode,
        router_architecture_id=router_arch_id,
        router_task_type=task_type,
    )

    # For histogram: counts by source and prediction
    # nq_preds[0] = count of Graph preds for NQ, nq_preds[1] = count of Dense preds for NQ
    nq_preds_counts = {0: 0, 1: 0}
    wiki_preds_counts = {0: 0, 1: 0}

    # Confusion Matrix: cm[true][pred]
    cm = np.zeros((2, 2), dtype=int)
    # Regret CM: regret_cm[true][pred]
    regret_cm = np.zeros((2, 2), dtype=float)

    found_any = False

    for split in ["train", "dev", "test"]:
        pred_path = model_paths.predictions(split)
        if not pred_path.is_file() and router_arch_id and "/" in router_arch_id:
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
                pred = int(row.get("predicted_class_id", -1))
                target = int(row.get("target_class_id", -1))

                if pred not in [0, 1] or target not in [0, 1]:
                    continue

                found_any = True
                source = qid_to_source.get(qid, "unknown")

                if source == "nq":
                    nq_preds_counts[pred] += 1
                elif source == "2wiki":
                    wiki_preds_counts[pred] += 1

                cm[target, pred] += 1

                # Calculate regret
                curve = row.get("oracle_curve", [])
                best_score = row.get("target_oracle_best_score", 0.0)
                if len(curve) >= 11:
                    # dense is index 10, graph is index 0
                    score = curve[-1] if pred == 1 else curve[0]
                    regret = max(0.0, best_score - score)
                    regret_cm[target, pred] += regret

    if not found_any:
        logging.error("No valid predictions found.")
        return

    # 1. Prediction Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    bar_width = 0.35

    nq_counts = [nq_preds_counts[0], nq_preds_counts[1]]
    wiki_counts = [wiki_preds_counts[0], wiki_preds_counts[1]]

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

    ax.set_ylabel("Count")
    ax.set_xlabel("Predicted Class")
    ax.set_title(
        "Classifier Prediction Distribution", color=PALETTE.get("text", "#1A1A1A")
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Graph (0)", "Dense (1)"])
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    hist_out = figures_dir / f"classifier_prediction_histogram.{ext}"
    try:
        fig.savefig(hist_out, bbox_inches="tight")
    except RuntimeError:
        hist_out = figures_dir / "classifier_prediction_histogram.png"
        fig.savefig(hist_out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Wrote prediction histogram to {hist_out}")

    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_heatmap(
        ax,
        cm,
        "Confusion Matrix",
        "Predicted Class",
        "True Class",
        ["Graph", "Dense"],
        ["Graph", "Dense"],
        cmap="Blues",
        fmt="d",
    )
    fig.tight_layout()
    cm_out = figures_dir / f"classifier_confusion_matrix.{ext}"
    try:
        fig.savefig(cm_out, bbox_inches="tight")
    except RuntimeError:
        cm_out = figures_dir / "classifier_confusion_matrix.png"
        fig.savefig(cm_out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Wrote confusion matrix to {cm_out}")

    # 3. Regret Matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_regret_cm = np.where(cm > 0, regret_cm / cm, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_heatmap(
        ax,
        mean_regret_cm,
        "Mean Regret Confusion Matrix",
        "Predicted Class",
        "True Class",
        ["Graph", "Dense"],
        ["Graph", "Dense"],
        cmap="Reds",
        fmt=".3f",
    )
    fig.tight_layout()
    regret_out = figures_dir / f"classifier_regret_matrix.{ext}"
    try:
        fig.savefig(regret_out, bbox_inches="tight")
    except RuntimeError:
        regret_out = figures_dir / "classifier_regret_matrix.png"
        fig.savefig(regret_out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Wrote regret matrix to {regret_out}")


if __name__ == "__main__":
    main()
