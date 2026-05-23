#!/usr/bin/env python3
"""Generate a bar chart of mean steady fuser scores on the test split."""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

from surf_rag.config.loader import load_pipeline_config
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

    benchmark_base = Path(cfg.paths.benchmark_base)
    benchmark_name = cfg.paths.benchmark_name
    benchmark_id = cfg.paths.benchmark_id

    # Output directory
    figures_dir = benchmark_base / benchmark_name / benchmark_id / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ext = "pgf"
    out_filename = f"steady_fuser_scores.{ext}"
    out_path = figures_dir / out_filename

    # Read the steady fuser scores
    metrics_dir = benchmark_base / benchmark_name / benchmark_id / "metrics"
    json_path = metrics_dir / "mean_steady_fuser_scores.json"

    if not json_path.is_file():
        logging.error(f"Steady fuser scores JSON not found: {json_path}")
        return

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "test" not in data:
        logging.error(f"'test' split not found in {json_path}")
        return

    test_data = data["test"]

    # Extract weights and means
    weights = []
    means = []
    stds = []

    # Sort the weights to ensure they are plotted in order
    for w_str in sorted(test_data.keys(), key=float):
        weights.append(w_str)
        means.append(test_data[w_str]["mean"])
        stds.append(test_data[w_str]["std"])

    x = np.arange(len(weights))

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars
    ax.bar(
        x, means, width=0.6, color=PALETTE.get("primary", "#2C7FB8"), edgecolor="white"
    )

    # Formatting
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("Fuser Weight")
    ax.set_title(
        "Steady Fuser Scores (Test Split)", color=PALETTE.get("text", "#1A1A1A")
    )

    ax.set_xticks(x)
    ax.set_xticklabels(weights)

    # Setting a more fitting Y limit between 0.2 and 0.8
    ax.set_ylim(0.2, 0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Wrote bar chart to {out_path}")


if __name__ == "__main__":
    main()
