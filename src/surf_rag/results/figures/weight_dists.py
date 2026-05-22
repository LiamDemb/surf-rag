"""Regressor predicted weight histogram (NQ vs 2Wiki), grouped bars on test split."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_REGRESSION,
    make_router_model_paths_for_cli,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import iter_predictions_jsonl
from surf_rag.results.tables.regressor_metrics import SkippedArtifact
from surf_rag.viz.theme import (
    DATASET_SOURCE_COLORS,
    DATASET_SOURCE_LABELS,
    PALETTE,
    bar_style,
    style_bar_axes,
)


def render_weight_dists(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    arch = get_router_arch(bundle, "regressor")
    if arch is None or not arch.architecture_id.strip():
        raise SkippedArtifact("results.router.regressor not configured")
    split = bundle.results.split
    paths = make_router_model_paths_for_cli(
        bundle.resolved.router_id,
        router_base=bundle.resolved.router_base,
        input_mode=arch.input_mode,
        router_architecture_id=arch.architecture_id,
        router_task_type=ROUTER_TASK_REGRESSION,
    )
    pred_path = paths.predictions(split)
    if not pred_path.is_file():
        raise SkippedArtifact(f"Missing predictions: {pred_path}")

    by_source: dict[str, list[float]] = {"nq": [], "2wiki": []}
    for row in iter_predictions_jsonl(pred_path):
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        src = bundle.qid_to_source.get(qid, "unknown")
        if src not in by_source:
            continue
        by_source[src].append(float(row.get("predicted_weight", 0.5)))

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bar_width = (bins[1] - bins[0]) * 0.42

    sources = ("nq", "2wiki")
    for i, src in enumerate(sources):
        weights = np.asarray(by_source[src], dtype=float)
        if weights.size == 0:
            continue
        counts, _ = np.histogram(weights, bins=bins)
        offset = (i - 0.5) * bar_width
        ax.bar(
            bin_centers + offset,
            counts,
            width=bar_width,
            label=DATASET_SOURCE_LABELS.get(src, src),
            **bar_style(color=DATASET_SOURCE_COLORS.get(src, PALETTE["primary"])),
        )

    ax.set_xlabel("Predicted dense weight")
    ax.set_ylabel("Count")
    # ax.set_title(f"Predicted weight by dataset ({split} split)")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{x:.1f}" for x in bins[:-1]])
    ax.legend(title="Dataset")
    style_bar_axes(ax)
    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {
            "split": split,
            "n_nq": len(by_source["nq"]),
            "n_2wiki": len(by_source["2wiki"]),
            "bin_width": 0.1,
            "layout": "grouped_bars",
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}
