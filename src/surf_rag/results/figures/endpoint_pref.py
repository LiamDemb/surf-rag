"""Endpoint preference delta histogram (NQ vs 2Wiki), side-by-side bars."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import load_oracle_rows
from surf_rag.results.metric_fields import resolve_oracle_metric_k
from surf_rag.results.oracle_diagnostics import per_query_diagnostics
from surf_rag.viz.theme import (
    DATASET_SOURCE_COLORS,
    DATASET_SOURCE_LABELS,
    PALETTE,
    bar_style,
    style_bar_axes,
)


def render_endpoint_pref(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    metric, k = resolve_oracle_metric_k(spec, bundle)
    oc = bundle.results.oracle
    rows = load_oracle_rows(bundle.oracle_scores_path)
    df = per_query_diagnostics(
        rows,
        metric=metric,
        k=k,
        diagnostic_ks=[],
        plateau_tau=oc.plateau_tau,
        qid_to_source=bundle.qid_to_source,
    )
    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(-1.0, 1.0, 31)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bar_width = (bins[1] - bins[0]) * 0.42

    sources = ("nq", "2wiki")
    for i, src in enumerate(sources):
        sub = df.loc[df["dataset_source"] == src, "delta"].to_numpy()
        if len(sub) == 0:
            continue
        counts, _ = np.histogram(sub, bins=bins)
        offset = (i - 0.5) * bar_width
        ax.bar(
            bin_centers + offset,
            counts,
            width=bar_width,
            label=DATASET_SOURCE_LABELS.get(src, src),
            **bar_style(color=DATASET_SOURCE_COLORS.get(src, PALETTE["primary"])),
        )

    ax.axvline(
        0.0, color=PALETTE["identity_line"], linestyle="--", linewidth=1.0, zorder=0
    )
    ax.set_xlabel(r"$\Delta_i$ (NDCG@k dense $-$ graph)")
    ax.set_ylabel("Count")
    # ax.set_title("Endpoint preference by dataset")
    ax.legend(title="Dataset")
    style_bar_axes(ax)
    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(
        meta_path,
        spec.id,
        {"metric": metric, "k": k, "n": len(df), "layout": "grouped_bars"},
    )
    return {"image": str(img_path), "meta": str(meta_path)}
