"""Sorted oracle dispersion curves by dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.loaders import load_oracle_rows
from surf_rag.results.metric_fields import resolve_oracle_metric_k
from surf_rag.results.oracle_diagnostics import per_query_diagnostics
from surf_rag.viz.theme import DATASET_SOURCE_COLORS, PALETTE


def render_dispersion_curve(
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
    for src, label in (("nq", "NQ"), ("2wiki", "2Wiki")):
        sub = df.loc[df["dataset_source"] == src, "dispersion"].sort_values(
            ascending=False
        )
        if len(sub) == 0:
            continue
        ax.plot(
            np.arange(len(sub)),
            sub.values,
            label=label,
            color=DATASET_SOURCE_COLORS.get(src, PALETTE["primary"]),
            linewidth=1.2,
        )
    ax.set_xlabel("Query rank (by dispersion)")
    ax.set_ylabel(r"$\delta_i$")
    # ax.set_title("Oracle dispersion (sorted)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)
    write_figure_meta(meta_path, spec.id, {"metric": metric, "k": k})
    return {"image": str(img_path), "meta": str(meta_path)}
