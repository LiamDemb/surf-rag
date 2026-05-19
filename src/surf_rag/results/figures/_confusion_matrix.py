"""Shared 2×2 routing confusion heatmap helpers for results figures."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from surf_rag.viz.theme import (
    HEATMAP_ANNOTATION_CONTRAST_THRESHOLD,
    PALETTE,
    sequential_cmap_from_palette,
)

BRANCH_LABELS: tuple[str, str] = ("GraphRAG", "DenseRAG")


def routing_regret_for_prediction(row: dict[str, Any], pred: int) -> float:
    """Oracle-curve regret when hard-routing to ``pred`` (0=graph, 1=dense)."""
    curve = list(row.get("oracle_curve") or [])
    if not curve:
        return 0.0
    best = float(row.get("target_oracle_best_score", 0.0))
    score = float(curve[-1] if pred == 1 else curve[0])
    return max(0.0, best - score)


def accumulate_confusion_counts(
    rows: list[tuple[int, int]],
) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for tgt_i, pred_i in rows:
        if pred_i in (0, 1) and tgt_i in (0, 1):
            cm[tgt_i, pred_i] += 1
    return cm


def accumulate_mean_regret_matrix(
    buckets: dict[tuple[int, int], list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_regret 2×2 with nan for empty, counts 2×2 int)."""
    mean_m = np.full((2, 2), np.nan, dtype=float)
    counts = np.zeros((2, 2), dtype=int)
    for (tgt_i, pred_i), regrets in buckets.items():
        if tgt_i not in (0, 1) or pred_i not in (0, 1) or not regrets:
            continue
        counts[tgt_i, pred_i] = len(regrets)
        mean_m[tgt_i, pred_i] = float(np.mean(regrets))
    return mean_m, counts


def render_binary_confusion_heatmap(
    matrix: np.ndarray,
    *,
    palette_key: str,
    title: str,
    xlabel: str = "Predicted",
    ylabel: str = "True",
    annotate: Callable[[int, int, float], str],
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    figsize: tuple[float, float] = (5.0, 4.0),
) -> Figure:
    """Draw a 2×2 heatmap with palette-based sequential shades (face → accent)."""
    cmap = sequential_cmap_from_palette(palette_key)
    fig, ax = plt.subplots(figsize=figsize)

    data = np.asarray(matrix, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmin_use, vmax_use = 0.0, 1.0
    else:
        vmin_use = 0.0 if vmin is None else float(vmin)
        vmax_use = float(np.max(finite)) if vmax is None else float(vmax)
        if vmax_use <= vmin_use:
            vmax_use = vmin_use + 1.0

    display = np.ma.masked_invalid(data)
    im = ax.imshow(
        display,
        cmap=cmap,
        vmin=vmin_use,
        vmax=vmax_use,
        aspect="equal",
    )
    ax.set_xticks([0, 1], labels=list(BRANCH_LABELS))
    ax.set_yticks([0, 1], labels=list(BRANCH_LABELS))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    norm = Normalize(vmin=vmin_use, vmax=vmax_use)
    threshold = HEATMAP_ANNOTATION_CONTRAST_THRESHOLD
    for i in range(2):
        for j in range(2):
            val = data[i, j]
            if not np.isfinite(val):
                ax.text(
                    j,
                    i,
                    annotate(i, j, float("nan")),
                    ha="center",
                    va="center",
                    color=PALETTE["text"],
                    fontsize=10,
                )
                continue
            level = float(norm(val))
            text_color = PALETTE["face"] if level >= threshold else PALETTE["text"]
            ax.text(
                j,
                i,
                annotate(i, j, val),
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    if cbar_label:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    fig.tight_layout()
    return fig


def new_regret_buckets() -> dict[tuple[int, int], list[float]]:
    return defaultdict(list)
