"""Binned calibration: QA accuracy vs retrieval-score bins, one line per pipeline."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.latency_metrics import bootstrap_mean_ci95
from surf_rag.results.bundle import ResultsBundle, policy_list
from surf_rag.results.figures.base import figure_paths, write_figure_meta
from surf_rag.results.figures.pipeline_retrieval_qa_scatter import (
    _POLICY_LABELS,
    _retrieval_axis_label,
)
from surf_rag.results.loaders import load_policy_metrics
from surf_rag.results.metric_fields import (
    e2e_retrieval_value,
    resolve_retrieval_metric_k,
)
from surf_rag.viz.theme import PALETTE

_CI_SAMPLES = 10_000
_CI_SEED = 42
_HIGHLIGHT_POLICY = "learned-soft"

_BIN_TICK_LABELS: tuple[str, ...] = (
    "Zero\n(0)",
    "Low\n(0, 0.25]",
    "Medium\n(0.25, 0.5]",
    "High\n(0.5, 1)",
    "Perfect\n(1)",
)
_N_BINS = len(_BIN_TICK_LABELS)

_PIPELINE_COLORS: dict[str, str] = {
    "dense-only": "#0072B2",
    "graph-only": "#009E73",
    "50-50": "#785EF0",
    "learned-soft": "#1F6EF5",
    "hard-routing": "#D55E00",
    "rrf": "#882255",
    "hybrid": "#CC79A7",
    "oracle-upper-bound": "#636363",
    "oracle-classification": "#8C8C8C",
}

_ORACLE_POLICIES = frozenset({"oracle-upper-bound", "oracle-classification"})
_Z_ORACLE = 2
_Z_PIPELINE = 6
_Z_HIGHLIGHT = 30


def _is_oracle_policy(policy: str) -> bool:
    return policy in _ORACLE_POLICIES or policy.startswith("oracle-")


def _policy_color(policy: str) -> str:
    if policy in _PIPELINE_COLORS:
        return _PIPELINE_COLORS[policy]
    idx = abs(hash(policy)) % 6
    fallback = ("#0072B2", "#009E73", "#785EF0", "#882255", "#CC79A7", "#56B4E9")
    return fallback[idx]


def _series_plot_order(series: list[dict]) -> list[dict]:
    """Oracles first, other pipelines, then SuRF-RAG last (on top)."""
    oracle = [s for s in series if s["oracle"]]
    highlight = [s for s in series if s["policy"] == _HIGHLIGHT_POLICY]
    rest = [s for s in series if not s["oracle"] and s["policy"] != _HIGHLIGHT_POLICY]
    return oracle + rest + highlight


def _zorder_for(policy: str, *, oracle: bool) -> int:
    if policy == _HIGHLIGHT_POLICY:
        return _Z_HIGHLIGHT
    return _Z_ORACLE if oracle else _Z_PIPELINE


def _assign_score_bin(score: float) -> int:
    """Map retrieval score in [0, 1] to bin 0..4 (zero / low / medium / high / perfect)."""
    v = float(score)
    if v <= 0.0:
        return 0
    if np.isclose(v, 1.0):
        return 4
    if v <= 0.25:
        return 1
    if v <= 0.5:
        return 2
    return 3


def _per_question_retrieval_accuracy(
    bundle: ResultsBundle,
    policy: str,
    *,
    metric: str,
    k: int,
) -> list[tuple[float, float]]:
    per_q = (
        load_policy_metrics(bundle.policies[policy].metrics_path).get("per_question")
        or []
    )
    pairs: list[tuple[float, float]] = []
    for row in per_q:
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        judge = row.get("qa_llm_judge")
        if not isinstance(judge, dict) or "correct" not in judge:
            continue
        score = e2e_retrieval_value(row, metric=metric, k=k)
        acc = 1.0 if bool(judge["correct"]) else 0.0
        pairs.append((score, acc))
    return pairs


def _bin_accuracy_stats(
    pairs: list[tuple[float, float]],
) -> list[dict[str, float | int | None]]:
    buckets: list[list[float]] = [[] for _ in range(_N_BINS)]
    for score, acc in pairs:
        buckets[_assign_score_bin(score)].append(acc)

    out: list[dict[str, float | int | None]] = []
    for b, accs in enumerate(buckets):
        if not accs:
            out.append(
                {
                    "bin": b,
                    "mean": None,
                    "ci_low": None,
                    "ci_high": None,
                    "n": 0,
                }
            )
            continue
        mean = float(np.mean(accs))
        lo, hi = bootstrap_mean_ci95(accs, samples=_CI_SAMPLES, seed=_CI_SEED)
        out.append(
            {
                "bin": b,
                "mean": mean,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n": len(accs),
            }
        )
    return out


def _legend_proxy(policy: str, label: str) -> Line2D:
    oracle = _is_oracle_policy(policy)
    highlight = policy == _HIGHLIGHT_POLICY
    return Line2D(
        [0],
        [0],
        color=_policy_color(policy),
        linestyle="--" if oracle else "-",
        linewidth=2.0 if highlight else (1.1 if oracle else 1.6),
        alpha=0.72 if oracle else 1.0,
        label=f"{label} (oracle)" if oracle else label,
    )


def _plot_series_on_axes(ax: Axes, series: list[dict]) -> None:
    x_all = np.arange(_N_BINS)

    for s in _series_plot_order(series):
        policy = str(s["policy"])
        oracle = _is_oracle_policy(policy)
        highlight = policy == _HIGHLIGHT_POLICY
        color = _policy_color(policy)
        linestyle = "--" if oracle else "-"
        linewidth = 2.2 if highlight else (1.1 if oracle else 1.6)
        alpha = 0.72 if oracle else 1.0
        z = _zorder_for(policy, oracle=oracle)

        xs: list[int] = []
        means: list[float] = []
        yerr_lo: list[float] = []
        yerr_hi: list[float] = []

        for row in s["bins"]:
            if row["mean"] is None:
                continue
            mean = float(row["mean"])
            lo = float(row["ci_low"])  # type: ignore[arg-type]
            hi = float(row["ci_high"])  # type: ignore[arg-type]
            xs.append(int(row["bin"]))
            means.append(mean)
            yerr_lo.append(max(0.0, mean - lo))
            yerr_hi.append(max(0.0, hi - mean))

        if not xs:
            continue

        ax.plot(
            xs,
            means,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=z,
        )
        ax.errorbar(
            xs,
            means,
            yerr=[yerr_lo, yerr_hi],
            color=color,
            linestyle="none",
            capsize=3,
            ecolor=color,
            elinewidth=1.2 if highlight else 0.9,
            capthick=1.2 if highlight else 0.9,
            alpha=alpha,
            zorder=z + 1,
        )
        ax.scatter(
            xs,
            means,
            s=64 if highlight else 52,
            color=color,
            edgecolors=PALETTE["text"],
            linewidths=0.35,
            zorder=z + 2,
        )

    ax.set_xticks(x_all, list(_BIN_TICK_LABELS))
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("QA accuracy (LLM judge)")


def _render_pipeline_qa_calibration_impl(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
    *,
    metric: str,
    k: int,
) -> dict[str, str]:
    split = bundle.results.split
    policies = policy_list(bundle)

    ordered = [p for p in policies if not _is_oracle_policy(p)] + [
        p for p in policies if _is_oracle_policy(p)
    ]

    series: list[dict] = []
    for policy in ordered:
        pairs = _per_question_retrieval_accuracy(bundle, policy, metric=metric, k=k)
        if not pairs:
            continue
        series.append(
            {
                "policy": policy,
                "label": _POLICY_LABELS.get(policy, policy),
                "oracle": _is_oracle_policy(policy),
                "bins": _bin_accuracy_stats(pairs),
            }
        )

    if not series:
        raise ValueError(
            f"No pipeline series with {metric}@{k} and QA judge labels on the split"
        )

    img_path, meta_path = figure_paths(bundle, spec.id)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    _plot_series_on_axes(ax, series)
    ax.set_xlabel(f"{_retrieval_axis_label(metric, k)} bin")
    ax.set_title(f"QA calibration by retrieval quality ({split} split)")

    legend_handles = [
        _legend_proxy(str(s["policy"]), str(s["label"]))
        for s in _series_plot_order(series)
    ]

    ncol = min(4, max(1, len(legend_handles)))
    fig.subplots_adjust(bottom=0.22)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=ncol,
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(img_path, format=bundle.image_format)
    plt.close(fig)

    write_figure_meta(
        meta_path,
        spec.id,
        {
            "split": split,
            "metric": metric,
            "k": k,
            "dataset_source": "all",
            "highlight_policy": _HIGHLIGHT_POLICY,
            "bin_tick_labels": list(_BIN_TICK_LABELS),
            "ci_method": "bootstrap_percentile",
            "ci_level": 0.95,
            "ci_samples": _CI_SAMPLES,
            "ci_seed": _CI_SEED,
            "series": [
                {"policy": s["policy"], "oracle": s["oracle"], "bins": s["bins"]}
                for s in series
            ],
        },
    )
    return {"image": str(img_path), "meta": str(meta_path)}


def render_pipeline_qa_calibration(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    metric, k = resolve_retrieval_metric_k(spec, bundle)
    return _render_pipeline_qa_calibration_impl(bundle, spec, metric=metric, k=k)


def render_pipeline_qa_calibration_recall(
    bundle: ResultsBundle, spec: ResultsArtifactSpec
) -> dict[str, str]:
    """Same calibration figure with bins defined by Recall@10 (not headline NDCG)."""
    k = int(spec.k if spec.k is not None else 10)
    recall_spec = replace(spec, metric="recall", k=k)
    return _render_pipeline_qa_calibration_impl(
        bundle, recall_spec, metric="recall", k=k
    )
