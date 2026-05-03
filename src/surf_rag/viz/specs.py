"""Typed figure specifications and YAML dict parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

SplitName = Literal["train", "dev", "test"]
HeatmapSplitName = Literal["all", "train", "dev", "test"]
HeatmapColorNorm = Literal["linear", "log", "power"]

_ROUTER_PRED_VS_ORACLE_ALLOWED = frozenset(
    {
        "kind",
        "split",
        "filter_invalid_only",
        "point_size",
        "alpha",
        "filename_stem",
        "add_density_hexbin",
    }
)

_ROUTER_PRED_VS_ORACLE_INTERVALS_ALLOWED = frozenset(
    {
        "kind",
        "split",
        "filter_invalid_only",
        "filename_stem",
        "max_queries",
        "subsample_seed",
        "rtol",
        "atol",
        "interval_bar_width_ratio",
        "interval_alpha",
        "dot_size",
    }
)

_BENCHMARK_ORACLE_HEATMAP_ALLOWED = frozenset(
    {
        "kind",
        "split",
        "exclude_all_zero_queries",
        "rtol",
        "atol",
        "filename_stem",
        "fig_width",
        "fig_height",
        "colormap",
        "color_norm",
        "color_power_gamma",
        "interior_peak_only",
        "exclude_all_one_queries",
        "mid_dense_band_strict_best",
        "show_plot_subtitle",
    }
)

_ORACLE_ARGMAX_WEIGHT_HISTOGRAM_ALLOWED = frozenset(
    {
        "kind",
        "split",
        "exclude_all_zero_queries",
        "rtol",
        "atol",
        "filename_stem",
        "fig_width",
        "fig_height",
        "hist_bins",
        "show_plot_subtitle",
    }
)

_KNOWN_KINDS_HINT = (
    "router_pred_vs_oracle, router_pred_vs_oracle_intervals, "
    "benchmark_oracle_ndcg_heatmap, oracle_argmax_weight_histogram"
)


@dataclass(frozen=True)
class BaseFigureSpec:
    """Base type for registered figure specs."""

    kind: str


@dataclass(frozen=True)
class RouterPredVsOracleSpec(BaseFigureSpec):
    """Scatter: x = predicted dense weight, y = distance to oracle argmax interval."""

    split: SplitName
    filter_invalid_only: bool = False
    point_size: float = 12.0
    alpha: float = 0.35
    filename_stem: str = "router_pred_vs_oracle"
    add_density_hexbin: bool = False

    def __post_init__(self) -> None:
        if self.kind != "router_pred_vs_oracle":
            raise ValueError(f"unexpected kind {self.kind!r}")
        if self.point_size <= 0:
            raise ValueError("point_size must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.add_density_hexbin:
            raise NotImplementedError(
                "add_density_hexbin is not implemented yet; keep it false in YAML."
            )

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> RouterPredVsOracleSpec:
        extra = frozenset(m.keys()) - _ROUTER_PRED_VS_ORACLE_ALLOWED
        if extra:
            raise ValueError(
                f"Unknown keys for router_pred_vs_oracle plot: {sorted(extra)}"
            )
        split = m.get("split", "test")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"split must be train|dev|test, got {split!r}")
        return RouterPredVsOracleSpec(
            kind="router_pred_vs_oracle",
            split=split,  # type: ignore[arg-type]
            filter_invalid_only=bool(m.get("filter_invalid_only", False)),
            point_size=float(m.get("point_size", 12.0)),
            alpha=float(m.get("alpha", 0.35)),
            filename_stem=str(m.get("filename_stem", "router_pred_vs_oracle")),
            add_density_hexbin=bool(m.get("add_density_hexbin", False)),
        )


@dataclass(frozen=True)
class RouterPredVsOracleIntervalsSpec(BaseFigureSpec):
    """Per-query plot: oracle optimal dense-weight intervals vs predicted dense weight."""

    split: SplitName
    filter_invalid_only: bool = False
    filename_stem: str = "router_pred_vs_oracle_intervals"
    max_queries: int | None = None
    subsample_seed: int = 42
    rtol: float = 1e-5
    atol: float = 1e-8
    interval_bar_width_ratio: float = 0.72
    interval_alpha: float = 0.45
    dot_size: float = 18.0

    def __post_init__(self) -> None:
        if self.kind != "router_pred_vs_oracle_intervals":
            raise ValueError(f"unexpected kind {self.kind!r}")
        if self.max_queries is not None and int(self.max_queries) < 1:
            raise ValueError("max_queries must be positive when set")
        if not (0.0 < self.interval_bar_width_ratio <= 1.0):
            raise ValueError("interval_bar_width_ratio must be in (0, 1]")
        if not (0.0 <= self.interval_alpha <= 1.0):
            raise ValueError("interval_alpha must be in [0, 1]")
        if self.dot_size <= 0:
            raise ValueError("dot_size must be positive")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("rtol/atol must be non-negative")

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> RouterPredVsOracleIntervalsSpec:
        extra = frozenset(m.keys()) - _ROUTER_PRED_VS_ORACLE_INTERVALS_ALLOWED
        if extra:
            raise ValueError(
                f"Unknown keys for router_pred_vs_oracle_intervals plot: {sorted(extra)}"
            )
        split = m.get("split", "test")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"split must be train|dev|test, got {split!r}")
        max_q_raw = m.get("max_queries", None)
        max_queries: int | None
        if max_q_raw is None or (
            isinstance(max_q_raw, str) and not str(max_q_raw).strip()
        ):
            max_queries = None
        else:
            max_queries = int(max_q_raw)
        return RouterPredVsOracleIntervalsSpec(
            kind="router_pred_vs_oracle_intervals",
            split=split,  # type: ignore[arg-type]
            filter_invalid_only=bool(m.get("filter_invalid_only", False)),
            filename_stem=str(
                m.get("filename_stem", "router_pred_vs_oracle_intervals")
            ),
            max_queries=max_queries,
            subsample_seed=int(m.get("subsample_seed", 42)),
            rtol=float(m.get("rtol", 1e-5)),
            atol=float(m.get("atol", 1e-8)),
            interval_bar_width_ratio=float(m.get("interval_bar_width_ratio", 0.72)),
            interval_alpha=float(m.get("interval_alpha", 0.45)),
            dot_size=float(m.get("dot_size", 18.0)),
        )


@dataclass(frozen=True)
class BenchmarkOracleHeatmapSpec(BaseFigureSpec):
    """Heatmap: x = queries, y = dense weight, color = oracle curve score.

    Queries are ordered by a **score-weighted centroid** on the weight grid:
    ``sum(score_i * w_i) / sum(score_i)`` (nonnegative scores), so columns flow
    from “mass on low dense weight” to “mass on high dense weight.”

    **color_norm**: ``log`` — logarithmic scale (compresses high scores on [0,1]);
    ``linear`` — uniform 0..max; ``power`` — :class:`~matplotlib.colors.PowerNorm`
    with ``color_power_gamma`` (default 2); **γ > 1** spreads differences among
    **high** scores and compresses the low end (inverse of log for typical NDCG).

    **interior_peak_only**: if true, keep only queries whose oracle **maximum**
    is attained at some dense weight that is not (within ``rtol``/``atol``) 0.0 or
    1.0 — i.e. the best score is not achieved solely at the endpoints of the
    weight grid (ties count: if any maximizing grid point is interior, the query
    stays).

    **exclude_all_one_queries**: if true, drop queries whose oracle curve is
    (within ``rtol``/``atol``) **1.0 at every weight bin** — uniform perfect scores.

    **mid_dense_band_strict_best**: if true, keep only queries where the maximum
    oracle score on **w ∈ [0.2, 0.8]** is **strictly greater** than the maximum on
    **w < 0.2** and strictly greater than the maximum on **w > 0.8** (tail bands
    use no grid points → treated as −∞ so the mid band wins if non-empty).

    **show_plot_subtitle**: if false, omit the small benchmark / filter summary
    text in the upper-left of the figure (defaults to true).
    """

    split: HeatmapSplitName
    exclude_all_zero_queries: bool = True
    exclude_all_one_queries: bool = False
    interior_peak_only: bool = False
    mid_dense_band_strict_best: bool = False
    rtol: float = 1e-5
    atol: float = 1e-8
    filename_stem: str = "benchmark_oracle_ndcg_heatmap"
    fig_width: float = 10.0
    fig_height: float = 4.5
    colormap: str = "RdYlGn"
    color_norm: HeatmapColorNorm = "log"
    color_power_gamma: float = 2.0
    show_plot_subtitle: bool = True

    def __post_init__(self) -> None:
        if self.kind != "benchmark_oracle_ndcg_heatmap":
            raise ValueError(f"unexpected kind {self.kind!r}")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("rtol/atol must be non-negative")
        if self.fig_width <= 0 or self.fig_height <= 0:
            raise ValueError("fig_width and fig_height must be positive")
        if not str(self.colormap).strip():
            raise ValueError("colormap must be non-empty")
        if self.color_norm not in ("linear", "log", "power"):
            raise ValueError(
                f"color_norm must be linear, log, or power, got {self.color_norm!r}"
            )
        if self.color_power_gamma <= 0:
            raise ValueError("color_power_gamma must be positive")

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> BenchmarkOracleHeatmapSpec:
        extra = frozenset(m.keys()) - _BENCHMARK_ORACLE_HEATMAP_ALLOWED
        if extra:
            raise ValueError(
                f"Unknown keys for benchmark_oracle_ndcg_heatmap plot: {sorted(extra)}"
            )
        split = m.get("split", "all")
        if split not in ("all", "train", "dev", "test"):
            raise ValueError(f"split must be all|train|dev|test, got {split!r}")
        cmap_raw = m.get("colormap", "RdYlGn")
        cmap_s = str(cmap_raw).strip() if cmap_raw is not None else "RdYlGn"
        if not cmap_s:
            cmap_s = "RdYlGn"
        cn_raw = m.get("color_norm", "log")
        cn_s = str(cn_raw).strip().lower() if cn_raw is not None else "log"
        if not cn_s:
            cn_s = "log"
        if cn_s not in ("linear", "log", "power"):
            raise ValueError(
                f"color_norm must be linear, log, or power, got {cn_raw!r}"
            )
        return BenchmarkOracleHeatmapSpec(
            kind="benchmark_oracle_ndcg_heatmap",
            split=split,  # type: ignore[arg-type]
            exclude_all_zero_queries=bool(m.get("exclude_all_zero_queries", True)),
            exclude_all_one_queries=bool(m.get("exclude_all_one_queries", False)),
            rtol=float(m.get("rtol", 1e-5)),
            atol=float(m.get("atol", 1e-8)),
            filename_stem=str(m.get("filename_stem", "benchmark_oracle_ndcg_heatmap")),
            fig_width=float(m.get("fig_width", 10.0)),
            fig_height=float(m.get("fig_height", 4.5)),
            colormap=cmap_s,
            color_norm=cn_s,  # type: ignore[arg-type]
            color_power_gamma=float(m.get("color_power_gamma", 2.0)),
            interior_peak_only=bool(m.get("interior_peak_only", False)),
            mid_dense_band_strict_best=bool(m.get("mid_dense_band_strict_best", False)),
            show_plot_subtitle=bool(m.get("show_plot_subtitle", True)),
        )


@dataclass(frozen=True)
class OracleArgmaxWeightHistogramSpec(BaseFigureSpec):
    """Stacked histogram of dense weights where the oracle curve attains its maximum.

    **Every** grid bin tied for the maximum is counted (not a single argmax).
    Green: query had a **unique** maximizing bin. Orange: query had **≥2** bins at
    the maximum — each tied bin gets one count.

    **hist_bins**: ``0`` (default) = one bar per ``weight_grid`` value in the dataset
    (e.g. 11 points 0..1). ``N > 0`` = ``N`` equal-width bins on ``[0, 1]``.
    """

    split: HeatmapSplitName
    exclude_all_zero_queries: bool = True
    hist_bins: int = 0
    rtol: float = 1e-5
    atol: float = 1e-8
    filename_stem: str = "oracle_argmax_weight_histogram"
    fig_width: float = 8.0
    fig_height: float = 4.5
    show_plot_subtitle: bool = True

    def __post_init__(self) -> None:
        if self.kind != "oracle_argmax_weight_histogram":
            raise ValueError(f"unexpected kind {self.kind!r}")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("rtol/atol must be non-negative")
        if self.fig_width <= 0 or self.fig_height <= 0:
            raise ValueError("fig_width and fig_height must be positive")
        if self.hist_bins > 500:
            raise ValueError("hist_bins must be <= 500 when using equal-width binning")

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> OracleArgmaxWeightHistogramSpec:
        extra = frozenset(m.keys()) - _ORACLE_ARGMAX_WEIGHT_HISTOGRAM_ALLOWED
        if extra:
            raise ValueError(
                f"Unknown keys for oracle_argmax_weight_histogram plot: {sorted(extra)}"
            )
        split = m.get("split", "all")
        if split not in ("all", "train", "dev", "test"):
            raise ValueError(f"split must be all|train|dev|test, got {split!r}")
        return OracleArgmaxWeightHistogramSpec(
            kind="oracle_argmax_weight_histogram",
            split=split,  # type: ignore[arg-type]
            exclude_all_zero_queries=bool(m.get("exclude_all_zero_queries", True)),
            hist_bins=int(m.get("hist_bins", 0)),
            rtol=float(m.get("rtol", 1e-5)),
            atol=float(m.get("atol", 1e-8)),
            filename_stem=str(m.get("filename_stem", "oracle_argmax_weight_histogram")),
            fig_width=float(m.get("fig_width", 8.0)),
            fig_height=float(m.get("fig_height", 4.5)),
            show_plot_subtitle=bool(m.get("show_plot_subtitle", True)),
        )


def figure_spec_from_mapping(plot: Mapping[str, Any]) -> BaseFigureSpec:
    """Dispatch on ``kind`` for one entry under ``figures.plots``."""
    kind = plot.get("kind")
    if not kind or not str(kind).strip():
        raise ValueError("Each plot entry must include non-empty 'kind'")
    key = str(kind).strip()
    if key == "router_pred_vs_oracle":
        return RouterPredVsOracleSpec.from_mapping(plot)
    if key == "router_pred_vs_oracle_intervals":
        return RouterPredVsOracleIntervalsSpec.from_mapping(plot)
    if key == "benchmark_oracle_ndcg_heatmap":
        return BenchmarkOracleHeatmapSpec.from_mapping(plot)
    if key == "oracle_argmax_weight_histogram":
        return OracleArgmaxWeightHistogramSpec.from_mapping(plot)
    raise ValueError(f"Unknown figure kind {key!r}. Known kinds: {_KNOWN_KINDS_HINT}")
