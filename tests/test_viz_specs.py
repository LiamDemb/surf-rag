from __future__ import annotations

import pytest

from surf_rag.viz.specs import (
    BenchmarkOracleHeatmapSpec,
    OracleArgmaxWeightHistogramSpec,
    RouterPredVsOracleIntervalsSpec,
    RouterPredVsOracleSpec,
    RouterTrainingLearningCurveSpec,
    figure_spec_from_mapping,
)


def test_router_pred_vs_oracle_from_mapping_defaults() -> None:
    s = RouterPredVsOracleSpec.from_mapping(
        {"kind": "router_pred_vs_oracle", "split": "test"}
    )
    assert s.split == "test"
    assert s.filter_invalid_only is False
    assert s.point_size == 12.0
    assert s.alpha == pytest.approx(0.35)
    assert s.filename_stem == "router_pred_vs_oracle"


def test_router_pred_vs_oracle_from_mapping_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        RouterPredVsOracleSpec.from_mapping(
            {"kind": "router_pred_vs_oracle", "split": "test", "extra_bad": 1}
        )


def test_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="split"):
        RouterPredVsOracleSpec.from_mapping(
            {"kind": "router_pred_vs_oracle", "split": "val"}
        )


def test_figure_spec_from_mapping_dispatches() -> None:
    spec = figure_spec_from_mapping({"kind": "router_pred_vs_oracle", "split": "dev"})
    assert isinstance(spec, RouterPredVsOracleSpec)
    assert spec.split == "dev"


def test_router_pred_vs_oracle_intervals_from_mapping() -> None:
    spec = RouterPredVsOracleIntervalsSpec.from_mapping(
        {"kind": "router_pred_vs_oracle_intervals", "split": "test"}
    )
    assert spec.max_queries is None
    assert spec.interval_bar_width_ratio == pytest.approx(0.72)


def test_figure_spec_from_mapping_intervals_dispatch() -> None:
    spec = figure_spec_from_mapping({"kind": "router_pred_vs_oracle_intervals"})
    assert isinstance(spec, RouterPredVsOracleIntervalsSpec)
    assert spec.split == "test"


def test_figure_spec_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown figure kind"):
        figure_spec_from_mapping({"kind": "not-a-real-plot"})


def test_benchmark_heatmap_from_mapping_defaults() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {"kind": "benchmark_oracle_ndcg_heatmap"}
    )
    assert s.split == "all"
    assert s.exclude_all_zero_queries is True
    assert s.colormap == "RdYlGn"
    assert s.color_norm == "log"
    assert s.color_power_gamma == 2.0
    assert s.interior_peak_only is False
    assert s.exclude_all_one_queries is False
    assert s.mid_dense_band_strict_best is False
    assert s.show_plot_subtitle is True


def test_benchmark_heatmap_show_plot_subtitle_from_yaml() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {
            "kind": "benchmark_oracle_ndcg_heatmap",
            "show_plot_subtitle": False,
        }
    )
    assert s.show_plot_subtitle is False


def test_benchmark_heatmap_mid_dense_band_strict_best_from_yaml() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {
            "kind": "benchmark_oracle_ndcg_heatmap",
            "mid_dense_band_strict_best": True,
        }
    )
    assert s.mid_dense_band_strict_best is True


def test_benchmark_heatmap_exclude_all_one_queries_from_yaml() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {
            "kind": "benchmark_oracle_ndcg_heatmap",
            "exclude_all_one_queries": True,
        }
    )
    assert s.exclude_all_one_queries is True


def test_benchmark_heatmap_color_norm_linear() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {"kind": "benchmark_oracle_ndcg_heatmap", "color_norm": "linear"}
    )
    assert s.color_norm == "linear"


def test_benchmark_heatmap_interior_peak_only_from_yaml() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {
            "kind": "benchmark_oracle_ndcg_heatmap",
            "interior_peak_only": True,
        }
    )
    assert s.interior_peak_only is True


def test_benchmark_heatmap_color_norm_power() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {
            "kind": "benchmark_oracle_ndcg_heatmap",
            "color_norm": "power",
            "color_power_gamma": 3.5,
        }
    )
    assert s.color_norm == "power"
    assert s.color_power_gamma == 3.5


def test_benchmark_heatmap_color_power_gamma_invalid() -> None:
    with pytest.raises(ValueError, match="color_power_gamma must be positive"):
        BenchmarkOracleHeatmapSpec(
            kind="benchmark_oracle_ndcg_heatmap",
            split="all",
            color_power_gamma=0.0,
        )


def test_benchmark_heatmap_colormap_from_yaml() -> None:
    s = BenchmarkOracleHeatmapSpec.from_mapping(
        {"kind": "benchmark_oracle_ndcg_heatmap", "colormap": "Blues"}
    )
    assert s.colormap == "Blues"


def test_figure_spec_benchmark_dispatch() -> None:
    spec = figure_spec_from_mapping({"kind": "benchmark_oracle_ndcg_heatmap"})
    assert isinstance(spec, BenchmarkOracleHeatmapSpec)


def test_oracle_argmax_histogram_from_mapping_defaults() -> None:
    s = OracleArgmaxWeightHistogramSpec.from_mapping(
        {"kind": "oracle_argmax_weight_histogram"}
    )
    assert s.split == "all"
    assert s.hist_bins == 0
    assert s.exclude_all_zero_queries is True


def test_figure_spec_oracle_argmax_histogram_dispatch() -> None:
    spec = figure_spec_from_mapping({"kind": "oracle_argmax_weight_histogram"})
    assert isinstance(spec, OracleArgmaxWeightHistogramSpec)


def test_router_training_learning_curve_defaults() -> None:
    s = RouterTrainingLearningCurveSpec.from_mapping(
        {"kind": "router_training_learning_curve"}
    )
    assert s.filename_stem == "router_training_learning_curve"
    assert s.include_dev is True
    assert s.show_loss is True
    assert s.show_regret is False


def test_router_training_learning_curve_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown keys"):
        RouterTrainingLearningCurveSpec.from_mapping(
            {"kind": "router_training_learning_curve", "bad": 1}
        )


def test_router_training_learning_curve_requires_metric() -> None:
    with pytest.raises(ValueError, match="At least one"):
        RouterTrainingLearningCurveSpec(
            kind="router_training_learning_curve",
            show_loss=False,
            show_regret=False,
        )


def test_figure_spec_router_training_learning_curve_dispatch() -> None:
    spec = figure_spec_from_mapping({"kind": "router_training_learning_curve"})
    assert isinstance(spec, RouterTrainingLearningCurveSpec)
