from __future__ import annotations

import pytest

from surf_rag.viz.specs import RouterPredVsOracleSpec, figure_spec_from_mapping


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


def test_figure_spec_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown figure kind"):
        figure_spec_from_mapping({"kind": "not-a-real-plot"})
