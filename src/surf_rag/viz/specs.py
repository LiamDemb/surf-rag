"""Typed figure specifications and YAML dict parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

SplitName = Literal["train", "dev", "test"]

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

_KNOWN_KINDS_HINT = "router_pred_vs_oracle, router_pred_vs_oracle_intervals"


@dataclass(frozen=True)
class BaseFigureSpec:
    """Base type for registered figure specs."""

    kind: str


@dataclass(frozen=True)
class RouterPredVsOracleSpec(BaseFigureSpec):
    """Scatter: x = oracle best grid weight, y = predicted dense weight."""

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
    raise ValueError(f"Unknown figure kind {key!r}. Known kinds: {_KNOWN_KINDS_HINT}")
