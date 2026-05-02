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


def figure_spec_from_mapping(plot: Mapping[str, Any]) -> BaseFigureSpec:
    """Dispatch on ``kind`` for one entry under ``figures.plots``."""
    kind = plot.get("kind")
    if not kind or not str(kind).strip():
        raise ValueError("Each plot entry must include non-empty 'kind'")
    key = str(kind).strip()
    if key == "router_pred_vs_oracle":
        return RouterPredVsOracleSpec.from_mapping(plot)
    raise ValueError(
        f"Unknown figure kind {key!r}. " f"Known kinds: router_pred_vs_oracle"
    )
