"""Polynomial-feature logistic regression router (linear head on monomials up to degree d)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from typing import Any

import torch
import torch.nn as nn

from surf_rag.router.model import parse_router_input_mode
from surf_rag.router.query_features import V1_FEATURE_NAMES

_ALLOWED_KW = frozenset({"degree", "max_expanded_features", "excluded_features"})

_DEFAULT_DEGREE = 2
_MAX_DEGREE = 12
_DEFAULT_MAX_EXPANDED = 150_000


def _normalize_excluded_features(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            "polyreg-v1 excluded_features must be a list of V1 feature name strings"
        )
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "polyreg-v1 excluded_features must be a list of V1 feature name strings"
        )
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "polyreg-v1 excluded_features entries must be non-empty strings"
            )
        out.append(str(item).strip())
    names = tuple(sorted(set(out)))
    for n in names:
        if n not in V1_FEATURE_NAMES:
            raise ValueError(
                f"polyreg-v1 excluded_features unknown name {n!r}; "
                f"allowed: {list(V1_FEATURE_NAMES)}"
            )
    return names


def _active_feature_column_indices(
    feature_dim: int, excluded: frozenset[str]
) -> tuple[int, ...]:
    fd = int(feature_dim)
    if fd > len(V1_FEATURE_NAMES):
        raise ValueError(
            f"feature_dim={feature_dim} exceeds V1 feature catalog ({len(V1_FEATURE_NAMES)})"
        )
    for name in excluded:
        idx = V1_FEATURE_NAMES.index(name)
        if idx >= fd:
            raise ValueError(
                f"polyreg-v1 cannot exclude {name!r}: not among the first {feature_dim} "
                "router feature columns for this dataset"
            )
    return tuple(i for i in range(fd) if V1_FEATURE_NAMES[i] not in excluded)


def expanded_monomial_count(n_inputs: int, degree: int) -> int:
    """Count monomials of total degree in 1..degree (matches sklearn PolynomialFeatures sans bias)."""
    if n_inputs <= 0:
        return 0
    total = 0
    for k in range(1, degree + 1):
        total += comb(n_inputs + k - 1, k)
    return int(total)


def _monomial_index_tuples(n_inputs: int, degree: int) -> list[tuple[int, ...]]:
    terms: list[tuple[int, ...]] = []
    for k in range(1, degree + 1):
        for tup in combinations_with_replacement(range(n_inputs), k):
            terms.append(tuple(tup))
    return terms


@dataclass(frozen=True)
class RouterPolyRegConfig:
    embedding_dim: int = 384
    feature_dim: int = 14
    input_mode: str = "both"
    degree: int = _DEFAULT_DEGREE
    excluded_features: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "feature_dim": int(self.feature_dim),
            "input_mode": str(self.input_mode),
            "degree": int(self.degree),
            "excluded_features": list(self.excluded_features),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RouterPolyRegConfig:
        ex_t = _normalize_excluded_features(d.get("excluded_features"))
        return cls(
            embedding_dim=int(d.get("embedding_dim", 384)),
            feature_dim=int(d.get("feature_dim", 14)),
            input_mode=parse_router_input_mode(str(d.get("input_mode", "both"))),
            degree=int(d.get("degree", _DEFAULT_DEGREE)),
            excluded_features=ex_t,
        )


class RouterPolyReg(nn.Module):
    """``Linear(poly_features(x))`` then sigmoid for ``predict_weight`` (same routing as logreg)."""

    def __init__(self, config: RouterPolyRegConfig) -> None:
        super().__init__()
        self.config = config
        mode = parse_router_input_mode(config.input_mode)
        excluded = frozenset(config.excluded_features)
        emb_d = int(config.embedding_dim)
        feat_d = int(config.feature_dim)

        if mode == "embedding":
            in_dim = emb_d
            feat_idx: tuple[int, ...] = ()
        elif mode == "query-features":
            feat_idx = _active_feature_column_indices(feat_d, excluded)
            if not feat_idx:
                raise ValueError(
                    "polyreg-v1 query-features mode needs at least one non-excluded feature"
                )
            in_dim = len(feat_idx)
        else:
            feat_idx = _active_feature_column_indices(feat_d, excluded)
            in_dim = emb_d + len(feat_idx)

        if in_dim <= 0:
            raise ValueError("polyreg-v1 requires a positive input dimension")

        self._mode = mode
        self._emb_dim = emb_d
        if feat_idx:
            idx_t = torch.tensor(feat_idx, dtype=torch.long)
            self.register_buffer("_feat_gather_idx", idx_t)
        else:
            self.register_buffer("_feat_gather_idx", torch.empty(0, dtype=torch.long))

        self._in_dim = in_dim
        deg = max(1, int(config.degree))
        self._term_indices = _monomial_index_tuples(in_dim, deg)
        expanded = len(self._term_indices)
        self.head = nn.Linear(expanded, 1)

    def _select_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        if self._feat_gather_idx.numel() == 0:
            return feature_vector[..., :0]
        return feature_vector.index_select(-1, self._feat_gather_idx)

    def _poly_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self._in_dim:
            raise ValueError(f"expected input dim {self._in_dim}, got {x.shape[-1]}")
        cols: list[torch.Tensor] = []
        for tup in self._term_indices:
            t = x[..., tup[0]]
            for j in tup[1:]:
                t = t * x[..., j]
            cols.append(t.unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    def forward(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        mode = self._mode
        if mode == "both":
            xf = self._select_features(feature_vector)
            x = torch.cat([query_embedding, xf], dim=-1)
        elif mode == "embedding":
            x = query_embedding
        else:
            x = self._select_features(feature_vector)
        phi = self._poly_features(x)
        return self.head(phi)

    def predict_weight(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        logits = self(query_embedding, feature_vector).squeeze(-1)
        return torch.sigmoid(logits)


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw) - _ALLOWED_KW)
    if unknown:
        raise ValueError(f"unknown architecture kwargs for polyreg-v1: {unknown}")
    deg = raw.get("degree", _DEFAULT_DEGREE)
    if isinstance(deg, bool) or not isinstance(deg, (int, float)):
        raise ValueError("polyreg-v1 degree must be int")
    deg_i = int(deg)
    if deg_i != float(deg):
        raise ValueError("polyreg-v1 degree must be integral")
    cap = raw.get("max_expanded_features", _DEFAULT_MAX_EXPANDED)
    if type(cap) is not int:
        raise ValueError("polyreg-v1 max_expanded_features must be int")
    if cap < 1:
        raise ValueError("polyreg-v1 max_expanded_features must be >= 1")
    if deg_i < 1 or deg_i > _MAX_DEGREE:
        raise ValueError(
            f"polyreg-v1 degree must be between 1 and {_MAX_DEGREE}, got {deg_i}"
        )
    excluded = _normalize_excluded_features(raw.get("excluded_features"))
    return {
        "degree": deg_i,
        "max_expanded_features": int(cap),
        "excluded_features": excluded,
    }


def _effective_input_dim_for_poly(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    excluded: frozenset[str],
) -> int:
    mode = parse_router_input_mode(input_mode)
    if mode == "embedding":
        return int(embedding_dim)
    if mode == "query-features":
        n_kept = len(_active_feature_column_indices(feature_dim, excluded))
        return n_kept
    emb_d = int(embedding_dim)
    n_kept = len(_active_feature_column_indices(feature_dim, excluded))
    return emb_d + n_kept


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    kwargs: dict[str, Any],
) -> RouterPolyRegConfig:
    opt = validate_kwargs(kwargs)
    excluded_tuple = opt["excluded_features"]
    excluded_fs = frozenset(excluded_tuple)
    mode = parse_router_input_mode(input_mode)

    if excluded_fs:
        for name in excluded_fs:
            idx = V1_FEATURE_NAMES.index(name)
            if idx >= int(feature_dim):
                raise ValueError(
                    f"polyreg-v1 cannot exclude {name!r}: index {idx} >= feature_dim={feature_dim}"
                )

    n_in = _effective_input_dim_for_poly(
        embedding_dim, feature_dim, input_mode, excluded_fs
    )
    if n_in <= 0:
        raise ValueError("polyreg-v1 needs a positive effective input size")
    cap = opt["max_expanded_features"]
    deg = opt["degree"]
    n_phi = expanded_monomial_count(n_in, deg)
    if n_phi > cap:
        raise ValueError(
            f"polyreg-v1 would expand ({n_in=}, {deg=}) to {n_phi} features, "
            f"above max_expanded_features={cap}. Lower degree, use fewer inputs "
            "(excluded_features, query-features-only), or raise max_expanded_features."
        )
    return RouterPolyRegConfig(
        embedding_dim=int(embedding_dim),
        feature_dim=int(feature_dim),
        input_mode=mode,
        degree=deg,
        excluded_features=excluded_tuple,
    )


def config_from_json(payload: dict[str, Any]) -> RouterPolyRegConfig:
    return RouterPolyRegConfig.from_json(payload)


def build_model(config: RouterPolyRegConfig) -> RouterPolyReg:
    return RouterPolyReg(config)
