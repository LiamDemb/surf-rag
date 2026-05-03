"""Logistic-regression router architecture baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from surf_rag.router.excluded_features import (
    active_feature_column_indices,
    normalize_excluded_features,
    validate_exclusions_for_input_mode,
)
from surf_rag.router.model import parse_router_input_mode


@dataclass(frozen=True)
class RouterLogRegConfig:
    embedding_dim: int = 384
    feature_dim: int = 14
    input_mode: str = "both"
    excluded_features: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "feature_dim": int(self.feature_dim),
            "input_mode": str(self.input_mode),
            "excluded_features": list(self.excluded_features),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RouterLogRegConfig":
        return cls(
            embedding_dim=int(d.get("embedding_dim", 384)),
            feature_dim=int(d.get("feature_dim", 14)),
            input_mode=parse_router_input_mode(str(d.get("input_mode", "both"))),
            excluded_features=normalize_excluded_features(d.get("excluded_features")),
        )


class RouterLogReg(nn.Module):
    """Single linear head over selected inputs."""

    def __init__(self, config: RouterLogRegConfig) -> None:
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
            feat_idx = active_feature_column_indices(feat_d, excluded)
            if not feat_idx:
                raise ValueError(
                    "logreg-v1 query-features mode needs at least one non-excluded feature"
                )
            in_dim = len(feat_idx)
        else:
            feat_idx = active_feature_column_indices(feat_d, excluded)
            in_dim = emb_d + len(feat_idx)

        self._mode = mode
        if feat_idx:
            self.register_buffer(
                "_feat_gather_idx", torch.tensor(feat_idx, dtype=torch.long)
            )
        else:
            self.register_buffer("_feat_gather_idx", torch.empty(0, dtype=torch.long))

        self.head = nn.Linear(in_dim, 1)

    def _select_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        if self._feat_gather_idx.numel() == 0:
            return feature_vector[..., :0]
        return feature_vector.index_select(-1, self._feat_gather_idx)

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
        return self.head(x)

    def predict_weight(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        logits = self(query_embedding, feature_vector).squeeze(-1)
        return torch.sigmoid(logits)


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(raw.keys()) - {"excluded_features"})
    if unknown:
        raise ValueError(
            f"logreg-v1 does not accept architecture kwargs {unknown}; "
            "only optional excluded_features is allowed."
        )
    excluded = normalize_excluded_features(raw.get("excluded_features"))
    return {"excluded_features": excluded}


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    kwargs: dict[str, Any],
) -> RouterLogRegConfig:
    opt = validate_kwargs(kwargs)
    mode = parse_router_input_mode(input_mode)
    excl = frozenset(opt["excluded_features"])
    if excl:
        active_feature_column_indices(feature_dim, excl)
    validate_exclusions_for_input_mode(
        mode,
        feature_dim,
        excl,
        query_features_need_one=True,
    )
    return RouterLogRegConfig(
        embedding_dim=int(embedding_dim),
        feature_dim=int(feature_dim),
        input_mode=mode,
        excluded_features=tuple(opt["excluded_features"]),
    )


def config_from_json(payload: dict[str, Any]) -> RouterLogRegConfig:
    return RouterLogRegConfig.from_json(payload)


def build_model(config: RouterLogRegConfig) -> RouterLogReg:
    return RouterLogReg(config)
