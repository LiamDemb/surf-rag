"""Logistic-regression router architecture baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from surf_rag.router.model import parse_router_input_mode


@dataclass(frozen=True)
class RouterLogRegConfig:
    embedding_dim: int = 384
    feature_dim: int = 14
    input_mode: str = "both"

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "feature_dim": int(self.feature_dim),
            "input_mode": str(self.input_mode),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RouterLogRegConfig":
        return cls(
            embedding_dim=int(d.get("embedding_dim", 384)),
            feature_dim=int(d.get("feature_dim", 14)),
            input_mode=parse_router_input_mode(str(d.get("input_mode", "both"))),
        )


class RouterLogReg(nn.Module):
    """Single linear head over selected inputs."""

    def __init__(self, config: RouterLogRegConfig) -> None:
        super().__init__()
        self.config = config
        mode = parse_router_input_mode(config.input_mode)
        in_dim = 0
        if mode in ("both", "embedding"):
            in_dim += int(config.embedding_dim)
        if mode in ("both", "query-features"):
            in_dim += int(config.feature_dim)
        self.head = nn.Linear(in_dim, 1)

    def forward(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        mode = parse_router_input_mode(self.config.input_mode)
        if mode == "both":
            x = torch.cat([query_embedding, feature_vector], dim=-1)
        elif mode == "embedding":
            x = query_embedding
        else:
            x = feature_vector
        return self.head(x)

    def predict_weight(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        logits = self(query_embedding, feature_vector).squeeze(-1)
        return torch.sigmoid(logits)


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    if raw:
        raise ValueError(
            "logreg-v1 does not accept architecture kwargs; expected empty mapping."
        )
    return {}


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    kwargs: dict[str, Any],
) -> RouterLogRegConfig:
    validate_kwargs(kwargs)
    return RouterLogRegConfig(
        embedding_dim=int(embedding_dim),
        feature_dim=int(feature_dim),
        input_mode=parse_router_input_mode(input_mode),
    )


def config_from_json(payload: dict[str, Any]) -> RouterLogRegConfig:
    return RouterLogRegConfig.from_json(payload)


def build_model(config: RouterLogRegConfig) -> RouterLogReg:
    return RouterLogReg(config)
