"""Compact MLP router: embedding and/or feature branch, 11-bin logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Folder names and checkpoint values for input ablations
ROUTER_INPUT_MODE_BOTH: str = "both"
ROUTER_INPUT_MODE_QUERY_FEATURES: str = "query-features"
ROUTER_INPUT_MODE_EMBEDDING: str = "embedding"

ALLOWED_ROUTER_INPUT_MODES: Tuple[str, ...] = (
    ROUTER_INPUT_MODE_BOTH,
    ROUTER_INPUT_MODE_QUERY_FEATURES,
    ROUTER_INPUT_MODE_EMBEDDING,
)


def parse_router_input_mode(value: str) -> str:
    s = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "both": ROUTER_INPUT_MODE_BOTH,
        "query-features": ROUTER_INPUT_MODE_QUERY_FEATURES,
        "query_features": ROUTER_INPUT_MODE_QUERY_FEATURES,
        "features": ROUTER_INPUT_MODE_QUERY_FEATURES,
        "embedding": ROUTER_INPUT_MODE_EMBEDDING,
        "embed": ROUTER_INPUT_MODE_EMBEDDING,
    }
    if s not in aliases:
        raise ValueError(
            f"input_mode must be one of {list(aliases.keys())!r}, got {value!r}"
        )
    return aliases[s]


def active_inputs_for_mode(input_mode: str) -> List[str]:
    m = parse_router_input_mode(input_mode)
    if m == ROUTER_INPUT_MODE_BOTH:
        return ["query_embedding", "feature_vector_norm"]
    if m == ROUTER_INPUT_MODE_QUERY_FEATURES:
        return ["feature_vector_norm"]
    return ["query_embedding"]


@dataclass(frozen=True)
class RouterMLPConfig:
    """Architecture hyperparameters."""

    embedding_dim: int = 384
    feature_dim: int = 14
    input_mode: str = ROUTER_INPUT_MODE_BOTH
    embed_proj_dim: int = 16
    feat_proj_dim: int = 16
    hidden_dim: int = 32
    num_bins: int = 11
    dropout: float = 0.1

    def to_json(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "input_mode": self.input_mode,
            "embed_proj_dim": self.embed_proj_dim,
            "feat_proj_dim": self.feat_proj_dim,
            "hidden_dim": self.hidden_dim,
            "num_bins": self.num_bins,
            "dropout": self.dropout,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "RouterMLPConfig":
        raw_mode = d.get("input_mode", ROUTER_INPUT_MODE_BOTH)
        if raw_mode is None or not isinstance(raw_mode, str):
            input_mode = ROUTER_INPUT_MODE_BOTH
        else:
            try:
                input_mode = parse_router_input_mode(raw_mode)
            except ValueError:
                input_mode = ROUTER_INPUT_MODE_BOTH
        return cls(
            embedding_dim=int(d.get("embedding_dim", 384)),
            feature_dim=int(d.get("feature_dim", 14)),
            input_mode=input_mode,
            embed_proj_dim=int(d.get("embed_proj_dim", 16)),
            feat_proj_dim=int(d.get("feat_proj_dim", 16)),
            hidden_dim=int(d.get("hidden_dim", 32)),
            num_bins=int(d.get("num_bins", 11)),
            dropout=float(d.get("dropout", 0.1)),
        )


class RouterMLP(nn.Module):
    """Classifies over the weight grid; uses one or both input branches by ``input_mode``."""

    def __init__(self, config: RouterMLPConfig) -> None:
        super().__init__()
        d = config
        self.config = d
        mode = parse_router_input_mode(d.input_mode)

        self.embed_ln: Optional[nn.Module]
        self.embed_proj: Optional[nn.Module]
        self.feat_proj: Optional[nn.Module]

        if mode in (ROUTER_INPUT_MODE_BOTH, ROUTER_INPUT_MODE_EMBEDDING):
            self.embed_ln = nn.LayerNorm(d.embedding_dim)
            self.embed_proj = nn.Linear(d.embedding_dim, d.embed_proj_dim)
        else:
            self.embed_ln = None
            self.embed_proj = None

        if mode in (ROUTER_INPUT_MODE_BOTH, ROUTER_INPUT_MODE_QUERY_FEATURES):
            self.feat_proj = nn.Linear(d.feature_dim, d.feat_proj_dim)
        else:
            self.feat_proj = None

        if mode == ROUTER_INPUT_MODE_BOTH:
            in_h = d.embed_proj_dim + d.feat_proj_dim
        elif mode == ROUTER_INPUT_MODE_QUERY_FEATURES:
            in_h = d.feat_proj_dim
        else:
            in_h = d.embed_proj_dim

        self.head = nn.Sequential(
            nn.Linear(in_h, d.hidden_dim),
            nn.GELU(),
            nn.Dropout(d.dropout),
            nn.Linear(d.hidden_dim, d.num_bins),
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        mode = parse_router_input_mode(self.config.input_mode)
        if mode == ROUTER_INPUT_MODE_BOTH:
            assert self.embed_ln is not None and self.embed_proj is not None
            assert self.feat_proj is not None
            z = self.embed_ln(query_embedding)
            e = F.gelu(self.embed_proj(z))
            f = F.gelu(self.feat_proj(feature_vector))
            h = torch.cat([e, f], dim=-1)
        elif mode == ROUTER_INPUT_MODE_QUERY_FEATURES:
            assert self.feat_proj is not None
            h = F.gelu(self.feat_proj(feature_vector))
        else:
            assert self.embed_ln is not None and self.embed_proj is not None
            z = self.embed_ln(query_embedding)
            h = F.gelu(self.embed_proj(z))
        return self.head(h)

    def predict_distribution(
        self,
        query_embedding: torch.Tensor,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        return F.softmax(self(query_embedding, feature_vector), dim=-1)


def expected_dense_weight(
    distribution: torch.Tensor,
    weight_grid: torch.Tensor,
) -> torch.Tensor:
    """Batch expected value ``sum p_i w_i``; ``weight_grid`` shape (num_bins,)."""
    if distribution.dim() == 1:
        distribution = distribution.unsqueeze(0)
    return (distribution * weight_grid.unsqueeze(0)).sum(dim=-1)


def stack_weight_grid(
    weight_grid: Sequence[float], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.tensor(list(weight_grid), device=device, dtype=dtype)


def list_to_feature_tensor(
    rows: List[List[float]], device: torch.device
) -> torch.Tensor:
    if not rows:
        return torch.zeros(0, 0, device=device, dtype=torch.float32)
    t = torch.tensor(rows, device=device, dtype=torch.float32)
    return t
