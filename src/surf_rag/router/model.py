"""Compact MLP router: query embedding projection, feature branch, 11-bin logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RouterMLPConfig:
    """Architecture hyperparameters."""

    embedding_dim: int = 384
    feature_dim: int = 14
    embed_proj_dim: int = 16
    feat_proj_dim: int = 16
    hidden_dim: int = 32
    num_bins: int = 11
    dropout: float = 0.1

    def to_json(self) -> Dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "embed_proj_dim": self.embed_proj_dim,
            "feat_proj_dim": self.feat_proj_dim,
            "hidden_dim": self.hidden_dim,
            "num_bins": self.num_bins,
            "dropout": self.dropout,
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "RouterMLPConfig":
        return cls(
            embedding_dim=int(d.get("embedding_dim", 384)),
            feature_dim=int(d.get("feature_dim", 14)),
            embed_proj_dim=int(d.get("embed_proj_dim", 16)),
            feat_proj_dim=int(d.get("feat_proj_dim", 16)),
            hidden_dim=int(d.get("hidden_dim", 32)),
            num_bins=int(d.get("num_bins", 11)),
            dropout=float(d.get("dropout", 0.1)),
        )


class RouterMLP(nn.Module):
    """Projects embedding and features, then classifies over the weight grid."""

    def __init__(self, config: RouterMLPConfig) -> None:
        super().__init__()
        self.config = config
        d = config
        self.embed_ln = nn.LayerNorm(d.embedding_dim)
        self.embed_proj = nn.Linear(d.embedding_dim, d.embed_proj_dim)
        self.feat_proj = nn.Linear(d.feature_dim, d.feat_proj_dim)
        in_h = d.embed_proj_dim + d.feat_proj_dim
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
        """Return logits of shape (batch, num_bins)."""
        z = self.embed_ln(query_embedding)
        e = F.gelu(self.embed_proj(z))
        f = F.gelu(self.feat_proj(feature_vector))
        h = torch.cat([e, f], dim=-1)
        return self.head(h)

    def predict_distribution(
        self,
        query_embedding: torch.Tensor,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Return probability distribution of shape (batch, num_bins)."""
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
