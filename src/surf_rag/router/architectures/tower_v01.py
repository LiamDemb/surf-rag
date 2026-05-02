"""Dual-tower router: embedding MLP (GELU) to linear ``feature_dim`` output; feature MLP; scalar head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from surf_rag.router.model import parse_router_input_mode

_DEFAULT_EMBED_DIMS: tuple[int, int, int] = (128, 64, 32)


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"feat_hidden", "dropout", "embed_dims"}
    unknown = sorted(set(raw.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"tower_v01 unknown architecture kwargs: {unknown}. Allowed: {sorted(allowed)}"
        )
    out = dict(raw)
    if "feat_hidden" in out:
        fh = int(out["feat_hidden"])
        if fh <= 0:
            raise ValueError("tower_v01 feat_hidden must be > 0")
        out["feat_hidden"] = fh
    if "dropout" in out:
        d = float(out["dropout"])
        if d < 0.0 or d >= 1.0:
            raise ValueError("tower_v01 dropout must be in [0, 1)")
        out["dropout"] = d
    if "embed_dims" in out and out["embed_dims"] is not None:
        ed = out["embed_dims"]
        if not isinstance(ed, (list, tuple)) or len(ed) != 3:
            raise ValueError(
                "tower_v01 embed_dims must be a length-3 list of positive ints "
                "[d384_to, d1, d2] before the final projection to feature_dim"
            )
        dims = [int(x) for x in ed]
        if any(x <= 0 for x in dims):
            raise ValueError("tower_v01 embed_dims entries must be > 0")
        out["embed_dims"] = tuple(dims)
    elif "embed_dims" in out:
        del out["embed_dims"]
    return {
        "feat_hidden": int(out.get("feat_hidden", 32)),
        "dropout": float(out.get("dropout", 0.0)),
        "embed_dims": (
            tuple(int(x) for x in out["embed_dims"])
            if "embed_dims" in out and out["embed_dims"] is not None
            else _DEFAULT_EMBED_DIMS
        ),
    }


@dataclass(frozen=True)
class RouterTowerV01Config:
    embedding_dim: int = 384
    feature_dim: int = 14
    input_mode: str = "both"
    feat_hidden: int = 32
    dropout: float = 0.0
    embed_dims: tuple[int, int, int] = _DEFAULT_EMBED_DIMS

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "feature_dim": int(self.feature_dim),
            "input_mode": str(self.input_mode),
            "feat_hidden": int(self.feat_hidden),
            "dropout": float(self.dropout),
            "embed_dims": list(self.embed_dims),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RouterTowerV01Config":
        ed = d.get("embed_dims", list(_DEFAULT_EMBED_DIMS))
        if isinstance(ed, (list, tuple)) and len(ed) == 3:
            embed_dims = tuple(int(x) for x in ed)
        else:
            embed_dims = _DEFAULT_EMBED_DIMS
        return cls(
            embedding_dim=int(d.get("embedding_dim", 384)),
            feature_dim=int(d.get("feature_dim", 14)),
            input_mode=parse_router_input_mode(str(d.get("input_mode", "both"))),
            feat_hidden=int(d.get("feat_hidden", 32)),
            dropout=float(d.get("dropout", 0.0)),
            embed_dims=embed_dims,
        )


class RouterTowerV01(nn.Module):
    """Embedding tower uses GELU on hidden blocks only; final ``Linear(d2, feature_dim)`` has no
    activation so the fusion head receives a full (signed) vector. Feature tower unchanged.
    """

    def __init__(self, config: RouterTowerV01Config) -> None:
        super().__init__()
        self.config = config
        mode = parse_router_input_mode(config.input_mode)
        d0, d1, d2 = config.embed_dims
        fd = int(config.feature_dim)
        fh = int(config.feat_hidden)
        drop = float(config.dropout)

        self.embed_ln: nn.Module | None = None
        self.embed_layers: nn.Module | None = None
        self.feat_layers: nn.Module | None = None
        self.head: nn.Linear

        if mode in ("both", "embedding"):
            emb_in = int(config.embedding_dim)
            self.embed_ln = nn.LayerNorm(emb_in)
            self.embed_layers = nn.Sequential(
                nn.Linear(emb_in, d0),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d0, d1),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d1, d2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d2, fd),
            )
        else:
            self.embed_ln = None
            self.embed_layers = None

        if mode in ("both", "query-features"):
            self.feat_layers = nn.Sequential(
                nn.Linear(fd, fh),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(fh, fd),
                nn.ReLU(),
            )
        else:
            self.feat_layers = None

        if mode == "both":
            self.head = nn.Linear(2 * fd, 1)
        else:
            self.head = nn.Linear(fd, 1)

    def forward(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        mode = parse_router_input_mode(self.config.input_mode)
        if mode == "both":
            assert self.embed_ln is not None and self.embed_layers is not None
            assert self.feat_layers is not None
            z = self.embed_ln(query_embedding)
            e = self.embed_layers(z)
            f = self.feat_layers(feature_vector)
            h = torch.cat([e, f], dim=-1)
        elif mode == "embedding":
            assert self.embed_ln is not None and self.embed_layers is not None
            z = self.embed_ln(query_embedding)
            h = self.embed_layers(z)
        else:
            assert self.feat_layers is not None
            h = self.feat_layers(feature_vector)
        return self.head(h)

    def predict_weight(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        logits = self(query_embedding, feature_vector).squeeze(-1)
        return torch.sigmoid(logits)


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    kwargs: dict[str, Any],
) -> RouterTowerV01Config:
    k = validate_kwargs(kwargs)
    return RouterTowerV01Config(
        embedding_dim=int(embedding_dim),
        feature_dim=int(feature_dim),
        input_mode=parse_router_input_mode(input_mode),
        feat_hidden=int(k["feat_hidden"]),
        dropout=float(k["dropout"]),
        embed_dims=tuple(int(x) for x in k["embed_dims"]),
    )


def config_from_json(payload: dict[str, Any]) -> RouterTowerV01Config:
    return RouterTowerV01Config.from_json(payload)


def build_model(config: RouterTowerV01Config) -> RouterTowerV01:
    return RouterTowerV01(config)
