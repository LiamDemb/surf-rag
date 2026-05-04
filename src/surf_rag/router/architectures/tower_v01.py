"""Dual-tower router: embedding MLP (GELU) to linear ``feature_dim`` output; feature MLP; scalar head."""

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
from surf_rag.router.model import (
    ROUTER_TASK_REGRESSION,
    parse_router_input_mode,
    parse_router_task_type,
)

_DEFAULT_EMBED_DIMS: tuple[int, int, int] = (128, 64, 32)


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"feat_hidden", "dropout", "embed_dims", "excluded_features"}
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
    excluded = normalize_excluded_features(out.get("excluded_features"))
    return {
        "feat_hidden": int(out.get("feat_hidden", 32)),
        "dropout": float(out.get("dropout", 0.0)),
        "embed_dims": (
            tuple(int(x) for x in out["embed_dims"])
            if "embed_dims" in out and out["embed_dims"] is not None
            else _DEFAULT_EMBED_DIMS
        ),
        "excluded_features": excluded,
    }


@dataclass(frozen=True)
class RouterTowerV01Config:
    embedding_dim: int = 384
    feature_dim: int = 14
    input_mode: str = "both"
    feat_hidden: int = 32
    dropout: float = 0.0
    embed_dims: tuple[int, int, int] = _DEFAULT_EMBED_DIMS
    excluded_features: tuple[str, ...] = ()
    task_type: str = ROUTER_TASK_REGRESSION

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding_dim": int(self.embedding_dim),
            "feature_dim": int(self.feature_dim),
            "input_mode": str(self.input_mode),
            "feat_hidden": int(self.feat_hidden),
            "dropout": float(self.dropout),
            "embed_dims": list(self.embed_dims),
            "excluded_features": list(self.excluded_features),
            "task_type": str(self.task_type),
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
            excluded_features=normalize_excluded_features(d.get("excluded_features")),
            task_type=parse_router_task_type(
                str(d.get("task_type", ROUTER_TASK_REGRESSION))
            ),
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
        excluded = frozenset(config.excluded_features)

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
            feat_idx = active_feature_column_indices(fd, excluded)
            if mode == "query-features" and not feat_idx:
                raise ValueError(
                    "tower_v01 query-features mode needs at least one non-excluded feature"
                )
            if feat_idx:
                nk = len(feat_idx)
                self.register_buffer(
                    "_feat_gather_idx", torch.tensor(feat_idx, dtype=torch.long)
                )
                self.feat_layers = nn.Sequential(
                    nn.Linear(nk, fh),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(fh, fd),
                    nn.ReLU(),
                )
            else:
                self.register_buffer(
                    "_feat_gather_idx", torch.empty(0, dtype=torch.long)
                )
                self.feat_layers = None
        else:
            self.register_buffer("_feat_gather_idx", torch.empty(0, dtype=torch.long))
            self.feat_layers = None

        if mode == "both":
            fusion_in = 2 * fd if self.feat_layers is not None else fd
            self.head = nn.Linear(
                fusion_in,
                (
                    2
                    if parse_router_task_type(config.task_type)
                    != ROUTER_TASK_REGRESSION
                    else 1
                ),
            )
        else:
            self.head = nn.Linear(
                fd,
                (
                    2
                    if parse_router_task_type(config.task_type)
                    != ROUTER_TASK_REGRESSION
                    else 1
                ),
            )

    def _select_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        if self._feat_gather_idx.numel() == 0:
            return feature_vector[..., :0]
        return feature_vector.index_select(-1, self._feat_gather_idx)

    def forward(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        mode = parse_router_input_mode(self.config.input_mode)
        if mode == "both":
            assert self.embed_ln is not None and self.embed_layers is not None
            z = self.embed_ln(query_embedding)
            e = self.embed_layers(z)
            if self.feat_layers is not None:
                xf = self._select_features(feature_vector)
                fvec = self.feat_layers(xf)
                h = torch.cat([e, fvec], dim=-1)
            else:
                h = e
        elif mode == "embedding":
            assert self.embed_ln is not None and self.embed_layers is not None
            z = self.embed_ln(query_embedding)
            h = self.embed_layers(z)
        else:
            assert self.feat_layers is not None
            xf = self._select_features(feature_vector)
            h = self.feat_layers(xf)
        return self.head(h)

    def predict_weight(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        if parse_router_task_type(self.config.task_type) != ROUTER_TASK_REGRESSION:
            raise ValueError(
                "predict_weight requires regression task_type, got "
                f"{self.config.task_type!r}"
            )
        logits = self(query_embedding, feature_vector).squeeze(-1)
        return torch.sigmoid(logits)

    def predict_class_logits(
        self, query_embedding: torch.Tensor, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        if parse_router_task_type(self.config.task_type) == ROUTER_TASK_REGRESSION:
            raise ValueError(
                "predict_class_logits requires classification task_type, got "
                f"{self.config.task_type!r}"
            )
        return self(query_embedding, feature_vector)


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    task_type: str,
    kwargs: dict[str, Any],
) -> RouterTowerV01Config:
    k = validate_kwargs(kwargs)
    mode = parse_router_input_mode(input_mode)
    excl = frozenset(k["excluded_features"])
    if excl:
        active_feature_column_indices(feature_dim, excl)
    validate_exclusions_for_input_mode(
        mode,
        feature_dim,
        excl,
        query_features_need_one=True,
    )
    return RouterTowerV01Config(
        embedding_dim=int(embedding_dim),
        feature_dim=int(feature_dim),
        input_mode=mode,
        feat_hidden=int(k["feat_hidden"]),
        dropout=float(k["dropout"]),
        embed_dims=tuple(int(x) for x in k["embed_dims"]),
        excluded_features=tuple(k["excluded_features"]),
        task_type=parse_router_task_type(task_type or ROUTER_TASK_REGRESSION),
    )


def config_from_json(payload: dict[str, Any]) -> RouterTowerV01Config:
    return RouterTowerV01Config.from_json(payload)


def build_model(config: RouterTowerV01Config) -> RouterTowerV01:
    return RouterTowerV01(config)
