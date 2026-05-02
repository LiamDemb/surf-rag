"""MLP router architecture (current default)."""

from __future__ import annotations

from typing import Any

from surf_rag.router.model import RouterMLP, RouterMLPConfig, parse_router_input_mode


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"embed_proj_dim", "feat_proj_dim", "hidden_dim", "dropout"}
    unknown = sorted(set(raw.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"mlp-v1 unknown architecture kwargs: {unknown}. Allowed: {sorted(allowed)}"
        )
    out = dict(raw)
    if "embed_proj_dim" in out and int(out["embed_proj_dim"]) <= 0:
        raise ValueError("mlp-v1 embed_proj_dim must be > 0")
    if "feat_proj_dim" in out and int(out["feat_proj_dim"]) <= 0:
        raise ValueError("mlp-v1 feat_proj_dim must be > 0")
    if "hidden_dim" in out and int(out["hidden_dim"]) <= 0:
        raise ValueError("mlp-v1 hidden_dim must be > 0")
    if "dropout" in out:
        d = float(out["dropout"])
        if d < 0.0 or d >= 1.0:
            raise ValueError("mlp-v1 dropout must be in [0, 1)")
    return {
        "embed_proj_dim": int(out.get("embed_proj_dim", 16)),
        "feat_proj_dim": int(out.get("feat_proj_dim", 16)),
        "hidden_dim": int(out.get("hidden_dim", 32)),
        "dropout": float(out.get("dropout", 0.1)),
    }


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    kwargs: dict[str, Any],
) -> RouterMLPConfig:
    k = validate_kwargs(kwargs)
    return RouterMLPConfig(
        embedding_dim=int(embedding_dim),
        feature_dim=int(feature_dim),
        input_mode=parse_router_input_mode(input_mode),
        embed_proj_dim=int(k["embed_proj_dim"]),
        feat_proj_dim=int(k["feat_proj_dim"]),
        hidden_dim=int(k["hidden_dim"]),
        dropout=float(k["dropout"]),
    )


def config_from_json(payload: dict[str, Any]) -> RouterMLPConfig:
    return RouterMLPConfig.from_json(payload)


def build_model(config: RouterMLPConfig) -> RouterMLP:
    return RouterMLP(config)
