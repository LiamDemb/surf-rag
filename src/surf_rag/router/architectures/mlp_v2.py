"""Embedding-only MLP router (LayerNorm + two hidden blocks + task head)."""

from __future__ import annotations

from typing import Any

from surf_rag.router.excluded_features import normalize_excluded_features
from surf_rag.router.model import (
    ROUTER_INPUT_MODE_EMBEDDING,
    ROUTER_TASK_REGRESSION,
    RouterMLPv2,
    RouterMLPv2Config,
    parse_router_input_mode,
    parse_router_task_type,
)


def validate_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "hidden_dim_1",
        "hidden_dim_2",
        "dropout_1",
        "dropout_2",
        "activation",
        "excluded_features",
    }
    unknown = sorted(set(raw.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"mlp-v2 unknown architecture kwargs: {unknown}. Allowed: {sorted(allowed)}"
        )
    out = dict(raw)
    if "hidden_dim_1" in out and int(out["hidden_dim_1"]) <= 0:
        raise ValueError("mlp-v2 hidden_dim_1 must be > 0")
    if "hidden_dim_2" in out and int(out["hidden_dim_2"]) <= 0:
        raise ValueError("mlp-v2 hidden_dim_2 must be > 0")
    if "dropout_1" in out:
        u = float(out["dropout_1"])
        if u < 0.0 or u >= 1.0:
            raise ValueError("mlp-v2 dropout_1 must be in [0, 1)")
    if "dropout_2" in out:
        u = float(out["dropout_2"])
        if u < 0.0 or u >= 1.0:
            raise ValueError("mlp-v2 dropout_2 must be in [0, 1)")
    if "activation" in out and out["activation"] is not None:
        a = str(out["activation"]).strip().lower()
        if a not in ("gelu", "relu"):
            raise ValueError("mlp-v2 activation must be 'gelu' or 'relu'")
    excluded = normalize_excluded_features(out.get("excluded_features"))
    if excluded:
        raise ValueError(
            "mlp-v2 is embedding-only and does not support excluded_features. "
            "Remove router.train.excluded_features / architecture_kwargs.excluded_features."
        )
    return {
        "hidden_dim_1": int(out.get("hidden_dim_1", 32)),
        "hidden_dim_2": int(out.get("hidden_dim_2", 8)),
        "dropout_1": float(out.get("dropout_1", 0.2)),
        "dropout_2": float(out.get("dropout_2", 0.1)),
        "activation": str(out.get("activation", "gelu")).strip().lower(),
        "excluded_features": (),
    }


def build_model_config(
    embedding_dim: int,
    feature_dim: int,
    input_mode: str,
    task_type: str,
    kwargs: dict[str, Any],
) -> RouterMLPv2Config:
    del feature_dim  # embedding-only
    mode = parse_router_input_mode(input_mode)
    if mode != ROUTER_INPUT_MODE_EMBEDDING:
        raise ValueError(
            "mlp-v2 only supports input_mode=embedding; "
            f"got {input_mode!r}. Use mlp-v1 (or another architecture) for "
            "'both' or 'query-features'."
        )
    k = validate_kwargs(kwargs)
    return RouterMLPv2Config(
        embedding_dim=int(embedding_dim),
        input_mode=mode,
        hidden_dim_1=int(k["hidden_dim_1"]),
        hidden_dim_2=int(k["hidden_dim_2"]),
        dropout_1=float(k["dropout_1"]),
        dropout_2=float(k["dropout_2"]),
        activation=str(k["activation"]),
        task_type=parse_router_task_type(task_type or ROUTER_TASK_REGRESSION),
    )


def config_from_json(payload: dict[str, Any]) -> RouterMLPv2Config:
    return RouterMLPv2Config.from_json(payload)


def build_model(config: RouterMLPv2Config) -> RouterMLPv2:
    return RouterMLPv2(config)
