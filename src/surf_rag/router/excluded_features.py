"""Train-level feature exclusions (V1 router columns dropped before the architecture)."""

from __future__ import annotations

from typing import Any

from surf_rag.router.query_features import V1_FEATURE_NAMES


def normalize_excluded_features(raw: object) -> tuple[str, ...]:
    """Validate names and return a sorted unique tuple (empty if unset)."""
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)):
        raise ValueError(
            "excluded_features must be a list of V1 feature name strings, not a single string"
        )
    if not isinstance(raw, (list, tuple)):
        raise ValueError("excluded_features must be a list of V1 feature name strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("excluded_features entries must be non-empty strings")
        out.append(str(item).strip())
    names = tuple(sorted(set(out)))
    for n in names:
        if n not in V1_FEATURE_NAMES:
            raise ValueError(
                f"excluded_features unknown name {n!r}; allowed: {list(V1_FEATURE_NAMES)}"
            )
    return names


def active_feature_column_indices(
    feature_dim: int, excluded: frozenset[str]
) -> tuple[int, ...]:
    """Column indices kept (first ``feature_dim`` V1 columns minus excluded names)."""
    fd = int(feature_dim)
    if fd > len(V1_FEATURE_NAMES):
        raise ValueError(
            f"feature_dim={feature_dim} exceeds V1 feature catalog ({len(V1_FEATURE_NAMES)})"
        )
    for name in excluded:
        idx = V1_FEATURE_NAMES.index(name)
        if idx >= fd:
            raise ValueError(
                f"cannot exclude {name!r}: index {idx} >= feature_dim={feature_dim}"
            )
    return tuple(i for i in range(fd) if V1_FEATURE_NAMES[i] not in excluded)


def validate_exclusions_for_input_mode(
    input_mode: str,
    feature_dim: int,
    excluded: frozenset[str],
    *,
    query_features_need_one: bool,
) -> None:
    """Raise if exclusions leave no usable feature columns when features are required."""
    from surf_rag.router.model import (
        ROUTER_INPUT_MODE_BOTH,
        ROUTER_INPUT_MODE_QUERY_FEATURES,
        parse_router_input_mode,
    )

    mode = parse_router_input_mode(input_mode)
    if mode == ROUTER_INPUT_MODE_QUERY_FEATURES:
        idx = active_feature_column_indices(feature_dim, excluded)
        if query_features_need_one and not idx:
            raise ValueError(
                "query-features mode requires at least one non-excluded V1 feature column"
            )
    elif mode == ROUTER_INPUT_MODE_BOTH and excluded:
        idx = active_feature_column_indices(feature_dim, excluded)
        if query_features_need_one and not idx:
            raise ValueError(
                "both mode requires at least one non-excluded feature column when using exclusions"
            )


def merged_architecture_kwargs_with_exclusions(
    architecture_kwargs: dict[str, Any] | None,
    train_excluded_features: tuple[str, ...] | None,
) -> dict[str, Any]:
    """Merge ``router.train.excluded_features`` over legacy ``architecture_kwargs.excluded_features``."""
    kw = dict(architecture_kwargs or {})
    arch_ex = kw.pop("excluded_features", None)
    train_ex = tuple(train_excluded_features or ())
    if train_ex:
        kw["excluded_features"] = list(train_ex)
    elif arch_ex is not None:
        kw["excluded_features"] = (
            list(arch_ex) if isinstance(arch_ex, (list, tuple)) else arch_ex
        )
    return kw
