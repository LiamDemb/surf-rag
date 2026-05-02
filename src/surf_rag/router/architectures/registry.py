"""Registry for pluggable router architectures."""

from __future__ import annotations

from surf_rag.router.architectures.base import RouterArchitectureDefinition

from . import logreg_v1 as _logreg_v1
from . import mlp_v1 as _mlp_v1

_REGISTRY: dict[str, RouterArchitectureDefinition] = {
    "mlp-v1": RouterArchitectureDefinition(
        name="mlp-v1",
        validate_kwargs=_mlp_v1.validate_kwargs,
        build_model_config=_mlp_v1.build_model_config,
        config_from_json=_mlp_v1.config_from_json,
        build_model=_mlp_v1.build_model,
    ),
    "logreg-v1": RouterArchitectureDefinition(
        name="logreg-v1",
        validate_kwargs=_logreg_v1.validate_kwargs,
        build_model_config=_logreg_v1.build_model_config,
        config_from_json=_logreg_v1.config_from_json,
        build_model=_logreg_v1.build_model,
    ),
}


def list_architectures() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_architecture(name: str) -> RouterArchitectureDefinition:
    key = (name or "").strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown router architecture {name!r}. Allowed: {list_architectures()}"
        )
    return _REGISTRY[key]
