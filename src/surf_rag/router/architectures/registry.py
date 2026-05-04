"""Registry for pluggable router architectures."""

from __future__ import annotations

from surf_rag.router.architectures.base import RouterArchitectureDefinition

from . import logreg_v1 as _logreg_v1
from . import mlp_v1 as _mlp_v1
from . import mlp_v2 as _mlp_v2
from . import polyreg_v1 as _polyreg_v1
from . import tower_v01 as _tower_v01

_REGISTRY: dict[str, RouterArchitectureDefinition] = {
    "mlp-v1": RouterArchitectureDefinition(
        name="mlp-v1",
        validate_kwargs=_mlp_v1.validate_kwargs,
        build_model_config=_mlp_v1.build_model_config,
        config_from_json=_mlp_v1.config_from_json,
        build_model=_mlp_v1.build_model,
    ),
    "mlp-v2": RouterArchitectureDefinition(
        name="mlp-v2",
        validate_kwargs=_mlp_v2.validate_kwargs,
        build_model_config=_mlp_v2.build_model_config,
        config_from_json=_mlp_v2.config_from_json,
        build_model=_mlp_v2.build_model,
    ),
    "logreg-v1": RouterArchitectureDefinition(
        name="logreg-v1",
        validate_kwargs=_logreg_v1.validate_kwargs,
        build_model_config=_logreg_v1.build_model_config,
        config_from_json=_logreg_v1.config_from_json,
        build_model=_logreg_v1.build_model,
    ),
    "polyreg-v1": RouterArchitectureDefinition(
        name="polyreg-v1",
        validate_kwargs=_polyreg_v1.validate_kwargs,
        build_model_config=_polyreg_v1.build_model_config,
        config_from_json=_polyreg_v1.config_from_json,
        build_model=_polyreg_v1.build_model,
    ),
    "tower_v01": RouterArchitectureDefinition(
        name="tower_v01",
        validate_kwargs=_tower_v01.validate_kwargs,
        build_model_config=_tower_v01.build_model_config,
        config_from_json=_tower_v01.config_from_json,
        build_model=_tower_v01.build_model,
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
