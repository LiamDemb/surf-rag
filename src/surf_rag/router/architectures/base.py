"""Base contracts for pluggable router architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn

ValidateKwargsFn = Callable[[dict[str, Any]], dict[str, Any]]
BuildModelConfigFn = Callable[[int, int, str, str, dict[str, Any]], Any]
ConfigFromJsonFn = Callable[[dict[str, Any]], Any]
BuildModelFn = Callable[[Any], nn.Module]


@dataclass(frozen=True)
class RouterArchitectureDefinition:
    """Registry entry describing one architecture family."""

    name: str
    validate_kwargs: ValidateKwargsFn
    build_model_config: BuildModelConfigFn
    config_from_json: ConfigFromJsonFn
    build_model: BuildModelFn
