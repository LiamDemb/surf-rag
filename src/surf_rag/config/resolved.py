"""Write ``resolved_config.yaml`` artifacts next to run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from surf_rag.config.loader import (
    PipelineConfig,
    ResolvedPaths,
    config_to_resolved_dict,
)


def write_resolved_config_yaml(
    path: Path,
    cfg: PipelineConfig,
    rp: ResolvedPaths,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write merged config + absolute paths as a YAML file."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = config_to_resolved_dict(cfg, rp)
    if extra:
        payload["extra"] = dict(extra)
    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    path.write_text(text, encoding="utf-8")
