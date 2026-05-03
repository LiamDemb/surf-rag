"""Shared types for figure outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FigureOutput:
    """Paths written by one figure render."""

    path_image: Path
    path_meta: Path
