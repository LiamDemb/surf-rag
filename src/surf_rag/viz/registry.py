"""Register and dispatch figure renderers by spec kind."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from surf_rag.viz.specs import BaseFigureSpec
from surf_rag.viz.types import FigureOutput

if TYPE_CHECKING:
    from surf_rag.viz.context import FigureRunContext

_RenderFn = Callable[[BaseFigureSpec, "FigureRunContext"], FigureOutput]

_REGISTRY: dict[str, _RenderFn] = {}


def register(kind: str, fn: _RenderFn) -> None:
    """Register a renderer for ``kind`` (replaces if duplicate)."""
    _REGISTRY[kind] = fn


def list_figure_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())


def render(spec: BaseFigureSpec, ctx: "FigureRunContext") -> FigureOutput:
    fn = _REGISTRY.get(spec.kind)
    if fn is None:
        allowed = ", ".join(list_figure_kinds()) or "(none)"
        raise ValueError(
            f"No renderer registered for kind {spec.kind!r}. Registered: {allowed}"
        )
    return fn(spec, ctx)
