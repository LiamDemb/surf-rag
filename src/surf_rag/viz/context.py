"""Resolved paths and IO targets for one figures run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surf_rag.config.loader import resolve_paths
from surf_rag.config.schema import PipelineConfig
from surf_rag.evaluation.router_model_artifacts import (
    RouterModelPaths,
    make_router_model_paths_for_cli,
)
from surf_rag.router.model import parse_router_input_mode
from surf_rag.viz.paths_layout import canonical_router_figure_dir


@dataclass(frozen=True)
class FigureRunContext:
    """Immutable snapshot of router bundle paths and figure output directory."""

    router_id: str
    router_architecture_id: str | None
    input_mode: str
    router_base: Path
    model_paths: RouterModelPaths
    output_dir: Path
    experiment_id: str | None
    image_format: str
    force: bool

    @staticmethod
    def from_pipeline(
        cfg: PipelineConfig,
        *,
        output_dir_override: Path | str | None = None,
        force: bool = False,
    ) -> FigureRunContext:
        """Build context from pipeline config (paths + router.train.input_mode)."""
        rp = resolve_paths(cfg)
        fig = cfg.figures
        rt = cfg.router.train
        input_mode = parse_router_input_mode(str(rt.input_mode or "both").strip())
        rid = str(cfg.paths.router_id).strip()
        if not rid:
            raise ValueError("paths.router_id is required for figure rendering")
        arch = (
            str(cfg.paths.router_architecture_id).strip()
            if cfg.paths.router_architecture_id
            else None
        )
        arch_id = arch if arch else None
        m_paths = make_router_model_paths_for_cli(
            rid,
            router_base=rp.router_base,
            input_mode=input_mode,
            router_architecture_id=arch_id,
        )
        if output_dir_override is not None:
            out = Path(output_dir_override).expanduser().resolve()
        elif fig.output_dir and str(fig.output_dir).strip():
            out = Path(str(fig.output_dir).strip()).expanduser().resolve()
        else:
            out = (
                canonical_router_figure_dir(
                    rp,
                    router_id=rid,
                    router_architecture_id=arch_id,
                    input_mode=input_mode,
                )
                .expanduser()
                .resolve()
            )
        img_fmt = str(fig.image_format or "png").strip().lower()
        if img_fmt not in ("png", "pdf"):
            raise ValueError(
                f"figures.image_format must be png or pdf, got {img_fmt!r}"
            )
        return FigureRunContext(
            router_id=rid,
            router_architecture_id=arch_id,
            input_mode=input_mode,
            router_base=rp.router_base,
            model_paths=m_paths,
            output_dir=out,
            experiment_id=(
                str(cfg.experiment_id).strip() if cfg.experiment_id else None
            ),
            image_format=img_fmt,
            force=force,
        )
