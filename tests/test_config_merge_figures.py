from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import yaml

from surf_rag.config.loader import config_to_resolved_dict, load_pipeline_config
from surf_rag.config.merge import merge_figures_render_args
from surf_rag.config.schema import (
    FiguresSection,
    FiguresThemeSection,
    PathsSection,
    PipelineConfig,
)


def test_merge_figures_render_args_fills_router_architecture_id_from_yaml() -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(
            PathsSection(),
            router_id="main-v01",
            router_architecture_id="tower-v01",
            router_base="data/router",
        ),
    )
    args = Namespace()
    merge_figures_render_args(args, cfg)
    assert args.router_id == "main-v01"
    assert args.router_architecture_id == "tower-v01"


def test_merge_figures_render_args_sets_output_dir_from_yaml(tmp_path: Path) -> None:
    cfg = replace(
        PipelineConfig(),
        paths=replace(PathsSection(), router_id="r"),
        figures=replace(FiguresSection(), output_dir=str(tmp_path / "out")),
    )
    args = Namespace()
    merge_figures_render_args(args, cfg)
    assert args.figures_output_dir == str(tmp_path / "out")


def test_load_pipeline_config_nested_figures_theme_is_dataclass(tmp_path: Path) -> None:
    """Regression: ``from __future__ import annotations`` stringifies field types; merge
    must still produce ``FiguresThemeSection`` for ``figures.theme``, not a plain dict.
    """
    yml = tmp_path / "nested_theme.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "paths": {"router_id": "x"},
                "figures": {
                    "enabled": True,
                    "theme": {
                        "name": "default",
                        "dpi": 175,
                        "overrides": {"axes.labelsize": 12},
                    },
                    "plots": [],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(yml)
    assert isinstance(cfg.figures.theme, FiguresThemeSection)
    assert cfg.figures.theme.dpi == 175
    assert cfg.figures.theme.overrides.get("axes.labelsize") == 12


def test_config_to_resolved_dict_includes_figures(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "paths": {"router_id": "x"},
                "figures": {"enabled": True, "image_format": "pdf", "plots": []},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(yml)
    from surf_rag.config.loader import resolve_paths

    d = config_to_resolved_dict(cfg, resolve_paths(cfg))
    assert "figures" in d
    assert d["figures"]["enabled"] is True
    assert d["figures"]["image_format"] == "pdf"
