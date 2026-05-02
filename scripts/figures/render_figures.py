#!/usr/bin/env python3
"""Render figures declared under ``figures:`` in a pipeline YAML (opt-in).

Configure a non-interactive matplotlib backend before any submodule imports
``pyplot`` (see tests for the same pattern).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from dotenv import load_dotenv

from surf_rag.config.env import apply_pipeline_env_from_config, load_app_env
from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.merge import merge_figures_render_args
from surf_rag.viz.runner import render_figures_from_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render figure kinds listed in pipeline YAML (figures.enabled)."
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pipeline YAML with paths.router_id and optional figures.plots.",
    )
    p.add_argument(
        "--figures-output-dir",
        default=None,
        help="Override figures.output_dir from YAML (directory for PNG/PDF + meta JSON).",
    )
    p.add_argument(
        "--only-figure",
        action="append",
        default=[],
        metavar="KIND",
        help="Render only these figure kind ids (repeatable). Default: all plots in YAML.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing image/meta files if present.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    load_app_env()
    load_dotenv()
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)

    cfg = load_pipeline_config(args.config.resolve())
    apply_pipeline_env_from_config(cfg)
    merge_figures_render_args(args, cfg)

    if not cfg.figures.enabled:
        log.error(
            "figures.enabled is false in %s; set figures.enabled: true to render.",
            args.config,
        )
        return 2

    only = frozenset(args.only_figure) if args.only_figure else None
    out_dir = getattr(args, "figures_output_dir", None)

    outputs = render_figures_from_config(
        cfg,
        output_dir_override=out_dir,
        only_kinds=only,
        force=bool(args.force),
    )
    for o in outputs:
        log.info("Wrote %s", o.path_image)
        log.info("Wrote %s", o.path_meta)
    if not outputs:
        log.warning(
            "No figures rendered (empty figures.plots or all filtered by --only-figure)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
