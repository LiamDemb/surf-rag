"""Learning curves from router ``training_history.json`` artifact."""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from surf_rag.evaluation.router_model_artifacts import read_json
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.specs import BaseFigureSpec, RouterTrainingLearningCurveSpec
from surf_rag.viz.theme import PALETTE
from surf_rag.viz.types import FigureOutput


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        out["surf_rag"] = metadata.version("surf_rag")
    except metadata.PackageNotFoundError:
        out["surf_rag"] = "unknown"
    try:
        import matplotlib as mpl

        out["matplotlib"] = mpl.__version__
    except Exception:
        out["matplotlib"] = "unknown"
    out["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    return out


def _load_training_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Router training history not found: {path}. Run router training first."
        )
    payload = read_json(path)
    rows = payload.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"training_history.json has empty or invalid 'history': {path}"
        )
    df = pd.DataFrame(rows)
    if "epoch" not in df.columns:
        raise ValueError(
            f"training_history.json missing 'epoch' in history rows: {path}"
        )
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.loc[df["epoch"].notna()].copy()
    if len(df) == 0:
        raise ValueError(f"No valid epoch rows found in training history: {path}")
    df["epoch"] = df["epoch"].astype(float)
    for c in ("train_loss", "dev_loss", "train_regret", "dev_regret"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("epoch", kind="mergesort").reset_index(drop=True)


def _task_type_from_manifest(path: Path) -> str:
    if not path.is_file():
        return "unknown"
    try:
        payload = read_json(path)
        return str(payload.get("task_type") or "unknown").strip().lower()
    except Exception:
        return "unknown"


def render_router_training_learning_curve(
    spec: BaseFigureSpec, ctx: FigureRunContext
) -> FigureOutput:
    if not isinstance(spec, RouterTrainingLearningCurveSpec):
        raise TypeError(f"Expected RouterTrainingLearningCurveSpec, got {type(spec)}")

    history_path = ctx.model_paths.training_history
    manifest_path = ctx.model_paths.manifest
    df = _load_training_history(history_path)
    task_type = _task_type_from_manifest(manifest_path)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ext = ctx.image_format
    stem = spec.filename_stem
    path_image = (ctx.output_dir / f"{stem}.{ext}").resolve()
    path_meta = (ctx.output_dir / f"{stem}.meta.json").resolve()
    if not ctx.force and (path_image.exists() or path_meta.exists()):
        raise FileExistsError(
            f"Refusing to overwrite {path_image} / {path_meta} without force=True"
        )

    import matplotlib.pyplot as plt

    x = df["epoch"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(spec.fig_width, spec.fig_height))
    plotted: list[str] = []
    try:
        if spec.show_loss and "train_loss" in df.columns:
            y = df["train_loss"].to_numpy(dtype=np.float64)
            if np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    label="train loss",
                    color=PALETTE["primary"],
                    linewidth=1.8,
                )
                plotted.append("train_loss")
        if spec.show_loss and spec.include_dev and "dev_loss" in df.columns:
            y = df["dev_loss"].to_numpy(dtype=np.float64)
            if np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    label="dev loss",
                    color=PALETTE["primary"],
                    linewidth=1.6,
                    linestyle="--",
                )
                plotted.append("dev_loss")
        reg_label = "error" if task_type == "classification" else "regret"
        if spec.show_regret and "train_regret" in df.columns:
            y = df["train_regret"].to_numpy(dtype=np.float64)
            if np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    label=f"train {reg_label}",
                    color=PALETTE["dark-blue"],
                    linewidth=1.8,
                )
                plotted.append("train_regret")
        if spec.show_regret and spec.include_dev and "dev_regret" in df.columns:
            y = df["dev_regret"].to_numpy(dtype=np.float64)
            if np.isfinite(y).any():
                ax.plot(
                    x,
                    y,
                    label=f"dev {reg_label}",
                    color=PALETTE["dark-blue"],
                    linewidth=1.6,
                    linestyle="--",
                )
                plotted.append("dev_regret")

        if not plotted:
            raise ValueError(
                "No training curves available to plot from training_history.json "
                f"for options include_dev={spec.include_dev}, "
                f"show_loss={spec.show_loss}, show_regret={spec.show_regret}"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.set_title("Router training learning curves", color=PALETTE["text"])
        if spec.show_plot_subtitle:
            sub = (
                f"router_id={ctx.router_id}"
                + (
                    f", arch_id={ctx.router_architecture_id}"
                    if ctx.router_architecture_id
                    else ""
                )
                + f", input_mode={ctx.input_mode}, task_type={task_type}, "
                f"epochs={len(df)}"
            )
            ax.text(0.01, 0.99, sub, transform=ax.transAxes, fontsize=9, va="top")
        ax.legend(loc="best", framealpha=0.92)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(path_image, bbox_inches="tight")
    finally:
        plt.close(fig)

    meta: dict[str, object] = {
        "figure_kind": spec.kind,
        "training_history_path": str(history_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "output_image": str(path_image),
        "router_id": ctx.router_id,
        "router_architecture_id": ctx.router_architecture_id,
        "input_mode": ctx.input_mode,
        "task_type": task_type,
        "n_epochs": int(len(df)),
        "metrics_plotted": plotted,
        "include_dev": spec.include_dev,
        "show_loss": spec.show_loss,
        "show_regret": spec.show_regret,
        "show_plot_subtitle": spec.show_plot_subtitle,
        "versions": _package_versions(),
    }
    path_meta.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return FigureOutput(path_image=path_image, path_meta=path_meta)
