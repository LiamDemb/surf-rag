"""Router training learning curves under ``results/<bundle>/figures/``."""

from __future__ import annotations

from pathlib import Path

from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_REGRESSION,
    make_router_model_paths_for_cli,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.tables.regressor_metrics import SkippedArtifact
from surf_rag.router.model import parse_router_input_mode, parse_router_task_type
from surf_rag.viz.context import FigureRunContext
from surf_rag.viz.learning_curve_labels import resolve_oracle_objective
from surf_rag.viz.renderers.router_training_learning_curve import (
    render_router_training_learning_curve,
)
from surf_rag.viz.specs import RouterTrainingLearningCurveSpec


def _context_for_router_role(
    bundle: ResultsBundle,
    role: str,
    *,
    output_dir: Path,
) -> FigureRunContext:
    arch = get_router_arch(bundle, role)
    if arch is None or not str(arch.architecture_id or "").strip():
        raise SkippedArtifact(f"results.router.{role} not configured")

    rp = bundle.resolved
    rid = str(rp.router_id).strip()
    if not rid:
        raise ValueError("paths.router_id is required for learning-curve figures")

    input_mode = parse_router_input_mode(str(arch.input_mode or "embedding").strip())
    arch_id = str(arch.architecture_id).strip()
    task_type = parse_router_task_type(str(arch.task_type or "regression"))
    train_loss = "regret" if task_type == ROUTER_TASK_REGRESSION else "cross_entropy"

    model_paths = make_router_model_paths_for_cli(
        rid,
        router_base=rp.router_base,
        input_mode=input_mode,
        router_architecture_id=arch_id,
        router_task_type=task_type,
    )
    oracle_metric, oracle_metric_k = resolve_oracle_objective(
        router_id=rid,
        config_metric=str(bundle.results.oracle.metric),
        config_k=int(bundle.results.oracle.k),
        oracle_summary_path=rp.router_oracle_dir / "summary.json",
    )
    ds_parquet = (rp.router_dataset_dir / "router_dataset.parquet").resolve()

    return FigureRunContext(
        router_id=rid,
        router_architecture_id=arch_id,
        input_mode=input_mode,
        router_base=rp.router_base,
        model_paths=model_paths,
        output_dir=output_dir.resolve(),
        experiment_id=(
            str(bundle.cfg.experiment_id).strip() if bundle.cfg.experiment_id else None
        ),
        image_format=bundle.image_format,
        force=True,
        resolved_paths=rp,
        router_dataset_parquet=ds_parquet,
        train_loss=train_loss,
        oracle_metric=oracle_metric,
        oracle_metric_k=oracle_metric_k,
    )


def _learning_curve_spec_from_results(
    spec: ResultsArtifactSpec,
) -> RouterTrainingLearningCurveSpec:
    plot: dict[str, object] = {
        "kind": "router_training_learning_curve",
        "filename_stem": (spec.filename_stem or spec.id).strip(),
    }
    if spec.fig_width is not None:
        plot["fig_width"] = float(spec.fig_width)
    if spec.fig_height is not None:
        plot["fig_height"] = float(spec.fig_height)
    if spec.show_plot_subtitle is not None:
        plot["show_plot_subtitle"] = bool(spec.show_plot_subtitle)
    if spec.include_dev is not None:
        plot["include_dev"] = bool(spec.include_dev)
    if spec.show_loss is not None:
        plot["show_loss"] = bool(spec.show_loss)
    if spec.show_regret is not None:
        plot["show_regret"] = bool(spec.show_regret)
    return RouterTrainingLearningCurveSpec.from_mapping(plot)


def render_router_training_learning_curve_results(
    bundle: ResultsBundle,
    spec: ResultsArtifactSpec,
) -> dict[str, str]:
    role = str(spec.router_role or "").strip()
    if not role:
        raise ValueError(
            f"Artifact {spec.id!r} requires router_role "
            "(e.g. regressor or classifier)"
        )

    lc_spec = _learning_curve_spec_from_results(spec)
    fig_dir = bundle.output_dir / "figures"
    ctx = _context_for_router_role(bundle, role, output_dir=fig_dir)
    out = render_router_training_learning_curve(lc_spec, ctx)
    return {"image": str(out.path_image), "meta": str(out.path_meta)}
