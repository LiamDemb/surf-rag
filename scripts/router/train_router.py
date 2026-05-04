"""Train the router MLP; artifacts under ``.../model/<input_mode>/``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from surf_rag.config.env import load_app_env, apply_pipeline_env_from_config
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.config.merge import merge_router_train_args
from surf_rag.config.resolved import write_resolved_config_yaml

from surf_rag.evaluation.router_dataset_artifacts import (
    make_router_dataset_paths_for_cli,
    read_router_dataset_manifest,
)
from surf_rag.evaluation.router_model_artifacts import (
    make_router_model_paths_for_cli,
    write_json,
    write_router_model_manifest,
    write_predictions_jsonl,
)
from surf_rag.router.embedding_config import parse_embedding_provider
from surf_rag.router.embedding_lock import infer_embedding_provider_from_model
from surf_rag.router.model import (
    ROUTER_TASK_REGRESSION,
    parse_router_input_mode,
    parse_router_task_type,
)
from surf_rag.router.training import (
    RouterTrainConfig,
    export_split_predictions,
    merged_architecture_kwargs,
    save_checkpoint,
    train_router,
    _weight_grid_from_df,
)
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train RouterMLP on router Parquet dataset."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config (router.train section).",
    )
    p.add_argument("--router-id", default=None)
    p.add_argument(
        "--router-architecture-id",
        default=None,
        help="Architecture artifact id under router models/<id>/...",
    )
    p.add_argument("--router-base", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument(
        "--architecture",
        default=None,
        help="Router architecture family (e.g., mlp-v1, mlp-v2, logreg-v1).",
    )
    p.add_argument(
        "--architecture-kwargs",
        default=None,
        help="JSON mapping for architecture-specific kwargs.",
    )
    p.add_argument(
        "--input-mode",
        default=None,
        help="both | query-features | embedding (default: ROUTER_INPUT_MODE or both)",
    )
    p.add_argument(
        "--router-task-type",
        default=None,
        help="regression | classification (default: config/env or regression)",
    )
    p.add_argument(
        "--router-loss",
        dest="loss",
        default=None,
        help="Training loss id (e.g. regret, hinge_squared_regret).",
    )
    p.add_argument(
        "--loss-kwargs",
        dest="loss_kwargs",
        default=None,
        help="JSON object of kwargs for the training loss.",
    )
    p.add_argument(
        "--midpoint-balance-masking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Undersample train rows so five equal-width midpoint buckets are balanced.",
    )
    p.add_argument(
        "--midpoint-balance-epsilon",
        type=float,
        default=None,
        help="Additive score tolerance for plateau ties (default 1e-6).",
    )
    p.add_argument(
        "--excluded-features",
        default=None,
        help="JSON list of V1 router feature names to drop (overrides config when set).",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> int:
    load_app_env()
    load_dotenv()
    import logging

    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    log = logging.getLogger(__name__)
    args._pipeline_cfg = None
    if args.config:
        pcfg = load_pipeline_config(args.config.resolve())
        args._pipeline_cfg = pcfg
        apply_pipeline_env_from_config(pcfg)
        merge_router_train_args(args, pcfg)
    if not args.router_id:
        log.error("Provide --config or --router-id.")
        return 2
    if not args.router_architecture_id:
        log.error(
            "Provide --router-architecture-id (or config paths.router_architecture_id)."
        )
        return 2

    ds_paths = make_router_dataset_paths_for_cli(
        args.router_id, router_base=args.router_base
    )
    if not ds_paths.router_dataset_parquet.is_file():
        log.error("Missing dataset: %s", ds_paths.router_dataset_parquet)
        return 1
    if not ds_paths.manifest.is_file():
        log.error("Missing dataset manifest: %s", ds_paths.manifest)
        return 1

    dmanifest = read_router_dataset_manifest(ds_paths)
    parquet_path = ds_paths.router_dataset_parquet

    raw_mode = (args.input_mode or "").strip() or os.getenv("ROUTER_INPUT_MODE", "both")
    input_mode = parse_router_input_mode(str(raw_mode))

    out_paths = make_router_model_paths_for_cli(
        args.router_id,
        router_base=args.router_base,
        input_mode=input_mode,
        router_architecture_id=args.router_architecture_id,
        router_task_type=parse_router_task_type(
            str(args.router_task_type or ROUTER_TASK_REGRESSION)
        ),
    )
    out_paths.ensure_dirs()

    device = args.device or os.getenv("ROUTER_TRAIN_DEVICE", "cpu")
    architecture = str(args.architecture or os.getenv("ROUTER_ARCHITECTURE", "mlp-v1"))
    architecture_kwargs: dict[str, object] = {}
    raw_ak = getattr(args, "architecture_kwargs", None)
    if isinstance(raw_ak, dict):
        architecture_kwargs = dict(raw_ak)
    elif isinstance(raw_ak, str) and raw_ak.strip():
        try:
            payload = json.loads(raw_ak)
            if not isinstance(payload, dict):
                raise ValueError("architecture kwargs must be a JSON object")
            architecture_kwargs = dict(payload)
        except Exception as exc:
            log.error("Invalid --architecture-kwargs JSON: %s", exc)
            return 2
    elif raw_ak is not None:
        log.error(
            "Invalid architecture_kwargs type %s (expected dict from --config or JSON string)",
            type(raw_ak).__name__,
        )
        return 2

    loss_raw = getattr(args, "loss", None)
    train_loss = str(
        (loss_raw if loss_raw is not None else os.getenv("ROUTER_TRAIN_LOSS", "regret"))
    ).strip()
    loss_kwargs: dict[str, object] = {}
    lk = getattr(args, "loss_kwargs", None)
    if lk is not None:
        if isinstance(lk, dict):
            loss_kwargs = dict(lk)
        else:
            try:
                payload = json.loads(str(lk))
                if not isinstance(payload, dict):
                    raise ValueError("loss kwargs must be a JSON object")
                loss_kwargs = dict(payload)
            except Exception as exc:
                log.error("Invalid --loss-kwargs JSON: %s", exc)
                return 2

    mid_mask_arg = getattr(args, "midpoint_balance_masking", None)
    if mid_mask_arg is None:
        midpoint_balance_masking = os.getenv(
            "ROUTER_MIDPOINT_BALANCE_MASKING", ""
        ).lower() in ("1", "true", "yes")
    else:
        midpoint_balance_masking = bool(mid_mask_arg)
    mid_eps_arg = getattr(args, "midpoint_balance_epsilon", None)
    midpoint_balance_epsilon = float(
        mid_eps_arg
        if mid_eps_arg is not None
        else os.getenv("ROUTER_MIDPOINT_BALANCE_EPSILON", "1e-6")
    )

    excluded_features = tuple(getattr(args, "excluded_features", ()) or ())
    task_type = parse_router_task_type(
        str(args.router_task_type or os.getenv("ROUTER_TASK_TYPE", "regression"))
    )
    cfg = RouterTrainConfig(
        parquet_path=parquet_path,
        router_id=args.router_id,
        output_dir=out_paths.run_root,
        epochs=int(args.epochs or os.getenv("ROUTER_EPOCHS", "100")),
        batch_size=int(args.batch_size or os.getenv("ROUTER_BATCH_SIZE", "32")),
        learning_rate=float(
            args.learning_rate or os.getenv("ROUTER_LEARNING_RATE", "0.001")
        ),
        seed=int(os.getenv("SEED", "42")),
        device=device,
        architecture=architecture,
        architecture_kwargs=architecture_kwargs,
        input_mode=input_mode,
        loss=train_loss,
        loss_kwargs=loss_kwargs,
        midpoint_balance_masking=midpoint_balance_masking,
        midpoint_balance_epsilon=midpoint_balance_epsilon,
        excluded_features=excluded_features,
        task_type=task_type,
    )

    result = train_router(cfg)
    merged_kw = merged_architecture_kwargs(cfg)
    df = pd.read_parquet(parquet_path)
    wg = _weight_grid_from_df(df)
    mcfg = result.model.config

    ds_emb_model = str(dmanifest.get("embedding_model") or "").strip()
    ds_prov_raw = dmanifest.get("embedding_provider")
    if str(ds_prov_raw or "").strip():
        ds_embedding_provider = parse_embedding_provider(str(ds_prov_raw))
    else:
        ds_embedding_provider = infer_embedding_provider_from_model(ds_emb_model)
    emb_src = str(
        (dmanifest.get("embedding_cache") or {}).get("embedding_source") or ""
    ).strip()
    if not emb_src:
        emb_src = "unknown"
    emb_dim_ds = dmanifest.get("embedding_dim")
    if emb_dim_ds is None and "embedding_dim" in df.columns and len(df):
        emb_dim_ds = int(df["embedding_dim"].iloc[0])
    if emb_dim_ds is None:
        emb_dim_ds = int(getattr(mcfg, "embedding_dim", 0) or 0)

    save_checkpoint(
        out_paths.checkpoint,
        result.model,
        mcfg,
        architecture=architecture,
        architecture_kwargs=merged_kw,
    )

    write_json(
        out_paths.metrics,
        {
            "router_id": args.router_id,
            "router_architecture_id": args.router_architecture_id,
            "architecture": architecture,
            "architecture_kwargs": merged_kw,
            "input_mode": input_mode,
            "task_type": task_type,
            "loss": result.loss_requested,
            "loss_kwargs": dict(result.loss_kwargs),
            "loss_effective": result.loss_effective,
            "loss_fallback": result.loss_fallback,
            "best_epoch": result.best_epoch,
            "splits": result.metrics,
            "router_quality_filtering": dict(
                result.metrics.get("router_quality_filtering") or {}
            ),
            "midpoint_balance_masking": result.midpoint_balance_report,
        },
    )
    write_json(
        out_paths.training_history,
        {
            "loss": result.loss_requested,
            "loss_effective": result.loss_effective,
            "loss_fallback": result.loss_fallback,
            "loss_kwargs": dict(result.loss_kwargs),
            "history": result.history,
        },
    )

    write_router_model_manifest(
        out_paths,
        router_id=args.router_id,
        router_architecture_id=args.router_architecture_id,
        input_mode=input_mode,
        task_type=task_type,
        architecture_name=architecture,
        architecture_kwargs=merged_kw,
        dataset_manifest_path=str(ds_paths.manifest.resolve()),
        model_config=mcfg.to_json(),
        training_config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "seed": cfg.seed,
            "device": cfg.device,
            "best_epoch": result.best_epoch,
            "loss": result.loss_requested,
            "loss_kwargs": dict(result.loss_kwargs),
            "loss_effective": result.loss_effective,
            "loss_fallback": result.loss_fallback,
            "midpoint_balance_masking": cfg.midpoint_balance_masking,
            "midpoint_balance_epsilon": cfg.midpoint_balance_epsilon,
        },
        feature_set_version=str(
            df["feature_set_version"].iloc[0]
            if "feature_set_version" in df.columns
            else "1"
        ),
        embedding_model=str(
            df["embedding_model"].iloc[0] if "embedding_model" in df.columns else ""
        ),
        embedding_provider=ds_embedding_provider,
        embedding_dim=int(emb_dim_ds) if emb_dim_ds is not None else None,
        embedding_source=emb_src,
        weight_grid=wg.tolist(),
        source_files={
            "router_dataset": str(parquet_path.resolve()),
        },
        target_spec={
            "name": (
                "oracle_curve"
                if task_type == ROUTER_TASK_REGRESSION
                else "oracle_binary_class_id"
            )
        },
        class_to_weight_map={"graph": 0.0, "dense": 1.0},
    )
    for split in ("train", "dev", "test"):
        rows = export_split_predictions(
            result.model, df, split, cfg.device, wg, task_type=task_type
        )
        if rows:
            write_predictions_jsonl(out_paths.predictions(split), rows)

    if getattr(args, "_pipeline_cfg", None) is not None:
        write_resolved_config_yaml(
            out_paths.run_root / "resolved_config.yaml",
            args._pipeline_cfg,
            resolve_paths(args._pipeline_cfg),
        )
    log.info("Wrote %s", out_paths.checkpoint)
    return 0


if __name__ == "__main__":
    sys.exit(main())
