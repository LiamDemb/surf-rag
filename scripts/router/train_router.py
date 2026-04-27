"""Train the router MLP; artifacts under ``.../model/<input_mode>/``."""

from __future__ import annotations

import argparse
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
from surf_rag.router.model import parse_router_input_mode
from surf_rag.router.training import (
    RouterTrainConfig,
    export_split_predictions,
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
    p.add_argument("--router-base", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument(
        "--input-mode",
        default=None,
        help="both | query-features | embedding (default: ROUTER_INPUT_MODE or both)",
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

    ds_paths = make_router_dataset_paths_for_cli(
        args.router_id, router_base=args.router_base
    )
    if not ds_paths.router_dataset_parquet.is_file():
        log.error("Missing dataset: %s", ds_paths.router_dataset_parquet)
        return 1
    if not ds_paths.manifest.is_file():
        log.error("Missing dataset manifest: %s", ds_paths.manifest)
        return 1

    read_router_dataset_manifest(ds_paths)
    parquet_path = ds_paths.router_dataset_parquet

    raw_mode = (args.input_mode or "").strip() or os.getenv("ROUTER_INPUT_MODE", "both")
    input_mode = parse_router_input_mode(str(raw_mode))

    out_paths = make_router_model_paths_for_cli(
        args.router_id, router_base=args.router_base, input_mode=input_mode
    )
    out_paths.ensure_dirs()

    device = args.device or os.getenv("ROUTER_TRAIN_DEVICE", "cpu")
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
        input_mode=input_mode,
    )

    result = train_router(cfg)
    df = pd.read_parquet(parquet_path)
    wg = _weight_grid_from_df(df)
    mcfg = result.model.config

    save_checkpoint(out_paths.checkpoint, result.model, mcfg)

    write_json(
        out_paths.metrics,
        {
            "router_id": args.router_id,
            "input_mode": input_mode,
            "best_epoch": result.best_epoch,
            "splits": result.metrics,
        },
    )
    write_json(out_paths.training_history, {"history": result.history})

    write_router_model_manifest(
        out_paths,
        router_id=args.router_id,
        input_mode=input_mode,
        dataset_manifest_path=str(ds_paths.manifest.resolve()),
        model_config=mcfg.to_json(),
        training_config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "seed": cfg.seed,
            "device": cfg.device,
            "best_epoch": result.best_epoch,
        },
        feature_set_version=str(
            df["feature_set_version"].iloc[0]
            if "feature_set_version" in df.columns
            else "1"
        ),
        embedding_model=str(
            df["embedding_model"].iloc[0] if "embedding_model" in df.columns else ""
        ),
        weight_grid=wg.tolist(),
        source_files={
            "router_dataset": str(parquet_path.resolve()),
        },
    )
    for split in ("train", "dev", "test"):
        rows = export_split_predictions(result.model, df, split, cfg.device, wg)
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
