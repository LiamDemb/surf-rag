"""Regressor quality table with regret baselines."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_REGRESSION,
    make_router_model_paths_for_cli,
)
from surf_rag.results.bundle import ResultsBundle, get_router_arch
from surf_rag.results.loaders import iter_predictions_jsonl
from surf_rag.results.tables.writer import write_table
from surf_rag.results.weight_grid import load_router_weight_grid, weight_grid_for_curve


class SkippedArtifact(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _interpolate_curve(curve: np.ndarray, w_hat: float) -> float:
    if len(curve) < 2:
        return float(curve[0]) if len(curve) == 1 else 0.0
    w = float(np.clip(w_hat, 0.0, 1.0))
    scaled = w * float(len(curve) - 1)
    lo = int(np.floor(scaled))
    lo = int(np.clip(lo, 0, len(curve) - 2))
    hi = lo + 1
    alpha = scaled - float(lo)
    return float((1.0 - alpha) * curve[lo] + alpha * curve[hi])


def _curve_regret(curve: list[float], grid: list[float], w: float) -> float:
    if not curve:
        return 0.0
    c = np.asarray(curve, dtype=np.float32)
    c_star = float(np.max(c))
    c_hat = _interpolate_curve(c, float(w))
    return float(c_star - c_hat)


def build_regressor_metrics(bundle: ResultsBundle) -> tuple[pd.DataFrame, dict]:
    arch = get_router_arch(bundle, "regressor")
    if arch is None or not arch.architecture_id.strip():
        raise SkippedArtifact("results.router.regressor not configured")

    paths = make_router_model_paths_for_cli(
        bundle.resolved.router_id,
        router_base=bundle.resolved.router_base,
        input_mode=arch.input_mode,
        router_architecture_id=arch.architecture_id,
        router_task_type=ROUTER_TASK_REGRESSION,
    )
    if not paths.checkpoint.is_file():
        raise SkippedArtifact(f"Regressor checkpoint missing: {paths.checkpoint}")

    split = bundle.results.split
    pred_path = paths.predictions(split)
    fallback_grid = load_router_weight_grid(
        bundle.resolved.router_dataset_dir / "router_dataset.parquet",
        model_paths=paths,
    )
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in iter_predictions_jsonl(pred_path):
        qid = str(row.get("question_id", "")).strip()
        if qid not in bundle.split_qids:
            continue
        src = bundle.qid_to_source.get(qid, "unknown")
        curve = list(row.get("oracle_curve") or [])
        if not curve:
            continue
        grid = weight_grid_for_curve(
            curve, list(row.get("weight_grid") or []), fallback_grid=fallback_grid
        )
        pred_w = float(row.get("predicted_weight", 0.5))
        wg = np.asarray(grid, dtype=float)
        idx_05 = int(np.argmin(np.abs(wg - 0.5)))
        buckets[(split, "all")].append(
            {
                "regret": _curve_regret(curve, grid, pred_w),
                "pred": pred_w,
                "baseline_05": _curve_regret(curve, grid, float(grid[idx_05])),
            }
        )
        if src in ("nq", "2wiki"):
            buckets[(split, src)].append(
                {
                    "regret": _curve_regret(curve, grid, pred_w),
                    "pred": pred_w,
                    "baseline_05": _curve_regret(curve, grid, float(grid[idx_05])),
                }
            )

    rows: list[dict] = []
    for (sp, src), items in sorted(buckets.items()):
        if not items:
            continue
        preds = [x["pred"] for x in items]
        rows.append(
            {
                "split": sp,
                "dataset_source": src,
                "mean_regret": float(np.mean([x["regret"] for x in items])),
                "mean_predicted_weight": float(np.mean(preds)),
                "std_predicted_weight": float(np.std(preds)),
                "baseline_05_regret": float(np.mean([x["baseline_05"] for x in items])),
                "oracle_regret": 0.0,
                "n": len(items),
            }
        )
    if not rows:
        raise SkippedArtifact(f"No regressor predictions on split {split!r}")
    df = pd.DataFrame(rows)
    paths_out = write_table(
        bundle,
        "regressor_metrics",
        df,
        {"architecture_id": arch.architecture_id},
    )
    return df, {"csv": str(paths_out[0]), "meta": str(paths_out[1])}
