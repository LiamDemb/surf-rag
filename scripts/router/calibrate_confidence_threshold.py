"""Sweep hybrid confidence thresholds on router dev split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from surf_rag.evaluation.artifact_paths import default_router_base
from surf_rag.router.inference_inputs import load_router_inference_context
from surf_rag.router.inference import (
    predict_batch,
    predict_class_id_batch,
    predict_class_probs_batch,
)


def _parse_thresholds(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    return sorted({min(1.0, max(0.0, v)) for v in vals})


def _oracle_value_for_weight(curve: list[float], grid: list[float], w: float) -> float:
    if not curve or not grid:
        return 0.0
    idx = int(np.argmin(np.abs(np.asarray(grid, dtype=np.float32) - float(w))))
    return float(curve[idx])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router-id", required=True)
    p.add_argument("--classifier-architecture-id", default=None)
    p.add_argument("--regressor-router-id", required=True)
    p.add_argument("--regressor-architecture-id", default=None)
    p.add_argument("--router-base", type=Path, default=None)
    p.add_argument("--input-mode", default="both")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="dev")
    p.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    rb = args.router_base or default_router_base()
    ds_path = rb / str(args.router_id) / "dataset" / "router_dataset.parquet"
    if not ds_path.is_file():
        raise FileNotFoundError(f"Missing router dataset parquet: {ds_path}")
    df = pd.read_parquet(ds_path)
    df = df[df["split"].astype(str) == str(args.split)]
    if df.empty:
        raise ValueError(f"No rows for split={args.split!r} in {ds_path}")

    clf = load_router_inference_context(
        args.router_id,
        router_architecture_id=args.classifier_architecture_id,
        input_mode=args.input_mode,
        router_base=rb,
        device=args.device,
        router_task_type="classification",
    )
    reg = load_router_inference_context(
        args.regressor_router_id,
        router_architecture_id=args.regressor_architecture_id,
        input_mode=args.input_mode,
        router_base=rb,
        device=args.device,
        router_task_type="regression",
    )

    qe = np.asarray(df["query_embedding"].tolist(), dtype=np.float32)
    qf = np.asarray(df["feature_vector_norm"].tolist(), dtype=np.float32)
    probs = predict_class_probs_batch(clf.router, qe, qf)
    cls = predict_class_id_batch(clf.router, qe, qf)
    conf = np.max(probs, axis=-1)
    reg_w = predict_batch(reg.router, qe, qf)

    thresholds = _parse_thresholds(args.thresholds)
    rows = []
    for th in thresholds:
        vals = []
        fallback_count = 0
        for i, row in df.reset_index(drop=True).iterrows():
            if float(conf[i]) >= th:
                w = 1.0 if int(cls[i]) == 1 else 0.0
            else:
                fallback_count += 1
                w = float(reg_w[i])
            curve = [float(x) for x in row.get("oracle_curve", [])]
            grid = [float(x) for x in row.get("weight_grid", [])]
            vals.append(_oracle_value_for_weight(curve, grid, w))
        rows.append(
            {
                "threshold": float(th),
                "mean_oracle_objective": float(np.mean(vals) if vals else 0.0),
                "fallback_rate": float(fallback_count / len(df)),
            }
        )
    rows = sorted(rows, key=lambda r: r["mean_oracle_objective"], reverse=True)
    payload = {"split": args.split, "n_rows": int(len(df)), "results": rows}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        print(str(args.output_json))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
