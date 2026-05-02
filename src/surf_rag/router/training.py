"""Train scalar router models from ``router_dataset.parquet``."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.architectures.registry import get_architecture
from surf_rag.router.model import parse_router_input_mode
from surf_rag.router.regret import regret_loss
from surf_rag.router.router_metrics import aggregate_router_metrics


@dataclass
class RouterTrainConfig:
    """Training hyperparameters and paths."""

    parquet_path: Path
    router_id: str
    output_dir: Path
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-5
    seed: int = 42
    device: str = "cpu"
    architecture: str = "mlp-v1"
    architecture_kwargs: dict[str, Any] | None = None
    num_workers: int = 0
    input_mode: str = "both"
    balance_training_sources: bool = True


def _seq_floats(val: object) -> List[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    return [float(x) for x in list(val)]


def _rows_to_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Stack embedding, feature_norm, oracle_curve, validity, question_ids, source."""
    emb_rows: List[List[float]] = []
    feat_rows: List[List[float]] = []
    curve_rows: List[List[float]] = []
    valid_rows: List[float] = []
    qids: List[str] = []
    sources: List[str] = []
    for _, row in df.iterrows():
        qids.append(str(row.get("question_id", "")))
        emb_rows.append(_seq_floats(row.get("query_embedding")))
        feat_rows.append(_seq_floats(row.get("feature_vector_norm")))
        curve_rows.append(_seq_floats(row.get("oracle_curve")))
        valid_rows.append(
            1.0 if bool(row.get("is_valid_for_router_training", False)) else 0.0
        )
        sources.append(str(row.get("dataset_source", "")))
    if not emb_rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
            [],
        )
    x_e = np.asarray(emb_rows, dtype=np.float32)
    x_f = np.asarray(feat_rows, dtype=np.float32)
    y = np.asarray(curve_rows, dtype=np.float32)
    valid = np.asarray(valid_rows, dtype=np.float32)
    return x_e, x_f, y, valid, qids, sources


def _split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"].astype(str).str.lower() == split.lower()].copy()


def _eligible_mask(df: pd.DataFrame) -> pd.Series:
    if "is_valid_for_router_training" not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    return df["is_valid_for_router_training"].fillna(False).astype(bool)


def _filter_router_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[_eligible_mask(df)].copy()


def _split_router_row_counts(df: pd.DataFrame, split: str) -> Dict[str, float]:
    sdf = _split_frame(df, split)
    total = int(len(sdf))
    eligible = int(_eligible_mask(sdf).sum()) if total > 0 else 0
    ignored = total - eligible
    return {
        "num_rows_total": float(total),
        "num_rows_router_eligible": float(eligible),
        "num_rows_router_ignored_all_zero": float(ignored),
    }


def build_model_config_from_df(df: pd.DataFrame, cfg: RouterTrainConfig) -> Any:
    sample = df.iloc[0]
    emb_dim = int(sample.get("embedding_dim", 0)) or len(
        _seq_floats(sample.get("query_embedding"))
    )
    feat_dim = len(_seq_floats(sample.get("feature_vector_norm")))
    arch = get_architecture(cfg.architecture)
    return arch.build_model_config(
        emb_dim,
        feat_dim,
        parse_router_input_mode(cfg.input_mode),
        dict(cfg.architecture_kwargs or {}),
    )


def _weight_grid_from_df(df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        return np.asarray(DEFAULT_DENSE_WEIGHT_GRID, dtype=np.float32)
    wg = df.iloc[0].get("weight_grid")
    if wg is None or (isinstance(wg, float) and pd.isna(wg)):
        return np.asarray(DEFAULT_DENSE_WEIGHT_GRID, dtype=np.float32)
    return np.asarray([float(x) for x in wg], dtype=np.float32)


@dataclass
class TrainRunResult:
    model: nn.Module
    history: List[Dict[str, float]]
    best_epoch: int
    metrics: Dict[str, Any]


def train_router(cfg: RouterTrainConfig) -> TrainRunResult:
    """Fit a router architecture; return model, history, and best epoch index."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    df = pd.read_parquet(cfg.parquet_path)
    train_df_total = _split_frame(df, "train")
    dev_df_total = _split_frame(df, "dev")
    train_df = _filter_router_eligible(train_df_total)
    dev_df = _filter_router_eligible(dev_df_total)
    if len(train_df) == 0:
        raise ValueError(
            "No router-eligible rows in train split. "
            "All rows are marked is_valid_for_router_training=false."
        )

    mcfg = build_model_config_from_df(train_df, cfg)
    arch = get_architecture(cfg.architecture)
    model = arch.build_model(mcfg).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    x_e, x_f, y, valid, _, sources = _rows_to_arrays(train_df)
    wg_np = _weight_grid_from_df(train_df)

    x_e_t = torch.tensor(x_e, device=cfg.device, dtype=torch.float32)
    x_f_t = torch.tensor(x_f, device=cfg.device, dtype=torch.float32)
    y_t = torch.tensor(y, device=cfg.device, dtype=torch.float32)
    valid_t = torch.tensor(valid, device=cfg.device, dtype=torch.float32)

    ds = TensorDataset(x_e_t, x_f_t, y_t, valid_t)
    sampler = None
    if cfg.balance_training_sources and len(sources) == len(ds):
        counts: Dict[str, int] = {}
        for s in sources:
            counts[s] = counts.get(s, 0) + 1
        sw = [1.0 / max(1, counts.get(s, 1)) for s in sources]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sw, dtype=torch.float64),
            num_samples=len(sw),
            replacement=True,
        )
    loader = DataLoader(
        ds,
        batch_size=min(cfg.batch_size, len(ds)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
    )

    x_ed, x_fd, yd, valid_d, _, _ = _rows_to_arrays(dev_df)
    has_dev = len(dev_df) > 0 and x_ed.shape[0] > 0
    if has_dev:
        x_ed = torch.tensor(x_ed, device=cfg.device, dtype=torch.float32)
        x_fd = torch.tensor(x_fd, device=cfg.device, dtype=torch.float32)
        yd = torch.tensor(yd, device=cfg.device, dtype=torch.float32)
        valid_d = torch.tensor(valid_d, device=cfg.device, dtype=torch.float32)

    history: List[Dict[str, float]] = []
    best_dev = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses: List[float] = []
        for xeb, xfb, yb, validb in loader:
            opt.zero_grad()
            w_hat = model.predict_weight(xeb, xfb)
            if bool((validb > 0.0).any().item()):
                mask = validb > 0.0
                loss = regret_loss(w_hat[mask], yb[mask])
            else:
                loss = regret_loss(w_hat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_losses.append(float(loss.item()))

        train_regret = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        dev_regret: Optional[float] = None
        if has_dev:
            model.eval()
            with torch.no_grad():
                w_hat_d = model.predict_weight(x_ed, x_fd)
                if bool((valid_d > 0.0).any().item()):
                    maskd = valid_d > 0.0
                    dev_regret = float(regret_loss(w_hat_d[maskd], yd[maskd]).item())
                else:
                    dev_regret = float(regret_loss(w_hat_d, yd).item())

        row: Dict[str, float] = {
            "epoch": float(epoch + 1),
            "train_regret": train_regret,
        }
        if dev_regret is not None:
            row["dev_regret"] = dev_regret
        history.append(row)

        if has_dev and dev_regret is not None:
            if dev_regret < best_dev - cfg.early_stopping_min_delta:
                best_dev = dev_regret
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.early_stopping_patience:
                    break
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})

    metrics = _eval_splits(model, df, wg_np, cfg.device, mcfg, cfg.architecture)
    return TrainRunResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        metrics=metrics,
    )


def _eval_splits(
    model: nn.Module,
    df: pd.DataFrame,
    wg_np: np.ndarray,
    device: str,
    mcfg: Any,
    architecture: str,
) -> Dict[str, Any]:
    model.eval()
    out: Dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        counts = _split_router_row_counts(df, split)
        sdf_total = _split_frame(df, split)
        if len(sdf_total) == 0:
            out[split] = {
                "num_rows": 0.0,
                **counts,
            }
            continue
        sdf = _filter_router_eligible(sdf_total)
        if len(sdf) == 0:
            out[split] = {
                "num_rows": 0.0,
                **counts,
            }
            continue
        x_e, x_f, y, valid, _, _ = _rows_to_arrays(sdf)
        x_e = torch.tensor(x_e, device=device, dtype=torch.float32)
        x_f = torch.tensor(x_f, device=device, dtype=torch.float32)
        valid_np = np.asarray(valid, dtype=np.float32)
        with torch.no_grad():
            pred_w = model.predict_weight(x_e, x_f).cpu().numpy()
        m = aggregate_router_metrics(
            oracle_curves=y,
            predicted_weights=pred_w,
            valid_mask=valid_np > 0.0,
            weight_grid=wg_np,
        )
        m["num_rows"] = float(len(sdf))
        m.update(counts)
        out[split] = m
    out["router_quality_filtering"] = {
        "reason": "all_zero_oracle_score",
        "num_rows_total": float(len(df)),
        "num_rows_router_eligible": float(_eligible_mask(df).sum()),
        "num_rows_router_ignored_all_zero": float(
            len(df) - int(_eligible_mask(df).sum())
        ),
    }
    out["config_summary"] = {
        "architecture": architecture,
        "model_config": mcfg.to_json(),
    }
    return out


def export_split_predictions(
    model: nn.Module,
    df: pd.DataFrame,
    split: str,
    device: str,
    weight_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    """One JSON object per row for ``predictions_<split>.jsonl``."""
    sdf = _split_frame(df, split)
    if len(sdf) == 0:
        return []
    x_e, x_f, y, valid, qids, _ = _rows_to_arrays(sdf)
    if x_e.shape[0] == 0:
        return []
    x_e = torch.tensor(x_e, device=device, dtype=torch.float32)
    x_f = torch.tensor(x_f, device=device, dtype=torch.float32)
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        pred_w = model.predict_weight(x_e, x_f).cpu().numpy()
    for i in range(len(qids)):
        curve = y[i]
        best_idx = int(np.argmax(curve)) if len(curve) else 0
        best_weight = float(weight_grid[best_idx]) if len(weight_grid) else 0.0
        best_score = float(np.max(curve)) if len(curve) else 0.0
        rows.append(
            {
                "question_id": qids[i],
                "oracle_curve": curve.tolist(),
                "target_oracle_best_weight": best_weight,
                "target_oracle_best_score": best_score,
                "predicted_weight": float(pred_w[i]),
                "is_valid_for_router_training": bool(valid[i] > 0.0),
            }
        )
    return rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    mcfg: Any,
    *,
    architecture: str,
    architecture_kwargs: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "architecture_kwargs": dict(architecture_kwargs or {}),
            "config": mcfg.to_json(),
        },
        path,
    )
