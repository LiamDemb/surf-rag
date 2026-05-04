"""Train scalar router models from ``router_dataset.parquet``."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.architectures.registry import get_architecture
from surf_rag.router.excluded_features import merged_architecture_kwargs_with_exclusions
from surf_rag.router.model import (
    ROUTER_TASK_REGRESSION,
    parse_router_input_mode,
    parse_router_task_type,
)
from surf_rag.router.losses import (
    resolve_router_classification_loss,
    resolve_router_training_loss,
)
from surf_rag.router.midpoint_balance import build_train_midpoint_balance_indices
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
    midpoint_balance_masking: bool = False
    midpoint_balance_epsilon: float = 1e-6
    loss: str = "regret"
    loss_kwargs: dict[str, Any] | None = None
    excluded_features: tuple[str, ...] = ()
    task_type: str = ROUTER_TASK_REGRESSION


def merged_architecture_kwargs(cfg: RouterTrainConfig) -> Dict[str, Any]:
    """Merge ``router.train.excluded_features`` into kwargs passed to architectures."""
    return merged_architecture_kwargs_with_exclusions(
        cfg.architecture_kwargs,
        cfg.excluded_features,
    )


def _seq_floats(val: object) -> List[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    return [float(x) for x in list(val)]


def _rows_to_arrays(
    df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    List[str],
]:
    """Stack embedding, feature_norm, oracle_curve, validity, question_ids, source."""
    emb_rows: List[List[float]] = []
    feat_rows: List[List[float]] = []
    curve_rows: List[List[float]] = []
    valid_rows: List[float] = []
    class_rows: List[int] = []
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
        class_rows.append(int(row.get("oracle_binary_class_id", 0)))
        sources.append(str(row.get("dataset_source", "")))
    if not emb_rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
            [],
        )
    x_e = np.asarray(emb_rows, dtype=np.float32)
    x_f = np.asarray(feat_rows, dtype=np.float32)
    y = np.asarray(curve_rows, dtype=np.float32)
    valid = np.asarray(valid_rows, dtype=np.float32)
    y_cls = np.asarray(class_rows, dtype=np.int64)
    return x_e, x_f, y, y_cls, valid, qids, sources


def _split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"].astype(str).str.lower() == split.lower()].copy()


def _eligible_mask(df: pd.DataFrame) -> pd.Series:
    if "is_valid_for_router_training" not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    return df["is_valid_for_router_training"].fillna(False).astype(bool)


def _eligible_mask_for_task(df: pd.DataFrame, task_type: str) -> pd.Series:
    task = parse_router_task_type(task_type)
    if task == ROUTER_TASK_REGRESSION:
        return _eligible_mask(df)
    col = "is_valid_for_router_training_classification"
    if col not in df.columns:
        return _eligible_mask(df)
    return df[col].fillna(False).astype(bool)


def _filter_router_eligible(df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    return df[_eligible_mask_for_task(df, task_type)].copy()


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
        parse_router_task_type(cfg.task_type),
        merged_architecture_kwargs(cfg),
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
    history: List[Dict[str, Any]]
    best_epoch: int
    metrics: Dict[str, Any]
    loss_requested: str = "regret"
    loss_effective: str = "regret"
    loss_fallback: bool = False
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)
    midpoint_balance_report: Dict[str, Any] | None = None


def train_router(cfg: RouterTrainConfig) -> TrainRunResult:
    """Fit a router architecture; return model, history, and best epoch index."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    task_type = parse_router_task_type(cfg.task_type)
    df = pd.read_parquet(cfg.parquet_path)
    train_df_total = _split_frame(df, "train")
    dev_df_total = _split_frame(df, "dev")
    train_df = _filter_router_eligible(train_df_total, task_type)
    dev_df = _filter_router_eligible(dev_df_total, task_type)
    if len(train_df) == 0:
        raise ValueError(
            "No router-eligible rows in train split. "
            "All rows are marked is_valid_for_router_training=false."
        )

    train_df = train_df.reset_index(drop=True)
    midpoint_balance_report: Optional[Dict[str, Any]] = None
    if cfg.midpoint_balance_masking:
        keep_idx, midpoint_balance_report = build_train_midpoint_balance_indices(
            train_df,
            epsilon=float(cfg.midpoint_balance_epsilon),
            seed=int(cfg.seed),
        )
        train_df = train_df.iloc[keep_idx].reset_index(drop=True)

    loss_kw = dict(cfg.loss_kwargs or {})
    if task_type == ROUTER_TASK_REGRESSION:
        loss_fn, loss_effective, loss_fallback = resolve_router_training_loss(
            cfg.loss, loss_kw
        )
    else:
        loss_fn, loss_effective, loss_fallback = resolve_router_classification_loss(
            cfg.loss, loss_kw
        )
    loss_requested = str(cfg.loss).strip() or "regret"

    mcfg = build_model_config_from_df(train_df, cfg)
    arch = get_architecture(cfg.architecture)
    model = arch.build_model(mcfg).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    x_e, x_f, y, y_cls, valid, _, sources = _rows_to_arrays(train_df)
    wg_np = _weight_grid_from_df(train_df)

    x_e_t = torch.tensor(x_e, device=cfg.device, dtype=torch.float32)
    x_f_t = torch.tensor(x_f, device=cfg.device, dtype=torch.float32)
    y_t = torch.tensor(y, device=cfg.device, dtype=torch.float32)
    y_cls_t = torch.tensor(y_cls, device=cfg.device, dtype=torch.long)
    valid_t = torch.tensor(valid, device=cfg.device, dtype=torch.float32)

    ds = TensorDataset(x_e_t, x_f_t, y_t, y_cls_t, valid_t)
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

    x_ed, x_fd, yd, yd_cls, valid_d, _, _ = _rows_to_arrays(dev_df)
    has_dev = len(dev_df) > 0 and x_ed.shape[0] > 0
    if has_dev:
        x_ed = torch.tensor(x_ed, device=cfg.device, dtype=torch.float32)
        x_fd = torch.tensor(x_fd, device=cfg.device, dtype=torch.float32)
        yd = torch.tensor(yd, device=cfg.device, dtype=torch.float32)
        yd_cls = torch.tensor(yd_cls, device=cfg.device, dtype=torch.long)
        valid_d = torch.tensor(valid_d, device=cfg.device, dtype=torch.float32)

    history: List[Dict[str, float]] = []
    best_dev = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_train_losses: List[float] = []
        epoch_train_regrets: List[float] = []
        for xeb, xfb, yb, yb_cls, validb in loader:
            opt.zero_grad()
            if task_type == ROUTER_TASK_REGRESSION:
                w_hat = model.predict_weight(xeb, xfb)
                if bool((validb > 0.0).any().item()):
                    mask = validb > 0.0
                    w_m = w_hat[mask]
                    y_m = yb[mask]
                else:
                    w_m = w_hat
                    y_m = yb
                loss = loss_fn(w_m, y_m)
                reg_log = regret_loss(w_m, y_m)
            else:
                logits = model.predict_class_logits(xeb, xfb)
                if bool((validb > 0.0).any().item()):
                    mask = validb > 0.0
                    logits = logits[mask]
                    yb_cls = yb_cls[mask]
                loss = loss_fn(logits, yb_cls)
                pred_cls = torch.argmax(logits, dim=-1)
                reg_log = (pred_cls != yb_cls).float().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_train_losses.append(float(loss.item()))
            epoch_train_regrets.append(float(reg_log.item()))

        train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else 0.0
        train_regret_metric = (
            float(np.mean(epoch_train_regrets)) if epoch_train_regrets else 0.0
        )
        dev_loss_v: Optional[float] = None
        dev_regret_v: Optional[float] = None
        if has_dev:
            model.eval()
            with torch.no_grad():
                if task_type == ROUTER_TASK_REGRESSION:
                    w_hat_d = model.predict_weight(x_ed, x_fd)
                    if bool((valid_d > 0.0).any().item()):
                        maskd = valid_d > 0.0
                        wd = w_hat_d[maskd]
                        ydv = yd[maskd]
                    else:
                        wd = w_hat_d
                        ydv = yd
                    dev_loss_v = float(loss_fn(wd, ydv).item())
                    dev_regret_v = float(regret_loss(wd, ydv).item())
                else:
                    logits_d = model.predict_class_logits(x_ed, x_fd)
                    if bool((valid_d > 0.0).any().item()):
                        maskd = valid_d > 0.0
                        logits_d = logits_d[maskd]
                        yd_cls_eval = yd_cls[maskd]
                    else:
                        yd_cls_eval = yd_cls
                    dev_loss_v = float(loss_fn(logits_d, yd_cls_eval).item())
                    pred_cls = torch.argmax(logits_d, dim=-1)
                    dev_regret_v = float(
                        (pred_cls != yd_cls_eval).float().mean().item()
                    )

        row: Dict[str, Any] = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "train_regret": train_regret_metric,
            "loss_name": loss_requested,
            "loss_effective": loss_effective,
        }
        if dev_loss_v is not None:
            row["dev_loss"] = dev_loss_v
            row["dev_regret"] = dev_regret_v
        history.append(row)

        if has_dev and dev_loss_v is not None:
            if dev_loss_v < best_dev - cfg.early_stopping_min_delta:
                best_dev = dev_loss_v
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

    metrics = _eval_splits(
        model, df, wg_np, cfg.device, mcfg, cfg.architecture, task_type=task_type
    )
    return TrainRunResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        metrics=metrics,
        loss_requested=loss_requested,
        loss_effective=loss_effective,
        loss_fallback=loss_fallback,
        loss_kwargs=loss_kw,
        midpoint_balance_report=midpoint_balance_report,
    )


def _eval_splits(
    model: nn.Module,
    df: pd.DataFrame,
    wg_np: np.ndarray,
    device: str,
    mcfg: Any,
    architecture: str,
    task_type: str = ROUTER_TASK_REGRESSION,
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
        sdf = _filter_router_eligible(sdf_total, task_type)
        if len(sdf) == 0:
            out[split] = {
                "num_rows": 0.0,
                **counts,
            }
            continue
        x_e, x_f, y, y_cls, valid, _, _ = _rows_to_arrays(sdf)
        x_e = torch.tensor(x_e, device=device, dtype=torch.float32)
        x_f = torch.tensor(x_f, device=device, dtype=torch.float32)
        valid_np = np.asarray(valid, dtype=np.float32)
        if parse_router_task_type(task_type) == ROUTER_TASK_REGRESSION:
            with torch.no_grad():
                pred_w = model.predict_weight(x_e, x_f).cpu().numpy()
            m = aggregate_router_metrics(
                oracle_curves=y,
                predicted_weights=pred_w,
                valid_mask=valid_np > 0.0,
                weight_grid=wg_np,
            )
        else:
            with torch.no_grad():
                logits = model.predict_class_logits(x_e, x_f).cpu().numpy()
            pred_cls = np.argmax(logits, axis=-1)
            m = {
                "classification_accuracy": float(np.mean(pred_cls == y_cls)),
                "classification_error": float(np.mean(pred_cls != y_cls)),
            }
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
        "task_type": parse_router_task_type(task_type),
    }
    return out


def export_split_predictions(
    model: nn.Module,
    df: pd.DataFrame,
    split: str,
    device: str,
    weight_grid: np.ndarray,
    task_type: str = ROUTER_TASK_REGRESSION,
) -> List[Dict[str, Any]]:
    """One JSON object per row for ``predictions_<split>.jsonl``."""
    sdf = _split_frame(df, split)
    if len(sdf) == 0:
        return []
    x_e, x_f, y, y_cls, valid, qids, _ = _rows_to_arrays(sdf)
    if x_e.shape[0] == 0:
        return []
    x_e = torch.tensor(x_e, device=device, dtype=torch.float32)
    x_f = torch.tensor(x_f, device=device, dtype=torch.float32)
    model.eval()
    rows: List[Dict[str, Any]] = []
    if parse_router_task_type(task_type) == ROUTER_TASK_REGRESSION:
        with torch.no_grad():
            pred_w = model.predict_weight(x_e, x_f).cpu().numpy()
    else:
        with torch.no_grad():
            logits = model.predict_class_logits(x_e, x_f).cpu().numpy()
        pred_cls = np.argmax(logits, axis=-1)
    for i in range(len(qids)):
        curve = y[i]
        best_score = float(np.max(curve)) if len(curve) else 0.0
        rows.append(
            {
                "question_id": qids[i],
                "oracle_curve": curve.tolist(),
                "target_oracle_best_score": best_score,
                "predicted_weight": (
                    float(pred_w[i])
                    if parse_router_task_type(task_type) == ROUTER_TASK_REGRESSION
                    else None
                ),
                "predicted_class_id": (
                    int(pred_cls[i])
                    if parse_router_task_type(task_type) != ROUTER_TASK_REGRESSION
                    else None
                ),
                "target_class_id": int(y_cls[i]),
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
