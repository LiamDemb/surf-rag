"""Train RouterMLP from ``router_dataset.parquet``."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.router.model import RouterMLP, RouterMLPConfig
from surf_rag.router.router_metrics import aggregate_router_metrics, kl_divergence_torch


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
    hidden_dim: int = 32
    embed_proj_dim: int = 16
    feat_proj_dim: int = 16
    dropout: float = 0.1
    num_workers: int = 0


def _seq_floats(val: object) -> List[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if hasattr(val, "tolist"):
        val = val.tolist()
    return [float(x) for x in list(val)]


def _rows_to_arrays(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Stack embedding, feature_norm, distribution; return question_ids."""
    emb_rows: List[List[float]] = []
    feat_rows: List[List[float]] = []
    dist_rows: List[List[float]] = []
    qids: List[str] = []
    for _, row in df.iterrows():
        qids.append(str(row.get("question_id", "")))
        emb_rows.append(_seq_floats(row.get("query_embedding")))
        feat_rows.append(_seq_floats(row.get("feature_vector_norm")))
        dist_rows.append(_seq_floats(row.get("distribution")))
    if not emb_rows:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            [],
        )
    x_e = np.asarray(emb_rows, dtype=np.float32)
    x_f = np.asarray(feat_rows, dtype=np.float32)
    y = np.asarray(dist_rows, dtype=np.float32)
    return x_e, x_f, y, qids


def _split_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"].astype(str).str.lower() == split.lower()].copy()


def build_model_config_from_df(
    df: pd.DataFrame, cfg: RouterTrainConfig
) -> RouterMLPConfig:
    sample = df.iloc[0]
    emb_dim = int(sample.get("embedding_dim", 0)) or len(
        _seq_floats(sample.get("query_embedding"))
    )
    feat_dim = len(_seq_floats(sample.get("feature_vector_norm")))
    return RouterMLPConfig(
        embedding_dim=emb_dim,
        feature_dim=feat_dim,
        embed_proj_dim=cfg.embed_proj_dim,
        feat_proj_dim=cfg.feat_proj_dim,
        hidden_dim=cfg.hidden_dim,
        num_bins=11,
        dropout=cfg.dropout,
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
    model: RouterMLP
    history: List[Dict[str, float]]
    best_epoch: int
    metrics: Dict[str, Any]


def train_router(cfg: RouterTrainConfig) -> TrainRunResult:
    """Fit RouterMLP; return model, per-epoch history, and best epoch index."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    df = pd.read_parquet(cfg.parquet_path)
    train_df = _split_frame(df, "train")
    dev_df = _split_frame(df, "dev")
    if len(train_df) == 0:
        raise ValueError("train split is empty in parquet")

    mcfg = build_model_config_from_df(train_df, cfg)
    model = RouterMLP(mcfg).to(cfg.device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    x_e, x_f, y, _ = _rows_to_arrays(train_df)
    wg_np = _weight_grid_from_df(train_df)
    wg = torch.tensor(wg_np, device=cfg.device, dtype=torch.float32)

    x_e_t = torch.tensor(x_e, device=cfg.device, dtype=torch.float32)
    x_f_t = torch.tensor(x_f, device=cfg.device, dtype=torch.float32)
    y_t = torch.tensor(y, device=cfg.device, dtype=torch.float32)

    ds = TensorDataset(x_e_t, x_f_t, y_t)
    loader = DataLoader(
        ds,
        batch_size=min(cfg.batch_size, len(ds)),
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    x_ed, x_fd, yd, _ = _rows_to_arrays(dev_df)
    has_dev = len(dev_df) > 0 and x_ed.shape[0] > 0
    if has_dev:
        x_ed = torch.tensor(x_ed, device=cfg.device, dtype=torch.float32)
        x_fd = torch.tensor(x_fd, device=cfg.device, dtype=torch.float32)
        yd = torch.tensor(yd, device=cfg.device, dtype=torch.float32)

    history: List[Dict[str, float]] = []
    best_dev = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses: List[float] = []
        for xeb, xfb, yb in loader:
            opt.zero_grad()
            logits = model(xeb, xfb)
            log_q = torch.log_softmax(logits, dim=-1)
            loss = kl_divergence_torch(log_q, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))

        train_kl = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        dev_kl: Optional[float] = None
        if has_dev:
            model.eval()
            with torch.no_grad():
                log_qd = torch.log_softmax(model(x_ed, x_fd), dim=-1)
                dev_kl = float(kl_divergence_torch(log_qd, yd).item())

        row: Dict[str, float] = {
            "epoch": float(epoch + 1),
            "train_kl": train_kl,
        }
        if dev_kl is not None:
            row["dev_kl"] = dev_kl
        history.append(row)

        if has_dev and dev_kl is not None:
            if dev_kl < best_dev - cfg.early_stopping_min_delta:
                best_dev = dev_kl
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

    metrics = _eval_splits(model, df, wg_np, cfg.device, mcfg)
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
    mcfg: RouterMLPConfig,
) -> Dict[str, Any]:
    model.eval()
    out: Dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        sdf = _split_frame(df, split)
        if len(sdf) == 0:
            out[split] = {"num_rows": 0}
            continue
        x_e, x_f, y, _ = _rows_to_arrays(sdf)
        x_e = torch.tensor(x_e, device=device, dtype=torch.float32)
        x_f = torch.tensor(x_f, device=device, dtype=torch.float32)
        with torch.no_grad():
            log_p = torch.log_softmax(model(x_e, x_f), dim=-1)
        log_np = log_p.cpu().numpy()
        t_list: List[np.ndarray] = [y[i] for i in range(len(y))]
        p_list: List[np.ndarray] = [log_np[i] for i in range(len(log_np))]
        m = aggregate_router_metrics(t_list, p_list, wg_np)
        m["num_rows"] = float(len(sdf))
        out[split] = m
    out["config_summary"] = mcfg.to_json()
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
    x_e, x_f, y, qids = _rows_to_arrays(sdf)
    if x_e.shape[0] == 0:
        return []
    x_e = torch.tensor(x_e, device=device, dtype=torch.float32)
    x_f = torch.tensor(x_f, device=device, dtype=torch.float32)
    wg = torch.tensor(weight_grid, device=device, dtype=torch.float32)
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        log_p = torch.log_softmax(model(x_e, x_f), dim=-1)
        dist_p = torch.exp(log_p)
        ev_p = (dist_p * wg.unsqueeze(0)).sum(dim=-1).cpu().numpy()
    y_t = y
    ev_t = (y_t * weight_grid.reshape(1, -1)).sum(axis=1)
    for i in range(len(qids)):
        pt = y_t[i]
        pp = dist_p[i].cpu().numpy()
        rows.append(
            {
                "question_id": qids[i],
                "target_distribution": pt.tolist(),
                "predicted_distribution": pp.tolist(),
                "target_expected_weight": float(ev_t[i]),
                "predicted_expected_weight": float(ev_p[i]),
            }
        )
    return rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    mcfg: RouterMLPConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": mcfg.to_json(),
        },
        path,
    )
