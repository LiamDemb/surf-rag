"""Load a trained router checkpoint and predict scalar per-query weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.evaluation.router_model_artifacts import (
    ROUTER_TASK_REGRESSION,
    parse_router_task_type,
    read_json,
)
from surf_rag.router.architectures.registry import get_architecture


@dataclass
class LoadedRouter:
    model: nn.Module
    config: Any
    architecture: str
    architecture_kwargs: Dict[str, Any]
    weight_grid: np.ndarray
    device: str
    manifest: Dict[str, Any]
    task_type: str = ROUTER_TASK_REGRESSION


def load_router_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    manifest_path: Optional[Path] = None,
    router_task_type: str | None = None,
) -> LoadedRouter:
    """Load ``model.pt`` (torch dict with ``state_dict`` and ``config``)."""
    try:
        pack = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        pack = torch.load(checkpoint_path, map_location=device)
    if isinstance(pack, dict) and "state_dict" in pack:
        architecture = str(pack.get("architecture") or "mlp-v1")
        architecture_kwargs = dict(pack.get("architecture_kwargs") or {})
        arch = get_architecture(architecture)
        cfg = arch.config_from_json(dict(pack.get("config") or {}))
        state = pack["state_dict"]
    else:
        raise ValueError("checkpoint must be a dict with state_dict and config")
    model = arch.build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    manifest: Dict[str, Any] = {}
    if manifest_path and manifest_path.is_file():
        manifest = read_json(manifest_path)
    manifest_task_type = parse_router_task_type(
        str(manifest.get("task_type", ROUTER_TASK_REGRESSION))
    )
    requested_task_type = (
        parse_router_task_type(router_task_type)
        if router_task_type is not None
        else manifest_task_type
    )
    if requested_task_type != manifest_task_type:
        raise ValueError(
            "Router task mismatch: requested "
            f"{requested_task_type!r}, manifest declares {manifest_task_type!r}"
        )
    wg = list(manifest.get("model", {}).get("weight_grid") or DEFAULT_DENSE_WEIGHT_GRID)
    weight_grid = np.asarray([float(x) for x in wg], dtype=np.float32)
    return LoadedRouter(
        model=model,
        config=cfg,
        architecture=architecture,
        architecture_kwargs=architecture_kwargs,
        weight_grid=weight_grid,
        device=device,
        manifest=manifest,
        task_type=manifest_task_type,
    )


@torch.inference_mode()
def predict_batch(
    router: LoadedRouter,
    query_embedding: np.ndarray,
    feature_vector: np.ndarray,
) -> np.ndarray:
    """Return predicted dense weights as numpy array, shape [B]."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]
    if feature_vector.ndim == 1:
        feature_vector = feature_vector[np.newaxis, :]
    x_e = torch.tensor(query_embedding, device=router.device, dtype=torch.float32)
    x_f = torch.tensor(feature_vector, device=router.device, dtype=torch.float32)
    if router.task_type != ROUTER_TASK_REGRESSION:
        raise ValueError(
            "predict_batch is regression-only; use classification helpers for "
            f"task_type={router.task_type!r}"
        )
    w_hat = router.model.predict_weight(x_e, x_f)
    return w_hat.cpu().numpy()


@torch.inference_mode()
def predict_class_logits_batch(
    router: LoadedRouter,
    query_embedding: np.ndarray,
    feature_vector: np.ndarray,
) -> np.ndarray:
    """Return class logits as numpy array, shape [B, C]."""
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]
    if feature_vector.ndim == 1:
        feature_vector = feature_vector[np.newaxis, :]
    x_e = torch.tensor(query_embedding, device=router.device, dtype=torch.float32)
    x_f = torch.tensor(feature_vector, device=router.device, dtype=torch.float32)
    if not hasattr(router.model, "predict_class_logits"):
        raise ValueError(
            "Loaded router model does not support classification logits "
            f"(task_type={router.task_type!r})"
        )
    logits = router.model.predict_class_logits(x_e, x_f)
    return logits.cpu().numpy()


@torch.inference_mode()
def predict_class_id_batch(
    router: LoadedRouter,
    query_embedding: np.ndarray,
    feature_vector: np.ndarray,
) -> np.ndarray:
    """Return predicted class ids, shape [B]."""
    logits = predict_class_logits_batch(router, query_embedding, feature_vector)
    return np.argmax(logits, axis=-1).astype(np.int64)


@torch.inference_mode()
def predict_class_probs_batch(
    router: LoadedRouter,
    query_embedding: np.ndarray,
    feature_vector: np.ndarray,
) -> np.ndarray:
    """Return class probabilities as numpy array, shape [B, 2]."""
    logits = predict_class_logits_batch(router, query_embedding, feature_vector)
    x = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    denom = ex.sum(axis=-1, keepdims=True)
    return ex / np.maximum(denom, 1e-12)


@torch.inference_mode()
def predict_class_confidence_batch(
    router: LoadedRouter,
    query_embedding: np.ndarray,
    feature_vector: np.ndarray,
) -> np.ndarray:
    """Return max class probability per row, shape [B]."""
    probs = predict_class_probs_batch(router, query_embedding, feature_vector)
    return np.max(probs, axis=-1).astype(np.float32)
