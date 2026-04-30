"""Load a trained router checkpoint and predict scalar per-query weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from surf_rag.evaluation.oracle_artifacts import DEFAULT_DENSE_WEIGHT_GRID
from surf_rag.evaluation.router_model_artifacts import read_json
from surf_rag.router.model import RouterMLP, RouterMLPConfig


@dataclass
class LoadedRouter:
    model: RouterMLP
    config: RouterMLPConfig
    weight_grid: np.ndarray
    device: str
    manifest: Dict[str, Any]


def load_router_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    manifest_path: Optional[Path] = None,
) -> LoadedRouter:
    """Load ``model.pt`` (torch dict with ``state_dict`` and ``config``)."""
    try:
        pack = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        pack = torch.load(checkpoint_path, map_location=device)
    if isinstance(pack, dict) and "state_dict" in pack:
        cfg = RouterMLPConfig.from_json(pack["config"])
        state = pack["state_dict"]
    else:
        raise ValueError("checkpoint must be a dict with state_dict and config")
    model = RouterMLP(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    manifest: Dict[str, Any] = {}
    if manifest_path and manifest_path.is_file():
        manifest = read_json(manifest_path)
    wg = list(manifest.get("model", {}).get("weight_grid") or DEFAULT_DENSE_WEIGHT_GRID)
    weight_grid = np.asarray([float(x) for x in wg], dtype=np.float32)
    return LoadedRouter(
        model=model,
        config=cfg,
        weight_grid=weight_grid,
        device=device,
        manifest=manifest,
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
    w_hat = router.model.predict_weight(x_e, x_f)
    return w_hat.cpu().numpy()
