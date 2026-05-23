"""Y-axis labels for router training learning-curve figures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from surf_rag.evaluation.router_model_artifacts import read_json

_LOSS_DISPLAY: dict[str, str] = {
    "regret": "Regret",
    "hinge_squared_regret": "Hinge Squared Regret",
    "boundary_magnet": "Boundary Magnet",
    "cross_entropy": "Cross Entropy",
    "ce": "Cross Entropy",
}

_ROUTER_ID_OBJECTIVE_RE = re.compile(r"^(ndcg|recall|hit)@(\d+)$", re.IGNORECASE)
_ROUTER_ID_METRIC = {
    "ndcg": "stateful_ndcg",
    "recall": "recall",
    "hit": "hit",
}


def format_metric_at_k(metric: str, k: int) -> str:
    """Human-readable objective name, e.g. ``NDCG@10``."""
    m = str(metric or "").strip().lower()
    k_int = int(k)
    if m in ("stateful_ndcg", "ndcg"):
        return f"NDCG@{k_int}"
    if m == "recall":
        return f"Recall@{k_int}"
    if m == "hit":
        return f"Hit@{k_int}"
    base = m.replace("_", " ").title() if m else "Metric"
    return f"{base}@{k_int}"


def format_loss_display_name(loss_id: str) -> str:
    key = str(loss_id or "").strip().lower().replace("-", "_")
    if not key:
        return "Loss"
    return _LOSS_DISPLAY.get(key, key.replace("_", " ").title())


def format_learning_curve_ylabel(
    loss_id: str,
    *,
    task_type: str,
    oracle_metric: str,
    oracle_metric_k: int,
) -> str:
    """Build Y-axis label from training loss and (for regression) oracle objective."""
    loss_name = format_loss_display_name(loss_id)
    task = str(task_type or "").strip().lower()
    if task == "classification":
        return f"{loss_name} Loss"
    metric_label = format_metric_at_k(oracle_metric, oracle_metric_k)
    return f"{metric_label} {loss_name} Loss"


def parse_router_id_objective(router_id: str) -> tuple[str, int] | None:
    m = _ROUTER_ID_OBJECTIVE_RE.match(str(router_id or "").strip())
    if not m:
        return None
    kind = m.group(1).lower()
    return _ROUTER_ID_METRIC[kind], int(m.group(2))


def resolve_oracle_objective(
    *,
    router_id: str,
    config_metric: str,
    config_k: int,
    oracle_summary_path: Path | None,
) -> tuple[str, int]:
    """Resolve oracle metric and k for regression loss labels."""
    if oracle_summary_path is not None and oracle_summary_path.is_file():
        try:
            data = read_json(oracle_summary_path)
            metric = str(data.get("oracle_metric") or "").strip()
            if metric:
                return metric, int(data.get("oracle_metric_k") or config_k)
        except Exception:
            pass
    parsed = parse_router_id_objective(router_id)
    if parsed is not None:
        return parsed
    return str(config_metric or "stateful_ndcg").strip(), int(config_k)


def resolve_training_loss_id(
    *,
    config_loss: str,
    history_payload: dict[str, Any] | None,
    manifest_path: Path | None,
) -> str:
    """Prefer artifact-recorded loss, then pipeline config."""
    if history_payload:
        for key in ("loss_effective", "loss"):
            val = str(history_payload.get(key) or "").strip()
            if val:
                return val
    if manifest_path is not None and manifest_path.is_file():
        try:
            training = read_json(manifest_path).get("training") or {}
            for key in ("loss_effective", "loss"):
                val = str(training.get(key) or "").strip()
                if val:
                    return val
        except Exception:
            pass
    return str(config_loss or "regret").strip() or "regret"
