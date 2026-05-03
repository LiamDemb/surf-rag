"""Process-wide cache for heavy HuggingFace SentenceTransformer / CrossEncoder models.

Avoids repeated construction (and repeated hub metadata checks) within one Python process.
Respects ``HF_HOME``, ``TRANSFORMERS_CACHE``, ``HF_HUB_OFFLINE``, etc. via the libraries.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Tuple

_lock = threading.Lock()
_sentence_transformers: Dict[Tuple[Any, ...], Any] = {}
_cross_encoders: Dict[Tuple[Any, ...], Any] = {}
logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    import torch

    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _resolve_cross_encoder_device(device: str | None) -> str | None:
    """Normalize cross-encoder device requests with safe MPS fallback."""
    raw = (device or "").strip()
    if not raw:
        return None
    normalized = raw.lower()
    if normalized != "mps":
        return normalized
    if _mps_available():
        return "mps"
    logger.warning(
        "CROSS_ENCODER_DEVICE=mps requested but MPS is unavailable; falling back to cpu."
    )
    return "cpu"


def get_sentence_transformer(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    device: str | None = None,
) -> Any:
    """Return a shared :class:`sentence_transformers.SentenceTransformer` instance."""
    name = (model_name or "").strip()
    if not name:
        raise ValueError("model_name must be non-empty")
    key = ("st", name, bool(trust_remote_code), revision, device)
    with _lock:
        cached = _sentence_transformers.get(key)
        if cached is not None:
            return cached
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision is not None:
        kwargs["revision"] = revision
    if device is not None:
        kwargs["device"] = device
    model = SentenceTransformer(name, **kwargs)
    with _lock:
        _sentence_transformers[key] = model
    return model


def get_cross_encoder(
    model_name: str,
    *,
    device: str | None = None,
    max_length: int | None = None,
) -> Any:
    """Return a shared :class:`sentence_transformers.CrossEncoder` instance."""
    name = (model_name or "").strip()
    if not name:
        raise ValueError("model_name must be non-empty")
    resolved_device = _resolve_cross_encoder_device(device)
    key = ("ce", name, resolved_device, max_length)
    with _lock:
        cached = _cross_encoders.get(key)
        if cached is not None:
            return cached
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {}
    if resolved_device is not None:
        kwargs["device"] = resolved_device
    if max_length is not None:
        kwargs["max_length"] = max_length
    model = CrossEncoder(name, **kwargs)
    with _lock:
        _cross_encoders[key] = model
    return model


def clear_model_caches() -> None:
    """Drop cached models (for tests only)."""
    with _lock:
        _sentence_transformers.clear()
        _cross_encoders.clear()
