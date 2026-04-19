"""Shared OpenAI Batch API shard limits (env-configurable)."""

from __future__ import annotations

import os

_DEFAULT_BATCH_MAX_REQUESTS = 50_000
_ENV_KEY = "OPENAI_BATCH_MAX_REQUESTS"


def batch_limit_requests() -> int:
    """Max requests per batch shard file (strategy + IE batch scripts).

    OpenAI's documented cap is 50,000 requests per batch input file; you can set
    a lower value via OPENAI_BATCH_MAX_REQUESTS to stay under other limits.
    """
    raw = os.environ.get(_ENV_KEY)
    if raw is None or not str(raw).strip():
        return _DEFAULT_BATCH_MAX_REQUESTS
    try:
        n = int(str(raw).strip(), 10)
    except ValueError:
        return _DEFAULT_BATCH_MAX_REQUESTS
    return max(1, n)
