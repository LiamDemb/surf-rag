"""Latency normalization and summary statistics for E2E reports."""

from __future__ import annotations

import math
import random
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Sequence

LATENCY_PROTOCOL_VERSION = "v1"


def canonicalize_latency_ms(
    *,
    retriever_name: str,
    latency_ms: Mapping[str, object] | None,
    routing_input_ms: float = 0.0,
) -> dict[str, float]:
    """Return canonical per-question latency keys.

    Canonical keys:
    - retrieval_stage_total_ms
    - routing_input_ms
    - router_predict_ms
    - dense_branch_ms (when dense branch ran)
    - graph_branch_ms (when graph branch ran)
    - fusion_ms (when both branches ran)
    """
    raw = dict(latency_ms or {})

    def _get(key: str, default: float = 0.0) -> float:
        v = raw.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    pipe_total = _get("total_ms", _get("total", 0.0))
    router_predict = _get("routing_predict_ms", 0.0)
    routing_input = float(max(0.0, routing_input_ms))

    out: dict[str, float] = {
        "routing_input_ms": routing_input,
        "router_predict_ms": router_predict,
    }
    name = (retriever_name or "").strip().lower()

    if name == "dense":
        out["dense_branch_ms"] = _get("total", pipe_total)
    elif name == "graph":
        out["graph_branch_ms"] = _get("total", pipe_total)
    else:
        out["fusion_ms"] = _get("fusion", 0.0)
        if "dense_total" in raw:
            out["dense_branch_ms"] = _get("dense_total")
        if "graph_total" in raw:
            out["graph_branch_ms"] = _get("graph_total")

    out["retrieval_stage_total_ms"] = max(0.0, pipe_total + routing_input)
    return out


def latency_values(
    rows: Iterable[Mapping[str, object]],
    *,
    key: str = "retrieval_stage_total_ms",
) -> list[float]:
    vals: list[float] = []
    for row in rows:
        lat = row.get("latency_ms")
        if not isinstance(lat, Mapping):
            continue
        v = lat.get(key)
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    return vals


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def bootstrap_mean_ci95(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> list[float]:
    xs = [float(v) for v in values]
    if not xs:
        return [0.0, 0.0]
    if len(xs) == 1:
        return [xs[0], xs[0]]
    rng = random.Random(seed)
    n = len(xs)
    means = [0.0] * samples
    for i in range(samples):
        draw = [xs[rng.randrange(n)] for _ in range(n)]
        means[i] = mean(draw)
    means.sort()
    lo = _percentile(means, 0.025)
    hi = _percentile(means, 0.975)
    return [float(lo), float(hi)]


def summarize_latency(
    values: Sequence[float],
    *,
    total_count: int | None = None,
    ci_samples: int = 10_000,
    ci_seed: int = 42,
) -> dict[str, float | int | list[float]]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    n_total = int(total_count) if total_count is not None else len(xs)
    missing = max(0, n_total - len(xs))
    if not xs:
        return {
            "count": n_total,
            "valid_count": 0,
            "missing_count": missing,
            "mean_ms": 0.0,
            "mean_ci95_ms": [0.0, 0.0],
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    m = mean(xs)
    sd = stdev(xs) if len(xs) > 1 else 0.0
    return {
        "count": n_total,
        "valid_count": len(xs),
        "missing_count": missing,
        "mean_ms": float(m),
        "mean_ci95_ms": bootstrap_mean_ci95(xs, samples=ci_samples, seed=ci_seed),
        "median_ms": float(median(xs)),
        "p90_ms": float(_percentile(xs, 0.90)),
        "p95_ms": float(_percentile(xs, 0.95)),
        "std_ms": float(sd),
        "min_ms": float(min(xs)),
        "max_ms": float(max(xs)),
    }
