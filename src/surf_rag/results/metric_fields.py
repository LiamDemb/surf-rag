"""Resolve metric values from oracle bins and E2E per-question rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surf_rag.config.schema import ResultsArtifactSpec
    from surf_rag.results.bundle import ResultsBundle

_DEFAULT_DIAGNOSTIC_KS: tuple[int, ...] = (5, 10, 20)


def _normalize_metric(metric: str) -> str:
    return str(metric or "").strip().lower()


def oracle_bin_value(bin_score: dict[str, Any], *, metric: str, k: int) -> float:
    """Read one weight-bin score for the requested metric and cut-off k."""
    m = _normalize_metric(metric)
    if m in ("stateful_ndcg", "ndcg"):
        diag = bin_score.get("diagnostic_ndcg") or {}
        key = str(int(k))
        if key in diag:
            return float(diag[key])
        if "oracle_objective_value" in bin_score:
            return float(bin_score.get("oracle_objective_value", 0.0))
        return float(bin_score.get("ndcg_primary", 0.0))
    if m == "hit":
        return float((bin_score.get("diagnostic_hit") or {}).get(str(int(k)), 0.0))
    if m == "recall":
        return float((bin_score.get("diagnostic_recall") or {}).get(str(int(k)), 0.0))
    if "oracle_objective_value" in bin_score:
        return float(bin_score["oracle_objective_value"])
    raise ValueError(f"Unsupported oracle metric: {metric!r}")


def oracle_curve_series(
    row: dict[str, Any], *, metric: str, k: int
) -> tuple[list[float], list[float]]:
    """Return (values, weight_grid) for one oracle question row."""
    scores = list(row.get("scores") or [])
    grid = [float(w) for w in (row.get("weight_grid") or [])]
    if len(scores) != len(grid):
        raise ValueError(
            f"oracle scores length mismatch for {row.get('question_id')}: "
            f"{len(scores)} vs {len(grid)}"
        )
    vals = [oracle_bin_value(s, metric=metric, k=k) for s in scores]
    return vals, grid


def e2e_retrieval_value(row: dict[str, Any], *, metric: str, k: int) -> float:
    """Read retrieval_before_ce metric for one per-question E2E row."""
    m = _normalize_metric(metric)
    block = (
        (row.get("retrieval_before_ce") or {}).get("retrieval", {}).get(str(int(k)), {})
    )
    if not isinstance(block, dict):
        return 0.0
    if m in ("stateful_ndcg", "ndcg"):
        return float(block.get("ndcg", 0.0))
    if m == "hit":
        return float(block.get("hit", 0.0))
    if m == "recall":
        return float(block.get("recall", 0.0))
    raise ValueError(f"Unsupported E2E retrieval metric: {metric!r}")


def resolve_retrieval_metric_k(
    spec: ResultsArtifactSpec | None,
    bundle: ResultsBundle,
) -> tuple[str, int]:
    """Resolve headline metric and k: artefact → results.oracle → schema defaults."""
    oc = bundle.results.oracle
    metric = _normalize_metric(
        spec.metric if spec is not None and spec.metric else oc.metric
    )
    k = int(spec.k if spec is not None and spec.k is not None else oc.k)
    return metric, k


resolve_oracle_metric_k = resolve_retrieval_metric_k


def resolve_retrieval_ks(
    spec: ResultsArtifactSpec | None,
    bundle: ResultsBundle,
) -> list[int]:
    """Resolve cut-off list for wide retrieval tables."""
    if spec is not None and spec.ks:
        return [int(x) for x in spec.ks]
    if spec is not None and spec.k is not None:
        return [int(spec.k)]
    diagnostic = bundle.results.oracle.diagnostic_ks
    if diagnostic:
        return [int(x) for x in diagnostic]
    return list(_DEFAULT_DIAGNOSTIC_KS)


def e2e_hit_recall_ndcg(row: dict[str, Any], *, k: int) -> tuple[float, float, float]:
    """Return (hit, recall, ndcg) at k from retrieval_before_ce."""
    block = (
        (row.get("retrieval_before_ce") or {}).get("retrieval", {}).get(str(int(k)), {})
    )
    if not isinstance(block, dict):
        return 0.0, 0.0, 0.0
    return (
        float(block.get("hit", 0.0)),
        float(block.get("recall", 0.0)),
        float(block.get("ndcg", 0.0)),
    )
