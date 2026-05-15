"""Resolve, validate, and load frozen dense/graph branch caches for e2e replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from surf_rag.evaluation.artifact_paths import router_oracle_dir
from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    read_manifest,
    read_retrieval_cache,
)
from surf_rag.retrieval.types import RetrievalResult
from surf_rag.router.policies import RoutingPolicyName


def _normalize_cache_mode(mode: str) -> str:
    s = (mode or "off").strip().lower()
    if s in ("explicit-jsonl", "explicit_jsonl"):
        return "explicit_jsonl"
    if s in ("router-oracle", "router_oracle"):
        return "router_oracle"
    if s in ("off", "none", ""):
        return "off"
    return s


def _resolved(p: Path) -> Path:
    try:
        return p.expanduser().resolve()
    except OSError:
        return p.expanduser()


def _paths_equal(a: Path, b: Path) -> bool:
    try:
        return _resolved(a) == _resolved(b)
    except OSError:
        return str(a).strip() == str(b).strip()


@dataclass(frozen=True)
class ResolvedFrozenBranchPaths:
    """Filesystem locations for dense/graph JSONL caches."""

    mode: str
    dense_jsonl: Path
    graph_jsonl: Path
    oracle_paths: Optional[OracleRunPaths]
    manifest: Dict[str, Any]


@dataclass(frozen=True)
class ResolvedFrozenBranchBundle:
    """Loaded question_id -> RetrievalResult maps plus provenance for manifests."""

    dense_by_qid: Dict[str, RetrievalResult]
    graph_by_qid: Dict[str, RetrievalResult]
    provenance: Dict[str, Any]
    paths: ResolvedFrozenBranchPaths


def _policy_needs_dense(policy: str) -> bool:
    p = (policy or "").strip().lower().replace("_", "-")
    return p in (
        RoutingPolicyName.DENSE_ONLY.value,
        RoutingPolicyName.EQUAL_50_50.value,
        RoutingPolicyName.LEARNED_SOFT.value,
        RoutingPolicyName.HARD_ROUTING.value,
        RoutingPolicyName.HYBRID.value,
    )


def _policy_needs_graph(policy: str) -> bool:
    p = (policy or "").strip().lower().replace("_", "-")
    return p in (
        RoutingPolicyName.GRAPH_ONLY.value,
        RoutingPolicyName.EQUAL_50_50.value,
        RoutingPolicyName.LEARNED_SOFT.value,
        RoutingPolicyName.HARD_ROUTING.value,
        RoutingPolicyName.HYBRID.value,
    )


def frozen_results_for_question(
    policy: str,
    question_id: str,
    *,
    dense_by_qid: Mapping[str, RetrievalResult],
    graph_by_qid: Mapping[str, RetrievalResult],
) -> Tuple[Optional[RetrievalResult], Optional[RetrievalResult]]:
    """Return (dense, graph) cached results for injection; ``None`` if branch unused."""
    qid = str(question_id or "").strip()
    need_d = _policy_needs_dense(policy)
    need_g = _policy_needs_graph(policy)
    dense: Optional[RetrievalResult] = None
    graph: Optional[RetrievalResult] = None
    missing: list[str] = []
    if need_d:
        dense = dense_by_qid.get(qid)
        if dense is None:
            missing.append(f"dense cache missing question_id={qid!r}")
    if need_g:
        graph = graph_by_qid.get(qid)
        if graph is None:
            missing.append(f"graph cache missing question_id={qid!r}")
    if missing:
        raise ValueError("; ".join(missing))
    return dense, graph


def resolve_frozen_branch_paths(
    *,
    mode: str,
    router_base: Path,
    paths_router_id: str,
    oracle_router_id: Optional[str],
    dense_jsonl: Optional[str | Path],
    graph_jsonl: Optional[str | Path],
) -> ResolvedFrozenBranchPaths:
    m = _normalize_cache_mode(mode)
    if m == "off":
        raise ValueError("resolve_frozen_branch_paths called with mode=off")
    if m == "router_oracle":
        rid = str(oracle_router_id or paths_router_id or "").strip()
        if not rid:
            raise ValueError(
                "branch_cache mode router_oracle requires paths.router_id or "
                "e2e.branch_cache.oracle_router_id"
            )
        oracle_paths = OracleRunPaths(
            run_root=router_oracle_dir(_resolved(router_base), rid)
        )
        return ResolvedFrozenBranchPaths(
            mode=m,
            dense_jsonl=oracle_paths.retrieval_dense,
            graph_jsonl=oracle_paths.retrieval_graph,
            oracle_paths=oracle_paths,
            manifest={},
        )
    if m == "explicit_jsonl":
        if not dense_jsonl or not graph_jsonl:
            raise ValueError(
                "branch_cache mode explicit_jsonl requires dense_jsonl and graph_jsonl"
            )
        d = Path(str(dense_jsonl)).expanduser()
        g = Path(str(graph_jsonl)).expanduser()
        return ResolvedFrozenBranchPaths(
            mode=m,
            dense_jsonl=d,
            graph_jsonl=g,
            oracle_paths=None,
            manifest={},
        )
    raise ValueError(
        f"Unknown branch_cache.mode {mode!r}; expected off, router_oracle, explicit_jsonl"
    )


def validate_frozen_branch_bundle(
    resolved: ResolvedFrozenBranchPaths,
    *,
    benchmark_path: Path,
    retrieval_asset_dir: Path,
    expect_branch_top_k: int,
    strict_manifest: bool,
) -> Dict[str, Any]:
    """Validate cache files and optional oracle manifest; return manifest subset for provenance."""
    if not resolved.dense_jsonl.is_file():
        raise FileNotFoundError(f"Frozen dense cache not found: {resolved.dense_jsonl}")
    if not resolved.graph_jsonl.is_file():
        raise FileNotFoundError(f"Frozen graph cache not found: {resolved.graph_jsonl}")

    manifest: Dict[str, Any] = {}
    if resolved.oracle_paths is not None and resolved.oracle_paths.manifest.is_file():
        manifest = read_manifest(resolved.oracle_paths)
    elif strict_manifest and resolved.mode == "router_oracle":
        raise FileNotFoundError(
            "strict_manifest requires oracle manifest.json next to retrieval caches at "
            f"{resolved.oracle_paths.run_root if resolved.oracle_paths else '?'}"
        )

    if not manifest:
        return {}

    bench_m = str(manifest.get("benchmark_path") or "").strip()
    corp_m = str(manifest.get("retrieval_asset_dir") or "").strip()
    btk = manifest.get("branch_top_k")

    if not strict_manifest:
        return {
            "manifest_schema_version": manifest.get("schema_version"),
            "manifest_branch_top_k": btk,
            "manifest_benchmark_path": bench_m or None,
            "manifest_retrieval_asset_dir": corp_m or None,
        }

    if bench_m:
        if not _paths_equal(Path(bench_m), benchmark_path):
            raise ValueError(
                "Frozen oracle manifest benchmark_path does not match e2e benchmark_path:\n"
                f"  manifest: {bench_m}\n"
                f"  e2e:      {benchmark_path}"
            )
    corp_cur = str(_resolved(retrieval_asset_dir))
    if corp_m and corp_cur != str(_resolved(Path(corp_m))):
        raise ValueError(
            "Frozen oracle manifest retrieval_asset_dir does not match e2e corpus dir:\n"
            f"  manifest: {corp_m}\n"
            f"  e2e:      {corp_cur}"
        )
    if btk is not None and int(btk) != int(expect_branch_top_k):
        raise ValueError(
            "Frozen oracle manifest branch_top_k does not match e2e.branch_top_k:\n"
            f"  manifest: {btk}\n"
            f"  e2e:      {expect_branch_top_k}"
        )
    return {
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_branch_top_k": btk,
        "manifest_benchmark_path": bench_m or None,
        "manifest_retrieval_asset_dir": corp_m or None,
    }


def load_frozen_branch_bundle(
    *,
    mode: str,
    router_base: Path,
    paths_router_id: str,
    oracle_router_id: Optional[str],
    dense_jsonl: Optional[str | Path],
    graph_jsonl: Optional[str | Path],
    benchmark_path: Path,
    retrieval_asset_dir: Path,
    expect_branch_top_k: int,
    strict_manifest: bool,
    question_ids: Optional[set[str]],
    policy: str,
) -> ResolvedFrozenBranchBundle:
    """Load dense+graph caches after path resolution and validation."""
    m = _normalize_cache_mode(mode)
    if m == "off":
        raise ValueError("load_frozen_branch_bundle requires mode != off")

    paths = resolve_frozen_branch_paths(
        mode=m,
        router_base=router_base,
        paths_router_id=paths_router_id,
        oracle_router_id=oracle_router_id,
        dense_jsonl=dense_jsonl,
        graph_jsonl=graph_jsonl,
    )
    manifest_meta = validate_frozen_branch_bundle(
        paths,
        benchmark_path=benchmark_path,
        retrieval_asset_dir=retrieval_asset_dir,
        expect_branch_top_k=expect_branch_top_k,
        strict_manifest=strict_manifest,
    )

    dense_by_qid = read_retrieval_cache(paths.dense_jsonl)
    graph_by_qid = read_retrieval_cache(paths.graph_jsonl)

    if question_ids:
        need_d = _policy_needs_dense(policy)
        need_g = _policy_needs_graph(policy)
        missing_d = sorted(q for q in question_ids if need_d and q not in dense_by_qid)
        missing_g = sorted(q for q in question_ids if need_g and q not in graph_by_qid)
        if missing_d or missing_g:
            parts = []
            if missing_d:
                parts.append(
                    f"dense cache missing {len(missing_d)} question_ids "
                    f"(e.g. {missing_d[:5]})"
                )
            if missing_g:
                parts.append(
                    f"graph cache missing {len(missing_g)} question_ids "
                    f"(e.g. {missing_g[:5]})"
                )
            raise ValueError("; ".join(parts))

    prov: Dict[str, Any] = {
        "mode": m,
        "dense_jsonl": str(paths.dense_jsonl.resolve()),
        "graph_jsonl": str(paths.graph_jsonl.resolve()),
        "strict_manifest": bool(strict_manifest),
        **manifest_meta,
    }
    if paths.oracle_paths is not None:
        prov["oracle_run_root"] = str(paths.oracle_paths.run_root.resolve())
    if paths.oracle_paths is not None and paths.oracle_paths.provenance.is_file():
        try:
            pjson = json.loads(
                paths.oracle_paths.provenance.read_text(encoding="utf-8")
            )
            prov["provenance_created_at"] = (pjson or {}).get("created_at")
        except Exception:
            pass

    return ResolvedFrozenBranchBundle(
        dense_by_qid=dense_by_qid,
        graph_by_qid=graph_by_qid,
        provenance=prov,
        paths=paths,
    )


__all__ = [
    "ResolvedFrozenBranchBundle",
    "ResolvedFrozenBranchPaths",
    "_normalize_cache_mode",
    "frozen_results_for_question",
    "load_frozen_branch_bundle",
    "resolve_frozen_branch_paths",
    "validate_frozen_branch_bundle",
]
