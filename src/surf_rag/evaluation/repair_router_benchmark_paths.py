"""Rewrite router oracle/dataset/model manifests when benchmark bundle paths move."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from surf_rag.evaluation.artifact_paths import (
    benchmark_bundle_dir,
    default_benchmark_base,
    default_router_base,
)
from surf_rag.evaluation.oracle_artifacts import (
    OracleRunPaths,
    build_oracle_run_root,
    read_manifest as read_oracle_manifest,
    utc_now_iso,
)
from surf_rag.evaluation.router_dataset_artifacts import (
    RouterDatasetPaths,
    build_router_dataset_root,
    read_router_dataset_manifest,
)

log = logging.getLogger(__name__)


def canonical_benchmark_paths(
    benchmark_name: str,
    benchmark_id: str,
    *,
    benchmark_base: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Return (benchmark.jsonl, corpus_dir) for the bundle."""
    bb = benchmark_base if benchmark_base is not None else default_benchmark_base()
    root = benchmark_bundle_dir(bb, benchmark_name, benchmark_id)
    return root / "benchmark" / "benchmark.jsonl", root / "corpus"


def repair_router_bundle_metadata(
    router_id: str,
    *,
    benchmark_name: str,
    benchmark_id: str,
    router_base: Optional[Path] = None,
    benchmark_base: Optional[Path] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Point manifests at ``benchmark_base/<name>/<id>/benchmark|corpus``."""
    rb = router_base if router_base is not None else default_router_base()
    bb = benchmark_base if benchmark_base is not None else default_benchmark_base()
    bench_path, corpus_dir = canonical_benchmark_paths(
        benchmark_name, benchmark_id, benchmark_base=bb
    )

    summary: Dict[str, Any] = {
        "dry_run": dry_run,
        "router_id": router_id,
        "benchmark_jsonl": str(bench_path.resolve()),
        "corpus_dir": str(corpus_dir.resolve()),
        "updated_files": [],
    }

    o_paths = OracleRunPaths(run_root=build_oracle_run_root(rb, router_id))
    if o_paths.manifest.is_file():
        data = read_oracle_manifest(o_paths)
        old_bp = data.get("benchmark_path")
        old_ra = data.get("retrieval_asset_dir")
        data["benchmark_name"] = benchmark_name
        data["benchmark_id"] = benchmark_id
        data["benchmark_path"] = str(bench_path.resolve())
        data["retrieval_asset_dir"] = str(corpus_dir.resolve())
        data["updated_at"] = utc_now_iso()
        if old_bp != data["benchmark_path"] or old_ra != data["retrieval_asset_dir"]:
            log.info("oracle manifest: %s", o_paths.manifest)
            if not dry_run:
                o_paths.manifest.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            summary["updated_files"].append(str(o_paths.manifest))

        prov_created = utc_now_iso()
        if o_paths.provenance.is_file():
            prev = json.loads(o_paths.provenance.read_text(encoding="utf-8"))
            prov_created = str(prev.get("created_at") or prov_created)
        prov: Dict[str, Any] = {
            "schema_version": 1,
            "created_at": prov_created,
            "router_id": router_id,
            "source_benchmark_bundle": {
                "name": benchmark_name,
                "id": benchmark_id,
                "benchmark_jsonl": str(bench_path.resolve()),
            },
            "source_corpus": {"retrieval_asset_dir": str(corpus_dir.resolve())},
        }
        prev_json = (
            o_paths.provenance.read_text(encoding="utf-8")
            if o_paths.provenance.is_file()
            else ""
        )
        new_json = json.dumps(prov, indent=2, ensure_ascii=False) + "\n"
        if prev_json != new_json:
            log.info("oracle provenance: %s", o_paths.provenance)
            if not dry_run:
                o_paths.provenance.write_text(new_json, encoding="utf-8")
            summary["updated_files"].append(str(o_paths.provenance))

    d_paths = RouterDatasetPaths(run_root=build_router_dataset_root(rb, router_id))
    if d_paths.manifest.is_file():
        dm = dict(read_router_dataset_manifest(d_paths))
        sb = dict(dm.get("source_benchmark") or {})
        sc = dict(dm.get("source_corpus") or {})
        want_bench = str(bench_path.resolve())
        want_corpus = str(corpus_dir.resolve())
        needs_ds = (
            sb.get("benchmark_path") != want_bench
            or sc.get("retrieval_asset_dir") != want_corpus
        )
        sb["name"] = benchmark_name
        sb["id"] = benchmark_id
        sb["benchmark_path"] = want_bench
        sc["retrieval_asset_dir"] = want_corpus
        dm["source_benchmark"] = sb
        dm["source_corpus"] = sc
        dm["updated_at"] = utc_now_iso()
        if needs_ds:
            log.info("dataset manifest: %s", d_paths.manifest)
            if not dry_run:
                d_paths.manifest.write_text(
                    json.dumps(dm, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            summary["updated_files"].append(str(d_paths.manifest))

    summary["updated_files"] = sorted(set(summary["updated_files"]))
    return summary
