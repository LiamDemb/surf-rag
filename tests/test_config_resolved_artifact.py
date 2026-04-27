"""resolved_config.yaml writing and manifest artifact links."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.config.resolved import write_resolved_config_yaml
from surf_rag.evaluation.e2e_runner import make_e2e_run_paths
from surf_rag.evaluation.manifest import update_manifest_artifacts, write_manifest
from surf_rag.router.policies import RoutingPolicyName


def test_write_resolved_config_yaml_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_pipeline_config(
        root / "configs" / "dev" / "examples" / "surf-bench-200-pipeline.yaml"
    )
    rp = resolve_paths(cfg)
    out = tmp_path / "resolved_config.yaml"
    write_resolved_config_yaml(out, cfg, rp)
    assert out.is_file()
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["paths"]["benchmark_name"] == "surf-bench"
    assert data["resolved_paths"]["corpus_dir"] == str(rp.corpus_dir)


def test_manifest_accepts_resolved_config_artifact(tmp_path: Path) -> None:
    """update_manifest_artifacts links resolved_config relative to run root."""
    root = Path(__file__).resolve().parents[1]
    cfg = load_pipeline_config(
        root / "configs" / "dev" / "examples" / "surf-bench-200-pipeline.yaml"
    )
    rp = resolve_paths(cfg)
    paths = make_e2e_run_paths(
        benchmark_base=tmp_path,
        benchmark_name=cfg.paths.benchmark_name,
        benchmark_id=cfg.paths.benchmark_id,
        policy=RoutingPolicyName.LEARNED_SOFT,
        run_id="run-1",
    )
    write_manifest(
        paths,
        run_id="run-1",
        benchmark=cfg.paths.benchmark_name,
        split="test",
        pipeline_name="test",
        retrieval_asset_dir=str(rp.corpus_dir),
        generator_model="gpt-4o-mini",
        include_graph_provenance=False,
        completion_window="24h",
        artifact_paths={
            "retrieval_results": "retrieval.jsonl",
            "batch_input": "batch_in.jsonl",
            "batch_state": "batch_state.json",
            "generation_answers": "answers.jsonl",
        },
        extra={},
    )
    write_resolved_config_yaml(paths.run_root / "resolved_config.yaml", cfg, rp)
    update_manifest_artifacts(paths, {"resolved_config": "resolved_config.yaml"})
    man = paths.manifest
    assert man.is_file()
    payload = json.loads(man.read_text(encoding="utf-8"))
    assert "artifacts" in payload
    assert payload["artifacts"].get("resolved_config") == "resolved_config.yaml"
