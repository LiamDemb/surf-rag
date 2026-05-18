from __future__ import annotations

from pathlib import Path

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.schema import ResultsSection


def test_load_results_section(tmp_path: Path) -> None:
    cfgp = tmp_path / "results.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "paths": {
                    "benchmark_name": "bench",
                    "benchmark_id": "v1",
                    "router_id": "rid",
                },
                "results": {
                    "bundle_id": "final-v1",
                    "split": "test",
                    "policies": {
                        "dense-only": "e2e-001",
                        "learned-soft": {
                            "run_id": "e2e-002",
                            "router_role": "regressor",
                        },
                    },
                    "router": {
                        "regressor": {
                            "architecture_id": "rg-001",
                            "task_type": "regression",
                        },
                    },
                    "artifacts": [
                        {"id": "bench_splits", "kind": "table"},
                        {
                            "id": "pairwise_a",
                            "kind": "figure",
                            "figure": "pairwise_retrieval",
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_pipeline_config(cfgp)
    assert cfg.results.bundle_id == "final-v1"
    assert cfg.results.split == "test"
    assert cfg.results.policies["dense-only"].run_id == "e2e-001"
    assert cfg.results.policies["learned-soft"].router_role == "regressor"
    assert cfg.results.router["regressor"].architecture_id == "rg-001"
    assert len(cfg.results.artifacts) == 2
    assert cfg.results.artifacts[0].id == "bench_splits"


def test_results_section_defaults() -> None:
    r = ResultsSection()
    assert r.output_root == "results"
    assert r.oracle.k == 5
