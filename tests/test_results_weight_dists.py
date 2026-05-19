from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import yaml

from surf_rag.config.loader import load_pipeline_config
from surf_rag.config.schema import ResultsArtifactSpec
from surf_rag.evaluation.router_model_artifacts import make_router_model_paths_for_cli
from surf_rag.results.bundle import ResultsBundle
from surf_rag.results.figures.weight_dists import render_weight_dists
from surf_rag.viz.theme import apply_theme


def _write_fixture(tmp_path: Path) -> Path:
    bench = tmp_path / "benchmarks" / "bench" / "v1"
    bench_jsonl = bench / "benchmark" / "benchmark.jsonl"
    bench_jsonl.parent.mkdir(parents=True, exist_ok=True)
    bench_jsonl.write_text(
        "\n".join(
            [
                '{"question_id":"q_test_nq","dataset_source":"nq"}',
                '{"question_id":"q_test_2wiki","dataset_source":"2wiki"}',
                '{"question_id":"q_train","dataset_source":"nq"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    router_base = tmp_path / "router"
    rid_dir = router_base / "rid"
    ds_dir = rid_dir / "dataset"
    oracle_dir = rid_dir / "oracle"
    ds_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "split_question_ids.json").write_text(
        json.dumps(
            {
                "train": ["q_train"],
                "dev": [],
                "test": ["q_test_nq", "q_test_2wiki"],
            }
        ),
        encoding="utf-8",
    )
    (oracle_dir / "oracle_scores.jsonl").write_text(
        '{"question_id":"q_test_nq"}\n', encoding="utf-8"
    )
    (oracle_dir / "retrieval_graph.jsonl").write_text(
        '{"question_id":"q_test_nq"}\n', encoding="utf-8"
    )

    paths = make_router_model_paths_for_cli(
        "rid",
        router_base=router_base,
        input_mode="embedding",
        router_architecture_id="rg-001",
        router_task_type="regression",
    )
    pred_path = paths.predictions("test")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        "\n".join(
            [
                json.dumps({"question_id": "q_test_nq", "predicted_weight": 0.15}),
                json.dumps({"question_id": "q_test_2wiki", "predicted_weight": 0.85}),
                json.dumps({"question_id": "q_train", "predicted_weight": 0.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = bench / "evaluations" / "dense-only" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text('{"per_question":[]}', encoding="utf-8")

    cfg_path = tmp_path / "results.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "data_base": str(tmp_path),
                    "benchmark_base": str(tmp_path / "benchmarks"),
                    "router_base": str(router_base),
                    "benchmark_name": "bench",
                    "benchmark_id": "v1",
                    "router_id": "rid",
                },
                "results": {
                    "bundle_id": "t",
                    "output_root": str(tmp_path / "out"),
                    "split": "test",
                    "image_format": "png",
                    "policies": {"dense-only": {"run_id": "run1"}},
                    "router": {
                        "regressor": {
                            "architecture_id": "rg-001",
                            "input_mode": "embedding",
                            "task_type": "regression",
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_weight_dists_test_split_grouped_bars(tmp_path: Path) -> None:
    apply_theme(dpi=100)
    cfg = load_pipeline_config(_write_fixture(tmp_path))
    bundle = ResultsBundle.from_config(cfg)
    assert bundle.results.split == "test"
    assert bundle.split_qids == {"q_test_nq", "q_test_2wiki"}

    spec = ResultsArtifactSpec(id="weight_dists", kind="figure", figure="weight_dists")
    paths = render_weight_dists(bundle, spec)

    img = Path(paths["image"])
    assert img.is_file()
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    assert meta["split"] == "test"
    assert meta["n_nq"] == 1
    assert meta["n_2wiki"] == 1
    assert meta["layout"] == "grouped_bars"
    assert meta["bin_width"] == 0.1
