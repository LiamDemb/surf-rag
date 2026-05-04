from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import yaml

from surf_rag.evaluation.artifact_paths import e2e_policy_run_dir


def test_llm_judge_submit_dry_run_writes_state(tmp_path: Path) -> None:
    bundle = tmp_path / "surf-bench" / "main"
    bench = bundle / "benchmark" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "What?",
                "gold_answers": ["a"],
                "answer": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_root = e2e_policy_run_dir(
        tmp_path / "bb", "surf-bench", "main", "dense-only", "run1"
    )
    gen = run_root / "generation"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "answers.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "What?",
                "gold_answers": ["a"],
                "answer": "a",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "experiment_id": "t",
                "seed": 1,
                "paths": {
                    "data_base": str(tmp_path),
                    "benchmark_base": str(tmp_path / "bb"),
                    "benchmark_name": "surf-bench",
                    "benchmark_id": "main",
                },
                "generation": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.0,
                    "max_tokens": 64,
                },
                "e2e": {
                    "run_id": "run1",
                    "policy": "dense-only",
                    "completion_window": "24h",
                },
            }
        ),
        encoding="utf-8",
    )

    from scripts.evaluation.llm_judge import cmd_submit

    ns = Namespace(
        config=cfgp,
        dry_run=True,
        force=True,
        benchmark_base=tmp_path / "bb",
        benchmark_name="surf-bench",
        benchmark_id="main",
        benchmark_path=bench,
        split="test",
        run_id="run1",
        policy="dense-only",
        retrieval_asset_dir=tmp_path,
        _loaded_cfg=__import__(
            "surf_rag.config.loader", fromlist=["load_pipeline_config"]
        ).load_pipeline_config(cfgp),
    )
    rc = cmd_submit(ns)
    assert rc == 0
    st = run_root / "llm_judge" / "batch_state.json"
    assert st.is_file()
    data = json.loads(st.read_text(encoding="utf-8"))
    assert data["kind"] == "llm_judge"
    assert data["total_requests"] == 1
