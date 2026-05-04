from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_answerability_submit_dry_run_writes_state_and_shards(tmp_path: Path) -> None:
    bundle = tmp_path / "surf-bench" / "main"
    bench = bundle / "benchmark" / "benchmark.jsonl"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "What?",
                "gold_answers": ["a"],
                "gold_support_sentences": ["evidence"],
                "dataset_source": "nq",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfgp = tmp_path / "audit.yaml"
    cfgp.write_text(
        yaml.safe_dump(
            {
                "schema_version": "surf-rag/pipeline/v1",
                "experiment_id": "t",
                "seed": 1,
                "paths": {
                    "data_base": str(tmp_path),
                    "benchmark_base": str(tmp_path),
                    "benchmark_name": "surf-bench",
                    "benchmark_id": "main",
                },
                "generation": {"model": "gpt-4o-mini", "temperature": 0.0},
                "e2e": {"completion_window": "24h"},
                "answerability": {
                    "prompt_file": str(
                        Path(__file__).resolve().parents[1]
                        / "prompts"
                        / "answerability_audit.txt"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    from scripts.audit.benchmark_answerability import cmd_submit
    from argparse import Namespace

    rc = cmd_submit(Namespace(config=cfgp, dry_run=True, force=False))
    assert rc == 0

    state_path = bundle / "audit" / "answerability" / "batch_state.json"
    assert state_path.is_file()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["kind"] == "answerability_audit"
    assert state["total_requests"] == 1
    assert state["shards"][0]["batch_id"].startswith("dry-run")
