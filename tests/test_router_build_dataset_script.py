from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd


class _DummyNormalizer:
    def to_json(self) -> dict:
        return {"version": "test"}


def test_build_router_dataset_writes_ignored_report(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts.router import build_router_dataset as mod

    router_base = tmp_path / "router"
    bench_path = tmp_path / "bench.jsonl"
    bench_path.write_text(
        json.dumps({"question_id": "q1", "question": "one", "dataset_source": "nq"})
        + "\n",
        encoding="utf-8",
    )

    oracle_labels = router_base / "r1" / "oracle" / "router_labels.jsonl"
    oracle_labels.parent.mkdir(parents=True, exist_ok=True)
    oracle_labels.write_text(
        json.dumps({"question_id": "q1", "oracle_best_score": 0.0}) + "\n",
        encoding="utf-8",
    )

    def _fake_parse_args() -> Namespace:
        return Namespace(
            config=None,
            router_id="r1",
            benchmark_name="b",
            benchmark_id="v1",
            benchmark_path=bench_path,
            retrieval_asset_dir=None,
            router_base=router_base,
            embedding_model="fake",
            split_seed=7,
            train_ratio=0.6,
            dev_ratio=0.2,
            test_ratio=0.2,
            log_level="INFO",
        )

    def _fake_build_router_dataframe(*args, **kwargs):
        df = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "question": "one",
                    "dataset_source": "nq",
                    "split": "train",
                    "is_valid_for_router_training": False,
                    "feature_set_version": "1",
                    "embedding_model": "fake",
                },
                {
                    "question_id": "q2",
                    "question": "two",
                    "dataset_source": "nq",
                    "split": "dev",
                    "is_valid_for_router_training": True,
                    "feature_set_version": "1",
                    "embedding_model": "fake",
                },
            ]
        )
        return df, _DummyNormalizer(), {"counts": {"train": 1, "dev": 1, "test": 0}}

    monkeypatch.setattr(mod, "load_app_env", lambda: None)
    monkeypatch.setattr(mod, "load_dotenv", lambda: None)
    monkeypatch.setattr(mod, "parse_args", _fake_parse_args)
    monkeypatch.setattr(mod, "build_router_dataframe", _fake_build_router_dataframe)

    rc = mod.main()
    assert rc == 0

    dataset_root = router_base / "r1" / "dataset"
    ignored_path = dataset_root / "ignored_router_questions.json"
    manifest_path = dataset_root / "manifest.json"
    assert ignored_path.is_file()
    assert manifest_path.is_file()

    ignored = json.loads(ignored_path.read_text(encoding="utf-8"))
    assert ignored["ignored_count_total"] == 1
    assert ignored["ignored_question_ids_by_split"]["train"] == ["q1"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count_total"] == 2
    assert manifest["row_count_router_eligible"] == 1
    assert manifest["row_count_router_ignored_all_zero"] == 1
    assert manifest["ignored_questions_report"] == "ignored_router_questions.json"
