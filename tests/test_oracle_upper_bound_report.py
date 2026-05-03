from __future__ import annotations

import json
from pathlib import Path

from scripts.router import report_oracle_upper_bound


def _write_benchmark(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "What?",
                "gold_answers": ["a"],
                "gold_support_sentences": ["s1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_oracle(router_base: Path, router_id: str) -> None:
    oracle = router_base / router_id / "oracle"
    oracle.mkdir(parents=True, exist_ok=True)
    row = {
        "question_id": "q1",
        "scores": [
            {
                "dense_weight": 0.2,
                "diagnostic_ndcg": {"5": 0.3, "10": 0.4, "20": 0.5},
                "diagnostic_hit": {"5": 1.0, "10": 1.0, "20": 1.0},
                "diagnostic_recall": {"5": 0.5, "10": 0.6, "20": 0.7},
            }
        ],
        "best_bin_index": 0,
    }
    (oracle / "oracle_scores.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )


def _write_split(router_base: Path, router_id: str, qids: list[str]) -> None:
    path = router_base / router_id / "dataset" / "split_question_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": [],
                "dev": [],
                "test": qids,
                "counts": {"train": 0, "dev": 0, "test": len(qids)},
            }
        ),
        encoding="utf-8",
    )


def test_report_oracle_upper_bound_happy_path(monkeypatch, tmp_path: Path) -> None:
    router_base = tmp_path / "router"
    router_id = "rid"
    _write_oracle(router_base, router_id)
    _write_split(router_base, router_id, ["q1"])
    bench = tmp_path / "benchmark.jsonl"
    _write_benchmark(bench)
    out = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "report_oracle_upper_bound.py",
            "--router-id",
            router_id,
            "--router-base",
            str(router_base),
            "--benchmark-path",
            str(bench),
            "--output",
            str(out),
        ],
    )
    rc = report_oracle_upper_bound.main()
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["router_id"] == router_id
    assert payload["evaluated_qids"] == 1
    assert payload["missing_qids"] == []
    assert payload["retrieval_at_k"]["10"]["ndcg"] == 0.4


def test_report_oracle_upper_bound_strict_missing_qid(
    monkeypatch, tmp_path: Path
) -> None:
    router_base = tmp_path / "router"
    router_id = "rid"
    _write_oracle(router_base, router_id)
    _write_split(router_base, router_id, ["q1", "q2"])
    bench = tmp_path / "benchmark.jsonl"
    _write_benchmark(bench)
    out = tmp_path / "report.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "report_oracle_upper_bound.py",
            "--router-id",
            router_id,
            "--router-base",
            str(router_base),
            "--benchmark-path",
            str(bench),
            "--output",
            str(out),
        ],
    )
    rc = report_oracle_upper_bound.main()
    assert rc == 1
    assert not out.exists()
