from __future__ import annotations

import json
from pathlib import Path

from surf_rag.benchmark.pipeline_audit import (
    resolve_pipeline_run_id,
    write_pipeline_step_report,
)


def test_write_pipeline_step_report_creates_and_updates_report(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark" / "benchmark.jsonl"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("", encoding="utf-8")

    run_id = "run-abc"
    report_path = write_pipeline_step_report(
        benchmark_path=benchmark_path,
        step_name="ingest",
        before=0,
        after=1000,
        run_id=run_id,
        details={"added": 1000},
    )
    write_pipeline_step_report(
        benchmark_path=benchmark_path,
        step_name="filter_benchmark",
        before=1000,
        after=800,
        run_id=run_id,
        details={"dropped_by_source": {"2wiki": 200}},
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert [s["name"] for s in payload["steps"]] == ["ingest", "filter_benchmark"]
    assert payload["totals"] == {
        "start": 0,
        "end": 800,
        "net_added": 800,
        "net_dropped": 0,
    }
    assert (benchmark_path.parent / "pipeline_counts.latest.json").is_file()


def test_write_pipeline_step_report_replaces_same_step(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text("", encoding="utf-8")
    run_id = "run-xyz"
    report_path = write_pipeline_step_report(
        benchmark_path=benchmark_path,
        step_name="ingest",
        before=100,
        after=200,
        run_id=run_id,
    )
    write_pipeline_step_report(
        benchmark_path=benchmark_path,
        step_name="ingest",
        before=100,
        after=250,
        run_id=run_id,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(payload["steps"]) == 1
    assert payload["steps"][0]["after"] == 250


def test_resolve_pipeline_run_id_prefers_explicit(monkeypatch) -> None:
    monkeypatch.setenv("PIPELINE_RUN_ID", "env-run")
    assert resolve_pipeline_run_id("explicit-run") == "explicit-run"
    assert resolve_pipeline_run_id(None) == "env-run"
