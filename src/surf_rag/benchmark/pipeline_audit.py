from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_pipeline_run_id(explicit_run_id: str | None = None) -> str:
    if explicit_run_id and str(explicit_run_id).strip():
        return str(explicit_run_id).strip()
    env_run_id = os.getenv("PIPELINE_RUN_ID", "").strip()
    if env_run_id:
        return env_run_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _initial_report(*, run_id: str, benchmark_path: Path) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "benchmark_path": str(benchmark_path),
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "steps": [],
        "totals": {
            "start": 0,
            "end": 0,
            "net_added": 0,
            "net_dropped": 0,
        },
    }


def _step_order_key(step_name: str) -> int:
    fixed = {
        "ingest": 0,
        "align_2wiki_support": 1,
        "build_corpus": 2,
        "filter_benchmark": 3,
    }
    return fixed.get(step_name, 100)


def _compute_totals(steps: List[Dict[str, Any]]) -> Dict[str, int]:
    if not steps:
        return {"start": 0, "end": 0, "net_added": 0, "net_dropped": 0}
    start = int(steps[0]["before"])
    end = int(steps[-1]["after"])
    return {
        "start": start,
        "end": end,
        "net_added": max(end - start, 0),
        "net_dropped": max(start - end, 0),
    }


def write_pipeline_step_report(
    *,
    benchmark_path: Path,
    step_name: str,
    before: int,
    after: int,
    run_id: str,
    details: Dict[str, Any] | None = None,
) -> Path:
    benchmark_dir = benchmark_path.parent
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    report_path = benchmark_dir / f"pipeline_counts.{run_id}.json"
    latest_path = benchmark_dir / "pipeline_counts.latest.json"

    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = _initial_report(run_id=run_id, benchmark_path=benchmark_path)

    steps = list(report.get("steps", []))
    now = _utc_now_iso()
    entry = {
        "name": step_name,
        "before": int(before),
        "after": int(after),
        "added": max(int(after) - int(before), 0),
        "dropped": max(int(before) - int(after), 0),
        "recorded_at": now,
    }
    if details:
        entry["details"] = details

    existing_index = next(
        (i for i, step in enumerate(steps) if str(step.get("name")) == step_name), -1
    )
    if existing_index >= 0:
        steps[existing_index] = entry
    else:
        steps.append(entry)

    indexed_steps = list(enumerate(steps))
    indexed_steps.sort(
        key=lambda pair: (_step_order_key(str(pair[1].get("name", ""))), pair[0])
    )
    steps = [step for _, step in indexed_steps]
    report["steps"] = steps
    report["updated_at"] = now
    report["totals"] = _compute_totals(steps)

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    report_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")
    return report_path
