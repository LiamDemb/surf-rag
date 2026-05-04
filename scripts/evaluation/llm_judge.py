#!/usr/bin/env python3
"""LLM-as-judge for one E2E run: submit batch, collect verdicts, merge into metrics.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from surf_rag.config.env import apply_pipeline_env_from_config, load_app_env
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.config.merge import merge_e2e_common_args
from surf_rag.core.openai_batch_submit import upload_sharded_chat_completion_batches
from surf_rag.evaluation.answerability_types import write_json
from surf_rag.evaluation.e2e_policies import parse_routing_policy
from surf_rag.evaluation.e2e_runner import make_e2e_run_paths
from surf_rag.evaluation.llm_judge_batch import (
    build_judge_body,
    parse_judge_batch_output_line,
    parse_llm_judge_custom_id,
    make_llm_judge_custom_id,
)
from surf_rag.evaluation.llm_judge_merge import merge_llm_judge_verdicts_into_metrics
from surf_rag.evaluation.manifest import update_manifest_artifacts
from surf_rag.evaluation.run_artifacts import RunArtifactPaths

log = logging.getLogger(__name__)
STATE_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_paths(args: argparse.Namespace) -> RunArtifactPaths:
    bb = args.benchmark_base
    policy = parse_routing_policy(args.policy)
    return make_e2e_run_paths(
        benchmark_base=bb,
        benchmark_name=args.benchmark_name,
        benchmark_id=args.benchmark_id,
        policy=policy,
        run_id=args.run_id,
    )


def _load_answers(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cmd_submit(args: argparse.Namespace) -> int:
    paths = _run_paths(args)
    answers_path = paths.generation_answers_jsonl()
    if not answers_path.is_file():
        log.error("Missing answers: %s", answers_path)
        return 2

    judge_dir = paths.run_root / "llm_judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    state_path = judge_dir / "batch_state.json"
    verdicts_path = judge_dir / "verdicts.jsonl"

    if verdicts_path.is_file() and not args.force:
        log.info("Verdicts exist at %s; use --force to re-submit.", verdicts_path)
        return 0

    prompt_path = (REPO_ROOT / "prompts" / "llm_judge.txt").resolve()
    if not prompt_path.is_file():
        log.error("Judge prompt not found: %s", prompt_path)
        return 2
    template = prompt_path.read_text(encoding="utf-8")

    cfg = args._loaded_cfg
    model = str(cfg.generation.model or "gpt-4o-mini")
    temperature = float(cfg.generation.temperature)
    max_tokens = int(getattr(cfg.generation, "max_tokens", 256) or 256)
    completion_window = str(cfg.e2e.completion_window or "24h")

    rows = _load_answers(answers_path)
    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid:
            continue
        body = build_judge_body(
            question=str(row.get("question", "") or ""),
            gold_answers=list(row.get("gold_answers") or []),
            prediction=str(row.get("answer", "") or ""),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_template=template,
        )
        records.append((make_llm_judge_custom_id(qid), body))

    shard_dir = judge_dir / "batch_shards"
    shards = upload_sharded_chat_completion_batches(
        records,
        work_dir=shard_dir,
        shard_filename_prefix="batch_input_llm_judge",
        completion_window=completion_window,
        dry_run=args.dry_run,
        batch_metadata={
            "description": "llm-judge",
            "run_root": str(paths.run_root),
        },
    )

    prompt_id = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    state = {
        "schema_version": STATE_VERSION,
        "kind": "llm_judge",
        "run_root": str(paths.run_root.resolve()),
        "judge_dir": str(judge_dir.resolve()),
        "shards": shards,
        "total_requests": len(records),
        "judge": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_path": str(prompt_path),
            "prompt_id": prompt_id,
        },
    }
    write_json(state_path, state)
    for s in shards:
        print(s.get("batch_id"))
    log.info("Wrote %s", state_path)
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    paths = _run_paths(args)
    judge_dir = paths.run_root / "llm_judge"
    state_path = judge_dir / "batch_state.json"
    if not state_path.is_file():
        log.error("Missing %s", state_path)
        return 2
    state = json.loads(state_path.read_text(encoding="utf-8"))
    answers_path = paths.generation_answers_jsonl()
    samples = _load_answers(answers_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY is not set.")
        return 1
    client = OpenAI(api_key=api_key)

    raw_lines: list[str] = []
    for shard in state.get("shards") or []:
        batch_id = shard.get("batch_id")
        if not batch_id or str(batch_id).startswith("dry-run"):
            continue
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            log.warning("Batch %s status=%s", batch_id, batch.status)
            continue
        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file", None
        )
        if not output_file_id:
            continue
        content = client.files.content(output_file_id)
        text = content.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        for ln in text.strip().split("\n"):
            if ln.strip():
                raw_lines.append(ln)

    if not raw_lines:
        log.error("No batch output collected.")
        return 1

    judge_dir.mkdir(parents=True, exist_ok=True)
    raw_path = judge_dir / "batch_output_raw.jsonl"
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    verdict_rows: list[dict[str, Any]] = []
    for row in samples:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid:
            continue
        verdict_rows.append({"question_id": qid, "correct": None})

    by_qid = {r["question_id"]: r for r in verdict_rows}
    for ln in raw_lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        cid = str(obj.get("custom_id", "") or "")
        qid = parse_llm_judge_custom_id(cid)
        if not qid or qid not in by_qid:
            continue
        cor = parse_judge_batch_output_line(obj)
        by_qid[qid]["correct"] = cor

    verdicts_path = judge_dir / "verdicts.jsonl"
    with verdicts_path.open("w", encoding="utf-8") as out:
        for row in verdict_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %s", verdicts_path)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    paths = _run_paths(args)
    metrics_path = paths.run_root / "metrics.json"
    if not metrics_path.is_file():
        log.error("Missing metrics: %s", metrics_path)
        return 2
    verdicts_path = paths.run_root / "llm_judge" / "verdicts.jsonl"
    if not verdicts_path.is_file():
        log.error("Missing verdicts: %s", verdicts_path)
        return 2

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    verdicts_by_qid: dict[str, bool] = {}
    with verdicts_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("question_id", "") or "").strip()
            c = row.get("correct")
            if qid and c is True:
                verdicts_by_qid[qid] = True
            elif qid and c is False:
                verdicts_by_qid[qid] = False

    merge_llm_judge_verdicts_into_metrics(metrics, verdicts_by_qid)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rel = str(verdicts_path.relative_to(paths.run_root))
    update_manifest_artifacts(paths, {"llm_judge_verdicts": rel})
    log.info("Updated %s", metrics_path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--benchmark-base", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("submit")
    p_s.add_argument("--dry-run", action="store_true")
    p_s.add_argument("--force", action="store_true")
    p_s.set_defaults(func=cmd_submit)

    sub.add_parser("collect").set_defaults(func=cmd_collect)
    sub.add_parser("merge").set_defaults(func=cmd_merge)

    args = ap.parse_args()
    load_app_env()
    cfg_path = Path(args.config).resolve()
    cfg = load_pipeline_config(cfg_path)
    apply_pipeline_env_from_config(cfg)
    ns = Namespace(
        benchmark_base=args.benchmark_base,
        benchmark_name=None,
        benchmark_id=None,
        benchmark_path=None,
        split=None,
        run_id=args.run_id,
        policy=args.policy,
        retrieval_asset_dir=None,
    )
    merge_e2e_common_args(ns, cfg)
    if not ns.run_id or not ns.policy:
        log.error(
            "e2e.run_id and e2e.policy must be set in config or pass --run-id / --policy."
        )
        return 2
    for k, v in vars(ns).items():
        setattr(args, k, v)
    args._loaded_cfg = cfg

    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
