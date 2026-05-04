#!/usr/bin/env python3
"""Answerability audit: submit OpenAI Batch, collect verdicts, build mask + manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from surf_rag.config.env import apply_pipeline_env_from_config, load_app_env
from surf_rag.config.loader import load_pipeline_config, resolve_paths
from surf_rag.core.openai_batch_submit import upload_sharded_chat_completion_batches
from surf_rag.evaluation.answerability_batch import (
    build_answerability_body,
    load_prompt_template,
    parse_answerability_batch_output_line,
)
from surf_rag.evaluation.answerability_batch_ids import (
    make_answerability_custom_id,
    parse_answerability_custom_id,
)
from surf_rag.evaluation.answerability_layout import (
    answerability_audit_dir,
    answerability_batch_state_path,
    answerability_manifest_path,
    answerability_mask_path,
    answerability_verdicts_path,
    bundle_root_from_benchmark_jsonl,
)
from surf_rag.evaluation.answerability_types import (
    audit_entries_from_verdicts,
    build_balance_mask,
    build_manifest_document,
    build_mask_document,
    iter_verdicts_jsonl,
    write_json,
)

log = logging.getLogger(__name__)

STATE_VERSION = 1


def _load_raw_answerability(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(raw.get("answerability") or {})


def _benchmark_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _verdict_qids(path: Path) -> Set[str]:
    s: set[str] = set()
    for row in iter_verdicts_jsonl(path):
        qid = str(row.get("question_id", "") or "").strip()
        if qid and row.get("answerable") is not None:
            s.add(qid)
    return s


def cmd_submit(args: argparse.Namespace) -> int:
    load_app_env()
    cfg_path = Path(args.config).resolve()
    cfg = load_pipeline_config(cfg_path)
    apply_pipeline_env_from_config(cfg)
    rp = resolve_paths(cfg)
    bench_path = Path(rp.benchmark_path)
    if not bench_path.is_file():
        log.error("Benchmark not found: %s", bench_path)
        return 2

    ans_cfg = _load_raw_answerability(cfg_path)
    model = str(ans_cfg.get("model") or cfg.generation.model or "gpt-4o-mini")
    temperature = float(ans_cfg.get("temperature", cfg.generation.temperature))
    max_tokens = int(ans_cfg.get("max_tokens", 256))
    completion_window = str(
        ans_cfg.get("completion_window") or cfg.e2e.completion_window or "24h"
    )
    prompt_path = Path(
        str(ans_cfg.get("prompt_file") or "prompts/answerability_audit.txt")
    ).resolve()
    if not prompt_path.is_file():
        log.error("Prompt file not found: %s", prompt_path)
        return 2
    template = load_prompt_template(prompt_path)

    audit_dir = answerability_audit_dir(bundle_root_from_benchmark_jsonl(bench_path))
    audit_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = answerability_verdicts_path(bench_path)
    state_path = answerability_batch_state_path(bench_path)

    samples = _benchmark_rows(bench_path)
    bench_qids = {
        str(r.get("question_id", "") or "").strip()
        for r in samples
        if r.get("question_id")
    }

    if verdicts_path.is_file() and not args.force:
        done = _verdict_qids(verdicts_path)
        if done >= bench_qids:
            log.info(
                "Verdicts already cover all %d benchmark questions; use --force to re-submit.",
                len(bench_qids),
            )
            return 0

    records: list[tuple[str, dict[str, Any]]] = []
    for row in samples:
        qid = str(row.get("question_id", "") or "").strip()
        if not qid:
            continue
        question = str(row.get("question", "") or "")
        gold_answers = list(row.get("gold_answers") or [])
        sents = list(row.get("gold_support_sentences") or [])
        support = "\n".join(str(s) for s in sents)
        body = build_answerability_body(
            question=question,
            gold_answers=[str(a) for a in gold_answers],
            gold_support=support,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_template=template,
        )
        records.append((make_answerability_custom_id(qid), body))

    shard_dir = audit_dir / "batch_shards"
    shards = upload_sharded_chat_completion_batches(
        records,
        work_dir=shard_dir,
        shard_filename_prefix="batch_input_answerability",
        completion_window=completion_window,
        dry_run=args.dry_run,
        batch_metadata={
            "description": "answerability-audit",
            "benchmark": str(bench_path),
        },
    )

    prompt_id = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    state: dict[str, Any] = {
        "schema_version": STATE_VERSION,
        "kind": "answerability_audit",
        "benchmark_path": str(bench_path.resolve()),
        "audit_dir": str(audit_dir.resolve()),
        "shards": shards,
        "total_requests": len(records),
        "answerability": {
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
    load_app_env()
    cfg_path = Path(args.config).resolve()
    cfg = load_pipeline_config(cfg_path)
    rp = resolve_paths(cfg)
    bench_path = Path(rp.benchmark_path)
    state_path = answerability_batch_state_path(bench_path)
    if not state_path.is_file():
        log.error("Missing batch state: %s", state_path)
        return 2

    state = json.loads(state_path.read_text(encoding="utf-8"))
    audit_dir = Path(state["audit_dir"])
    verdicts_path = answerability_verdicts_path(bench_path)
    samples = _benchmark_rows(bench_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        log.error("OPENAI_API_KEY is not set.")
        return 1

    raw_lines: list[str] = []
    client = OpenAI(api_key=api_key) if api_key else None
    for shard in state.get("shards") or []:
        batch_id = shard.get("batch_id")
        if not batch_id or str(batch_id).startswith("dry-run"):
            continue
        if client is None:
            continue
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            log.warning(
                "Shard batch %s not completed (status=%s).", batch_id, batch.status
            )
            continue
        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file", None
        )
        if not output_file_id:
            log.warning("Shard batch %s has no output file.", batch_id)
            continue
        content = client.files.content(output_file_id)
        text = content.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        for ln in text.strip().split("\n"):
            if ln.strip():
                raw_lines.append(ln)

    if not raw_lines:
        log.error("No batch output lines collected; batches may still be running.")
        return 1

    raw_path = audit_dir / "batch_output_raw.jsonl"
    raw_path.write_text(
        "\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8"
    )

    verdict_rows: list[dict[str, Any]] = []
    for sample in samples:
        qid = str(sample.get("question_id", "") or "").strip()
        if not qid:
            continue
        verdict_rows.append(
            {
                "question_id": qid,
                "answerable": None,
                "dataset_source": str(sample.get("dataset_source", "") or "").strip()
                or "unknown",
                "question": str(sample.get("question", "") or ""),
            }
        )

    by_qid_row = {r["question_id"]: r for r in verdict_rows}
    for ln in raw_lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        cid = str(obj.get("custom_id", "") or "")
        qid = parse_answerability_custom_id(cid)
        if not qid or qid not in by_qid_row:
            continue
        ab = parse_answerability_batch_output_line(obj)
        by_qid_row[qid]["answerable"] = ab

    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    with verdicts_path.open("w", encoding="utf-8") as out:
        for row in verdict_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %s", verdicts_path)
    return 0


def cmd_balance(args: argparse.Namespace) -> int:
    load_app_env()
    cfg_path = Path(args.config).resolve()
    cfg = load_pipeline_config(cfg_path)
    rp = resolve_paths(cfg)
    bench_path = Path(rp.benchmark_path)
    verdicts_path = answerability_verdicts_path(bench_path)
    if not verdicts_path.is_file():
        log.error("Missing verdicts: %s (run collect first)", verdicts_path)
        return 2

    ans_cfg = _load_raw_answerability(cfg_path)
    balance_cfg = dict(ans_cfg.get("balance") or {})
    balance_enabled = bool(balance_cfg.get("enabled", False))
    seed = int(balance_cfg.get("seed", cfg.seed))
    policy = str(balance_cfg.get("policy") or "equal_per_source_min")

    verdict_rows = iter_verdicts_jsonl(verdicts_path)
    audit_entries = audit_entries_from_verdicts(verdict_rows)
    balance_entries: list[dict[str, str]] = []
    if balance_enabled:
        balance_entries = build_balance_mask(verdict_rows, seed=seed, policy=policy)

    mask_doc = build_mask_document(
        audit_entries=audit_entries, balance_entries=balance_entries
    )
    mask_path = answerability_mask_path(bench_path)
    write_json(mask_path, mask_doc)

    state_path = answerability_batch_state_path(bench_path)
    audit_model = "unknown"
    prompt_id = "unknown"
    if state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        meta = st.get("answerability") or {}
        audit_model = str(meta.get("model") or "unknown")
        prompt_id = str(meta.get("prompt_id") or "unknown")

    manifest = build_manifest_document(
        benchmark_path=bench_path,
        audit_model=audit_model,
        prompt_id=prompt_id,
        verdict_rows=verdict_rows,
        mask_entries=mask_doc["entries"],
        balance_enabled=balance_enabled,
        balance_policy=policy,
        balance_seed=seed if balance_enabled else None,
    )
    write_json(answerability_manifest_path(bench_path), manifest)
    log.info("Wrote %s and %s", mask_path, answerability_manifest_path(bench_path))
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML with paths + answerability block",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit", help="Build batch shards and submit to OpenAI")
    p_sub.add_argument("--dry-run", action="store_true")
    p_sub.add_argument(
        "--force", action="store_true", help="Re-submit even if verdicts exist"
    )
    p_sub.set_defaults(func=cmd_submit)

    p_col = sub.add_parser("collect", help="Download batch outputs → verdicts.jsonl")
    p_col.add_argument(
        "--dry-run", action="store_true", help="Reserved; collect needs API"
    )
    p_col.set_defaults(func=cmd_collect)

    p_bal = sub.add_parser(
        "balance", help="Build mask.json + manifest.json from verdicts"
    )
    p_bal.set_defaults(func=cmd_balance)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
