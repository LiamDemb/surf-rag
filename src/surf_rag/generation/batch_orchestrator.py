"""Batch submit/collect for a single pipeline run."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI
from tqdm.auto import tqdm

from surf_rag.core.openai_batch_limits import batch_limit_requests
from surf_rag.core.prompts import get_generator_prompt
from surf_rag.evaluation.manifest import update_manifest_artifacts, write_manifest
from surf_rag.evaluation.retrieval_jsonl import write_retrieval_line
from surf_rag.evaluation.run_artifacts import (
    RunArtifactPaths,
    build_run_root,
    default_evaluation_base,
    make_generation_custom_id,
    parse_generation_custom_id,
)
from surf_rag.generation.batch import (
    build_batch_line,
    build_completion_body,
    parse_generation_output,
)
from surf_rag.generation.generator_tool import GenerationParseResult
from surf_rag.generation.batch_compiler import BatchRequestRecord
from surf_rag.generation.prompt_renderer import PromptRenderer
from surf_rag.strategies.factory import build_dense_retriever, build_graph_retriever

logger = logging.getLogger(__name__)

BATCH_LIMIT_BYTES = 200 * 1024 * 1024
STATE_VERSION = 2


def _normalize_pipeline_name(name: str) -> str:
    p = name.strip().lower()
    if p in ("dense", "d"):
        return "dense"
    if p in ("graph", "g"):
        return "graph"
    raise ValueError(f"Unknown pipeline_name={name!r}; expected 'dense' or 'graph'.")


def _load_benchmark(path: Path, limit: int | None) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
            if limit and len(samples) >= limit:
                break
    return samples


def load_question_ids_from_labeled_jsonl(path: Path) -> set[str]:
    """Load unique question_id values from a labeled JSONL (e.g. labeled_test.jsonl)."""
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id", "")
            if qid:
                ids.add(str(qid))
    return ids


def _load_completed_question_ids_from_answers(answers_path: Path) -> Set[str]:
    """Question IDs that already have a successful collected generation row (resume).

    Rows with ``generation_parse_error`` set are not considered complete and will
    be re-submitted on a new batch prep.
    """
    done: Set[str] = set()
    if not answers_path.is_file():
        return done
    with answers_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("question_id", "")).strip()
            if not qid or "answer" not in row:
                continue
            if row.get("generation_parse_error"):
                continue
            done.add(qid)
    return done


def _default_include_graph_provenance() -> bool:
    return os.getenv("INCLUDE_GRAPH_PATHS_IN_PROMPT", "").strip().lower() == "true"


def submit_batches(
    benchmark_path: Path,
    *,
    retrieval_asset_dir: Path,
    paths: RunArtifactPaths,
    benchmark: str,
    split: str,
    pipeline_name: str,
    run_id: str,
    limit: Optional[int] = None,
    only_question_ids: Optional[set[str]] = None,
    completion_window: str = "24h",
    include_graph_provenance: Optional[bool] = None,
    dry_run: bool = False,
) -> int:
    """Run retrieval for one branch, compile batch JSONL, submit shard(s) under run_root/batch/."""
    pipeline = _normalize_pipeline_name(pipeline_name)
    if include_graph_provenance is None:
        include_graph_provenance = _default_include_graph_provenance()

    paths.ensure_dirs()
    asset_dir = retrieval_asset_dir.resolve()
    run_root = paths.run_root.resolve()

    if not dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set (required unless dry_run).")
        return 1

    effective_limit = None if only_question_ids else limit
    samples = _load_benchmark(benchmark_path, limit=effective_limit)
    if only_question_ids:
        before = len(samples)
        samples = [
            s
            for s in samples
            if str(s.get("question_id", "")).strip() in only_question_ids
        ]
        logger.info(
            "Filtered benchmark %d → %d rows matching %d question IDs.",
            before,
            len(samples),
            len(only_question_ids),
        )
    if not samples:
        logger.error(
            "No samples to process (check --limit, --only-question-ids-from, and benchmark file)."
        )
        return 1

    logger.info("Loaded %d benchmark samples.", len(samples))

    answers_path = paths.generation_answers_jsonl()
    completed_qids = _load_completed_question_ids_from_answers(answers_path)
    if completed_qids:
        logger.info(
            "Found %d already-collected questions in %s (resume).",
            len(completed_qids),
            answers_path,
        )

    base_prompt = get_generator_prompt()
    model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("GENERATOR_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("GENERATOR_MAX_TOKENS", "512"))
    renderer = PromptRenderer(
        base_prompt=base_prompt, include_graph_provenance=include_graph_provenance
    )

    logger.info("Building %s retriever from %s ...", pipeline, asset_dir)
    if pipeline == "dense":
        retriever = build_dense_retriever(str(asset_dir))
    else:
        retriever = build_graph_retriever(str(asset_dir))

    write_manifest(
        paths,
        run_id=run_id,
        benchmark=benchmark,
        split=split,
        pipeline_name=pipeline,
        retrieval_asset_dir=str(asset_dir),
        generator_model=model_id,
        include_graph_provenance=include_graph_provenance,
        completion_window=completion_window,
        artifact_paths={
            "retrieval_results": str(
                paths.retrieval_results_jsonl().relative_to(run_root)
            ),
            "batch_input": str(paths.batch_input_jsonl().relative_to(run_root)),
            "batch_state": str(paths.batch_state_json().relative_to(run_root)),
            "generation_answers": str(
                paths.generation_answers_jsonl().relative_to(run_root)
            ),
        },
    )

    records: List[BatchRequestRecord] = []
    skipped = 0
    pending_samples = [
        s
        for s in samples
        if s.get("question", "").strip()
        and str(s.get("question_id", "")).strip() not in completed_qids
    ]

    progress = None
    if pending_samples:
        progress = tqdm(
            total=len(pending_samples),
            desc="Retrieval + batch prep",
            unit="question",
            dynamic_ncols=True,
        )

    retrieval_fp = paths.retrieval_results_jsonl().open("a", encoding="utf-8")
    try:
        for sample in samples:
            question = sample.get("question", "").strip()
            qid = str(sample.get("question_id", "")).strip()
            if not question:
                continue
            if qid in completed_qids:
                skipped += 1
                continue

            rr = retriever.retrieve(question)
            write_retrieval_line(retrieval_fp, rr, qid)

            custom_id = make_generation_custom_id(
                run_id, benchmark, split, pipeline, qid
            )
            messages = renderer.to_messages(question, rr)
            body = build_completion_body(
                messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            records.append(BatchRequestRecord(custom_id=custom_id, body=body))

            if progress is not None:
                progress.update(1)
    finally:
        retrieval_fp.close()

    if progress is not None:
        progress.close()

    if skipped:
        logger.info(
            "Skipped %d already-collected questions (%d new to process).",
            skipped,
            len(samples) - skipped,
        )

    return finalize_batch_submission(
        records,
        samples,
        benchmark_path=benchmark_path,
        paths=paths,
        benchmark=benchmark,
        split=split,
        pipeline_name=pipeline,
        run_id=run_id,
        retrieval_asset_dir=asset_dir,
        completion_window=completion_window,
        dry_run=dry_run,
    )


def finalize_batch_submission(
    records: List[BatchRequestRecord],
    samples: list[dict],
    *,
    benchmark_path: Path,
    paths: RunArtifactPaths,
    benchmark: str,
    split: str,
    pipeline_name: str,
    run_id: str,
    retrieval_asset_dir: Path,
    completion_window: str,
    dry_run: bool,
) -> int:
    """Write ``batch_input.jsonl``, shard, optionally upload, write ``batch_state.json``."""
    run_root = paths.run_root.resolve()
    asset_dir = retrieval_asset_dir.resolve()

    batch_input_path = paths.batch_input_jsonl()
    with batch_input_path.open("w", encoding="utf-8") as bf:
        for rec in records:
            line = build_batch_line(rec.custom_id, rec.body)
            bf.write(json.dumps(line, ensure_ascii=False) + "\n")

    if not records:
        logger.info("All questions already collected. Nothing to submit.")
        state = _build_state(
            paths=paths,
            benchmark_path=benchmark_path,
            samples=samples,
            shards=[],
            retrieval_asset_dir=asset_dir,
            benchmark=benchmark,
            split=split,
            pipeline_name=pipeline_name,
            run_id=run_id,
        )
        paths.batch_state_json().write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0

    logger.info("Recorded %d requests. Sharding batch JSONL...", len(records))

    limit_requests = batch_limit_requests()
    shards: list[dict] = []
    shard_idx = 0
    shard_count = 0
    shard_bytes = 0
    shard_file = paths.batch_shard_path(shard_idx)
    batch_out = shard_file.open("w", encoding="utf-8")

    def flush_shard() -> None:
        nonlocal shard_idx, shard_count, shard_bytes, shard_file, batch_out
        if shard_count == 0:
            return
        batch_out.close()
        if dry_run:
            logger.info(
                "[dry-run] Would upload shard %d (%d requests) at %s",
                shard_idx,
                shard_count,
                shard_file,
            )
            shards.append(
                {
                    "batch_id": f"dry-run-{shard_idx}",
                    "input_path": str(shard_file),
                    "request_count": shard_count,
                }
            )
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("Uploading shard %d (%d requests)...", shard_idx, shard_count)
            with shard_file.open("rb") as r:
                uploaded = client.files.create(file=r, purpose="batch")
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
                metadata={
                    "description": f"Generation ({pipeline_name})",
                    "benchmark": benchmark,
                    "split": split,
                    "run_id": run_id,
                    "shard": str(shard_idx),
                },
            )
            shards.append(
                {
                    "batch_id": batch.id,
                    "input_path": str(shard_file),
                    "request_count": shard_count,
                }
            )
            logger.info("Shard %d batch created: %s", shard_idx, batch.id)
        shard_idx += 1
        shard_count = 0
        shard_bytes = 0
        shard_file = paths.batch_shard_path(shard_idx)
        batch_out = shard_file.open("w", encoding="utf-8")

    for rec in records:
        line = build_batch_line(rec.custom_id, rec.body)
        line_json = json.dumps(line, ensure_ascii=False) + "\n"
        line_bytes = len(line_json.encode("utf-8"))

        if shard_count > 0 and (
            shard_count >= limit_requests
            or shard_bytes + line_bytes > BATCH_LIMIT_BYTES
        ):
            flush_shard()

        batch_out.write(line_json)
        shard_count += 1
        shard_bytes += line_bytes

    if shard_count > 0:
        flush_shard()
    else:
        batch_out.close()

    state = _build_state(
        paths=paths,
        benchmark_path=benchmark_path,
        samples=samples,
        shards=shards,
        retrieval_asset_dir=asset_dir,
        benchmark=benchmark,
        split=split,
        pipeline_name=pipeline_name,
        run_id=run_id,
    )
    paths.batch_state_json().write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    update_manifest_artifacts(
        paths,
        {
            "batch_output_raw": str(
                paths.batch_output_raw_jsonl().relative_to(run_root)
            ),
            "batch_output_parsed": str(
                paths.batch_output_parsed_jsonl().relative_to(run_root)
            ),
        },
    )

    logger.info(
        "Submitted %d shard(s), %d requests. State: %s",
        len(shards),
        state["total_requests"],
        paths.batch_state_json(),
    )
    for s in shards:
        print(s["batch_id"])
    return 0


def _write_outputs_from_raw_lines(
    *,
    paths: RunArtifactPaths,
    samples: list[dict],
    raw_lines: list[str],
) -> tuple[int, int]:
    """Persist raw/parsed outputs and merge into answers.jsonl."""
    raw_path = paths.batch_output_raw_jsonl()
    parsed_path = paths.batch_output_parsed_jsonl()
    answers_path = paths.generation_answers_jsonl()

    raw_path.write_text(
        "\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8"
    )

    existing_by_qid: Dict[str, Dict[str, Any]] = {}
    if answers_path.is_file():
        with answers_path.open("r", encoding="utf-8") as ef:
            for line in ef:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = str(row.get("question_id", "")).strip()
                if qid:
                    existing_by_qid[qid] = row

    answers_by_cid: Dict[str, GenerationParseResult] = {}
    errors_by_cid: Dict[str, Any] = {}
    completed = 0
    failed = 0
    parsed_rows: List[Dict[str, Any]] = []

    for ln in raw_lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        result = parse_generation_output(obj)
        cid = str(obj.get("custom_id", "") or result.custom_id or "")
        if cid:
            answers_by_cid[cid] = result
        if obj.get("error"):
            failed += 1
            if cid:
                errors_by_cid[cid] = obj.get("error")
        elif result.generation_parse_error:
            failed += 1
        else:
            completed += 1
        parsed_rows.append(
            {
                "custom_id": cid or result.custom_id,
                "answer": result.answer,
                "candidate_answer_span": result.candidate_answer_span,
                "support_quote": result.support_quote,
                "generation_reasoning": result.reasoning,
                "generation_output_format": result.generation_output_format,
                "generation_parse_error": result.generation_parse_error,
                "error": obj.get("error"),
                "raw_status_code": (obj.get("response") or {}).get("status_code"),
            }
        )

    with parsed_path.open("w", encoding="utf-8") as pf:
        for pr in parsed_rows:
            pf.write(json.dumps(pr, ensure_ascii=False) + "\n")

    by_qid: Dict[str, Dict[str, Any]] = {}
    for cid, result in answers_by_cid.items():
        meta = parse_generation_custom_id(cid)
        if not meta:
            logger.warning("Unparseable custom_id (skipped): %s", cid)
            continue
        qid = meta["question_id"]
        by_qid[qid] = {
            "answer": result.answer,
            "candidate_answer_span": result.candidate_answer_span,
            "support_quote": result.support_quote,
            "generation_reasoning": result.reasoning,
            "generation_output_format": result.generation_output_format,
            "generation_parse_error": result.generation_parse_error,
            "custom_id": cid,
            "batch_error": errors_by_cid.get(cid),
        }

    logger.info("Writing %s (one line per benchmark row)...", answers_path)
    with answers_path.open("w", encoding="utf-8") as out:
        for sample in samples:
            row = dict(sample)
            qid = str(row.get("question_id", "")).strip()
            if qid in by_qid:
                row.update(by_qid[qid])
            elif qid in existing_by_qid:
                row.update(existing_by_qid[qid])
            else:
                row.setdefault("answer", "")
                row.setdefault("custom_id", "")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    update_manifest_artifacts(
        paths,
        {
            "batch_output_raw": str(
                paths.batch_output_raw_jsonl().relative_to(paths.run_root)
            ),
            "batch_output_parsed": str(
                paths.batch_output_parsed_jsonl().relative_to(paths.run_root)
            ),
            "generation_answers": str(
                paths.generation_answers_jsonl().relative_to(paths.run_root)
            ),
        },
    )
    return completed, failed


def finalize_sync_submission(
    records: List[BatchRequestRecord],
    samples: list[dict],
    *,
    benchmark_path: Path,
    paths: RunArtifactPaths,
    benchmark: str,
    split: str,
    pipeline_name: str,
    run_id: str,
    retrieval_asset_dir: Path,
) -> int:
    """Submit one chat completion per record and ingest like collect_batches."""
    run_root = paths.run_root.resolve()
    asset_dir = retrieval_asset_dir.resolve()

    batch_input_path = paths.batch_input_jsonl()
    with batch_input_path.open("w", encoding="utf-8") as bf:
        for rec in records:
            line = build_batch_line(rec.custom_id, rec.body)
            bf.write(json.dumps(line, ensure_ascii=False) + "\n")

    state = _build_state(
        paths=paths,
        benchmark_path=benchmark_path,
        samples=samples,
        shards=[],
        retrieval_asset_dir=asset_dir,
        benchmark=benchmark,
        split=split,
        pipeline_name=pipeline_name,
        run_id=run_id,
    )
    paths.batch_state_json().write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if not records:
        logger.info("All questions already collected. Nothing to submit.")
        return 0

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    raw_lines: list[str] = []
    for rec in tqdm(records, desc="E2E sync generation", unit="q", dynamic_ncols=True):
        body = rec.body
        try:
            response = client.chat.completions.create(
                model=body["model"],
                messages=body["messages"],
                temperature=body["temperature"],
                max_tokens=body["max_tokens"],
                tools=body["tools"],
                tool_choice=body["tool_choice"],
                parallel_tool_calls=body.get("parallel_tool_calls", False),
            )
            response_body = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else dict(response)
            )
            line_obj = {
                "custom_id": rec.custom_id,
                "response": {"status_code": 200, "body": response_body},
            }
        except Exception as e:  # noqa: BLE001
            line_obj = {
                "custom_id": rec.custom_id,
                "error": {"message": str(e)},
            }
        raw_lines.append(json.dumps(line_obj, ensure_ascii=False))

    completed, failed = _write_outputs_from_raw_lines(
        paths=paths,
        samples=samples,
        raw_lines=raw_lines,
    )
    logger.info(
        "Sync submit+collect complete: completed=%d failed=%d; run_root=%s",
        completed,
        failed,
        run_root,
    )
    return 0


def _build_state(
    *,
    paths: RunArtifactPaths,
    benchmark_path: Path,
    samples: list[dict],
    shards: list[dict],
    retrieval_asset_dir: Path,
    benchmark: str,
    split: str,
    pipeline_name: str,
    run_id: str,
) -> Dict[str, Any]:
    return {
        "schema_version": STATE_VERSION,
        "run_root": str(paths.run_root.resolve()),
        "benchmark_path": str(benchmark_path),
        "benchmark": benchmark,
        "split": split,
        "pipeline_name": pipeline_name,
        "run_id": run_id,
        "retrieval_asset_dir": str(retrieval_asset_dir),
        "shards": shards,
        "samples_count": len(samples),
        "samples": samples,
        "total_requests": sum(s.get("request_count", 0) for s in shards),
    }


def collect_batches(
    *,
    state_path: Optional[Path] = None,
    run_root: Optional[Path] = None,
) -> int:
    """Download shard outputs, persist raw + parsed batch lines, merge generation/answers.jsonl."""
    if state_path is None:
        if run_root is None:
            raise ValueError("collect_batches requires state_path or run_root")
        state_path = RunArtifactPaths(run_root).batch_state_json()

    if not state_path.is_file():
        logger.error("State file not found: %s", state_path)
        return 1

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    paths = RunArtifactPaths(Path(state["run_root"]))
    samples = state.get("samples") or []
    shards = state.get("shards") or []
    if not samples:
        logger.error("No samples in state file.")
        return 1

    paths.generation_dir.mkdir(parents=True, exist_ok=True)
    paths.batch_dir.mkdir(parents=True, exist_ok=True)

    answers_path = paths.generation_answers_jsonl()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)

    raw_lines: List[str] = []
    for shard in shards:
        batch_id = shard.get("batch_id")
        if not batch_id or str(batch_id).startswith("dry-run"):
            continue
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            logger.warning(
                "Shard batch %s not completed (status=%s). Skipping.",
                batch_id,
                batch.status,
            )
            continue

        output_file_id = getattr(batch, "output_file_id", None) or getattr(
            batch, "output_file", None
        )
        if not output_file_id:
            logger.warning("Shard batch %s has no output file.", batch_id)
            continue

        logger.info("Downloading shard output for %s...", batch_id)
        content = client.files.content(output_file_id)
        text = content.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        for ln in text.strip().split("\n"):
            if ln.strip():
                raw_lines.append(ln)

    completed, failed = _write_outputs_from_raw_lines(
        paths=paths,
        samples=samples,
        raw_lines=raw_lines,
    )

    logger.info(
        "Summary: completed=%d failed=%d; wrote %d rows to %s",
        completed,
        failed,
        len(samples),
        answers_path,
    )
    print(f"Output: {answers_path}")
    return 0


def make_run_paths_for_cli(
    benchmark: str,
    split: str,
    pipeline_name: str,
    run_id: str,
    evaluation_base: Optional[Path] = None,
) -> RunArtifactPaths:
    """Convenience: default evaluation base + build RunArtifactPaths."""
    base = evaluation_base if evaluation_base is not None else default_evaluation_base()
    run_root = build_run_root(base, benchmark, split, pipeline_name, run_id)
    return RunArtifactPaths(run_root)
