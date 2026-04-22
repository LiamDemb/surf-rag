"""Orchestrate in-place 2Wiki benchmark alignment (title-localized, thresholded)."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from surf_rag.benchmark.corpus_filter import iter_jsonl, write_jsonl
from surf_rag.benchmark.support_alignment import (
    GoldSupportAnchor,
    SupportAlignmentDecision,
    align_gold_support_anchors,
    build_title_to_candidate_sentences,
)
from surf_rag.core.docstore import DocStore
from surf_rag.core.docstore_sentence_index import (
    build_title_to_candidate_sentences_from_docstore,
)
from surf_rag.core.embedder import SentenceTransformersEmbedder
from surf_rag.core.schemas import BenchmarkItem, parse_benchmark_support_fields

logger = logging.getLogger(__name__)


class AlignmentStats(TypedDict):
    total_rows: int
    two_wiki_rows: int
    two_wiki_with_facts: int
    two_wiki_kept: int
    two_wiki_dropped: int
    facts_replaced: int
    skipped_no_provenance: int


def row_to_benchmark_item(row: Dict[str, Any]) -> BenchmarkItem:
    sentences, titles, sent_ids = parse_benchmark_support_fields(row)
    return BenchmarkItem(
        question_id=row["question_id"],
        question=row["question"],
        gold_answers=row["gold_answers"],
        dataset_source=row["dataset_source"],
        gold_support_sentences=sentences,
        gold_support_titles=titles,
        gold_support_sent_ids=sent_ids,
        dataset_version=row.get("dataset_version"),
    )


def _anchors_from_item(item: BenchmarkItem) -> List[GoldSupportAnchor]:
    sentences = item.gold_support_sentences
    n = len(sentences)
    ts = (item.gold_support_titles + [""] * n)[:n]
    sids = (item.gold_support_sent_ids + [-1] * n)[:n]
    return [
        GoldSupportAnchor(title=ts[i], sent_id=sids[i], sentence=sentences[i])
        for i in range(n)
    ]


def _two_wiki_row_fully_resolved(decisions: List[SupportAlignmentDecision]) -> bool:
    """Keep row only if every line is exact match or replaced under thresholds."""
    return all(d.replaced or d.reason == "already_present" for d in decisions)


def _outcome_label(decision_reason: str, replaced: bool) -> str:
    if replaced:
        return "Replaced (thresholds passed)"
    if decision_reason == "already_present":
        return "Unchanged (exact or substring match in cleaned article sentences)"
    if decision_reason == "below_thresholds":
        return "Unchanged (below thresholds; showing nearest candidate)"
    if decision_reason == "no_corpus_chunks_for_title":
        return "Unchanged (no fetched HTML / no sentences for this title in DocStore)"
    return f"Unchanged ({decision_reason})"


def _line_block_from_decision(dec: SupportAlignmentDecision) -> Dict[str, Any]:
    return {
        "title": dec.title,
        "sent_id": dec.sent_id,
        "original": dec.original,
        "outcome": _outcome_label(dec.reason, dec.replaced),
        "reason": dec.reason,
        "replaced": dec.replaced,
        "replacement": dec.replacement,
        "semantic_cosine": dec.semantic_cosine,
        "rouge_l": dec.rouge_l,
        "nearest_candidate": dec.nearest_candidate,
        "nearest_semantic_cosine": dec.nearest_semantic_cosine,
        "nearest_rouge_l": dec.nearest_rouge_l,
    }


def render_full_alignment_report_markdown(
    *,
    row_blocks: List[Dict[str, Any]],
    tau_sem: float,
    tau_lex: float,
    model_name: str,
    created_at: str,
    dropped_rows: List[Dict[str, Any]],
) -> str:
    """One section per 2Wiki row; every support line with outcome and nearest neighbor when applicable."""
    lines = [
        "# 2Wiki gold support alignment report (full)",
        "",
        f"- Created: {created_at}",
        f"- Model: `{model_name}`",
        f"- Thresholds: `semantic_cosine >= {tau_sem}` and `rouge_l >= {tau_lex}`",
        f"- Rows in this report: **{len(row_blocks)}** (2Wiki questions processed for alignment)",
        f"- Dropped (unresolved) rows: **{len(dropped_rows)}**",
        "",
    ]
    if dropped_rows:
        lines.extend(["## Dropped 2Wiki rows (unresolved gold support)", ""])
        for dr in dropped_rows:
            lines.append(f"- `{dr['question_id']}`: {dr['detail']}")
        lines.append("")

    if not row_blocks:
        lines.append("_No 2Wiki rows with provenance were processed._")
        return "\n".join(lines) + "\n"

    for block in row_blocks:
        lines.extend(
            [
                f"## `{block['question_id']}`",
                "",
                f"- **Question:** {block['question']}",
                "",
            ]
        )
        for i, ln in enumerate(block["lines"], start=1):
            sid = ln["sent_id"]
            sid_s = str(sid) if sid >= 0 else "n/a"
            lines.extend(
                [
                    f"### Support line {i} — `{ln['title']}` (dataset sent_id: {sid_s})",
                    "",
                    f"- **Outcome:** {ln['outcome']}",
                    "",
                    "| Field | Value |",
                    "| --- | --- |",
                    f"| Original | {ln['original']} |",
                ]
            )
            if ln["replaced"] and ln["replacement"] is not None:
                lines.extend(
                    [
                        f"| Replacement | {ln['replacement']} |",
                        f"| semantic_cosine (accepted) | {ln['semantic_cosine']:.4f} |",
                        f"| rouge_l (accepted) | {ln['rouge_l']:.4f} |",
                    ]
                )
            elif ln["reason"] == "already_present":
                lines.append(
                    "| Note | Gold text already present in title-localized article sentences (no embedding search). |"
                )
            elif ln["nearest_candidate"]:
                lines.extend(
                    [
                        f"| Nearest article sentence | {ln['nearest_candidate']} |",
                        f"| semantic_cosine (nearest) | {ln['nearest_semantic_cosine']:.4f} |",
                        f"| rouge_l (nearest) | {ln['nearest_rouge_l']:.4f} |",
                    ]
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def render_replacement_report_markdown(
    *,
    replaced_entries: List[Dict[str, Any]],
    tau_sem: float,
    tau_lex: float,
    model_name: str,
    created_at: str,
    dropped_rows: List[Dict[str, Any]],
) -> str:
    lines = [
        "# 2Wiki gold support alignment report",
        "",
        f"- Created: {created_at}",
        f"- Model: `{model_name}`",
        f"- Thresholds: `semantic_cosine >= {tau_sem}` and `rouge_l >= {tau_lex}`",
        f"- Replacements: **{len(replaced_entries)}** (unchanged rows are omitted from detail below)",
        f"- Dropped (unresolved) rows: **{len(dropped_rows)}**",
        "",
    ]
    if dropped_rows:
        lines.extend(["## Dropped 2Wiki rows", ""])
        for dr in dropped_rows:
            lines.append(f"- `{dr['question_id']}`: {dr['detail']}")
        lines.append("")

    if not replaced_entries:
        lines.append("_No sentences were replaced._")
        return "\n".join(lines) + "\n"

    for block in replaced_entries:
        lines.extend(
            [
                f"## `{block['question_id']}`",
                "",
                f"- **Question:** {block['question']}",
                f"- **Title:** {block['title']}",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Original | {block['original']} |",
                f"| Replacement | {block['replacement']} |",
                f"| semantic_cosine | {block['semantic_cosine']:.4f} |",
                f"| rouge_l | {block['rouge_l']:.4f} |",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _collect_two_wiki_titles_needed(rows: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for row in rows:
        if str(row.get("dataset_source", "")).strip().lower() != "2wiki":
            continue
        _, titles, _ = parse_benchmark_support_fields(row)
        for t in titles:
            key = str(t).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def run_2wiki_support_alignment(
    benchmark_path: Path,
    *,
    backup_path: Path,
    report_path: Path,
    docstore_path: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    model_name: str = "all-MiniLM-L6-v2",
    tau_sem: float = 0.92,
    tau_lex: float = 0.75,
    embedder: Optional[SentenceTransformersEmbedder] = None,
    full_report: bool = False,
    drop_unresolved: bool = True,
    chunk_min_tokens: int = 500,
    chunk_max_tokens: int = 800,
    chunk_overlap_tokens: int = 100,
) -> AlignmentStats:
    """
    Backup ``benchmark_path`` to ``backup_path``, rewrite benchmark in place,
    write ``report_path`` markdown.

    Candidate sentences come from ``docstore_path`` (preferred: same clean+chunk
    path as corpus build) or legacy ``corpus.jsonl`` via ``corpus_path``.

    When ``drop_unresolved`` is True, 2Wiki rows that still have any support line
    with reason other than ``already_present`` / successful replacement are
    removed from the benchmark (audited in the report).
    """
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")
    if docstore_path is None and (corpus_path is None or not corpus_path.is_file()):
        raise ValueError(
            "Provide docstore_path to a DocStore SQLite file (preferred) or corpus_path to corpus.jsonl."
        )

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(benchmark_path, backup_path)
    logger.info("Backed up original benchmark to %s", backup_path)

    all_rows = list(iter_jsonl(benchmark_path))
    titles_needed = _collect_two_wiki_titles_needed(all_rows)

    if docstore_path is not None:
        docstore = DocStore(str(docstore_path))
        try:
            title_to_candidates = build_title_to_candidate_sentences_from_docstore(
                docstore,
                titles_needed,
                chunk_min_tokens=chunk_min_tokens,
                chunk_max_tokens=chunk_max_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        finally:
            docstore.close()
        logger.info(
            "Indexed %d Wikipedia titles from DocStore (chunk-shaped sentences)",
            len(title_to_candidates),
        )
    else:
        corpus_rows = list(iter_jsonl(corpus_path))  # type: ignore[arg-type]
        title_to_candidates = build_title_to_candidate_sentences(corpus_rows)
        logger.info(
            "Indexed %d Wikipedia titles from corpus chunks (legacy mode)",
            len(title_to_candidates),
        )

    emb = embedder or SentenceTransformersEmbedder(model_name=model_name)

    rows_out: List[Dict[str, Any]] = []
    replaced_report: List[Dict[str, Any]] = []
    full_row_blocks: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []
    stats: AlignmentStats = {
        "total_rows": len(all_rows),
        "two_wiki_rows": 0,
        "two_wiki_with_facts": 0,
        "two_wiki_kept": 0,
        "two_wiki_dropped": 0,
        "facts_replaced": 0,
        "skipped_no_provenance": 0,
    }

    for row in all_rows:
        ds = str(row.get("dataset_source", "")).strip().lower()
        if ds != "2wiki":
            rows_out.append(dict(row))
            continue

        stats["two_wiki_rows"] += 1
        item = row_to_benchmark_item(row)
        anchors = _anchors_from_item(item)

        def _drop(
            detail: str, decisions: Optional[List[SupportAlignmentDecision]] = None
        ) -> None:
            stats["two_wiki_dropped"] += 1
            dropped_rows.append(
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "detail": detail,
                    "decisions": (
                        [_line_block_from_decision(d) for d in decisions]
                        if decisions
                        else None
                    ),
                }
            )

        if not anchors or not any(a.title.strip() for a in anchors):
            stats["skipped_no_provenance"] += 1
            if drop_unresolved:
                _drop("missing gold_support_titles / provenance")
            else:
                rows_out.append(dict(row))
                stats["two_wiki_kept"] += 1
            continue

        stats["two_wiki_with_facts"] += 1
        decisions = align_gold_support_anchors(
            anchors,
            title_to_candidates,
            emb,
            tau_sem=tau_sem,
            tau_lex=tau_lex,
        )

        if full_report:
            full_row_blocks.append(
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "lines": [_line_block_from_decision(d) for d in decisions],
                }
            )

        resolved = _two_wiki_row_fully_resolved(decisions)
        if not resolved and drop_unresolved:
            reasons = ", ".join(f"{d.title}:{d.reason}" for d in decisions)
            _drop(f"not fully resolved ({reasons})", decisions)
            continue

        new_sentences = list(item.gold_support_sentences)
        for i, dec in enumerate(decisions):
            if dec.replaced and dec.replacement is not None:
                stats["facts_replaced"] += 1
                new_sentences[i] = dec.replacement
                replaced_report.append(
                    {
                        "question_id": item.question_id,
                        "question": item.question,
                        "title": dec.title,
                        "original": dec.original,
                        "replacement": dec.replacement,
                        "semantic_cosine": dec.semantic_cosine,
                        "rouge_l": dec.rouge_l,
                    }
                )

        updated = BenchmarkItem(
            question_id=item.question_id,
            question=item.question,
            gold_answers=item.gold_answers,
            dataset_source=item.dataset_source,
            gold_support_sentences=new_sentences,
            gold_support_titles=item.gold_support_titles,
            gold_support_sent_ids=item.gold_support_sent_ids,
            dataset_version=item.dataset_version,
        )
        rows_out.append(updated.to_json())
        stats["two_wiki_kept"] += 1

    write_jsonl(benchmark_path, rows_out)

    created_at = datetime.now(timezone.utc).isoformat()
    if full_report:
        report_body = render_full_alignment_report_markdown(
            row_blocks=full_row_blocks,
            tau_sem=tau_sem,
            tau_lex=tau_lex,
            model_name=model_name,
            created_at=created_at,
            dropped_rows=dropped_rows,
        )
    else:
        report_body = render_replacement_report_markdown(
            replaced_entries=replaced_report,
            tau_sem=tau_sem,
            tau_lex=tau_lex,
            model_name=model_name,
            created_at=created_at,
            dropped_rows=dropped_rows,
        )
    report_path.write_text(report_body, encoding="utf-8")

    logger.info(
        "Wrote benchmark: %s (rows=%d, 2wiki=%d, kept=%d, dropped=%d, with_facts=%d, facts_replaced=%d)",
        benchmark_path,
        stats["total_rows"],
        stats["two_wiki_rows"],
        stats["two_wiki_kept"],
        stats["two_wiki_dropped"],
        stats["two_wiki_with_facts"],
        stats["facts_replaced"],
    )
    logger.info("Wrote alignment report: %s", report_path)
    return stats
