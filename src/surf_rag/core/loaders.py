from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from surf_rag.benchmark import (
    extract_2wiki_support_lines,
    extract_nq_support_sentences,
    nq_document_title,
)
from surf_rag.benchmark.sentence_utils import build_sentencizer

from .schemas import BenchmarkItem, sha256_text

logger = logging.getLogger(__name__)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def nq_row_question_text(row: Dict[str, Any]) -> Optional[str]:
    """
    Primary: Hugging Face NQ uses ``question`` as ``{\"text\": \"...\"}``.
    Legacy fallbacks for older exports.
    """
    qb = row.get("question")
    if isinstance(qb, dict):
        t = qb.get("text")
        if t is not None and str(t).strip():
            return str(t).strip()
    return _first_non_empty(
        row.get("question_text"),
        row.get("questionText"),
        qb if isinstance(qb, str) else None,
    )


def _nq_token_arrays(
    document: Optional[Dict[str, Any]],
) -> Optional[Tuple[List[Any], List[Any]]]:
    """Return (token_strings, is_html_flags) from HF-style document.tokens dict."""
    if not isinstance(document, dict):
        return None
    toks = document.get("tokens")
    if not isinstance(toks, dict):
        return None
    token_vals = toks.get("token")
    is_html = toks.get("is_html")
    if not isinstance(token_vals, list) or not isinstance(is_html, list):
        return None
    return token_vals, is_html


def _nq_span_index(raw: Any) -> Optional[int]:
    """Normalize start_token / end_token (scalar or single-element list) to int."""
    if raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _nq_answer_strings_from_short_answer_entry(
    item: Dict[str, Any],
    document: Optional[Dict[str, Any]],
) -> List[str]:
    """
    One NQ ``short_answers`` element.

    Hugging Face / json export uses ``text`` as a **list** of strings (often one
    element), not a plain string. Do not ``str()`` the list (that produces
    ``\"['foo']\"``). If ``text`` is empty, recover span from ``document.tokens``.
    """
    out: List[str] = []
    raw_text = item.get("text")
    if isinstance(raw_text, list):
        for x in raw_text:
            s = str(x).strip()
            if s:
                out.append(s)
    elif isinstance(raw_text, str):
        s = raw_text.strip()
        if s:
            out.append(s)

    if out:
        return out

    st = _nq_span_index(item.get("start_token"))
    et = _nq_span_index(item.get("end_token"))
    if st is None or et is None or document is None:
        return []
    pair = _nq_token_arrays(document)
    if not pair:
        return []
    token_vals, is_html = pair
    n = min(len(token_vals), len(is_html))
    if st < 0 or st >= n or et < 0 or et >= n or st > et:
        return []
    parts: List[str] = []
    for i in range(st, et + 1):
        if not is_html[i]:
            t = token_vals[i]
            if t is not None and str(t).strip():
                parts.append(str(t).strip())
    joined = " ".join(parts).strip()
    return [joined] if joined else []


def _nq_short_answer_texts_from_annotations(row: Dict[str, Any]) -> List[str]:
    """
    Gold short answers: ``annotations`` → ``short_answers`` (list of dicts).

    Each dict may have ``text`` as a string or list of strings (HF JSONL), plus
    optional ``start_token`` / ``end_token`` into ``document.tokens``.
    """
    out: List[str] = []
    document = row.get("document") if isinstance(row.get("document"), dict) else None
    ann = row.get("annotations")
    if ann is None:
        return out
    if isinstance(ann, dict):
        ann_iter: List[Any] = [ann]
    elif isinstance(ann, list):
        ann_iter = list(ann)
    else:
        return out

    for a in ann_iter:
        if not isinstance(a, dict):
            continue
        sa = a.get("short_answers", [])
        if isinstance(sa, dict):
            sa_iter = [sa]
        elif isinstance(sa, list):
            sa_iter = sa
        else:
            continue
        for item in sa_iter:
            if isinstance(item, dict):
                out.extend(_nq_answer_strings_from_short_answer_entry(item, document))
            elif isinstance(item, str) and item.strip():
                out.append(item.strip())
    # Dedupe, preserve order
    return list(dict.fromkeys(out))


def _has_nq_document_context(row: Dict[str, Any]) -> bool:
    """
    Hugging Face NQ: HTML and tokens live under ``document`` (``document.html``,
    ``document.tokens``). Legacy top-level context fields kept as fallback.
    """
    doc_block = row.get("document")
    if isinstance(doc_block, dict):
        if _first_non_empty(doc_block.get("html")):
            return True
        tokens = doc_block.get("tokens")
        if isinstance(tokens, dict):
            token_vals = tokens.get("token")
            is_html = tokens.get("is_html")
            if isinstance(token_vals, list) and isinstance(is_html, list):
                for i, ih in enumerate(is_html):
                    if i < len(token_vals) and not ih and str(token_vals[i]).strip():
                        return True
        elif isinstance(tokens, list):
            for t in tokens:
                if isinstance(t, dict) and t.get("token") and not t.get("is_html"):
                    return True

    if _first_non_empty(
        row.get("context"),
        row.get("document_text"),
        row.get("document") if isinstance(row.get("document"), str) else None,
        row.get("paragraph"),
    ):
        return True
    if _first_non_empty(row.get("document_html")):
        return True
    return False


def nq_row_fail_reasons(row: Dict[str, Any]) -> List[str]:
    """
    Return human-readable reasons this row would be skipped by :func:`load_nq`.
    Empty list means the row would be ingested.
    """
    reasons: List[str] = []
    question = nq_row_question_text(row)
    if not question:
        reasons.append(
            "missing question (expected question.text or legacy question fields)"
        )

    answers = _nq_short_answer_texts_from_annotations(row)
    if not answers:
        reasons.append(
            "no gold short answers (expected annotations[].short_answers with non-empty "
            "text or token span)"
        )

    if not _has_nq_document_context(row):
        reasons.append(
            "missing document context (expected document.html or document.tokens with text tokens)"
        )

    return reasons


def _iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict) and "data" in data:
            for item in data["data"]:
                yield item
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
        return
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_nq(
    path: str,
    dataset_version: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[BenchmarkItem]:
    """
    Load Natural Questions JSON/JSONL (Hugging Face export shape).

    Expects ``question.text``, document HTML/tokens under ``document``, and gold
    from ``annotations[].short_answers`` (``text`` as string or list of strings;
    HF JSONL uses a list). See
    :func:`nq_row_fail_reasons` for validation details.
    """
    source = "nq"
    count = 0
    sentencizer = build_sentencizer()
    for row in _iter_json_records(Path(path)):
        question = nq_row_question_text(row)
        if not question:
            continue

        answers = _nq_short_answer_texts_from_annotations(row)
        support_sentences = extract_nq_support_sentences(
            row=row,
            sentencizer=sentencizer,
        )
        if not answers or not _has_nq_document_context(row) or not support_sentences:
            continue

        doc_title = nq_document_title(row)
        n_sup = len(support_sentences)
        yield BenchmarkItem(
            question_id=sha256_text(question),
            question=question,
            gold_answers=answers,
            dataset_source=source,
            gold_support_sentences=support_sentences,
            gold_support_titles=[doc_title] * n_sup,
            gold_support_sent_ids=[-1] * n_sup,
            dataset_version=dataset_version,
        )
        count += 1
        if max_rows and count >= max_rows:
            break


def load_2wiki(
    path: str,
    dataset_version: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[BenchmarkItem]:
    """Load 2WikiMultiHopQA data from JSON/JSONL."""
    source = "2wiki"
    count = 0
    for row in _iter_json_records(Path(path)):
        question = row.get("question")
        answers = _as_list(row.get("answer"))
        if not question or not answers:
            continue

        supporting_facts = row.get("supporting_facts")
        if not supporting_facts:
            continue
        lines = extract_2wiki_support_lines(row)
        support_sentences = [t[2] for t in lines]
        if not support_sentences:
            continue

        yield BenchmarkItem(
            question_id=sha256_text(question),
            question=question,
            gold_answers=answers,
            dataset_source=source,
            gold_support_sentences=support_sentences,
            gold_support_titles=[t[0] for t in lines],
            gold_support_sent_ids=[t[1] for t in lines],
            dataset_version=dataset_version,
        )
        count += 1
        if max_rows and count >= max_rows:
            break
