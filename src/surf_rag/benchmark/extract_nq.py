from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from surf_rag.core.canonical_clean import clean_html_to_structured_doc
from surf_rag.core.schemas import sha256_text

from .sentence_utils import (
    build_sentencizer,
    dedupe_preserve_order,
    sentence_span_for_char_span,
    sentence_spans,
)

_TOKEN_RE = re.compile(r"\w+")
_NON_WORD_RE = re.compile(r"[^\w]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _nq_span_index(raw: Any) -> Optional[int]:
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


def _nq_token_arrays(
    document: Optional[Dict[str, Any]],
) -> Optional[Tuple[List[Any], List[Any]]]:
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


def nq_document_title(row: Dict[str, Any]) -> str:
    """Wikipedia page title from HF NQ ``document.title`` (or legacy top-level fallbacks)."""
    document = row.get("document")
    if isinstance(document, dict):
        t = document.get("title")
        if t is not None and str(t).strip():
            return str(t).strip()
    for key in ("document_title", "title"):
        v = row.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _iter_short_answer_spans(row: Dict[str, Any]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    ann = row.get("annotations")
    if ann is None:
        return spans
    ann_iter: List[Any]
    if isinstance(ann, dict):
        ann_iter = [ann]
    elif isinstance(ann, list):
        ann_iter = list(ann)
    else:
        return spans

    for a in ann_iter:
        if not isinstance(a, dict):
            continue
        sa = a.get("short_answers", [])
        sa_iter: List[Any]
        if isinstance(sa, dict):
            sa_iter = [sa]
        elif isinstance(sa, list):
            sa_iter = sa
        else:
            continue
        for item in sa_iter:
            if not isinstance(item, dict):
                continue
            st = _nq_span_index(item.get("start_token"))
            et = _nq_span_index(item.get("end_token"))
            if st is None or et is None:
                continue
            # Existing loaders interpret end_token as inclusive.
            if st < 0 or et < st:
                continue
            spans.append((st, et))
    return spans


def _iter_short_answer_entries(
    row: Dict[str, Any],
) -> List[Tuple[int, int, List[str]]]:
    out: List[Tuple[int, int, List[str]]] = []
    ann = row.get("annotations")
    if ann is None:
        return out
    ann_iter: List[Any]
    if isinstance(ann, dict):
        ann_iter = [ann]
    elif isinstance(ann, list):
        ann_iter = list(ann)
    else:
        return out

    for a in ann_iter:
        if not isinstance(a, dict):
            continue
        sa = a.get("short_answers", [])
        sa_iter: List[Any]
        if isinstance(sa, dict):
            sa_iter = [sa]
        elif isinstance(sa, list):
            sa_iter = sa
        else:
            continue
        for item in sa_iter:
            if not isinstance(item, dict):
                continue
            st = _nq_span_index(item.get("start_token"))
            et = _nq_span_index(item.get("end_token"))
            if st is None or et is None or st < 0 or et < st:
                continue
            texts: List[str] = []
            raw_text = item.get("text")
            if isinstance(raw_text, list):
                texts.extend(str(x).strip() for x in raw_text if str(x).strip())
            elif isinstance(raw_text, str) and raw_text.strip():
                texts.append(raw_text.strip())
            out.append((st, et, list(dict.fromkeys(texts))))
    return out


def _reconstruct_text_with_offsets(
    token_vals: Sequence[Any], is_html: Sequence[Any]
) -> Tuple[str, Dict[int, Tuple[int, int]], Dict[int, int], List[Tuple[int, int, int]]]:
    """
    Returns:
      text
      token_to_char: original_token_idx -> (char_start, char_end) for visible tokens
      token_to_visible_pos: original_token_idx -> position among visible tokens
      visible_tokens: list of (original_idx, char_start, char_end)
    """
    n = min(len(token_vals), len(is_html))
    pieces: List[str] = []
    token_to_char: Dict[int, Tuple[int, int]] = {}
    token_to_visible_pos: Dict[int, int] = {}
    visible_tokens: List[Tuple[int, int, int]] = []
    char_pos = 0
    visible_pos = 0
    for i in range(n):
        if bool(is_html[i]):
            continue
        token_text = str(token_vals[i]).strip()
        if not token_text:
            continue
        if pieces:
            pieces.append(" ")
            char_pos += 1
        start = char_pos
        pieces.append(token_text)
        char_pos += len(token_text)
        end = char_pos
        token_to_char[i] = (start, end)
        token_to_visible_pos[i] = visible_pos
        visible_tokens.append((i, start, end))
        visible_pos += 1
    return ("".join(pieces), token_to_char, token_to_visible_pos, visible_tokens)


def _span_to_char(
    span: Tuple[int, int],
    token_to_char: Dict[int, Tuple[int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    st, et = span
    first_idx = None
    first_char = None
    last_idx = None
    last_char = None
    for idx in range(st, et + 1):
        char_span = token_to_char.get(idx)
        if char_span is None:
            continue
        if first_idx is None:
            first_idx = idx
            first_char = char_span[0]
        last_idx = idx
        last_char = char_span[1]
    if first_idx is None or first_char is None or last_idx is None or last_char is None:
        return None
    return (first_idx, last_idx, first_char, last_char)


def _normalize_match_text(text: str) -> str:
    s = text.casefold()
    s = s.replace("|", " ")
    s = re.sub(r"\s+([,.;:!?%])", r"\1", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _keyword_terms(text: str) -> set[str]:
    terms = {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}
    return {t for t in terms if t not in _STOPWORDS and len(t) > 1}


def _context_terms(
    token_vals: Sequence[Any],
    is_html: Sequence[Any],
    span: Tuple[int, int],
    window_tokens: int = 20,
) -> set[str]:
    st, et = span
    left = max(0, st - window_tokens)
    right = min(min(len(token_vals), len(is_html)) - 1, et + window_tokens)
    terms: set[str] = set()
    for i in range(left, right + 1):
        if bool(is_html[i]):
            continue
        tok = str(token_vals[i]).strip()
        if not tok:
            continue
        tok_clean = _NON_WORD_RE.sub("", tok).lower()
        if not tok_clean or tok_clean in _STOPWORDS or len(tok_clean) <= 1:
            continue
        terms.add(tok_clean)
    return terms


def _cleaned_candidate_sentences(
    row: Dict[str, Any],
    sentencizer,
) -> List[str]:
    document = row.get("document") if isinstance(row.get("document"), dict) else {}
    html = document.get("html") or row.get("document_html")
    if not isinstance(html, str) or not html.strip():
        return []
    title = document.get("title") or row.get("document_title") or row.get("title")
    question = row.get("question")
    q_text = question.get("text") if isinstance(question, dict) else question
    doc_id = sha256_text(f"{title or ''}:{q_text or ''}:{len(html)}")
    structured = clean_html_to_structured_doc(
        html=html,
        doc_id=doc_id,
        title=title,
        url=document.get("url"),
        anchors={"outgoing_titles": [], "incoming_stub": []},
        source="nq",
        dataset_origin="nq",
    )
    candidates: List[str] = []
    for block in structured.blocks:
        block_text = (block.text or "").strip()
        if not block_text:
            continue
        for start, end in sentence_spans(block_text, sentencizer):
            sentence = block_text[start:end].strip()
            if sentence:
                candidates.append(sentence)
    return dedupe_preserve_order(candidates)


def _score_candidate(
    sentence: str,
    context_terms: set[str],
    question_terms: set[str],
) -> Tuple[int, int, int]:
    sent_terms = _keyword_terms(sentence)
    context_overlap = len(sent_terms & context_terms)
    question_overlap = len(sent_terms & question_terms)
    return (context_overlap, question_overlap, -len(sentence))


def _question_phrases(question_text: str) -> List[str]:
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(question_text)]
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    phrases: List[str] = []
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]} {tokens[i + 1]}")
    return list(dict.fromkeys(phrases))


def _trim_long_sentence(
    sentence: str,
    answer_texts: Sequence[str],
    question_text: str,
) -> str:
    if len(sentence) < 250:
        return sentence
    s_cf = sentence.casefold()
    answer_pos = -1
    for answer in sorted(answer_texts, key=len, reverse=True):
        a = answer.casefold().strip()
        if not a:
            continue
        pos = s_cf.find(a)
        if pos >= 0:
            answer_pos = pos
            break
    if answer_pos < 0:
        for answer in answer_texts:
            for tok in sorted(_TOKEN_RE.findall(answer), key=len, reverse=True):
                t = tok.casefold()
                if len(t) <= 3:
                    continue
                pos = s_cf.find(t)
                if pos >= 0 and (answer_pos < 0 or pos < answer_pos):
                    answer_pos = pos
    if answer_pos < 0:
        return sentence

    best_phrase_pos = -1
    for phrase in _question_phrases(question_text):
        pos = s_cf.find(phrase)
        if 0 <= pos <= answer_pos and pos > best_phrase_pos:
            best_phrase_pos = pos
    if best_phrase_pos >= 0:
        trimmed = sentence[best_phrase_pos:].strip(" |-\n\t")
        if trimmed:
            return trimmed
    return sentence


def _select_best_cleaned_sentence(
    candidates: Sequence[str],
    answer_texts: Sequence[str],
    context_terms: set[str],
    question_terms: set[str],
    question_text: str,
) -> Optional[str]:
    norm_candidates = [(c, _normalize_match_text(c)) for c in candidates]
    norm_answers = [_normalize_match_text(a) for a in answer_texts if a.strip()]
    norm_answers = [a for a in norm_answers if a]
    if not norm_answers:
        return None
    matched: List[str] = []
    for original, norm in norm_candidates:
        if any(ans in norm for ans in norm_answers):
            matched.append(original)
    if not matched:
        return None
    best = max(
        matched,
        key=lambda s: _score_candidate(
            s, context_terms=context_terms, question_terms=question_terms
        ),
    )
    return _trim_long_sentence(
        best, answer_texts=answer_texts, question_text=question_text
    )


def _extract_sentence_for_span(
    text: str,
    answer_char_span: Tuple[int, int],
    first_visible_pos: int,
    last_visible_pos: int,
    visible_tokens: Sequence[Tuple[int, int, int]],
    sentencizer,
    initial_window_tokens: int = 100,
    max_window_tokens: int = 512,
) -> Optional[str]:
    if not text or not visible_tokens:
        return None
    window = max(1, initial_window_tokens)
    n_visible = len(visible_tokens)
    ans_start, ans_end = answer_char_span

    while window <= max_window_tokens:
        left = max(0, first_visible_pos - window)
        right = min(n_visible - 1, last_visible_pos + window)
        _, window_start, _ = visible_tokens[left]
        _, _, window_end = visible_tokens[right]
        local_text = text[window_start:window_end]
        local_answer = (ans_start - window_start, ans_end - window_start)
        spans = sentence_spans(local_text, sentencizer)
        sent_span = sentence_span_for_char_span(spans, local_answer)
        if sent_span is not None:
            sent_start, sent_end = sent_span
            touches_left = sent_start == 0 and left > 0
            touches_right = sent_end == len(local_text) and right < (n_visible - 1)
            if not touches_left and not touches_right:
                sentence = local_text[sent_start:sent_end].strip()
                return sentence or None
        window *= 2

    full_spans = sentence_spans(text, sentencizer)
    sent_span = sentence_span_for_char_span(full_spans, answer_char_span)
    if sent_span is None:
        return None
    sent_start, sent_end = sent_span
    sentence = text[sent_start:sent_end].strip()
    return sentence or None


def extract_nq_support_sentences(
    row: Dict[str, Any],
    sentencizer=None,
    initial_window_tokens: int = 100,
    max_window_tokens: int = 512,
) -> List[str]:
    document = row.get("document") if isinstance(row.get("document"), dict) else None
    token_pair = _nq_token_arrays(document)
    if token_pair is None:
        return []
    token_vals, is_html = token_pair
    (
        text,
        token_to_char,
        token_to_visible_pos,
        visible_tokens,
    ) = _reconstruct_text_with_offsets(token_vals, is_html)
    if not text:
        return []

    spans = _iter_short_answer_entries(row)
    if not spans:
        return []

    sentencizer = sentencizer or build_sentencizer()
    cleaned_candidates = _cleaned_candidate_sentences(row=row, sentencizer=sentencizer)
    question = row.get("question")
    question_text = (
        question.get("text") if isinstance(question, dict) else str(question or "")
    )
    question_terms = _keyword_terms(question_text)
    extracted: List[str] = []
    for st, et, span_answer_texts in spans:
        span = (st, et)
        mapped = _span_to_char(span, token_to_char)
        if mapped is None:
            continue
        first_idx, last_idx, char_start, char_end = mapped
        first_pos = token_to_visible_pos.get(first_idx)
        last_pos = token_to_visible_pos.get(last_idx)
        if first_pos is None or last_pos is None:
            continue
        span_surface = text[char_start:char_end].strip()
        answer_texts = list(span_answer_texts)
        if span_surface:
            answer_texts.append(span_surface)

        context = _context_terms(token_vals, is_html, span=span)
        sentence = _select_best_cleaned_sentence(
            candidates=cleaned_candidates,
            answer_texts=answer_texts,
            context_terms=context,
            question_terms=question_terms,
            question_text=question_text,
        )
        if sentence:
            extracted.append(sentence)

    return dedupe_preserve_order(extracted)
