from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_sentencizer():
    try:
        import spacy

        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except ImportError:
        return _RegexSentencizer()


def sentence_spans(text: str, sentencizer) -> List[Tuple[int, int]]:
    if not text.strip():
        return []
    doc = sentencizer(text)
    spans: List[Tuple[int, int]] = []
    for sent in doc.sents:
        start = int(sent.start_char)
        end = int(sent.end_char)
        if end > start:
            spans.append((start, end))
    return spans


class _SimpleSentence:
    def __init__(self, start_char: int, end_char: int):
        self.start_char = start_char
        self.end_char = end_char


class _SimpleDoc:
    def __init__(self, sents: List[_SimpleSentence]):
        self.sents = sents


class _RegexSentencizer:
    _split_re = re.compile(r"(?<=[.!?])\s+")

    def __call__(self, text: str) -> _SimpleDoc:
        spans: List[_SimpleSentence] = []
        start = 0
        for match in self._split_re.finditer(text):
            end = match.start()
            if end > start:
                spans.append(_SimpleSentence(start, end))
            start = match.end()
        if start < len(text):
            spans.append(_SimpleSentence(start, len(text)))
        return _SimpleDoc(spans)


def sentence_for_char_span(
    text: str,
    sent_spans: Sequence[Tuple[int, int]],
    answer_span: Tuple[int, int],
) -> str | None:
    span = sentence_span_for_char_span(sent_spans, answer_span)
    if span is None:
        return None
    start, end = span
    snippet = text[start:end].strip()
    if snippet:
        return snippet
    return None


def sentence_span_for_char_span(
    sent_spans: Sequence[Tuple[int, int]],
    answer_span: Tuple[int, int],
) -> Tuple[int, int] | None:
    ans_start, ans_end = answer_span
    if ans_end <= ans_start:
        return None
    for start, end in sent_spans:
        if ans_start >= start and ans_end <= end:
            return (start, end)
    return None
