from __future__ import annotations

from typing import Any, Dict, List, Tuple


def extract_2wiki_support_lines(row: Dict[str, Any]) -> List[Tuple[str, int, str]]:
    """
    Extract ordered (title, sent_id, sentence) tuples from a 2Wiki row.

    Sentences are deduplicated in order (same behavior as legacy string extraction).
    """
    supporting_facts = row.get("supporting_facts") or {}
    context = row.get("context") or {}
    sf_titles = supporting_facts.get("title")
    sf_sent_ids = supporting_facts.get("sent_id")
    ctx_titles = context.get("title")
    ctx_sentences = context.get("sentences")
    if not isinstance(sf_titles, list) or not isinstance(sf_sent_ids, list):
        return []
    if not isinstance(ctx_titles, list) or not isinstance(ctx_sentences, list):
        return []

    title_to_indices: Dict[str, List[int]] = {}
    for idx, title in enumerate(ctx_titles):
        key = str(title).strip()
        if not key:
            continue
        title_to_indices.setdefault(key, []).append(idx)

    extracted: List[Tuple[str, int, str]] = []
    for raw_title, raw_sent_id in zip(sf_titles, sf_sent_ids):
        title = str(raw_title).strip()
        if not title:
            continue
        try:
            sent_id = int(raw_sent_id)
        except (TypeError, ValueError):
            continue
        if sent_id < 0:
            continue

        for ctx_idx in title_to_indices.get(title, []):
            article_sentences = (
                ctx_sentences[ctx_idx] if ctx_idx < len(ctx_sentences) else None
            )
            if not isinstance(article_sentences, list):
                continue
            if sent_id >= len(article_sentences):
                continue
            sentence = str(article_sentences[sent_id]).strip()
            if sentence:
                extracted.append((title, sent_id, sentence))
                break

    seen: set[str] = set()
    unique: List[Tuple[str, int, str]] = []
    for title, sent_id, sentence in extracted:
        if sentence in seen:
            continue
        seen.add(sentence)
        unique.append((title, sent_id, sentence))
    return unique


def extract_2wiki_support_sentences(row: Dict[str, Any]) -> List[str]:
    """Sentence strings only, deduped (same order as :func:`extract_2wiki_support_lines`)."""
    return [t[2] for t in extract_2wiki_support_lines(row)]
