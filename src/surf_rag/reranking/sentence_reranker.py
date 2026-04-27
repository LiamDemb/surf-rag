"""Cross-encoder reranking over individual sentences from chunk shortlists."""

from __future__ import annotations

from typing import Any

from surf_rag.benchmark.sentence_utils import build_sentencizer, sentence_spans
from surf_rag.core.model_cache import get_cross_encoder
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult

PROMPT_EVIDENCE_KEY = "prompt_evidence"
PROMPT_EVIDENCE_SENTENCE_SHORTLIST = "sentence_shortlist"


def _word_count(s: str) -> int:
    return len(s.split())


def _collect_sentence_candidates(
    chunks: list[RetrievedChunk],
    sentencizer: Any,
    *,
    max_sentences: int,
) -> list[dict[str, Any]]:
    """Ordered sentence candidates from chunks in rank order."""
    out: list[dict[str, Any]] = []
    for parent_rank, ch in enumerate(chunks):
        text = ch.text or ""
        if not text.strip():
            continue
        spans = sentence_spans(text, sentencizer)
        for si, (start, end) in enumerate(spans):
            if len(out) >= max_sentences:
                return out
            frag = text[start:end].strip()
            if not frag:
                continue
            out.append(
                {
                    "parent": ch,
                    "parent_rank": parent_rank,
                    "sentence_index_in_chunk": si,
                    "start": int(start),
                    "end": int(end),
                    "text": frag,
                }
            )
        if len(out) >= max_sentences:
            break
    return out


def apply_sentence_rerank(
    query: str,
    chunk_ranked: RetrievalResult,
    *,
    cross_encoder_model: str,
    top_k: int = 20,
    max_sentences: int = 48,
    max_words: int = 1280,
    include_title_attr: bool = True,
) -> RetrievalResult:
    """Rerank individual sentences with a cross-encoder; output chunks are sentence items.

    Expects ``chunk_ranked`` to be the chunk-level shortlist (e.g. after CE rerank).
    Sets ``debug_info["prompt_evidence"] == "sentence_shortlist"`` for prompt rendering.
    Sentence candidates are *not* truncated by top-k: all candidates are kept and sorted
    by CE score. ``top_k`` is accepted for backward-compatible call signatures.
    """
    _ = top_k
    if chunk_ranked.status != "OK" or not chunk_ranked.chunks:
        return chunk_ranked

    sentencizer = build_sentencizer()
    candidates = _collect_sentence_candidates(
        list(chunk_ranked.chunks),
        sentencizer,
        max_sentences=max_sentences,
    )
    base_debug: dict[str, Any] = dict(chunk_ranked.debug_info or {})

    if not candidates:
        base_debug["sentence_rerank_skipped"] = "no_sentences"
        return RetrievalResult(
            query=chunk_ranked.query,
            retriever_name=chunk_ranked.retriever_name,
            status=chunk_ranked.status,
            chunks=list(chunk_ranked.chunks),
            latency_ms=dict(chunk_ranked.latency_ms),
            error=chunk_ranked.error,
            debug_info=base_debug,
        )

    model = get_cross_encoder(cross_encoder_model)
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    for i, cand in enumerate(candidates):
        cand["_ce_score"] = float(scores[i])

    candidates.sort(key=lambda c: -c["_ce_score"])

    out_chunks: list[RetrievedChunk] = []
    total_words = 0
    for cand in candidates:
        sent_text = cand["text"]
        wc = _word_count(sent_text)
        if total_words + wc > max_words and out_chunks:
            break
        total_words += wc
        rnk = len(out_chunks)

        parent: RetrievedChunk = cand["parent"]
        pmeta = dict(parent.metadata or {})
        parent_rerank = pmeta.get("rerank_score", parent.score)

        sid = f"{parent.chunk_id}#s{cand['sentence_index_in_chunk']}"
        meta: dict[str, Any] = {
            "evidence_kind": "sentence",
            "parent_chunk_id": parent.chunk_id,
            "parent_chunk_rank": cand["parent_rank"],
            "sentence_index_in_chunk": cand["sentence_index_in_chunk"],
            "char_start": cand["start"],
            "char_end": cand["end"],
            "parent_rerank_score": float(parent_rerank),
            "parent_retrieval_score": pmeta.get("retrieval_score"),
            "branch": pmeta.get("branch"),
        }
        if pmeta.get("graph_path_lines") is not None:
            meta["graph_path_lines"] = pmeta["graph_path_lines"]
        title = pmeta.get("title")
        src = pmeta.get("source")
        if include_title_attr and title:
            meta["title"] = str(title)
        if src:
            meta["source_dataset"] = str(src)

        out_chunks.append(
            RetrievedChunk(
                chunk_id=sid,
                text=sent_text,
                score=cand["_ce_score"],
                rank=rnk,
                metadata=meta,
            )
        )

    base_debug[PROMPT_EVIDENCE_KEY] = PROMPT_EVIDENCE_SENTENCE_SHORTLIST

    return RetrievalResult(
        query=chunk_ranked.query,
        retriever_name=f"{chunk_ranked.retriever_name}+sentence_rerank",
        status="OK",
        chunks=out_chunks,
        latency_ms=dict(chunk_ranked.latency_ms),
        error=chunk_ranked.error,
        debug_info=base_debug,
    )
