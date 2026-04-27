"""Sentence-window construction, cross-encoder scoring, and diversity selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from surf_rag.benchmark.sentence_utils import build_sentencizer, sentence_spans
from surf_rag.core.model_cache import get_cross_encoder
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class SentenceWindowConfig:
    """Controls sentence-window context selection (reader stage, not retrieval)."""

    radius: int = 1
    max_windows: int = 12
    min_windows: int = 8
    max_words: int = 1280
    max_subwindow_words: int = 180
    min_top_chunk_coverage: int = 3
    min_distinct_parent_chunks: int = 4
    max_windows_per_chunk: int = 2
    iou_select_threshold: float = 0.35
    premerge_iou: float = 0.35
    premerge_max_gap_chars: int = 48
    ce_relax_margin: float = 3.0
    ce_filler_top_ranks: int = 3
    filler_title_overlap: bool = True
    filler_novel_parent_max_rank: int = 10
    merge_overlaps: bool = True
    duplicate_filter: bool = True
    include_title: bool = True


@dataclass(frozen=True)
class _WindowCandidate:
    chunk_id: str
    chunk_rank: int
    window_index: int
    text: str  # body only; title is in chunk metadata for prompt headers
    ce_score: float
    sent_start: int
    sent_end: int
    char_start: int
    char_end: int


def _word_count(s: str) -> int:
    return len(s.split()) if s.strip() else 0


def _normalize_for_dup(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().casefold())


def _raw_span_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    inter = max(0, min(e1, e2) - max(s1, s2))
    len1, len2 = e1 - s1, e2 - s2
    if len1 <= 0 or len2 <= 0:
        return 0.0
    union = len1 + len2 - inter
    return float(inter) / float(union) if union else 0.0


def _char_iou(a: _WindowCandidate, b: _WindowCandidate) -> float:
    """IoU on character spans within the same parent chunk."""
    if a.chunk_id != b.chunk_id:
        return 0.0
    return _raw_span_iou(a.char_start, a.char_end, b.char_start, b.char_end)


def _refine_span_to_subspans(
    full: str, c0: int, c1: int, max_w: int
) -> List[Tuple[int, int]]:
    """Split a char span until each piece has at most max_w words."""
    if c1 <= c0:
        return []
    seg = full[c0:c1].strip()
    if not seg:
        return []
    if _word_count(full[c0:c1]) <= max_w:
        return [(c0, c1)]
    # Avoid infinite recursion on unsplittable whitespace-only middle
    sub = full[c0:c1]
    for pat in (r"\n\n+", r"\n+", r";\s+"):
        if re.search(pat, sub):
            parts: List[Tuple[int, int]] = []
            pos = 0
            for m in re.finditer(pat, sub):
                chunk = sub[pos : m.start()]
                if chunk.strip():
                    s0, s1 = c0 + pos, c0 + m.start()
                    parts.extend(_refine_span_to_subspans(full, s0, s1, max_w))
                pos = m.end()
            if pos < len(sub) and sub[pos:].strip():
                parts.extend(_refine_span_to_subspans(full, c0 + pos, c1, max_w))
            if len(parts) >= 1:
                return parts
    return _word_chunk_spans(full, c0, c1, max_w)


def _word_chunk_spans(full: str, c0: int, c1: int, max_w: int) -> List[Tuple[int, int]]:
    sub = full[c0:c1]
    it = list(_TOKEN_RE.finditer(sub))
    if not it:
        return []
    out: List[Tuple[int, int]] = []
    i = 0
    ntok = len(it)
    while i < ntok:
        start_local = it[i].start()
        w = 0
        j = i
        while j < ntok and w < max_w:
            w += 1
            j += 1
        end_local = it[j - 1].end()
        s0, s1 = c0 + start_local, c0 + end_local
        if s1 > s0:
            out.append((s0, s1))
        i = j
    return out if out else [(c0, c1)]


def _build_windows_for_chunk(
    chunk_text: str,
    sentencizer: object,
    radius: int,
) -> List[Tuple[int, int, int, int]]:
    """Return (sent_start, sent_end, char_start, char_end) per window."""
    if not (chunk_text or "").strip():
        return []
    spans = sentence_spans(chunk_text, sentencizer)
    if not spans:
        return [(0, 0, 0, len(chunk_text))]
    n = len(spans)
    out: List[Tuple[int, int, int, int]] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n - 1, i + radius)
        c0 = spans[lo][0]
        c1 = spans[hi][1]
        out.append((lo, hi, c0, c1))
    return out


def _merge_windows(
    windows: List[Tuple[int, int, int, int]], merge: bool
) -> List[Tuple[int, int, int, int]]:
    if not merge or len(windows) <= 1:
        return windows
    seen: set[Tuple[int, int]] = set()
    merged: List[Tuple[int, int, int, int]] = []
    for w in windows:
        key = (w[0], w[1])
        if key in seen:
            continue
        seen.add(key)
        merged.append(w)
    return merged


def _distinct_chunk_ids(selected: List[_WindowCandidate]) -> int:
    return len({c.chunk_id for c in selected})


def _norm_tokens(s: str) -> set[str]:
    t = re.sub(r"[^\w\s]", " ", s.lower())
    return {x for x in t.split() if len(x) > 2}


def _title_query_overlap(query: str, title: str) -> bool:
    if not (query or "").strip() or not (title or "").strip():
        return False
    qt = _norm_tokens(query)
    tt = _norm_tokens(title)
    if not qt or not tt:
        return False
    return bool(qt & tt)


def _filler_add_ok(
    w: _WindowCandidate,
    query: str,
    all_cands: List[_WindowCandidate],
    cfg: SentenceWindowConfig,
    chunk_titles: Dict[str, str],
    selected_ids: set[str],
) -> bool:
    if not all_cands:
        return True
    ref = max(c.ce_score for c in all_cands)
    if w.ce_score >= ref - cfg.ce_relax_margin:
        return True
    if w.chunk_rank < cfg.ce_filler_top_ranks:
        return True
    t = str(chunk_titles.get(w.chunk_id) or "")
    if cfg.filler_title_overlap and _title_query_overlap(query, t):
        return True
    if (
        w.chunk_id not in selected_ids
        and w.chunk_rank < cfg.filler_novel_parent_max_rank
    ):
        return True
    return False


def _spans_conflict(
    w: _WindowCandidate, k: _WindowCandidate, *, iou_t: float, max_gap: int
) -> bool:
    if w.chunk_id != k.chunk_id:
        return False
    iou = _raw_span_iou(k.char_start, k.char_end, w.char_start, w.char_end)
    gap0 = w.char_start - k.char_end
    gap1 = k.char_start - w.char_end
    adjacent = 0 <= gap0 <= max_gap or 0 <= gap1 <= max_gap
    return bool(iou > iou_t or (iou > 0.02 and adjacent))


def _premerge_per_chunk(
    cands: List[_WindowCandidate],
    *,
    iou_t: float,
    max_gap: int,
) -> List[_WindowCandidate]:
    """Remove overlapping or adjacent same-chunk near-duplicates; keep higher CE."""
    if len(cands) <= 1:
        return cands
    by: Dict[str, List[_WindowCandidate]] = {}
    for c in cands:
        by.setdefault(c.chunk_id, []).append(c)
    out: List[_WindowCandidate] = []
    for _cid, group in by.items():
        order = sorted(group, key=lambda w: (-w.ce_score, w.char_start))
        kept: List[_WindowCandidate] = []
        for w in order:
            blocked = any(
                _spans_conflict(w, k, iou_t=iou_t, max_gap=max_gap)
                and k.ce_score >= w.ce_score
                for k in kept
            )
            if blocked:
                continue
            rem = [
                j
                for j, k in enumerate(kept)
                if _spans_conflict(w, k, iou_t=iou_t, max_gap=max_gap)
                and k.ce_score < w.ce_score
            ]
            for j in sorted(rem, reverse=True):
                del kept[j]
            kept.append(w)
        for wi, w in enumerate(sorted(kept, key=lambda x: (x.char_start, -x.ce_score))):
            out.append(
                _WindowCandidate(
                    chunk_id=w.chunk_id,
                    chunk_rank=w.chunk_rank,
                    window_index=wi,
                    text=w.text,
                    ce_score=w.ce_score,
                    sent_start=-1,
                    sent_end=-1,
                    char_start=w.char_start,
                    char_end=w.char_end,
                )
            )
    return out


def _select_diverse(
    cands: List[_WindowCandidate],
    chunk_order: Sequence[str],
    query: str,
    chunk_titles: Dict[str, str],
    cfg: SentenceWindowConfig,
) -> List[Tuple[_WindowCandidate, float]]:
    """Return (candidate, selection_score) with higher = earlier in prompt order."""
    if not cands:
        return []
    by_id: Dict[str, List[_WindowCandidate]] = {}
    for c in cands:
        by_id.setdefault(c.chunk_id, []).append(c)
    for lst in by_id.values():
        lst.sort(key=lambda w: -w.ce_score)

    selected: List[_WindowCandidate] = []
    seen_keys: set[Tuple[str, int]] = set()
    seen_text: set[str] = set()
    counts: Dict[str, int] = {}
    used_words = 0
    iou_threshold = float(cfg.iou_select_threshold)

    def overlaps_selected(w: _WindowCandidate) -> bool:
        for s in selected:
            if _char_iou(s, w) > iou_threshold:
                return True
        return False

    def try_add(w: _WindowCandidate, *, use_filler_rules: bool = False) -> bool:
        nonlocal used_words
        wk = (w.chunk_id, w.window_index)
        if wk in seen_keys:
            return False
        if len(selected) >= cfg.max_windows:
            return False
        if counts.get(w.chunk_id, 0) >= cfg.max_windows_per_chunk:
            return False
        if overlaps_selected(w):
            return False
        if use_filler_rules:
            sel_ids = {c.chunk_id for c in selected}
            if not _filler_add_ok(w, query, cands, cfg, chunk_titles, sel_ids):
                return False
        wc = _word_count(w.text)
        if used_words + wc > cfg.max_words:
            return False
        if cfg.duplicate_filter:
            k = _normalize_for_dup(w.text)
            if k in seen_text:
                return False
            seen_text.add(k)
        seen_keys.add(wk)
        selected.append(w)
        counts[w.chunk_id] = counts.get(w.chunk_id, 0) + 1
        used_words += wc
        return True

    coverage_n = min(cfg.min_top_chunk_coverage, len(chunk_order))
    for cid in chunk_order[:coverage_n]:
        pool = by_id.get(cid) or []
        if not pool:
            continue
        best = max(pool, key=lambda x: x.ce_score)
        try_add(best, use_filler_rules=False)

    target_distinct = min(cfg.min_distinct_parent_chunks, len(chunk_order))
    safety = 0
    while (
        _distinct_chunk_ids(selected) < target_distinct
        and len(selected) < cfg.max_windows
        and safety < len(chunk_order) * 3
    ):
        safety += 1
        before = len(selected)
        for cid in chunk_order:
            if _distinct_chunk_ids(selected) >= target_distinct:
                break
            if cid in {c.chunk_id for c in selected}:
                continue
            pool = by_id.get(cid) or []
            for w in sorted(pool, key=lambda x: -x.ce_score):
                if try_add(w, use_filler_rules=False):
                    break
        if len(selected) == before:
            break

    sel_keys = {(c.chunk_id, c.window_index) for c in selected}
    pool_all = [c for c in cands if (c.chunk_id, c.window_index) not in sel_keys]
    pool_all.sort(key=lambda w: -w.ce_score)
    for w in pool_all:
        if len(selected) >= cfg.max_windows:
            break
        try_add(w, use_filler_rules=True)

    if len(selected) < cfg.min_windows:
        taken = {(c.chunk_id, c.window_index) for c in selected}
        pool_rest = [c for c in cands if (c.chunk_id, c.window_index) not in taken]
        pool_rest.sort(key=lambda w: -w.ce_score)
        for w in pool_rest:
            if len(selected) >= cfg.max_windows:
                break
            if len(selected) >= cfg.min_windows:
                break
            try_add(w, use_filler_rules=True)

    n = len(selected)
    return [(c, float(n - i)) for i, c in enumerate(selected)]


class SentenceWindowReranker:
    """Cross-encode overlapping sentence windows; select a diverse, budgeted set."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        config: SentenceWindowConfig | None = None,
    ) -> None:
        self._model = get_cross_encoder(model_name)
        self._model_name = model_name
        self._config = config or SentenceWindowConfig()
        self._sentencizer = build_sentencizer()

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self, query: str, result: RetrievalResult, top_k: int
    ) -> RetrievalResult:
        del top_k
        if result.status != "OK" or not result.chunks:
            return result
        cfg = self._config
        flat: List[_WindowCandidate] = []
        for chunk_rank, ch in enumerate(result.chunks):
            raw = ch.text or ""
            wspans = _build_windows_for_chunk(
                raw, self._sentencizer, max(0, cfg.radius)
            )
            wspans = _merge_windows(wspans, cfg.merge_overlaps)
            subwi = 0
            for _wi, (lo, hi, c0, c1) in enumerate(wspans):
                for s0, s1 in _refine_span_to_subspans(
                    raw, c0, c1, cfg.max_subwindow_words
                ):
                    body = raw[s0:s1].strip()
                    if not body:
                        continue
                    flat.append(
                        _WindowCandidate(
                            chunk_id=ch.chunk_id,
                            chunk_rank=chunk_rank,
                            window_index=subwi,
                            text=body,
                            ce_score=0.0,
                            sent_start=lo,
                            sent_end=hi,
                            char_start=s0,
                            char_end=s1,
                        )
                    )
                    subwi += 1
        if not flat:
            return RetrievalResult(
                query=result.query,
                retriever_name=f"{result.retriever_name}+sw",
                status="OK",
                chunks=[],
                latency_ms=dict(result.latency_ms),
                error=result.error,
                debug_info=dict(result.debug_info) if result.debug_info else None,
            )

        pairs = [[query, c.text] for c in flat]
        scores = self._model.predict(pairs, show_progress_bar=False)
        rescored: List[_WindowCandidate] = []
        for i, c in enumerate(flat):
            sc = float(scores[i]) if i < len(scores) else 0.0
            rescored.append(
                _WindowCandidate(
                    chunk_id=c.chunk_id,
                    chunk_rank=c.chunk_rank,
                    window_index=c.window_index,
                    text=c.text,
                    ce_score=sc,
                    sent_start=c.sent_start,
                    sent_end=c.sent_end,
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
            )
        rescored = _premerge_per_chunk(
            rescored,
            iou_t=cfg.premerge_iou,
            max_gap=cfg.premerge_max_gap_chars,
        )
        chunk_order = [c.chunk_id for c in result.chunks]
        chunk_titles: Dict[str, str] = {}
        for c in result.chunks:
            m = c.metadata or {}
            chunk_titles[c.chunk_id] = str(m.get("title") or "")
        selected = _select_diverse(rescored, chunk_order, query, chunk_titles, cfg)
        return _build_sw_result(
            result, selected, include_title_in_headers=cfg.include_title
        )


def _build_sw_result(
    result: RetrievalResult,
    selected: List[Tuple[_WindowCandidate, float]],
    *,
    include_title_in_headers: bool,
) -> RetrievalResult:
    out_chunks: list[RetrievedChunk] = []
    chunk_by_id = {c.chunk_id: c for c in result.chunks}
    for order, (cand, pos_score) in enumerate(selected):
        orig = chunk_by_id.get(cand.chunk_id)
        ometa = dict(orig.metadata) if orig else {}
        title = ometa.get("title")
        wmeta: dict[str, object] = {
            **ometa,
            "context_unit": "sentence_window",
            "original_chunk_id": cand.chunk_id,
            "window_index": cand.window_index,
            "sentence_start": cand.sent_start,
            "sentence_end": cand.sent_end,
            "char_start": cand.char_start,
            "char_end": cand.char_end,
            "window_rerank_score": cand.ce_score,
            "retrieval_score": float(orig.score) if orig else 0.0,
            "original_chunk_rank": cand.chunk_rank,
            "prompt_w_label": f"W{order + 1}",
            "prompt_include_title": include_title_in_headers,
        }
        if include_title_in_headers and title is not None:
            wmeta["title"] = title
        cid = f"{cand.chunk_id}::sw{cand.window_index}"
        out_chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                text=cand.text,
                score=float(pos_score),
                rank=order,
                metadata=wmeta,
            )
        )
    return RetrievalResult(
        query=result.query,
        retriever_name=f"{result.retriever_name}+sw",
        status="OK",
        chunks=out_chunks,
        latency_ms=dict(result.latency_ms),
        error=result.error,
        debug_info=dict(result.debug_info) if result.debug_info else None,
    )
