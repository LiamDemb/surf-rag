"""Title-localized alignment of 2Wiki gold support sentences to modern corpus text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Sequence

import numpy as np

from surf_rag.benchmark.corpus_filter import contains_normalized, normalize_for_matching
from surf_rag.benchmark.sentence_utils import build_sentencizer
from surf_rag.core.embedder import SentenceTransformersEmbedder


class GoldSupportAnchor(NamedTuple):
    """One gold support line with Wikipedia title and optional dataset sent index."""

    title: str
    sent_id: int
    sentence: str


# ROUGE-L (sentence-level F1 based on longest common subsequence of tokens)
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_prev = dp[i - 1]
        row_cur = dp[i]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row_cur[j] = row_prev[j - 1] + 1
            else:
                row_cur[j] = max(row_prev[j], row_cur[j - 1])
    return dp[n][m]


def rouge_l_f1(reference: str, candidate: str) -> float:
    """ROUGE-L F1 on whitespace-tokenized, casefolded words."""
    r = reference.casefold().split()
    c = candidate.casefold().split()
    if not r or not c:
        return 0.0
    lcs = _lcs_length(r, c)
    if lcs == 0:
        return 0.0
    prec = lcs / len(c)
    rec = lcs / len(r)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _tokenize_for_dedupe(text: str) -> str:
    return normalize_for_matching(text)


# Corpus title index
def sentences_from_chunk_text(text: str, sentencizer) -> List[str]:
    """Split chunk text into sentences; drop empties."""
    from surf_rag.benchmark.sentence_utils import sentence_spans

    spans = sentence_spans(text, sentencizer)
    out: List[str] = []
    for start, end in spans:
        s = text[start:end].strip()
        if s:
            out.append(s)
    return out


def build_title_to_candidate_sentences(
    corpus_rows: Sequence[Mapping[str, Any]],
    *,
    sentencizer=None,
) -> Dict[str, List[str]]:
    """Group all unique sentences from corpus chunks by normalized title key."""
    if sentencizer is None:
        sentencizer = build_sentencizer()
    title_to_raw: Dict[str, List[str]] = {}
    for row in corpus_rows:
        title = row.get("title")
        if title is None or not str(title).strip():
            continue
        key = str(title).strip()
        text = str(row.get("text", "") or "")
        if not text.strip():
            continue
        title_to_raw.setdefault(key, []).append(text)

    result: Dict[str, List[str]] = {}
    for title, texts in title_to_raw.items():
        seen: set[str] = set()
        ordered: List[str] = []
        for t in texts:
            for sent in sentences_from_chunk_text(t, sentencizer):
                dedupe_key = _tokenize_for_dedupe(sent)
                if not dedupe_key or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                ordered.append(sent)
        result[title] = ordered
    return result


@dataclass(frozen=True)
class SupportAlignmentDecision:
    """Outcome for one support fact."""

    original: str
    replacement: str | None
    title: str
    sent_id: int
    semantic_cosine: float
    rouge_l: float
    reason: str
    # Best corpus candidate
    nearest_candidate: str | None = None
    nearest_semantic_cosine: float | None = None
    nearest_rouge_l: float | None = None

    @property
    def replaced(self) -> bool:
        return self.replacement is not None and self.replacement != self.original


def _already_present_in_candidates(gold: str, candidates: Sequence[str]) -> bool:
    """Gold is considered present if it matches or is contained in a candidate."""
    g = normalize_for_matching(gold)
    if not g:
        return True
    for c in candidates:
        if normalize_for_matching(c) == g:
            return True
        if contains_normalized(c, gold):
            return True
    return False


def _best_neighbor_index(
    gold: str, candidates: Sequence[str], sem_scores: np.ndarray
) -> int:
    """Argmax by highest cosine, then ROUGE-L, then lexicographic candidate text."""
    return int(
        min(
            range(len(candidates)),
            key=lambda i: (
                -float(sem_scores[i]),
                -rouge_l_f1(gold, candidates[i]),
                candidates[i],
            ),
        )
    )


def _cosine_matrix(
    embedder: SentenceTransformersEmbedder, texts: Sequence[str]
) -> np.ndarray:
    """Rows are normalized embedding vectors."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = embedder.model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def align_one_support_fact(
    anchor: GoldSupportAnchor,
    title_to_candidates: Mapping[str, Sequence[str]],
    embedder: SentenceTransformersEmbedder,
    *,
    tau_sem: float,
    tau_lex: float,
) -> SupportAlignmentDecision:
    """
    Pick a replacement sentence from corpus candidates for ``anchor.title`` only.

    Accepts iff ROUGE-L >= tau_lex AND cosine >= tau_sem (both required).
    If multiple candidates qualify, picks highest cosine, then ROUGE-L, then text.
    """
    title = anchor.title.strip()
    gold = anchor.sentence
    candidates = list(title_to_candidates.get(title, []))

    if not candidates:
        return SupportAlignmentDecision(
            original=gold,
            replacement=None,
            title=title,
            sent_id=anchor.sent_id,
            semantic_cosine=0.0,
            rouge_l=0.0,
            reason="no_corpus_chunks_for_title",
        )

    if _already_present_in_candidates(gold, candidates):
        return SupportAlignmentDecision(
            original=gold,
            replacement=None,
            title=title,
            sent_id=anchor.sent_id,
            semantic_cosine=1.0,
            rouge_l=1.0,
            reason="already_present",
        )

    # Encode gold once + all candidates; cosine = dot product (normalized).
    all_texts = [gold] + candidates
    mat = _cosine_matrix(embedder, all_texts)
    gold_vec = mat[0]
    cand_vecs = mat[1:]
    sem_scores = (cand_vecs @ gold_vec).astype(np.float64)

    best_i = _best_neighbor_index(gold, candidates, sem_scores)
    nearest = candidates[best_i]
    nearest_sem = float(sem_scores[best_i])
    nearest_rl = rouge_l_f1(gold, nearest)

    # Deterministic selection: sort by (-sem, -rl, cand text)
    qualifying: List[tuple[float, float, str, int]] = []
    for i, cand in enumerate(candidates):
        sem = float(sem_scores[i])
        rl = rouge_l_f1(gold, cand)
        if sem >= tau_sem and rl >= tau_lex:
            qualifying.append((sem, rl, cand, i))

    if not qualifying:
        return SupportAlignmentDecision(
            original=gold,
            replacement=None,
            title=title,
            sent_id=anchor.sent_id,
            semantic_cosine=nearest_sem,
            rouge_l=nearest_rl,
            reason="below_thresholds",
            nearest_candidate=nearest,
            nearest_semantic_cosine=nearest_sem,
            nearest_rouge_l=nearest_rl,
        )

    qualifying.sort(key=lambda t: (-t[0], -t[1], t[2]))
    _sem, _rl, replacement, idx = qualifying[0]
    rep_sem = float(sem_scores[idx])
    rep_rl = rouge_l_f1(gold, replacement)
    return SupportAlignmentDecision(
        original=gold,
        replacement=replacement,
        title=title,
        sent_id=anchor.sent_id,
        semantic_cosine=rep_sem,
        rouge_l=rep_rl,
        reason="aligned",
        nearest_candidate=replacement,
        nearest_semantic_cosine=rep_sem,
        nearest_rouge_l=rep_rl,
    )


def align_gold_support_anchors(
    anchors: Sequence[GoldSupportAnchor],
    title_to_candidates: Mapping[str, Sequence[str]],
    embedder: SentenceTransformersEmbedder,
    *,
    tau_sem: float,
    tau_lex: float,
) -> List[SupportAlignmentDecision]:
    """Align each anchor independently (order preserved)."""
    return [
        align_one_support_fact(
            a, title_to_candidates, embedder, tau_sem=tau_sem, tau_lex=tau_lex
        )
        for a in anchors
    ]
