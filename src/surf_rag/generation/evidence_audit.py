"""Preflight checks on rendered user prompts before OpenAI batch submission."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# Default: evidence v2 hard cap per window body (see sentence_window max_subwindow_words).
DEFAULT_MAX_WINDOW_BLOCK_WORDS = 220


def _count_w_headers(text: str) -> int:
    return len(re.findall(r"^\[W\d+ \|", text, flags=re.MULTILINE))


def _word_count(s: str) -> int:
    return len(s.split()) if s.strip() else 0


def _window_block_word_counts(ctx: str) -> list[int]:
    """Word counts for text under each [Wn | ...] header (body only)."""
    parts = re.split(r"^\[W\d+ \|[^\n]*\]\s*", ctx, flags=re.MULTILINE)
    counts: list[int] = []
    for p in parts[1:]:
        body = p.strip()
        if body:
            counts.append(_word_count(body))
    return counts


@dataclass(frozen=True)
class EvidenceAuditResult:
    """Batch-level + per-prompt heuristics for evidence shape."""

    n_prompts: int
    w_header_counts: list[int]
    mean_w_headers: float
    mean_context_words: float
    mean_block_words: list[float]  # per-prompt: mean window body words
    max_block_words_per_prompt: list[int]  # largest single window body in each prompt
    batch_max_block_words: float
    suspicious_flags: list[str]
    level: Literal["ok", "warn", "fail"]

    def summary_line(self) -> str:
        return (
            f"evidence_audit level={self.level} prompts={self.n_prompts} "
            f"mean_w_headers={self.mean_w_headers:.2f} mean_ctx_words={self.mean_context_words:.1f} "
            f"batch_max_block_words={self.batch_max_block_words:.1f} "
            f"flags={self.suspicious_flags or '[]'}"
        )


def _context_segment(user_text: str) -> str:
    """Best-effort slice to the evidence portion (for word stats)."""
    low = user_text.lower()
    for key in (
        "evidence (selected windows):",
        "evidence:\n",
        "context:\n",
    ):
        j = low.find(key)
        if j >= 0:
            seg = user_text[j + len(key) :]
            qm = re.search(r"\n\nQuestion:\s*", seg, re.IGNORECASE)
            if qm:
                return seg[: qm.start()]
            return seg
    return user_text


def audit_user_prompt(
    user_text: str,
    *,
    expect_sentence_windows: bool,
    max_window_block_words: int = DEFAULT_MAX_WINDOW_BLOCK_WORDS,
) -> dict[str, Any]:
    """Heuristics for a **single** user message (after rendering)."""
    ctx = _context_segment(user_text)
    w_n = _count_w_headers(user_text)
    big_seps = ctx.count("---\n\n")
    ctx_words = _word_count(ctx)
    wblock_counts = _window_block_word_counts(ctx) if w_n > 0 else []
    max_block = max(wblock_counts) if wblock_counts else 0
    mean_b = float(sum(wblock_counts) / len(wblock_counts)) if wblock_counts else 0.0
    out: dict[str, Any] = {
        "w_headers": w_n,
        "context_words": ctx_words,
        "mean_window_words": mean_b,
        "max_block_words": max_block,
        "triple_break_sections": big_seps,
    }
    flags: list[str] = []
    if expect_sentence_windows:
        if w_n == 0 and ctx_words > 200:
            flags.append("no_W_headers_but_substantial_context")
        if w_n > 0 and w_n < 3:
            flags.append("fewer_than_3_windows")
        if w_n > 0 and mean_b > 200:
            flags.append("mean_window_words_very_high")
        if w_n > 0 and max_block > max_window_block_words:
            flags.append("megawindow_block_exceeds_cap")
        if w_n == 0 and big_seps >= 4 and ctx_words > 2000:
            flags.append("looks_like_five_full_chunk_separators")
    out["suspicious"] = bool(flags)
    out["flags"] = flags
    return out


def audit_sentence_window_batch(
    user_messages: list[str],
    *,
    max_window_block_words: int = DEFAULT_MAX_WINDOW_BLOCK_WORDS,
) -> EvidenceAuditResult:
    """Summarize a batch of user prompts; intended for `reranker=sentence_window` runs."""
    if not user_messages:
        return EvidenceAuditResult(
            n_prompts=0,
            w_header_counts=[],
            mean_w_headers=0.0,
            mean_context_words=0.0,
            mean_block_words=[],
            max_block_words_per_prompt=[],
            batch_max_block_words=0.0,
            suspicious_flags=[],
            level="ok",
        )
    wcs: list[int] = []
    cwords: list[float] = []
    bwords: list[float] = []
    max_per: list[int] = []
    flags: list[str] = []
    for u in user_messages:
        r = audit_user_prompt(
            u,
            expect_sentence_windows=True,
            max_window_block_words=max_window_block_words,
        )
        wcs.append(int(r["w_headers"]))
        cwords.append(float(r["context_words"]))
        bwords.append(float(r.get("mean_window_words") or 0.0))
        max_per.append(int(r.get("max_block_words") or 0))
        for f in r.get("flags") or []:
            if f not in flags:
                flags.append(f)
    n = len(user_messages)
    mean_w = sum(wcs) / n
    mean_ctx = sum(cwords) / n
    mean_b = sum(bwords) / n if bwords else 0.0
    batch_max_block = float(max(max_per)) if max_per else 0.0

    level: Literal["ok", "warn", "fail"] = "ok"
    if mean_w < 3 and mean_ctx > 400:
        level = "fail"
        if "no_W_headers" not in " ".join(flags):
            flags.append("batch_no_or_few_w_headers")
    if mean_w > 0 and (mean_b > 200 or min(wcs) < 3):
        level = "warn" if level == "ok" else level
        if mean_b > 200:
            flags.append("batch_mean_window_words_high")
    if "megawindow_block_exceeds_cap" in flags or batch_max_block > float(
        max_window_block_words
    ):
        level = "fail"
        if "megawindow_block_exceeds_cap" not in flags:
            flags.append("batch_contains_megawindow")
    if "looks_like_five_full_chunk_separators" in flags or (
        "no_W_headers_but_substantial_context" in str(flags) and mean_ctx > 2200
    ):
        level = "fail"

    return EvidenceAuditResult(
        n_prompts=n,
        w_header_counts=wcs,
        mean_w_headers=mean_w,
        mean_context_words=mean_ctx,
        mean_block_words=bwords,
        max_block_words_per_prompt=max_per,
        batch_max_block_words=batch_max_block,
        suspicious_flags=flags,
        level=level,
    )
