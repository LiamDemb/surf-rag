"""Turn structured retrieval results into OpenAI chat messages."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from surf_rag.core.prompts import GENERATOR_SYSTEM_MESSAGE, get_generator_prompt
from surf_rag.retrieval.types import RetrievedChunk, RetrievalResult

_ENV_INCLUDE_GRAPH_PATHS = "INCLUDE_GRAPH_PATHS_IN_PROMPT"


def _include_graph_paths_from_env() -> bool:
    """When INCLUDE_GRAPH_PATHS_IN_PROMPT is 1/true/yes, attach graph path lines to prompts."""
    v = os.getenv(_ENV_INCLUDE_GRAPH_PATHS, "").strip().lower()
    return v in ("1", "true", "yes")


def _format_sentence_window_block(ch: RetrievedChunk) -> str:
    """Format a compact window with parent rank, title, score, and source id."""
    meta = ch.metadata or {}
    wlab = str(meta.get("prompt_w_label") or "W?").strip()
    if wlab and not wlab.startswith("W"):
        wlab = f"W{wlab}"
    rank = int(meta.get("original_chunk_rank", 0))
    wscore = float(meta.get("window_rerank_score", 0.0))
    show_title = bool(meta.get("prompt_include_title", True))
    title_raw = str(meta.get("title") or "").strip() if show_title else ""
    title_esc = title_raw.replace("|", " ") if title_raw else "—"
    source_id = str(meta.get("original_chunk_id", meta.get("source_doc_id", "")) or "")[
        :200
    ]
    header = (
        f"[{wlab} | chunk_rank={rank} | title={title_esc} | "
        f"window_score={wscore:.4f} | source_chunk_id={source_id}]"
    )
    return f"{header}\n{ch.text}"


def _context_strings_from_retrieval(
    retrieval: RetrievalResult, *, include_graph_provenance: bool
) -> List[str]:
    """Build per-chunk context blocks (dense-like text only unless provenance is on)."""
    out: List[str] = []
    for ch in retrieval.chunks:
        meta = ch.metadata or {}
        if str(meta.get("context_unit", "")) == "sentence_window":
            out.append(_format_sentence_window_block(ch))
            continue
        if (
            include_graph_provenance
            and meta.get("branch") == "graph"
            and meta.get("graph_path_lines")
        ):
            lines = meta["graph_path_lines"]
            header = "[{}]".format(" | ".join(lines))
            out.append(f"{header}\n\n{ch.text}")
        else:
            out.append(ch.text)
    return out


@dataclass
class PromptRenderer:
    """Renders QA prompts from retrieval.

    Graph retrieval still stores ``graph_path_lines`` on chunks; by default the prompt
    uses **chunk text only** (same shape as dense). Set ``include_graph_provenance=True``
    or ``INCLUDE_GRAPH_PATHS_IN_PROMPT=1`` to prepend path headers for trace-aware runs.
    """

    base_prompt: str | None = None
    include_graph_provenance: bool = field(
        default_factory=_include_graph_paths_from_env
    )

    def __post_init__(self) -> None:
        if self.base_prompt is None:
            self.base_prompt = get_generator_prompt()

    def to_messages(
        self, query: str, retrieval: RetrievalResult
    ) -> List[Dict[str, str]]:
        """Return OpenAI-style chat messages: system + user."""
        if retrieval.status != "OK" or not retrieval.chunks:
            joined = ""
        else:
            ctx = _context_strings_from_retrieval(
                retrieval, include_graph_provenance=self.include_graph_provenance
            )
            joined = "\n\n---\n\n".join(ctx)

        base = self.base_prompt or ""

        if "{context}" in base and "{question}" in base:
            prompt = base.format(context=joined, question=query)
            system = GENERATOR_SYSTEM_MESSAGE
        else:
            prompt = (
                f"{base}\n\n"
                "Context:\n" + joined + f"\n\nQuestion: {query}\n\n"
                "Call the format_answer tool with your reasoning and a short final answer."
            )
            system = base

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def hashable_user_content(self, query: str, retrieval: RetrievalResult) -> str:
        """User message text only (for logging); full hash should include system."""
        msgs = self.to_messages(query, retrieval)
        return msgs[1]["content"] if len(msgs) > 1 else ""
