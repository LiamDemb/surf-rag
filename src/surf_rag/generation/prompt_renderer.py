"""Turn structured retrieval results into OpenAI chat messages."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass, field
from typing import Dict, List

from surf_rag.core.prompts import GENERATOR_SYSTEM_MESSAGE, get_generator_prompt
from surf_rag.reranking.sentence_reranker import (
    PROMPT_EVIDENCE_KEY,
    PROMPT_EVIDENCE_SENTENCE_SHORTLIST,
)
from surf_rag.retrieval.types import RetrievalResult

_ENV_INCLUDE_GRAPH_PATHS = "INCLUDE_GRAPH_PATHS_IN_PROMPT"


def _include_graph_paths_from_env() -> bool:
    """When INCLUDE_GRAPH_PATHS_IN_PROMPT is 1/true/yes, attach graph path lines to prompts."""
    v = os.getenv(_ENV_INCLUDE_GRAPH_PATHS, "").strip().lower()
    return v in ("1", "true", "yes")


def _evidence_shortlist_xml(retrieval: RetrievalResult) -> str:
    """Render sentence-ranked evidence as structured XML-like lines."""
    lines: List[str] = ["<evidence_shortlist>"]
    for i, ch in enumerate(retrieval.chunks):
        meta = ch.metadata or {}
        sid = f"S{i + 1:03d}"
        parent = str(meta.get("parent_chunk_id") or ch.chunk_id or "")
        src = html.escape(parent, quote=True)
        rank = int(meta.get("parent_chunk_rank", 0)) + 1
        score = float(ch.score)
        title = meta.get("title")
        title_attr = ""
        if title:
            title_attr = f' title="{html.escape(str(title), quote=True)}"'
        body = html.escape(ch.text or "", quote=False)
        lines.append(
            f'  <S id="{sid}" source="{src}" rank="{rank}" score="{score:.4f}"{title_attr}>'
            f"{body}</S>"
        )
    lines.append("</evidence_shortlist>")
    return "\n".join(lines)


def _sentence_context_from_retrieval(
    retrieval: RetrievalResult,
    *,
    include_graph_provenance: bool,
    prompt_style: str,
) -> str:
    if (
        (retrieval.debug_info or {}).get(PROMPT_EVIDENCE_KEY)
        == PROMPT_EVIDENCE_SENTENCE_SHORTLIST
        and prompt_style.strip().lower() == "structured"
    ):
        return _evidence_shortlist_xml(retrieval)
    if (
        (retrieval.debug_info or {}).get(PROMPT_EVIDENCE_KEY)
        == PROMPT_EVIDENCE_SENTENCE_SHORTLIST
    ):
        # Plain join for non-XML runs
        parts: List[str] = []
        for ch in retrieval.chunks:
            meta = ch.metadata or {}
            bits = [f"[{meta.get('parent_chunk_id', '')}]", (ch.text or "").strip()]
            parts.append(" ".join(b for b in bits if b))
        return "\n\n---\n\n".join(parts)
    return "\n\n---\n\n".join(
        _context_strings_from_retrieval(
            retrieval, include_graph_provenance=include_graph_provenance
        )
    )


def _context_strings_from_retrieval(
    retrieval: RetrievalResult, *, include_graph_provenance: bool
) -> List[str]:
    """Build per-chunk context blocks (dense-like text only unless provenance is on)."""
    out: List[str] = []
    for ch in retrieval.chunks:
        meta = ch.metadata or {}
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

    When ``retrieval.debug_info["prompt_evidence"] == "sentence_shortlist"``, the
    context is rendered from sentence items (see ``sentence_rerank_prompt_style``).
    """

    base_prompt: str | None = None
    include_graph_provenance: bool = field(
        default_factory=_include_graph_paths_from_env
    )
    sentence_rerank_prompt_style: str = "structured"

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
            joined = _sentence_context_from_retrieval(
                retrieval,
                include_graph_provenance=self.include_graph_provenance,
                prompt_style=self.sentence_rerank_prompt_style,
            )

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
