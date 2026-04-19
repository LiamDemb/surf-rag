"""LLM-based query entity extractor for GraphRetriever."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List

from openai import OpenAI

from surf_rag.core.enrich_entities import normalize_key

logger = logging.getLogger(__name__)


def _parse_entity_list(response_text: str) -> List[str]:
    """Extract JSON array of entity strings from LLM response."""
    text = (response_text or "").strip()
    if not text:
        return []
    # Try to find JSON array
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
    except json.JSONDecodeError:
        pass
    return []


class LLMQueryEntityExtractor:
    """Extract query entities via LLM. Returns normalized strings for graph lookup."""

    def __init__(
        self,
        alias_resolver,
        model_id: str = "",
        prompt_template: str = "",
        max_tokens: int = 256,
    ):
        self.alias_resolver = alias_resolver
        self.model_id = model_id or os.getenv("QUERY_ENTITY_MODEL", "gpt-4o-mini")
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def extract(self, query: str) -> List[str]:
        """Extract entities from query via LLM; return normalized strings."""
        if not query or not query.strip():
            return []

        from surf_rag.core.prompts import get_query_entity_extraction_prompt

        template = self.prompt_template or get_query_entity_extraction_prompt()
        prompt = template.format(query=query.strip())

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.warning("LLM query entity extraction failed: %s", e)
            return []

        content = ""
        if response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", "") or ""

        raw_entities = _parse_entity_list(content)
        seen: set[str] = set()
        result: List[str] = []
        for surface in raw_entities:
            if not surface:
                continue
            norm = (
                self.alias_resolver.normalize(surface)
                if self.alias_resolver
                else normalize_key(surface)
            )
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return sorted(result)
