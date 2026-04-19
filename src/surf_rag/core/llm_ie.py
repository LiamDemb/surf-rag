"""One-pass LLM extraction: entities + triples in a single call.

Uses OpenAI Chat API with tool calling to extract both entity inventory and
relational triples from chunk text. The LLM is the sole authority.
Post-processing uses normalize_key() only (no alias canonicalization) for
duplicate-entity-node graph representation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from surf_rag.core.enrich_entities import normalize_key

logger = logging.getLogger(__name__)

RULE_ID = "LLM_IE_V1"

_EXTRACT_IE_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities_and_triples",
        "description": "Emit extracted entities and triples from the text in one pass.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "surface": {
                                "type": "string",
                                "description": "Entity as mentioned in text",
                            },
                            "type": {
                                "type": "string",
                                "description": "PERSON, ORG, GPE, LOC, EVENT, WORK_OF_ART, NOUN_CHUNK",
                            },
                        },
                        "required": ["surface", "type"],
                    },
                    "description": "All named entities found in the text",
                },
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subj_surface": {"type": "string"},
                            "pred": {"type": "string"},
                            "obj_surface": {"type": "string"},
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["subj_surface", "pred", "obj_surface"],
                    },
                    "description": "Subject-predicate-object triples",
                },
            },
            "required": ["entities", "triples"],
        },
    },
}


def _find_evidence_span(evidence: str, text: str) -> tuple[int, int]:
    """Locate evidence string in text. Returns (start, end) or (-1, -1) if not found."""
    if not evidence or not text:
        return (-1, -1)
    idx = text.find(evidence)
    if idx >= 0:
        return (idx, idx + len(evidence))
    ev_norm = " ".join(evidence.split())
    txt_norm = " ".join(text.split())
    idx = txt_norm.find(ev_norm)
    if idx >= 0:
        return (idx, idx + len(ev_norm))
    return (-1, -1)


def _post_process_ie(
    raw_entities: List[Dict[str, Any]],
    raw_triples: List[Dict[str, Any]],
    text: str,
    chunk_id: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert raw LLM output into corpus metadata schema.

    Uses normalize_key() only (no alias_map) per duplicate-entity-node design.
    Ensures all triple endpoints exist in the entity list.
    """
    entity_by_norm: Dict[str, Dict[str, Any]] = {}
    banned_predicates = {
        "is",
        "was",
        "has",
        "had",
        "mentions",
        "discusses",
        "related_to",
    }

    for e in raw_entities or []:
        surface = (e.get("surface") or "").strip()
        etype = (e.get("type") or "NOUN_CHUNK").strip().upper()
        if not surface:
            continue
        norm = normalize_key(surface)
        if not norm:
            continue
        if norm not in entity_by_norm:
            entity_by_norm[norm] = {
                "surface": surface,
                "norm": norm,
                "type": etype if etype else "NOUN_CHUNK",
                "qid": None,
            }

    seen_triples: set[tuple[str, str, str]] = set()
    relations: List[Dict[str, Any]] = []

    for t in raw_triples or []:
        subj_surface = (t.get("subj_surface") or "").strip()
        pred = (t.get("pred") or "").strip()
        obj_surface = (t.get("obj_surface") or "").strip()
        confidence = t.get("confidence")
        if confidence is None:
            confidence = 0.8
        else:
            confidence = max(0.0, min(1.0, float(confidence)))
        evidence = (t.get("evidence") or "").strip()

        if not subj_surface or not pred or not obj_surface:
            continue
        if pred.lower() in banned_predicates:
            continue

        subj_norm = normalize_key(subj_surface)
        obj_norm = normalize_key(obj_surface)
        if not subj_norm or not obj_norm:
            continue

        key = (subj_norm, pred, obj_norm)
        if key in seen_triples:
            continue
        seen_triples.add(key)

        start_char, end_char = _find_evidence_span(evidence, text)
        match_text = evidence or f"{subj_surface} {pred} {obj_surface}"

        rec: Dict[str, Any] = {
            "subj_surface": subj_surface,
            "obj_surface": obj_surface,
            "subj_norm": subj_norm,
            "pred": pred,
            "obj_norm": obj_norm,
            "source": "llm",
            "rule_id": RULE_ID,
            "confidence": confidence,
            "match_text": match_text,
            "start_char": start_char,
            "end_char": end_char,
        }
        if chunk_id:
            rec["chunk_id"] = chunk_id
        relations.append(rec)

        if subj_norm not in entity_by_norm:
            entity_by_norm[subj_norm] = {
                "surface": subj_surface,
                "norm": subj_norm,
                "type": "NOUN_CHUNK",
                "qid": None,
            }
        if obj_norm not in entity_by_norm:
            entity_by_norm[obj_norm] = {
                "surface": obj_surface,
                "norm": obj_norm,
                "type": "NOUN_CHUNK",
                "qid": None,
            }

    entities = [entity_by_norm[k] for k in sorted(entity_by_norm)]
    return entities, relations


@dataclass
class LLMInformationExtractor:
    """One-pass LLM extractor: entities + triples in a single API call."""

    model_id: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    prompt_template: str = ""
    _client: Optional[OpenAI] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = os.getenv("LLM_IE_MODEL", "gpt-4o-mini")
        if self.temperature == 0.0:
            env_temp = os.getenv("LLM_IE_TEMPERATURE")
            if env_temp is not None:
                self.temperature = float(env_temp)
        if self.max_tokens <= 0:
            self.max_tokens = int(os.getenv("LLM_IE_MAX_TOKENS", "2048"))
        if not self.prompt_template:
            from surf_rag.core.prompts import get_ie_extraction_prompt

            self.prompt_template = get_ie_extraction_prompt()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def extract(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        seed_titles_in_chunk: Optional[List[str]] = None,
        title: str = "N/A",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract entities and triples in one LLM call.

        Returns (entities, relations) in corpus metadata schema.
        """
        text = text or ""
        if not text.strip():
            return [], []

        seed_list = seed_titles_in_chunk or []
        seed_str = (
            "\n".join(f"- {s}" for s in seed_list) if seed_list else "(none detected)"
        )

        prompt = self.prompt_template.format(
            title=title,
            text=text,
            seed_titles_in_chunk=seed_str,
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            tools=[_EXTRACT_IE_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "extract_entities_and_triples"},
            },
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_entities: List[Dict[str, Any]] = []
        raw_triples: List[Dict[str, Any]] = []

        msg = response.choices[0].message if response.choices else None
        if msg and msg.tool_calls:
            for tc in msg.tool_calls:
                if getattr(tc, "function", None) is None:
                    continue
                if getattr(tc.function, "name", None) != "extract_entities_and_triples":
                    continue
                args_str = getattr(tc.function, "arguments", None) or "{}"
                try:
                    args = json.loads(args_str)
                    raw_entities.extend(args.get("entities", []))
                    raw_triples.extend(args.get("triples", []))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse IE tool args: %s", e)

        return _post_process_ie(raw_entities, raw_triples, text, chunk_id)


def build_ie_chat_request(
    prompt: str,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """Build body dict for /v1/chat/completions used by Batch API."""
    model_id = model_id or os.getenv("LLM_IE_MODEL", "gpt-4o-mini")
    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("LLM_IE_TEMPERATURE", "0"))
    )
    max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.getenv("LLM_IE_MAX_TOKENS", "2048"))
    )
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [_EXTRACT_IE_TOOL],
        "tool_choice": {
            "type": "function",
            "function": {"name": "extract_entities_and_triples"},
        },
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def build_ie_batch_line(custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single line for the Batch API input JSONL file."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def parse_ie_batch_output_line(
    line: Dict[str, Any],
) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a Batch API output line into (custom_id, entities, relations)."""
    custom_id = line.get("custom_id") or ""
    raw_entities: List[Dict[str, Any]] = []
    raw_triples: List[Dict[str, Any]] = []

    if line.get("error"):
        return (custom_id, [], [])

    response = line.get("response")
    if not response or response.get("status_code") != 200:
        return (custom_id, [], [])

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return (custom_id, [], [])

    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function") or {}
        if func.get("name") != "extract_entities_and_triples":
            continue
        args_str = func.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
            raw_entities.extend(args.get("entities", []))
            raw_triples.extend(args.get("triples", []))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse ie batch output for %s: %s", custom_id, e)

    return (custom_id, raw_entities, raw_triples)
