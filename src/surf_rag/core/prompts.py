"""Central prompt definitions for LLM-based components of the system"""

from __future__ import annotations

import os
from pathlib import Path

# Generator Prompt
GENERATOR_SYSTEM_MESSAGE = (
    "You are a strict QA system. Answer based ONLY on the provided context. "
    "You must call the format_answer tool with reasoning and a short final answer."
)

BASE_PROMPT = (
    "You are a strict QA system. Answer based ONLY on the provided context."
    "\n\n"
    "You must call the function format_answer with:\n"
    "- reasoning: concise rationale grounded in the context only.\n"
    "- answer: the final short answer only (suitable for exact-match evaluation).\n\n"
    "EXAMPLES:\n"
    "Context: 'Toy Story features a boy named Andy who has a younger sister named Molly.'\n"
    "Question: what is andy's sisters name in toy story\n"
    "→ reasoning: The context states Andy's younger sister is named Molly.\n"
    "→ answer: Molly\n\n"
    "Context: 'The PUMA 560 was the first robot used in a surgery in 1983.'\n"
    "Question: when was the first robot used in surgery\n"
    "→ reasoning: The context gives 1983 for the first surgical robot.\n"
    "→ answer: 1983\n\n"
    "YOUR TASK:\n"
    "Context: {context}\n"
    "Question: {question}\n\n"
    "Call format_answer with your reasoning and short final answer."
)


def get_generator_prompt() -> str:
    """Return the generator base prompt. Override via GENERATOR_BASE_PROMPT_FILE env."""
    path = os.getenv("GENERATOR_BASE_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return BASE_PROMPT


# Information Extraction Prompt for KG Construction
DEFAULT_IE_PROMPT = """You are a High-Fidelity Knowledge Graph Extractor. In a SINGLE pass, extract BOTH the entity inventory AND the relational triples from the text. You are the sole authority on what entities and relations exist in the chunk.

YOUR TASK:
1. ENTITY INVENTORY: Identify all named entities (people, organizations, places, events, works, concepts) that appear in the text.
2. RELATIONAL TRIPLES: Extract subject-predicate-object triples that connect these entities.

SEED ANCHORING (Wikipedia titles):
The following Wikipedia page titles were detected in this chunk. When an entity in the text matches one of these titles (or is an obvious variant), PREFER using that exact title as the entity surface form. This improves consistency with the knowledge base.
- If no match exists, create a new entity using the exact form from the text.
SEED TITLES IN CHUNK:
{seed_titles_in_chunk}

EXTRACTION STRATEGY:
1. LINE-BY-LINE & LIST PARSING: Process the text line-by-line. Bulleted lists and glossaries are dense with facts.
   - When encountering "Item - Description" or "Item : Description", the "Item" is the Subject for ALL relations in that line.
2. DECENTRALIZED SEARCH: Seek relationships between secondary and tertiary entities. NEVER assign a local fact to the main document topic when the text names a different subject.
3. PROMOTE ROLES TO PREDICATES: Titles, roles, and structural relationships become the Predicate. BAD: (Tim Cook) --[is]--> (CEO of Apple). GOOD: (Tim Cook) --[ceo_of]--> (Apple).
4. ATOMIC ENTITIES: Keep Subjects and Objects short and atomic. Strip descriptive modifiers.

STRICT GROUNDING (Anti-Hallucination):
1. ZERO EXTERNAL KNOWLEDGE: Extract exact, verbatim strings from the text. Do NOT canonicalize names, use real names instead of stage names, or resolve aliases.
2. VERBATIM EVIDENCE: The `evidence` field is mandatory for every triple. Quote the exact phrase that proves it.
   - If Subject/Object do not appear in your evidence quote, you have hallucinated. Fix or drop the triple.
3. Extract ALL valid relational pairs you can find, from the first word to the last line.

PREDICATE RULES:
- Predicates must be concise snake_case (e.g., born_in, ceo_of, member_of).
- BANNED: is, was, has, had, mentions, discusses, related_to.
- EXCEPTION: instance_of, subclass_of, has_profession ONLY when the text explicitly assigns a category.

OUTPUT VIA THE STRUCTURED TOOL:
- entities: list of {{surface, type}} for each entity (type: PERSON, ORG, GPE, LOC, EVENT, WORK_OF_ART, or NOUN_CHUNK for other).
- triples: list of {{subj_surface, pred, obj_surface, confidence, evidence}}.

PAGE TITLE (for context):
{title}

INPUT TEXT:
{text}
"""


def get_ie_extraction_prompt() -> str:
    """Return the one-pass extraction prompt. Override via LLM_IE_PROMPT_FILE env."""
    path = os.getenv("LLM_IE_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_IE_PROMPT


# Query Entity Extraction Prompt
DEFAULT_QUERY_ENTITY_PROMPT = """Extract the key entities (people, places, organizations, events, works, concepts) that the user is asking about in this question.

TASK: Return a JSON list of entity strings. Use the exact phrasing from the question when possible. Keep entities atomic and short.

RULES:
- Extract only entities explicitly mentioned or clearly implied.
- Do not infer or add entities not in the question.
- Return an empty list if no clear entities.

QUESTION:
{query}

Output a JSON array of entity strings, e.g. ["Albert Einstein", "Germany"].
"""


def get_query_entity_extraction_prompt() -> str:
    """Return the query entity extraction prompt. Override via QUERY_ENTITY_PROMPT_FILE env."""
    path = os.getenv("QUERY_ENTITY_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_QUERY_ENTITY_PROMPT
