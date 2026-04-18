"""Central prompt definitions for LLM-based components of the system"""

from __future__ import annotations

import os
from pathlib import Path


# Generator Prompt
GENERATOR_SYSTEM_MESSAGE = (
    "You are a strict QA system. Answer based ONLY on the provided context."
)

BASE_PROMPT = (
    "You are a strict QA system. Answer based ONLY on the provided context."
    "\n\n"
    "EXAMPLES:"
    "Context: 'Toy Story features a boy named Andy who has a younger sister named Molly.'\n"
    "Question: what is andy's sisters name in toy story\n"
    "Answer: Molly\n\n"
    "Context: 'The PUMA 560 was the first robot used in a surgery, assisting in a biopsy in 1983.'\n"
    "Question: when was the first robot used in surgery\n"
    "Answer: 1983\n\n"
    "Context: 'Donovan Mitchell was selected with the 13th overall pick in the 2017 NBA draft.'\n"
    "Question: where was donovan mitchell picked in the draft\n"
    "Answer: 13th\n\n"
    "Context: 'Gabriela Mistral was a Chilean poet. G. K. Chesterton was an English writer and philosopher.'\n"
    "Question: Were both Gabriela Mistral and G. K. Chesterton authors?\n"
    "Answer: yes"
    "\n\n"
    "YOUR TASK:"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)


def get_generator_prompt() -> str:
    """Return the generator base prompt. Override via GENERATOR_BASE_PROMPT_FILE env."""
    path = os.getenv("GENERATOR_BASE_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return BASE_PROMPT


# LLM-As-The-Judge Prompt (Legacy)
DEFAULT_JUDGE_PROMPT_OLD = """You are an expert evaluator grading an AI's answer against a Ground Truth reference. 
Focus strictly on factual and semantic equivalence. Ignore differences in wording, length, or conversational boilerplate.

Grade the AI Answer on this strict 0-2 scale:
[0] Incorrect / Refusal: The answer is factually wrong, completely misses the core meaning, or states that the context is insufficient.
[1] Partial / Superficial: The answer contains relevant keywords or partial facts, but misses the specific reasoning, causality, or completeness of the Ground Truth.
[2] Comprehensive / Correct: The answer fully captures the core facts and logical meaning of the Ground Truth.

Original Question: {question}
Ground Truth: {gold_answers}
AI Answer: {predicted_answer}

Output ONLY a single integer (0, 1, or 2). Do not output any other text or explanation."""

DEFAULT_JUDGE_PROMPT = """You are an expert evaluator grading an AI's answer against a Ground Truth reference for a factual Question Answering task. 
Your only goal is to determine if the AI successfully retrieved the core fact requested by the question.

CRITICAL GRADING RULES:
- Entity Resolution: Treat aliases, initials, acronyms, and missing middle names as equivalent (e.g., "E.L. Doctorow" perfectly matches "Edgar Lawrence 'E. L.' Doctorow").
- Embedded Answers: If the Ground Truth is explicitly present in the AI Answer, it is correct. Ignore extra conversational text or hallucinated context as long as the exact answer to the question is provided.

Grade the AI Answer strictly on this 0-2 scale:
[2] Correct: The AI Answer contains the exact Ground Truth fact, entity, or date. It successfully answers the question, regardless of minor formatting differences or extra text.
[1] Partial: The AI Answer contains some correct elements (e.g., 1 out of 2 items in a list) but is incomplete, or it is slightly too broad to be a perfect match.
[0] Incorrect: The AI Answer completely misses the Ground Truth, contradicts it, or states "insufficient context".

Original Question: {question}
Ground Truth: {gold_answers}
AI Answer: {predicted_answer}

Output ONLY a single integer (0, 1, or 2). Do not output any other text or explanation."""


def get_judge_prompt() -> str:
    """Return the judge prompt template. Override via LLM_JUDGE_PROMPT_FILE env."""
    path = os.getenv("LLM_JUDGE_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_JUDGE_PROMPT


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
