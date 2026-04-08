"""Post-session insight extraction pipeline.

Runs after a conversation session ends to:
1. Generate a session summary
2. Identify mentioned entities
3. Extract structured reasoning (explicit/deductive/inductive/abductive)
4. Update entity cards
5. Detect contradictions with existing conclusions
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from deep_memory.store.db import DeepMemoryDB
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    SESSION_REASONING_PROMPT,
    SESSION_SUMMARY_PROMPT,
)

logger = logging.getLogger("deep_memory.reasoning")


def _parse_json_response(text: str) -> Any:
    """Extract JSON from an LLM response that may contain markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Last resort: find first { or [ and parse from there
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        end = text.rfind(end_char)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


# Type alias for the LLM call function
# Signature: (prompt: str) -> str
LLMCallFn = Callable[[str], str]


def extract_session_summary(
    transcript: str,
    llm_call: LLMCallFn,
) -> dict:
    """Generate a structured summary of a conversation session.
    
    Returns: {"short_summary": str, "key_decisions": list, "entities_mentioned": list}
    """
    prompt = SESSION_SUMMARY_PROMPT.format(transcript=transcript)
    response = llm_call(prompt)
    return _parse_json_response(response)


def extract_entities(
    transcript: str,
    llm_call: LLMCallFn,
) -> list[dict]:
    """Identify entities mentioned in a conversation.
    
    Returns: [{"name": str, "type": str, "relevance": str}, ...]
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(transcript=transcript)
    response = llm_call(prompt)
    result = _parse_json_response(response)
    if isinstance(result, list):
        return result
    return result.get("entities", []) if isinstance(result, dict) else []


def extract_reasoning(
    entity_name: str,
    transcript: str,
    existing_conclusions: list[dict],
    llm_call: LLMCallFn,
) -> dict:
    """Extract structured reasoning about an entity from a conversation.
    
    Returns dict with keys: explicit, deductive, inductive, abductive,
    contradictions, card_updates
    """
    # Format existing conclusions for context
    if existing_conclusions:
        conclusions_text = "\n".join(
            f"- [{c['type']}] {c['content']} (confidence: {c.get('confidence', 1.0)})"
            for c in existing_conclusions
        )
    else:
        conclusions_text = "(No existing knowledge)"

    prompt = SESSION_REASONING_PROMPT.format(
        entity_name=entity_name,
        existing_conclusions=conclusions_text,
        transcript=transcript,
    )
    response = llm_call(prompt)
    return _parse_json_response(response)


def _slugify(name: str) -> str:
    """Convert a name to a simple entity ID."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def process_session(
    session_id: str,
    transcript: str,
    llm_call: LLMCallFn,
    db: Optional[DeepMemoryDB] = None,
) -> dict:
    """Full post-session processing pipeline.
    
    1. Summarize the session
    2. Extract entities
    3. For each entity, extract reasoning
    4. Store everything in the database
    
    Args:
        session_id: Unique session identifier
        transcript: The full conversation text
        llm_call: Function that takes a prompt string and returns LLM response
        db: Optional DeepMemoryDB instance (creates default if None)
    
    Returns: Summary dict with stats about what was extracted
    """
    own_db = db is None
    if own_db:
        db = DeepMemoryDB()

    try:
        stats = {
            "session_id": session_id,
            "entities_found": 0,
            "conclusions_added": 0,
            "contradictions_found": 0,
            "card_updates": 0,
        }

        # Step 1: Session summary
        logger.info("Extracting session summary for %s", session_id)
        summary = extract_session_summary(transcript, llm_call)
        
        entity_names = summary.get("entities_mentioned", [])
        db.add_summary(
            session_id=session_id,
            short_summary=summary.get("short_summary", ""),
            key_decisions=summary.get("key_decisions"),
            entities_mentioned=entity_names,
        )

        # Step 2: Extract entities (use LLM for richer extraction)
        logger.info("Extracting entities for %s", session_id)
        entities = extract_entities(transcript, llm_call)
        stats["entities_found"] = len(entities)

        # Ensure all entities exist in DB
        for entity_info in entities:
            name = entity_info.get("name", "")
            if not name:
                continue
            entity_id = _slugify(name)
            etype = entity_info.get("type", "person")
            db.upsert_entity(entity_id, name, etype)

        # Step 3: Extract reasoning per entity
        for entity_info in entities:
            name = entity_info.get("name", "")
            if not name:
                continue
            entity_id = _slugify(name)

            # Get existing conclusions for context
            existing = db.get_conclusions(entity_id=entity_id, limit=30)

            logger.info("Extracting reasoning about %s", name)
            reasoning = extract_reasoning(name, transcript, existing, llm_call)

            # Store explicit conclusions
            for item in reasoning.get("explicit", []):
                db.add_conclusion(
                    entity_id=entity_id,
                    type="explicit",
                    content=item["content"],
                    confidence=item.get("confidence", 1.0),
                    source_sessions=[session_id],
                )
                stats["conclusions_added"] += 1

            # Store deductive conclusions
            for item in reasoning.get("deductive", []):
                db.add_conclusion(
                    entity_id=entity_id,
                    type="deductive",
                    content=item["conclusion"],
                    premises=item.get("premises"),
                    confidence=item.get("confidence", 0.9),
                    source_sessions=[session_id],
                )
                stats["conclusions_added"] += 1

            # Store inductive conclusions
            for item in reasoning.get("inductive", []):
                db.add_conclusion(
                    entity_id=entity_id,
                    type="inductive",
                    content=item["pattern"],
                    premises=item.get("observations"),
                    confidence=item.get("confidence", 0.7),
                    source_sessions=[session_id],
                )
                stats["conclusions_added"] += 1

            # Store abductive conclusions
            for item in reasoning.get("abductive", []):
                content = f"{item['explanation']} (observed: {item['behavior']})"
                db.add_conclusion(
                    entity_id=entity_id,
                    type="abductive",
                    content=content,
                    confidence=item.get("confidence", 0.6),
                    source_sessions=[session_id],
                )
                stats["conclusions_added"] += 1

            # Handle contradictions
            for contradiction in reasoning.get("contradictions", []):
                stats["contradictions_found"] += 1
                logger.info(
                    "Contradiction found for %s: %s vs %s",
                    name,
                    contradiction.get("existing"),
                    contradiction.get("new_evidence"),
                )

            # Update entity card
            card_updates = reasoning.get("card_updates")
            if card_updates and any(v for v in card_updates.values() if v):
                # Filter out empty values
                card = {k: v for k, v in card_updates.items() if v}
                if card:
                    db.update_entity_card(entity_id, card)
                    stats["card_updates"] += 1

        logger.info(
            "Session %s processed: %d entities, %d conclusions, %d contradictions",
            session_id,
            stats["entities_found"],
            stats["conclusions_added"],
            stats["contradictions_found"],
        )
        return stats

    finally:
        if own_db:
            db.close()
