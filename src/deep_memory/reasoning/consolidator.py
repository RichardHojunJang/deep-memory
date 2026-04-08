"""Contradiction detection and conclusion merging.

Periodically reviews an entity's conclusions to:
- Remove redundant conclusions
- Resolve contradictions
- Consolidate multiple observations into higher-level patterns
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from deep_memory.store.db import DeepMemoryDB
from .prompts import CONSOLIDATION_PROMPT

logger = logging.getLogger("deep_memory.reasoning")

LLMCallFn = Callable[[str], str]


def _parse_json_response(text: str):
    """Extract JSON from an LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    for sc, ec in [("{", "}"), ("[", "]")]:
        start = text.find(sc)
        if start == -1:
            continue
        end = text.rfind(ec)
        if end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def consolidate_entity(
    entity_id: str,
    llm_call: LLMCallFn,
    db: Optional[DeepMemoryDB] = None,
    min_conclusions: int = 5,
) -> dict:
    """Review and consolidate an entity's conclusions.
    
    Args:
        entity_id: The entity to consolidate
        llm_call: Function that takes a prompt and returns LLM response
        db: Optional DB instance
        min_conclusions: Skip if entity has fewer than this many conclusions
    
    Returns: Stats about what was consolidated
    """
    own_db = db is None
    if own_db:
        db = DeepMemoryDB()

    try:
        stats = {
            "entity_id": entity_id,
            "redundant_removed": 0,
            "contradictions_resolved": 0,
            "consolidated": 0,
        }

        entity = db.get_entity(entity_id)
        if not entity:
            logger.warning("Entity %s not found", entity_id)
            return stats

        conclusions = db.get_conclusions(entity_id=entity_id, active_only=True, limit=100)
        if len(conclusions) < min_conclusions:
            logger.info(
                "Entity %s has only %d conclusions, skipping consolidation",
                entity_id, len(conclusions),
            )
            return stats

        # Format conclusions for the prompt
        conclusions_text = "\n".join(
            f"- [id={c['id']}] [{c['type']}] {c['content']} (confidence: {c.get('confidence', 1.0)})"
            for c in conclusions
        )

        prompt = CONSOLIDATION_PROMPT.format(
            entity_name=entity.get("name", entity_id),
            conclusions=conclusions_text,
        )

        logger.info("Consolidating %d conclusions for %s", len(conclusions), entity_id)
        response = llm_call(prompt)
        result = _parse_json_response(response)

        # Handle redundant pairs
        for pair in result.get("redundant_pairs", []):
            ids = pair.get("ids", [])
            keep_id = pair.get("keep")
            if len(ids) == 2 and keep_id is not None:
                remove_id = ids[0] if ids[1] == keep_id else ids[1]
                db.supersede_conclusion(remove_id, keep_id)
                stats["redundant_removed"] += 1
                logger.debug("Redundant: %d superseded by %d", remove_id, keep_id)

        # Handle contradictions
        for contradiction in result.get("contradictions", []):
            ids = contradiction.get("ids", [])
            keep_id = contradiction.get("keep")
            if len(ids) == 2 and keep_id is not None:
                remove_id = ids[0] if ids[1] == keep_id else ids[1]
                db.supersede_conclusion(remove_id, keep_id)
                stats["contradictions_resolved"] += 1
                logger.debug("Contradiction resolved: keeping %d", keep_id)

        # Handle consolidated conclusions
        for consolidation in result.get("consolidated", []):
            from_ids = consolidation.get("from_ids", [])
            new_content = consolidation.get("new_content", "")
            new_type = consolidation.get("type", "inductive")
            confidence = consolidation.get("confidence", 0.8)

            if not from_ids or not new_content:
                continue

            # Add the new consolidated conclusion
            new_id = db.add_conclusion(
                entity_id=entity_id,
                type=new_type,
                content=new_content,
                confidence=confidence,
            )

            # Supersede the originals
            for old_id in from_ids:
                db.supersede_conclusion(old_id, new_id)

            stats["consolidated"] += 1
            logger.debug(
                "Consolidated %d conclusions into %d: %s",
                len(from_ids), new_id, new_content[:80],
            )

        logger.info(
            "Consolidation for %s: %d redundant, %d contradictions, %d consolidated",
            entity_id,
            stats["redundant_removed"],
            stats["contradictions_resolved"],
            stats["consolidated"],
        )
        return stats

    finally:
        if own_db:
            db.close()


def consolidate_all(
    llm_call: LLMCallFn,
    db: Optional[DeepMemoryDB] = None,
    min_conclusions: int = 5,
) -> list[dict]:
    """Consolidate conclusions for all entities that have enough data.
    
    Returns: List of stats dicts, one per processed entity.
    """
    own_db = db is None
    if own_db:
        db = DeepMemoryDB()

    try:
        entities = db.list_entities()
        results = []
        for entity in entities:
            stats = consolidate_entity(
                entity["id"],
                llm_call=llm_call,
                db=db,
                min_conclusions=min_conclusions,
            )
            if any(v for k, v in stats.items() if k != "entity_id"):
                results.append(stats)
        return results
    finally:
        if own_db:
            db.close()
