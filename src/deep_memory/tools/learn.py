"""Manual insight storage tool for Hermes."""

from __future__ import annotations

import json
import re
from typing import Any


def _slugify(name: str) -> str:
    """Convert a name to a simple entity ID."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def learn(entity: str, insight: str, type: str = "explicit", task_id: str = None) -> str:
    """Store a new insight about an entity in deep memory.

    Auto-creates the entity if it doesn't exist.
    Valid types: explicit, deductive, inductive, abductive.
    """
    from deep_memory.store import DeepMemoryDB

    valid_types = ("explicit", "deductive", "inductive", "abductive")
    if type not in valid_types:
        return json.dumps({"error": f"Invalid type \'{type}\'. Must be one of: {valid_types}"})

    db = DeepMemoryDB()
    try:
        entity_id = _slugify(entity)

        # Auto-create entity if it doesn't exist
        existing = db.get_entity(entity_id)
        if not existing:
            db.upsert_entity(entity_id, entity, "person")

        # Store the conclusion
        conclusion_id = db.add_conclusion(
            entity_id=entity_id,
            type=type,
            content=insight,
        )

        return json.dumps({
            "stored": True,
            "conclusion_id": conclusion_id,
            "entity_id": entity_id,
            "entity_name": entity,
            "type": type,
            "content": insight,
        })
    finally:
        db.close()


TOOL_SCHEMA = {
    "name": "learn",
    "description": (
        "Store a new insight or conclusion about an entity (person, project, concept) "
        "in deep memory. The entity is auto-created if it doesn't exist. Use this to "
        "explicitly record important facts, deductions, or patterns discovered during "
        "conversations. More structured than basic memory — insights are typed, "
        "searchable, and linked to specific entities."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Name of the entity (person, project, concept) this insight is about.",
            },
            "insight": {
                "type": "string",
                "description": "The insight or conclusion to store.",
            },
            "type": {
                "type": "string",
                "enum": ["explicit", "deductive", "inductive", "abductive"],
                "description": "Type of insight: explicit (stated fact), deductive (certain conclusion from premises), inductive (pattern from multiple observations), abductive (simplest explanation for behavior). Default: explicit.",
                "default": "explicit",
            },
        },
        "required": ["entity", "insight"],
    },
}
