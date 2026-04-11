"""Manual insight storage tool for Hermes."""

from __future__ import annotations

import json

from deep_memory.api import LearnRequest, create_service


def learn(entity: str, insight: str, type: str = "explicit", task_id: str = None) -> str:
    """Store a new insight about an entity in deep memory.

    Auto-creates the entity if it doesn't exist.
    Valid types: explicit, deductive, inductive, abductive.
    """
    service = create_service()
    try:
        result = service.learn(
            LearnRequest(
                entity=entity,
                insight=insight,
                conclusion_type=type,
            )
        )
        entity_record = result["entity"]
        conclusion_record = result["conclusion"]
        return json.dumps({
            "stored": True,
            "conclusion_id": result["conclusion_id"],
            "entity_id": entity_record["entity_id"],
            "entity_name": entity_record["name"],
            "type": conclusion_record["conclusion_type"],
            "content": conclusion_record["content"],
        })
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    finally:
        service.close()


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
