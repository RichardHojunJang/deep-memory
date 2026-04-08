"""Entity profile management tool for Hermes."""

from __future__ import annotations

import json
import re
from typing import Any


def _slugify(name: str) -> str:
    """Convert a name to a simple entity ID."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def entities(action: str, name: str | None = None, card: dict | None = None, task_id: str = None) -> str:
    """Manage entity profiles in deep memory.

    Actions:
    - list: List all tracked entities
    - get: Get entity profile with card and recent conclusions
    - update: Update entity card (biographical info, preferences)
    """
    from deep_memory.store import DeepMemoryDB

    db = DeepMemoryDB()
    try:
        if action == "list":
            ents = db.list_entities()
            return json.dumps({
                "entities": [
                    {"id": e["id"], "name": e["name"], "type": e["type"],
                     "updated_at": e["updated_at"]}
                    for e in ents
                ],
                "total": len(ents),
            })

        elif action == "get":
            if not name:
                return json.dumps({"error": "Name is required for 'get' action."})
            entity_id = _slugify(name)
            entity = db.get_entity(entity_id)
            if not entity:
                return json.dumps({"error": f"Entity \'{name}\' not found."})
            conclusions = db.get_conclusions(entity_id=entity_id, limit=20)
            return json.dumps({
                "entity": entity,
                "conclusions": conclusions,
                "total_conclusions": len(conclusions),
            }, default=str)

        elif action == "update":
            if not name:
                return json.dumps({"error": "Name is required for 'update' action."})
            if not card:
                return json.dumps({"error": "Card data is required for 'update' action."})
            entity_id = _slugify(name)
            existing = db.get_entity(entity_id)
            if not existing:
                entity = db.upsert_entity(entity_id, name, card=card)
            else:
                entity = db.update_entity_card(entity_id, card)
            return json.dumps({"updated": True, "entity": entity}, default=str)

        else:
            return json.dumps({"error": f"Unknown action \'{action}\'. Use: list, get, update"})

    finally:
        db.close()


TOOL_SCHEMA = {
    "name": "entities",
    "description": (
        "Manage entity profiles in deep memory. Entities represent people, projects, "
        "or concepts that are tracked over time. Use 'list' to see all tracked entities, "
        "'get' to view an entity's profile and recent conclusions, or 'update' to "
        "modify an entity's biographical card."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "get", "update"],
                "description": "Action to perform: list all entities, get a specific entity, or update a card.",
            },
            "name": {
                "type": "string",
                "description": "Entity name (required for get/update).",
            },
            "card": {
                "type": "object",
                "description": "Card data to update (for update action). Key biographical info like role, preferences, etc.",
            },
        },
        "required": ["action"],
    },
}
