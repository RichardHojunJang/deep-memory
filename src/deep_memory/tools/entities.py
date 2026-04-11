"""Entity profile management tool for Hermes."""

from __future__ import annotations

import json

from deep_memory.api import EntityQuery, EntityUpdate, EntityUpsert, create_service
from deep_memory.api.service import DeepMemoryService


def entities(action: str, name: str | None = None, card: dict | None = None, task_id: str = None) -> str:
    """Manage entity profiles in deep memory.

    Actions:
    - list: List all tracked entities
    - get: Get entity profile with card and recent conclusions
    - update: Update entity card (biographical info, preferences)
    """
    service = create_service()
    try:
        if action == "list":
            ents = service.list_entities()
            return json.dumps({
                "entities": [item.to_dict() for item in ents],
                "total": len(ents),
            })

        if action == "get":
            if not name:
                return json.dumps({"error": "Name is required for 'get' action."})
            detail = service.get_entity(EntityQuery(name=name, include_conclusions=True, conclusion_limit=20))
            payload = detail.to_dict()
            payload["total_conclusions"] = len(detail.conclusions)
            return json.dumps(payload, default=str)

        if action == "update":
            if not name:
                return json.dumps({"error": "Name is required for 'update' action."})
            if not card:
                return json.dumps({"error": "Card data is required for 'update' action."})
            entity_id = DeepMemoryService.slugify(name)
            try:
                entity = service.update_entity(EntityUpdate(entity_id=entity_id, card=card))
            except KeyError:
                entity = service.upsert_entity(EntityUpsert(entity_id=entity_id, name=name, card=card))
            return json.dumps({"updated": True, "entity": entity.to_dict()}, default=str)

        return json.dumps({"error": f"Unknown action '{action}'. Use: list, get, update"})
    except KeyError:
        return json.dumps({"error": f"Entity '{name}' not found."})
    finally:
        service.close()


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
