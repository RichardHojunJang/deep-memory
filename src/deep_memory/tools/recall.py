"""Semantic recall tool for Hermes — search deep memory for relevant insights."""

from __future__ import annotations

import json

from deep_memory.api import RecallRequest, create_service


def recall(query: str, entity: str | None = None, limit: int = 10, task_id: str = None) -> str:
    """Search deep memory for relevant insights about entities.

    Uses hybrid FTS5 keyword + sqlite-vec semantic search.
    Returns ranked conclusions as JSON.
    """
    service = create_service()
    try:
        results = service.recall(RecallRequest(query=query, entity_id=entity, limit=limit))
        if not results:
            return json.dumps({"results": [], "message": "No matching insights found."})

        return json.dumps({
            "results": [r.to_dict() for r in results],
            "total": len(results),
        })
    finally:
        service.close()


TOOL_SCHEMA = {
    "name": "recall",
    "description": (
        "Search deep memory for relevant insights and conclusions about entities "
        "(people, projects, concepts). Uses hybrid keyword + semantic search across "
        "all stored reasoning conclusions. More powerful than basic memory — returns "
        "structured insights extracted through formal reasoning over past conversations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for — a question, topic, or keywords.",
            },
            "entity": {
                "type": "string",
                "description": "Optional entity name/ID to filter results to a specific person, project, or concept.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10).",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}
