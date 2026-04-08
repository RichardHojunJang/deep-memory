"""Hermes Agent integration — register deep memory tools and hooks.

This module is the bridge between deep-memory and Hermes Agent.
Import it from Hermes's tool discovery to register the toolset.

Usage in Hermes model_tools.py _discover_tools():
    try:
        import deep_memory.hermes_integration  # noqa: F401
    except ImportError:
        pass
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("deep_memory")


def _get_config() -> dict:
    """Read deep_memory config from Hermes config.yaml."""
    try:
        from pathlib import Path
        import yaml

        config_path = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("deep_memory", {})
    except Exception:
        pass
    return {}


def _check_requirements() -> bool:
    """Check if deep-memory is available."""
    try:
        from deep_memory.store.db import DeepMemoryDB
        return True
    except ImportError:
        return False


def _recall_handler(args: dict, **kwargs) -> str:
    """Handler for the recall tool."""
    from deep_memory.tools.recall import recall
    return recall(
        query=args.get("query", ""),
        entity=args.get("entity"),
        limit=args.get("limit", 10),
        task_id=kwargs.get("task_id"),
    )


def _learn_handler(args: dict, **kwargs) -> str:
    """Handler for the learn tool."""
    from deep_memory.tools.learn import learn
    return learn(
        entity=args.get("entity", ""),
        insight=args.get("insight", ""),
        type=args.get("type", "explicit"),
        task_id=kwargs.get("task_id"),
    )


def _entities_handler(args: dict, **kwargs) -> str:
    """Handler for the entities tool."""
    from deep_memory.tools.entities import entities
    return entities(
        action=args.get("action", "list"),
        name=args.get("name"),
        card=args.get("card"),
        task_id=kwargs.get("task_id"),
    )


def register_tools():
    """Register deep-memory tools with Hermes tool registry."""
    try:
        from tools.registry import registry
    except ImportError:
        logger.debug("Hermes tool registry not available, skipping registration")
        return

    from deep_memory.tools.recall import TOOL_SCHEMA as RECALL_SCHEMA
    from deep_memory.tools.learn import TOOL_SCHEMA as LEARN_SCHEMA
    from deep_memory.tools.entities import TOOL_SCHEMA as ENTITIES_SCHEMA

    registry.register(
        name="recall",
        toolset="deep_memory",
        schema=RECALL_SCHEMA,
        handler=_recall_handler,
        check_fn=_check_requirements,
    )

    registry.register(
        name="learn",
        toolset="deep_memory",
        schema=LEARN_SCHEMA,
        handler=_learn_handler,
        check_fn=_check_requirements,
    )

    registry.register(
        name="entities",
        toolset="deep_memory",
        schema=ENTITIES_SCHEMA,
        handler=_entities_handler,
        check_fn=_check_requirements,
    )

    logger.info("Deep memory tools registered: recall, learn, entities")


def build_deep_memory_context(entity_id: str = None) -> str:
    """Build context string for system prompt injection.
    
    Returns a formatted block similar to Hermes's MEMORY block,
    containing the entity's card and recent key conclusions.
    """
    from deep_memory.store.db import DeepMemoryDB

    db = DeepMemoryDB()
    try:
        parts = []

        if entity_id:
            entity = db.get_entity(entity_id)
            if entity and entity.get("card"):
                card = entity["card"]
                card_lines = []
                for k, v in card.items():
                    if isinstance(v, list):
                        card_lines.append(f"**{k.title()}:** {', '.join(str(x) for x in v)}")
                    elif v:
                        card_lines.append(f"**{k.title()}:** {v}")
                if card_lines:
                    parts.append("## Entity Profile\n" + "\n".join(card_lines))

            # Get recent high-confidence conclusions
            conclusions = db.get_conclusions(
                entity_id=entity_id, active_only=True, limit=15
            )
            if conclusions:
                conclusion_lines = []
                for c in conclusions:
                    if c.get("confidence", 1.0) >= 0.7:
                        prefix = {"explicit": "📌", "deductive": "🔗", "inductive": "📊", "abductive": "💡"}.get(c["type"], "•")
                        conclusion_lines.append(f"{prefix} {c['content']}")
                if conclusion_lines:
                    parts.append("## Key Insights\n" + "\n".join(conclusion_lines))

        if not parts:
            return ""

        return (
            "\n══════════════════════════════════════════════\n"
            "DEEP MEMORY (reasoning-based insights)\n"
            "══════════════════════════════════════════════\n"
            + "\n\n".join(parts)
            + "\n"
        )
    except Exception as e:
        logger.debug("Failed to build deep memory context: %s", e)
        return ""
    finally:
        db.close()


# Auto-register on import
register_tools()
