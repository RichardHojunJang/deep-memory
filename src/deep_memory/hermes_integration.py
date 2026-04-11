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

import logging

from deep_memory.adapters.hermes_tools import is_runtime_available, register_with_registry
from deep_memory.api import EntityQuery, create_service

logger = logging.getLogger("deep_memory")


def _check_requirements() -> bool:
    """Check if deep-memory is available."""
    return is_runtime_available()


def register_tools():
    """Register deep-memory tools with Hermes tool registry."""
    try:
        from tools.registry import registry
    except ImportError:
        logger.debug("Hermes tool registry not available, skipping registration")
        return

    names = register_with_registry(registry)
    logger.info("Deep memory tools registered: %s", ", ".join(names))


def build_deep_memory_context(entity_id: str = None) -> str:
    """Build context string for system prompt injection.

    Returns a formatted block similar to Hermes's MEMORY block,
    containing the entity's card and recent key conclusions.
    """
    if not entity_id:
        return ""

    service = create_service()
    try:
        detail = service.get_entity(
            EntityQuery(entity_id=entity_id, include_conclusions=True, conclusion_limit=15)
        )
    except KeyError:
        return ""
    except Exception as exc:
        logger.debug("Failed to build deep memory context: %s", exc)
        return ""
    finally:
        service.close()

    parts = []
    card = detail.entity.card or {}
    if card:
        card_lines = []
        for k, v in card.items():
            if isinstance(v, list):
                card_lines.append(f"**{k.title()}:** {', '.join(str(x) for x in v)}")
            elif v:
                card_lines.append(f"**{k.title()}:** {v}")
        if card_lines:
            parts.append("## Entity Profile\n" + "\n".join(card_lines))

    conclusion_lines = []
    for conclusion in detail.conclusions:
        if conclusion.confidence >= 0.7:
            prefix = {
                "explicit": "📌",
                "deductive": "🔗",
                "inductive": "📊",
                "abductive": "💡",
            }.get(conclusion.conclusion_type, "•")
            conclusion_lines.append(f"{prefix} {conclusion.content}")
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


# Auto-register on import
register_tools()
