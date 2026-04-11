"""Thin helpers for exposing deep-memory through Hermes tool registration.

The adapter layer is intentionally small: Hermes sees schemas + callables, while
Deep Memory keeps the real behavior behind a stable service API. That makes the
current toolset integration work today and leaves a clean seam for future
backend/provider hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, Iterable, List, Optional

from deep_memory.api import (
    DeepMemoryService,
    EntityQuery,
    EntityUpdate,
    EntityUpsert,
    LearnRequest,
    RecallRequest,
    create_service,
)
from deep_memory.tools.entities import TOOL_SCHEMA as ENTITIES_SCHEMA
from deep_memory.tools.learn import TOOL_SCHEMA as LEARN_SCHEMA
from deep_memory.tools.recall import TOOL_SCHEMA as RECALL_SCHEMA

ToolHandler = Callable[..., str]
AvailabilityCheck = Callable[[], bool]


@dataclass(frozen=True)
class HermesToolAdapter:
    """Registration-ready description of a deep-memory tool."""

    name: str
    schema: Dict[str, Any]
    handler: ToolHandler
    availability_check: AvailabilityCheck
    service_dependency: str = "deep_memory.api.create_service"


def load_service_api() -> Callable[..., DeepMemoryService]:
    """Return the service constructor used by current and future adapters."""
    return create_service


def is_runtime_available() -> bool:
    """Return whether the current Python environment can execute deep-memory."""
    try:
        service = create_service()
        service.close()
        return True
    except Exception:
        return False


def _with_service(callback: Callable[[DeepMemoryService], Dict[str, Any]]) -> str:
    service = create_service()
    try:
        return json.dumps(callback(service), default=str)
    except KeyError as exc:
        return json.dumps({"error": str(exc)})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    finally:
        service.close()


def _handle_recall(args: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    payload = args or {}

    def _run(service: DeepMemoryService) -> Dict[str, Any]:
        results = service.recall(
            RecallRequest(
                query=payload.get("query", ""),
                entity_id=payload.get("entity"),
                limit=payload.get("limit", 10),
            )
        )
        if not results:
            return {"results": [], "message": "No matching insights found."}
        return {"results": [item.to_dict() for item in results], "total": len(results)}

    return _with_service(_run)


def _handle_learn(args: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    payload = args or {}

    def _run(service: DeepMemoryService) -> Dict[str, Any]:
        return service.learn(
            LearnRequest(
                entity=payload.get("entity", ""),
                insight=payload.get("insight", ""),
                conclusion_type=payload.get("type", "explicit"),
            )
        )

    return _with_service(_run)


def _handle_entities(args: Optional[Dict[str, Any]] = None, **_: Any) -> str:
    payload = args or {}
    action = payload.get("action", "list")

    def _run(service: DeepMemoryService) -> Dict[str, Any]:
        if action == "list":
            entities = service.list_entities()
            return {
                "entities": [item.to_dict() for item in entities],
                "total": len(entities),
            }
        if action == "get":
            name = payload.get("name")
            if not name:
                raise ValueError("Name is required for 'get' action.")
            detail = service.get_entity(
                EntityQuery(name=name, include_conclusions=True, conclusion_limit=20)
            )
            data = detail.to_dict()
            data["total_conclusions"] = len(detail.conclusions)
            return data
        if action == "update":
            name = payload.get("name")
            card = payload.get("card")
            if not name:
                raise ValueError("Name is required for 'update' action.")
            if not card:
                raise ValueError("Card data is required for 'update' action.")
            entity_id = DeepMemoryService.slugify(name)
            try:
                entity = service.update_entity(EntityUpdate(entity_id=entity_id, card=card))
            except KeyError:
                entity = service.upsert_entity(EntityUpsert(entity_id=entity_id, name=name, card=card))
            return {"updated": True, "entity": entity.to_dict()}
        raise ValueError(f"Unknown action '{action}'. Use: list, get, update")

    return _with_service(_run)


def iter_tool_adapters() -> List[HermesToolAdapter]:
    """Return the current Hermes-facing adapter definitions."""
    return [
        HermesToolAdapter(
            name="recall",
            schema=dict(RECALL_SCHEMA),
            handler=_handle_recall,
            availability_check=is_runtime_available,
        ),
        HermesToolAdapter(
            name="learn",
            schema=dict(LEARN_SCHEMA),
            handler=_handle_learn,
            availability_check=is_runtime_available,
        ),
        HermesToolAdapter(
            name="entities",
            schema=dict(ENTITIES_SCHEMA),
            handler=_handle_entities,
            availability_check=is_runtime_available,
        ),
    ]


def register_with_registry(registry: Any, adapters: Optional[Iterable[HermesToolAdapter]] = None) -> List[str]:
    """Register the adapters with a Hermes-like registry object."""
    tool_adapters = list(adapters or iter_tool_adapters())
    registered: List[str] = []
    for adapter in tool_adapters:
        registry.register(
            name=adapter.name,
            toolset="deep_memory",
            schema=adapter.schema,
            handler=adapter.handler,
            check_fn=adapter.availability_check,
        )
        registered.append(adapter.name)
    return registered
