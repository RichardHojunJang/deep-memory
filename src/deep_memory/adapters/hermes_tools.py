"""Thin helpers for exposing deep-memory through Hermes tool registration.

This module intentionally stays small and backend-agnostic. It translates the
package's current tool functions into a registration-friendly shape while also
leaving room for a future service layer or provider/backend hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


ToolHandler = Callable[..., Any]
AvailabilityCheck = Callable[[], bool]

_SERVICE_MODULE_CANDIDATES = (
    "deep_memory.service_api",
    "deep_memory.service",
    "deep_memory.services",
)


@dataclass(frozen=True)
class HermesToolAdapter:
    """Registration-ready description of a deep-memory tool."""

    name: str
    schema: Dict[str, Any]
    handler: ToolHandler
    availability_check: AvailabilityCheck
    service_dependency: Optional[str] = None


def load_service_api() -> Optional[Any]:
    """Load an optional service-layer module or factory.

    The current branch may not expose a dedicated service API yet. Returning
    ``None`` keeps the adapter layer usable for today's toolset/plugin mode
    while documenting where a backend-facing service layer can plug in later.
    """

    for module_name in _SERVICE_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        if hasattr(module, "get_service"):
            return module.get_service  # type: ignore[return-value]
        if hasattr(module, "create_service"):
            return module.create_service  # type: ignore[return-value]
        if hasattr(module, "DeepMemoryService"):
            return module.DeepMemoryService  # type: ignore[return-value]
        return module
    return None


def is_runtime_available() -> bool:
    """Return whether the current Python environment can execute deep-memory."""

    try:
        importlib.import_module("deep_memory.store.db")
        return True
    except Exception:
        return False


def _build_tool_handler(module_name: str, function_name: str, field_map: Mapping[str, str]) -> ToolHandler:
    """Create a lazy handler that adapts Hermes-style args to tool call kwargs."""

    def _handler(args: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Any:
        payload = dict(args or {})
        task_id = kwargs.get("task_id")
        if task_id is not None and "task_id" not in payload:
            payload["task_id"] = task_id

        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        call_kwargs = {target: payload.get(source) for source, target in field_map.items()}
        return fn(**call_kwargs)

    return _handler


def iter_tool_adapters() -> List[HermesToolAdapter]:
    """Return the current Hermes-facing adapter definitions.

    The tools remain the source of truth today. A future service layer can sit
    behind the same adapter contract without changing Hermes registration code.
    """

    recall_module = importlib.import_module("deep_memory.tools.recall")
    learn_module = importlib.import_module("deep_memory.tools.learn")
    entities_module = importlib.import_module("deep_memory.tools.entities")

    adapters = [
        HermesToolAdapter(
            name="recall",
            schema=dict(recall_module.TOOL_SCHEMA),
            handler=_build_tool_handler(
                "deep_memory.tools.recall",
                "recall",
                {"query": "query", "entity": "entity", "limit": "limit", "task_id": "task_id"},
            ),
            availability_check=is_runtime_available,
            service_dependency="optional-service-api",
        ),
        HermesToolAdapter(
            name="learn",
            schema=dict(learn_module.TOOL_SCHEMA),
            handler=_build_tool_handler(
                "deep_memory.tools.learn",
                "learn",
                {"entity": "entity", "insight": "insight", "type": "type", "task_id": "task_id"},
            ),
            availability_check=is_runtime_available,
            service_dependency="optional-service-api",
        ),
        HermesToolAdapter(
            name="entities",
            schema=dict(entities_module.TOOL_SCHEMA),
            handler=_build_tool_handler(
                "deep_memory.tools.entities",
                "entities",
                {"action": "action", "name": "name", "card": "card", "task_id": "task_id"},
            ),
            availability_check=is_runtime_available,
            service_dependency="optional-service-api",
        ),
    ]
    return adapters


def register_with_registry(registry: Any, adapters: Optional[Iterable[HermesToolAdapter]] = None) -> List[str]:
    """Register the adapters with a Hermes-like registry object.

    The registry is expected to expose a ``register(...)`` method compatible
    with Hermes's tool registry. The function returns the registered names so
    tests and callers can validate the resulting surface.
    """

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
