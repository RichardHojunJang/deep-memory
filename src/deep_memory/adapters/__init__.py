"""Thin adapter layer for host/runtime integrations.

The core package stores and reasons over memory. Adapters expose those
capabilities to specific runtimes without coupling the engine to one host.
"""

from .hermes_plugin import DeepMemorySessionPlugin, build_prompt_context, process_session_messages
from .hermes_tools import (
    HermesToolAdapter,
    iter_tool_adapters,
    load_service_api,
    register_with_registry,
)

__all__ = [
    "DeepMemorySessionPlugin",
    "HermesToolAdapter",
    "build_prompt_context",
    "iter_tool_adapters",
    "load_service_api",
    "process_session_messages",
    "register_with_registry",
]
