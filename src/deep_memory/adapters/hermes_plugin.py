"""Thin helpers for prompt-context and session-hook style integrations.

These adapters let deep-memory operate today as a Hermes plugin/toolset while
keeping the integration surface narrow enough to swap in future backend or
provider hooks when Hermes exposes them more directly.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

ContextBuilder = Callable[..., str]
SessionProcessor = Callable[..., Any]
MessageLoader = Callable[[str], List[dict]]


def _load_context_builder() -> Optional[ContextBuilder]:
    try:
        module = importlib.import_module("deep_memory.hermes_integration")
        return getattr(module, "build_deep_memory_context", None)
    except Exception:
        return None


def _load_session_processor() -> Optional[SessionProcessor]:
    try:
        module = importlib.import_module("deep_memory.session_hook")
        return getattr(module, "process_session_async", None)
    except Exception:
        return None


def build_prompt_context(entity_id: Optional[str] = None, builder: Optional[ContextBuilder] = None) -> str:
    """Build prompt-context text for the active entity/session when available."""

    context_builder = builder or _load_context_builder()
    if context_builder is None:
        return ""
    try:
        return context_builder(entity_id=entity_id)
    except TypeError:
        return context_builder(entity_id)
    except Exception:
        logger.debug("deep-memory context builder failed", exc_info=True)
        return ""


def process_session_messages(
    session_id: str,
    messages: Iterable[Mapping[str, Any]],
    processor: Optional[SessionProcessor] = None,
) -> bool:
    """Forward session messages to an async/background processor if available."""

    message_list = [dict(message) for message in messages]
    if not session_id or not message_list:
        return False

    session_processor = processor or _load_session_processor()
    if session_processor is None:
        return False

    try:
        session_processor(session_id=session_id, messages=message_list)
        return True
    except TypeError:
        session_processor(session_id, message_list)
        return True
    except Exception:
        logger.debug("deep-memory session processor failed", exc_info=True)
        return False


def _default_message_loader(session_id: str) -> List[dict]:
    try:
        from hermes_state import SessionDB
    except Exception:
        return []

    try:
        session_db = SessionDB()
        return list(session_db.get_session_messages(session_id) or [])
    except Exception:
        logger.debug("failed to load session messages from Hermes", exc_info=True)
        return []


class DeepMemorySessionPlugin:
    """Small plugin wrapper for Hermes-style session lifecycle hooks."""

    name = "deep_memory"
    description = "Forward completed Hermes sessions into deep-memory reasoning"

    def __init__(self, *, message_loader: Optional[MessageLoader] = None, processor: Optional[SessionProcessor] = None) -> None:
        self._message_loader = message_loader or _default_message_loader
        self._processor = processor

    def on_session_end(self, session_id: Optional[str] = None, **_: Any) -> bool:
        if not session_id:
            return False
        messages = self._message_loader(session_id)
        return process_session_messages(session_id, messages, processor=self._processor)
