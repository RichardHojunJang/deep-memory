"""Post-session hook for Hermes Agent.

Integrates with Hermes's plugin system to automatically run
reasoning after each conversation session.

Install as a Hermes plugin:
    # In ~/.hermes/plugins/ or via plugin registration
    from deep_memory.session_hook import DeepMemoryPlugin
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, List, Optional

logger = logging.getLogger("deep_memory.session_hook")


def _get_llm_call_fn() -> Optional[Callable[[str], str]]:
    """Create an LLM call function using Hermes's auxiliary client.
    
    Falls back to direct OpenAI/Anthropic API if auxiliary client unavailable.
    """
    # Try Hermes auxiliary client first
    try:
        from agent.auxiliary_client import get_auxiliary_client
        client = get_auxiliary_client()
        if client:
            def llm_call(prompt: str) -> str:
                response = client.chat.completions.create(
                    model=client.default_model or "gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content
            return llm_call
    except ImportError:
        pass

    # Fallback: direct API call
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        import json
        import urllib.request

        def llm_call(prompt: str) -> str:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                }).encode(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
        return llm_call

    return None


def _format_messages_as_transcript(messages: List[dict]) -> str:
    """Convert Hermes message list to a readable transcript."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            continue  # Skip system prompts
        if role == "tool":
            continue  # Skip raw tool outputs
        if not content or not isinstance(content, str):
            continue
        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        label = {"user": "User", "assistant": "Assistant"}.get(role, role.title())
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def process_session_async(
    session_id: str,
    messages: List[dict],
    llm_call: Optional[Callable] = None,
) -> None:
    """Run post-session reasoning in a background thread.
    
    This is fire-and-forget — errors are logged but don't block the session.
    """
    if not messages:
        return

    transcript = _format_messages_as_transcript(messages)
    if len(transcript.strip()) < 50:
        logger.debug("Session %s too short for reasoning, skipping", session_id)
        return

    if llm_call is None:
        llm_call = _get_llm_call_fn()
    if llm_call is None:
        logger.debug("No LLM available for post-session reasoning")
        return

    def _run():
        try:
            from deep_memory.reasoning.extractor import process_session
            stats = process_session(
                session_id=session_id,
                transcript=transcript,
                llm_call=llm_call,
            )
            logger.info(
                "Deep memory processed session %s: %d entities, %d conclusions",
                session_id,
                stats.get("entities_found", 0),
                stats.get("conclusions_added", 0),
            )
        except Exception:
            logger.exception("Failed to process session %s for deep memory", session_id)

    thread = threading.Thread(target=_run, name=f"deep-memory-{session_id}", daemon=True)
    thread.start()


class DeepMemoryPlugin:
    """Hermes plugin that hooks into session lifecycle.
    
    Register with Hermes's plugin system to automatically
    extract insights after each conversation session.
    """
    
    name = "deep_memory"
    description = "Reasoning-based memory extraction after sessions"

    def on_session_end(self, session_id: str = None, **kwargs) -> None:
        """Called by Hermes at the end of each session."""
        # Note: we'd need to get messages from the session DB
        # This hook provides session_id but not messages directly
        try:
            from hermes_state import SessionDB
            sdb = SessionDB()
            messages = sdb.get_session_messages(session_id) if session_id else []
            if messages:
                process_session_async(session_id, messages)
        except Exception:
            logger.debug("Could not retrieve session messages for deep memory processing")
