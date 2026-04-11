"""Tests for the thin Hermes adapter layer."""

from __future__ import annotations

import json

from deep_memory.adapters import hermes_plugin, hermes_tools


def test_iter_tool_adapters_exposes_current_toolset():
    adapters = hermes_tools.iter_tool_adapters()
    assert [adapter.name for adapter in adapters] == ["recall", "learn", "entities"]
    for adapter in adapters:
        assert adapter.schema["name"] == adapter.name
        assert callable(adapter.handler)
        assert callable(adapter.availability_check)
        assert adapter.service_dependency == "deep_memory.api.create_service"


def test_register_with_registry_registers_deep_memory_toolset():
    registered_calls = []

    class FakeRegistry:
        def register(self, **kwargs):
            registered_calls.append(kwargs)

    names = hermes_tools.register_with_registry(FakeRegistry())
    assert names == ["recall", "learn", "entities"]
    assert all(call["toolset"] == "deep_memory" for call in registered_calls)
    assert [call["name"] for call in registered_calls] == names


def test_tool_handlers_use_service_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_tools, "create_service", lambda: hermes_tools.DeepMemoryService(db_path=tmp_path / "tools.db"))

    learn_adapter = next(item for item in hermes_tools.iter_tool_adapters() if item.name == "learn")
    recall_adapter = next(item for item in hermes_tools.iter_tool_adapters() if item.name == "recall")

    learn_result = json.loads(
        learn_adapter.handler({"entity": "Alice", "insight": "Likes SQLite", "type": "explicit"})
    )
    assert learn_result["stored"] is True

    recall_result = json.loads(recall_adapter.handler({"query": "SQLite", "limit": 5}))
    assert any(item["content"] == "Likes SQLite" for item in recall_result["results"])


def test_build_prompt_context_uses_supplied_builder():
    context = hermes_plugin.build_prompt_context("alice", builder=lambda entity_id=None: f"ctx:{entity_id}")
    assert context == "ctx:alice"


def test_build_prompt_context_is_defensive_when_builder_fails():
    def boom(entity_id=None):
        raise RuntimeError("nope")

    assert hermes_plugin.build_prompt_context("alice", builder=boom) == ""


def test_process_session_messages_returns_false_without_processor(monkeypatch):
    monkeypatch.setattr(hermes_plugin, "_load_session_processor", lambda: None)
    assert hermes_plugin.process_session_messages("session-1", [{"role": "user", "content": "hi"}], processor=None) is False


def test_process_session_messages_invokes_processor_with_keyword_style():
    received = {}

    def processor(*, session_id, messages):
        received["session_id"] = session_id
        received["messages"] = messages

    ok = hermes_plugin.process_session_messages(
        "session-2",
        [{"role": "user", "content": "hello"}],
        processor=processor,
    )
    assert ok is True
    assert received["session_id"] == "session-2"
    assert received["messages"][0]["content"] == "hello"


def test_plugin_uses_loader_and_processor():
    calls = {}

    def loader(session_id):
        calls["loaded"] = session_id
        return [{"role": "user", "content": "hello"}]

    def processor(session_id, messages):
        calls["processed"] = (session_id, messages)

    plugin = hermes_plugin.DeepMemorySessionPlugin(message_loader=loader, processor=processor)
    assert plugin.on_session_end("session-3") is True
    assert calls["loaded"] == "session-3"
    assert calls["processed"][0] == "session-3"
