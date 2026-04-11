"""Adapter/service contract tests for the backend-ready integration surface."""

from __future__ import annotations

import types

from deep_memory.adapters.hermes_tools import load_service_api


def test_load_service_api_returns_none_when_service_layer_is_absent():
    assert load_service_api() is None


def test_load_service_api_accepts_future_factory(monkeypatch):
    fake_module = types.SimpleNamespace(create_service=lambda: "service")

    def fake_import(name):
        if name == "deep_memory.service_api":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr("deep_memory.adapters.hermes_tools.importlib.import_module", fake_import)
    service_api = load_service_api()
    assert callable(service_api)
    assert service_api() == "service"


def test_load_service_api_accepts_future_class(monkeypatch):
    class FakeService:
        pass

    fake_module = types.SimpleNamespace(DeepMemoryService=FakeService)

    def fake_import(name):
        if name == "deep_memory.service_api":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr("deep_memory.adapters.hermes_tools.importlib.import_module", fake_import)
    assert load_service_api() is FakeService
