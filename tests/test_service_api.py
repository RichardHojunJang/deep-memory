"""Tests for the stable, backend-ready service API."""

from __future__ import annotations

from deep_memory.adapters.hermes_tools import load_service_api
from deep_memory.api import DeepMemoryService, LearnRequest, RecallRequest


def test_load_service_api_returns_constructor():
    service_factory = load_service_api()
    assert callable(service_factory)


def test_service_factory_builds_deep_memory_service(tmp_path):
    service_factory = load_service_api()
    service = service_factory(db_path=tmp_path / "service.db")
    try:
        assert isinstance(service, DeepMemoryService)
    finally:
        service.close()


def test_service_learn_and_recall_round_trip(tmp_path):
    service = DeepMemoryService(db_path=tmp_path / "memory.db")
    try:
        stored = service.learn(
            LearnRequest(entity="Alice", insight="Alice likes SQLite", conclusion_type="explicit")
        )
        assert stored["stored"] is True
        results = service.recall(RecallRequest(query="SQLite", limit=5))
        assert any(item.content == "Alice likes SQLite" for item in results)
    finally:
        service.close()
