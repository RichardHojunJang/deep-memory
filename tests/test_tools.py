"""Tests for Hermes tool definitions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_memory.store.db import DeepMemoryDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh DB and patch DeepMemoryDB to use it."""
    test_db = DeepMemoryDB(db_path=tmp_path / "test.db")
    yield test_db
    test_db.close()


@pytest.fixture(autouse=True)
def patch_db(db):
    """Patch DeepMemoryDB constructor to return our test DB."""
    with patch("deep_memory.store.db.DEFAULT_DB_PATH", db.db_path):
        yield


class TestRecallTool:
    def test_recall_empty(self):
        from deep_memory.tools.recall import recall
        result = json.loads(recall("anything"))
        assert result["results"] == []

    def test_recall_finds_match(self, db):
        from deep_memory.tools.recall import recall
        db.upsert_entity("u1", "Alice")
        db.add_conclusion("u1", "explicit", "Alice is a Python developer")
        result = json.loads(recall("Python developer"))
        assert len(result["results"]) >= 1
        assert "Python" in result["results"][0]["content"]

    def test_recall_entity_filter(self, db):
        from deep_memory.tools.recall import recall
        db.upsert_entity("u1", "Alice")
        db.upsert_entity("u2", "Bob")
        db.add_conclusion("u1", "explicit", "Loves coffee")
        db.add_conclusion("u2", "explicit", "Loves coffee too")
        result = json.loads(recall("coffee", entity="u1"))
        assert all(r["entity_id"] == "u1" for r in result["results"])

    def test_recall_schema(self):
        from deep_memory.tools.recall import TOOL_SCHEMA
        assert TOOL_SCHEMA["name"] == "recall"
        assert "query" in TOOL_SCHEMA["parameters"]["properties"]


class TestLearnTool:
    def test_learn_creates_entity(self, db):
        from deep_memory.tools.learn import learn
        result = json.loads(learn("Alice", "Alice is an engineer"))
        assert result["stored"] is True
        assert result["entity_id"] == "alice"
        entity = db.get_entity("alice")
        assert entity is not None
        assert entity["name"] == "Alice"

    def test_learn_invalid_type(self):
        from deep_memory.tools.learn import learn
        result = json.loads(learn("Alice", "test", type="invalid"))
        assert "error" in result

    def test_learn_deductive(self, db):
        from deep_memory.tools.learn import learn
        result = json.loads(learn("Bob", "Bob is likely busy", type="deductive"))
        assert result["type"] == "deductive"
        conclusions = db.get_conclusions(entity_id="bob")
        assert len(conclusions) == 1

    def test_learn_schema(self):
        from deep_memory.tools.learn import TOOL_SCHEMA
        assert TOOL_SCHEMA["name"] == "learn"
        assert "entity" in TOOL_SCHEMA["parameters"]["properties"]


class TestEntitiesTool:
    def test_list_empty(self):
        from deep_memory.tools.entities import entities
        result = json.loads(entities("list"))
        assert result["total"] == 0

    def test_list_with_entities(self, db):
        from deep_memory.tools.entities import entities
        db.upsert_entity("u1", "Alice", "person")
        db.upsert_entity("p1", "Project X", "project")
        result = json.loads(entities("list"))
        assert result["total"] == 2

    def test_get_entity(self, db):
        from deep_memory.tools.entities import entities
        db.upsert_entity("alice", "Alice", card={"role": "engineer"})
        db.add_conclusion("alice", "explicit", "Alice likes Python")
        result = json.loads(entities("get", name="Alice"))
        assert result["entity"]["name"] == "Alice"
        assert result["total_conclusions"] == 1

    def test_get_nonexistent(self):
        from deep_memory.tools.entities import entities
        result = json.loads(entities("get", name="Nobody"))
        assert "error" in result

    def test_update_card(self, db):
        from deep_memory.tools.entities import entities
        db.upsert_entity("alice", "Alice")
        result = json.loads(entities("update", name="Alice", card={"role": "CTO"}))
        assert result["updated"] is True
        assert result["entity"]["card"]["role"] == "CTO"

    def test_update_creates_if_missing(self):
        from deep_memory.tools.entities import entities
        result = json.loads(entities("update", name="NewPerson", card={"role": "dev"}))
        assert result["updated"] is True

    def test_invalid_action(self):
        from deep_memory.tools.entities import entities
        result = json.loads(entities("delete"))
        assert "error" in result

    def test_schema(self):
        from deep_memory.tools.entities import TOOL_SCHEMA
        assert TOOL_SCHEMA["name"] == "entities"
