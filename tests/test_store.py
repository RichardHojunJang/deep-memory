"""Tests for the storage layer."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from deep_memory.store.schema import init_schema, get_schema_version, SCHEMA_VERSION
from deep_memory.store.db import DeepMemoryDB
from deep_memory.store.search import fts_search, hybrid_search


@pytest.fixture
def db(tmp_path):
    """Create a fresh DeepMemoryDB for each test."""
    db = DeepMemoryDB(db_path=tmp_path / "test.db")
    yield db
    db.close()


class TestSchema:
    def test_init_creates_tables(self, db):
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert "entities" in table_names
        assert "conclusions" in table_names
        assert "summaries" in table_names
        assert "conclusions_fts" in table_names

    def test_schema_version(self, db):
        version = get_schema_version(db.conn)
        assert version == SCHEMA_VERSION


class TestEntityCRUD:
    def test_upsert_and_get(self, db):
        entity = db.upsert_entity("user-1", "Alice", "person", {"role": "engineer"})
        assert entity["id"] == "user-1"
        assert entity["name"] == "Alice"
        assert entity["card"]["role"] == "engineer"

    def test_upsert_updates_existing(self, db):
        db.upsert_entity("user-1", "Alice", "person", {"role": "engineer"})
        db.upsert_entity("user-1", "Alice B", "person", {"role": "senior engineer"})
        entity = db.get_entity("user-1")
        assert entity["name"] == "Alice B"
        assert entity["card"]["role"] == "senior engineer"

    def test_get_nonexistent(self, db):
        assert db.get_entity("nope") is None

    def test_list_entities(self, db):
        db.upsert_entity("u1", "Alice", "person")
        db.upsert_entity("u2", "Bob", "person")
        db.upsert_entity("p1", "Project X", "project")
        assert len(db.list_entities()) == 3
        assert len(db.list_entities(type="person")) == 2
        assert len(db.list_entities(type="project")) == 1

    def test_delete_entity(self, db):
        db.upsert_entity("u1", "Alice")
        assert db.delete_entity("u1") is True
        assert db.get_entity("u1") is None
        assert db.delete_entity("u1") is False

    def test_update_card_merges(self, db):
        db.upsert_entity("u1", "Alice", card={"role": "eng", "lang": "python"})
        db.update_entity_card("u1", {"role": "senior eng", "team": "infra"})
        entity = db.get_entity("u1")
        assert entity["card"]["role"] == "senior eng"
        assert entity["card"]["lang"] == "python"
        assert entity["card"]["team"] == "infra"


class TestConclusionCRUD:
    def test_add_and_get(self, db):
        db.upsert_entity("u1", "Alice")
        cid = db.add_conclusion("u1", "explicit", "Alice is an engineer")
        conclusions = db.get_conclusions(entity_id="u1")
        assert len(conclusions) == 1
        assert conclusions[0]["content"] == "Alice is an engineer"
        assert conclusions[0]["id"] == cid

    def test_filter_by_type(self, db):
        db.upsert_entity("u1", "Alice")
        db.add_conclusion("u1", "explicit", "Fact 1")
        db.add_conclusion("u1", "deductive", "Deduction 1")
        db.add_conclusion("u1", "inductive", "Pattern 1")
        assert len(db.get_conclusions(entity_id="u1", type="explicit")) == 1
        assert len(db.get_conclusions(entity_id="u1", type="deductive")) == 1

    def test_supersede(self, db):
        db.upsert_entity("u1", "Alice")
        old = db.add_conclusion("u1", "explicit", "Alice likes Java")
        new = db.add_conclusion("u1", "explicit", "Alice prefers Python over Java")
        db.supersede_conclusion(old, new)
        active = db.get_conclusions(entity_id="u1", active_only=True)
        assert len(active) == 1
        assert active[0]["content"] == "Alice prefers Python over Java"
        all_c = db.get_conclusions(entity_id="u1", active_only=False)
        assert len(all_c) == 2

    def test_with_premises(self, db):
        db.upsert_entity("u1", "Alice")
        db.add_conclusion(
            "u1", "deductive", "Alice is busy",
            premises=["Alice has 5 meetings today", "Alice skipped lunch"],
        )
        c = db.get_conclusions(entity_id="u1")[0]
        assert c["premises"] == ["Alice has 5 meetings today", "Alice skipped lunch"]


class TestSummaryCRUD:
    def test_add_and_get(self, db):
        sid = db.add_summary(
            "session-123",
            short_summary="Discussed project architecture",
            key_decisions=["Use SQLite", "Skip Redis"],
            entities_mentioned=["u1", "p1"],
        )
        summaries = db.get_summaries(session_id="session-123")
        assert len(summaries) == 1
        assert summaries[0]["key_decisions"] == ["Use SQLite", "Skip Redis"]


class TestFTSSearch:
    def test_basic_search(self, db):
        db.upsert_entity("u1", "Alice")
        db.add_conclusion("u1", "explicit", "Alice enjoys Python programming")
        db.add_conclusion("u1", "explicit", "Alice has a cat named Whiskers")
        results = fts_search(db.conn, "Python programming")
        assert len(results) >= 1
        assert "Python" in results[0].content

    def test_search_with_entity_filter(self, db):
        db.upsert_entity("u1", "Alice")
        db.upsert_entity("u2", "Bob")
        db.add_conclusion("u1", "explicit", "Likes coffee")
        db.add_conclusion("u2", "explicit", "Likes coffee too")
        results = fts_search(db.conn, "coffee", entity_id="u1")
        assert len(results) == 1
        assert results[0].entity_id == "u1"

    def test_hybrid_fts_only(self, db):
        db.upsert_entity("u1", "Alice")
        db.add_conclusion("u1", "explicit", "Alice works at a startup")
        db.add_conclusion("u1", "inductive", "Alice prefers small teams")
        results = hybrid_search(db.conn, "startup small teams")
        assert len(results) >= 1
