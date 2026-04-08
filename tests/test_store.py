"""Tests for the deep-memory storage layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deep_memory.store.db import (
    add_conclusion,
    add_summary,
    get_conclusions,
    get_entity,
    get_summary,
    init_db,
    list_entities,
    list_summaries,
    supersede_conclusion,
    upsert_entity,
)
from deep_memory.store.search import hybrid_search


@pytest.fixture()
def conn(tmp_path: Path):
    """Create a fresh temp DB for each test."""
    db_path = tmp_path / "test.db"
    c = init_db(db_path)
    yield c
    c.close()


def _make_embedding(dim: int = 384, seed: int = 42) -> bytes:
    """Generate a random float32 embedding as bytes."""
    vec = np.random.default_rng(seed).random(dim, dtype=np.float32)
    return vec.tobytes()


# ---- Schema tests ----


class TestSchema:
    def test_tables_created(self, conn):
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
        }
        assert "entities" in tables
        assert "conclusions" in tables
        assert "summaries" in tables
        assert "conclusions_fts" in tables
        assert "conclusions_vec" in tables

    def test_idempotent_init(self, tmp_path):
        """Calling init_db twice should not raise."""
        db = tmp_path / "idem.db"
        c1 = init_db(db)
        c1.close()
        c2 = init_db(db)
        c2.close()


# ---- Entity CRUD ----


class TestEntityCRUD:
    def test_upsert_and_get(self, conn):
        upsert_entity(conn, "u1", "Alice", "person", "card text")
        e = get_entity(conn, "u1")
        assert e is not None
        assert e["name"] == "Alice"
        assert e["type"] == "person"
        assert e["card"] == "card text"

    def test_upsert_updates(self, conn):
        upsert_entity(conn, "u1", "Alice")
        upsert_entity(conn, "u1", "Alice Updated", "org")
        e = get_entity(conn, "u1")
        assert e["name"] == "Alice Updated"
        assert e["type"] == "org"

    def test_get_missing(self, conn):
        assert get_entity(conn, "nope") is None

    def test_list_entities(self, conn):
        upsert_entity(conn, "u1", "Alice")
        upsert_entity(conn, "u2", "Bob")
        ents = list_entities(conn)
        assert len(ents) == 2


# ---- Conclusion CRUD ----


class TestConclusionCRUD:
    def test_add_and_get(self, conn):
        upsert_entity(conn, "u1", "Alice")
        cid = add_conclusion(conn, "u1", "explicit", "Alice likes coffee")
        rows = get_conclusions(conn, entity_id="u1")
        assert len(rows) == 1
        assert rows[0]["id"] == cid
        assert rows[0]["content"] == "Alice likes coffee"

    def test_add_with_embedding(self, conn):
        upsert_entity(conn, "u1", "Alice")
        emb = _make_embedding()
        cid = add_conclusion(
            conn,
            "u1",
            "deductive",
            "Alice is a morning person",
            premises=["She wakes at 5am", "She likes sunrises"],
            confidence=0.9,
            source_sessions=["sess-1"],
            embedding=emb,
        )
        assert cid > 0

    def test_supersede(self, conn):
        upsert_entity(conn, "u1", "Alice")
        old = add_conclusion(conn, "u1", "inductive", "Alice likes tea")
        new = add_conclusion(conn, "u1", "explicit", "Alice likes coffee")
        supersede_conclusion(conn, old, new)

        active = get_conclusions(conn, entity_id="u1", include_superseded=False)
        assert len(active) == 1
        assert active[0]["id"] == new

        all_rows = get_conclusions(conn, entity_id="u1", include_superseded=True)
        assert len(all_rows) == 2


# ---- Summary CRUD ----


class TestSummaryCRUD:
    def test_add_and_get(self, conn):
        sid = add_summary(
            conn,
            session_id="sess-1",
            short_summary="Setup project",
            long_summary="We set up the deep-memory project with SQLite.",
            key_decisions=["Use FTS5", "Use sqlite-vec"],
            entities_mentioned=["u1"],
        )
        s = get_summary(conn, "sess-1")
        assert s is not None
        assert s["short_summary"] == "Setup project"
        assert s["id"] == sid

    def test_list_summaries(self, conn):
        add_summary(conn, "s1", short_summary="First")
        add_summary(conn, "s2", short_summary="Second")
        sums = list_summaries(conn, limit=10)
        assert len(sums) == 2

    def test_get_missing_summary(self, conn):
        assert get_summary(conn, "nope") is None


# ---- Search tests ----


class TestSearch:
    def test_fts_search(self, conn):
        upsert_entity(conn, "u1", "Alice")
        add_conclusion(conn, "u1", "explicit", "Alice enjoys morning coffee")
        add_conclusion(conn, "u1", "explicit", "Alice reads books at night")

        results = hybrid_search(conn, "coffee", entity_id="u1")
        assert len(results) >= 1
        assert "coffee" in results[0]["content"].lower()

    def test_fts_search_no_match(self, conn):
        upsert_entity(conn, "u1", "Alice")
        add_conclusion(conn, "u1", "explicit", "Alice enjoys morning coffee")

        results = hybrid_search(conn, "xyzzyzzy_nomatch")
        assert len(results) == 0

    def test_hybrid_search_with_embedding(self, conn):
        upsert_entity(conn, "u1", "Alice")
        rng = np.random.default_rng(0)
        emb1 = rng.random(384, dtype=np.float32).tobytes()
        emb2 = rng.random(384, dtype=np.float32).tobytes()
        query_emb = emb1  # identical to first -> should rank highest

        add_conclusion(
            conn, "u1", "explicit", "Alice enjoys morning coffee", embedding=emb1
        )
        add_conclusion(
            conn, "u1", "explicit", "Alice reads books at night", embedding=emb2
        )

        results = hybrid_search(conn, "coffee", embedding=query_emb, entity_id="u1")
        assert len(results) >= 1

    def test_entity_id_filter(self, conn):
        upsert_entity(conn, "u1", "Alice")
        upsert_entity(conn, "u2", "Bob")
        add_conclusion(conn, "u1", "explicit", "Alice enjoys coffee")
        add_conclusion(conn, "u2", "explicit", "Bob enjoys coffee")

        results = hybrid_search(conn, "coffee", entity_id="u1")
        assert all(r["entity_id"] == "u1" for r in results)
