"""Connection management and CRUD operations for the deep-memory store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from .schema import create_tables

DEFAULT_DB_PATH = Path.home() / ".hermes" / "deep_memory" / "memory.db"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with sqlite-vec loaded."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Initialise the database: create tables and return the connection."""
    conn = get_connection(db_path)
    create_tables(conn)
    return conn


# ---------- entities ----------


def get_entity(conn: sqlite3.Connection, entity_id: str) -> dict | None:
    """Fetch an entity by id. Returns dict or None."""
    row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
    return dict(row) if row else None


def upsert_entity(
    conn: sqlite3.Connection,
    entity_id: str,
    name: str,
    entity_type: str = "person",
    card: str | None = None,
) -> None:
    """Insert or update an entity."""
    conn.execute(
        """
        INSERT INTO entities (id, name, type, card)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            type = excluded.type,
            card = excluded.card,
            updated_at = CURRENT_TIMESTAMP
        """,
        (entity_id, name, entity_type, card),
    )
    conn.commit()


def list_entities(conn: sqlite3.Connection) -> list[dict]:
    """Return all entities."""
    rows = conn.execute("SELECT * FROM entities ORDER BY name").fetchall()
    return [dict(r) for r in rows]


# ---------- conclusions ----------


def add_conclusion(
    conn: sqlite3.Connection,
    entity_id: str,
    conclusion_type: str,
    content: str,
    premises: list[str] | None = None,
    confidence: float = 1.0,
    source_sessions: list[str] | None = None,
    embedding: bytes | None = None,
) -> int:
    """Add a conclusion and sync to FTS5 + vec0. Returns the new row id."""
    cur = conn.execute(
        """
        INSERT INTO conclusions (entity_id, type, content, premises, confidence,
                                 source_sessions, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entity_id,
            conclusion_type,
            content,
            json.dumps(premises) if premises else None,
            confidence,
            json.dumps(source_sessions) if source_sessions else None,
            embedding,
        ),
    )
    row_id = cur.lastrowid

    # Sync FTS5
    conn.execute(
        "INSERT INTO conclusions_fts (rowid, content, type) VALUES (?, ?, ?)",
        (row_id, content, conclusion_type),
    )

    # Sync vec0 if embedding provided
    if embedding is not None:
        conn.execute(
            "INSERT INTO conclusions_vec (conclusion_id, embedding) VALUES (?, ?)",
            (row_id, embedding),
        )

    conn.commit()
    return row_id


def get_conclusions(
    conn: sqlite3.Connection,
    entity_id: str | None = None,
    include_superseded: bool = False,
) -> list[dict]:
    """Fetch conclusions, optionally filtered by entity_id."""
    query = "SELECT * FROM conclusions WHERE 1=1"
    params: list[Any] = []
    if entity_id is not None:
        query += " AND entity_id = ?"
        params.append(entity_id)
    if not include_superseded:
        query += " AND superseded_by IS NULL"
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def supersede_conclusion(
    conn: sqlite3.Connection,
    old_id: int,
    new_id: int,
) -> None:
    """Mark a conclusion as superseded by another."""
    conn.execute(
        "UPDATE conclusions SET superseded_by = ? WHERE id = ?",
        (new_id, old_id),
    )
    conn.commit()


# ---------- summaries ----------


def add_summary(
    conn: sqlite3.Connection,
    session_id: str,
    short_summary: str | None = None,
    long_summary: str | None = None,
    key_decisions: list[str] | None = None,
    entities_mentioned: list[str] | None = None,
) -> int:
    """Add a session summary. Returns the new row id."""
    cur = conn.execute(
        """
        INSERT INTO summaries (session_id, short_summary, long_summary,
                               key_decisions, entities_mentioned)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            short_summary,
            long_summary,
            json.dumps(key_decisions) if key_decisions else None,
            json.dumps(entities_mentioned) if entities_mentioned else None,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_summary(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Fetch the most recent summary for a session."""
    row = conn.execute(
        "SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def list_summaries(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Return recent summaries."""
    rows = conn.execute(
        "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]
