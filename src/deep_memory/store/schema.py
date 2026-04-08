"""SQLite schema definitions and migrations for Deep Memory."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'person',
    card TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK(type IN ('explicit', 'deductive', 'inductive', 'abductive')),
    content TEXT NOT NULL,
    premises TEXT,
    confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    source_sessions TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    superseded_by INTEGER REFERENCES conclusions(id)
);

CREATE INDEX IF NOT EXISTS idx_conclusions_entity ON conclusions(entity_id);
CREATE INDEX IF NOT EXISTS idx_conclusions_type ON conclusions(type);
CREATE INDEX IF NOT EXISTS idx_conclusions_active ON conclusions(entity_id) WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    short_summary TEXT,
    long_summary TEXT,
    key_decisions TEXT,
    entities_mentioned TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS conclusions_fts USING fts5(
    content,
    type,
    content='',
    tokenize='porter unicode61'
);
"""

FTS_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS conclusions_ai AFTER INSERT ON conclusions BEGIN
    INSERT INTO conclusions_fts(rowid, content, type)
    VALUES (new.id, new.content, new.type);
END;

CREATE TRIGGER IF NOT EXISTS conclusions_ad AFTER DELETE ON conclusions BEGIN
    INSERT INTO conclusions_fts(conclusions_fts, rowid, content, type)
    VALUES ('delete', old.id, old.content, old.type);
END;

CREATE TRIGGER IF NOT EXISTS conclusions_au AFTER UPDATE OF content, type ON conclusions BEGIN
    INSERT INTO conclusions_fts(conclusions_fts, rowid, content, type)
    VALUES ('delete', old.id, old.content, old.type);
    INSERT INTO conclusions_fts(rowid, content, type)
    VALUES (new.id, new.content, new.type);
END;
"""


def init_schema(conn: sqlite3.Connection, embedding_dim: int = 384) -> None:
    """Create all tables, indexes, FTS, and triggers.

    Args:
        conn: SQLite connection.
        embedding_dim: Dimension for the vec0 embedding column (default 384).
    """
    conn.executescript(TABLES_SQL)
    conn.executescript(FTS_SQL)
    conn.executescript(FTS_TRIGGERS_SQL)
    # Try to load sqlite-vec for vector search
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        # Check if vec table already exists with a different dimension
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name='conclusions_vec'"
            ).fetchone()
            if row and row[0]:
                existing_sql = row[0]
                # Extract dimension from existing CREATE statement
                import re

                m = re.search(r"float\[(\d+)\]", existing_sql)
                if m:
                    existing_dim = int(m.group(1))
                    if existing_dim != embedding_dim:
                        logger.warning(
                            "Vec table exists with dimension %d but embedder "
                            "uses %d. Drop conclusions_vec to recreate.",
                            existing_dim,
                            embedding_dim,
                        )
        except Exception:
            pass

        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS conclusions_vec "
            f"USING vec0(embedding float[{embedding_dim}])"
        )
    except (ImportError, Exception):
        pass  # sqlite-vec not available; vector search disabled
    # Record schema version
    conn.execute(
        "INSERT OR REPLACE INTO schema_version(version) VALUES (?)",
        (SCHEMA_VERSION,),
    )
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return current schema version, or 0 if not initialized."""
    try:
        row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0

