"""SQLite schema definitions for the deep-memory storage layer."""

import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT DEFAULT 'person',
    card TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT REFERENCES entities(id),
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    premises TEXT,
    confidence REAL DEFAULT 1.0,
    source_sessions TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    superseded_by INTEGER REFERENCES conclusions(id)
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    short_summary TEXT,
    long_summary TEXT,
    key_decisions TEXT,
    entities_mentioned TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS conclusions_fts USING fts5(content, type);
"""

VEC_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS conclusions_vec USING vec0(
    conclusion_id INTEGER PRIMARY KEY,
    embedding float[384]
);
"""


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables, FTS5 virtual table, and vec0 virtual table."""
    conn.executescript(SCHEMA_SQL)
    conn.executescript(FTS_SQL)
    conn.executescript(VEC_SQL)
