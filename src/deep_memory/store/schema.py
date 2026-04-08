"""SQLite schema definitions and migrations for Deep Memory."""

import sqlite3

SCHEMA_VERSION = 1

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
    content_rowid='id',
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


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indexes, FTS, and triggers."""
    conn.executescript(TABLES_SQL)
    conn.executescript(FTS_SQL)
    conn.executescript(FTS_TRIGGERS_SQL)
    # Try to load sqlite-vec for vector search
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conclusions_vec USING vec0(
                embedding float[384]
            )
        """)
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

