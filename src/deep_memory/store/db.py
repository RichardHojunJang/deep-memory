"""Database connection management and CRUD operations for Deep Memory."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from .schema import init_schema, get_schema_version, SCHEMA_VERSION


DEFAULT_DB_PATH = Path.home() / ".hermes" / "deep_memory" / "memory.db"


class DeepMemoryDB:
    """Single-file SQLite store for deep memory."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            if get_schema_version(self._conn) < SCHEMA_VERSION:
                init_schema(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── Entity CRUD ──────────────────────────────────────────

    def upsert_entity(
        self, entity_id: str, name: str, type: str = "person", card: dict | None = None
    ) -> dict:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO entities (id, name, type, card, updated_at)
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(id) DO UPDATE SET
                     name = excluded.name,
                     type = excluded.type,
                     card = COALESCE(excluded.card, entities.card),
                     updated_at = CURRENT_TIMESTAMP""",
                (entity_id, name, type, json.dumps(card) if card else None),
            )
        return self.get_entity(entity_id)

    def get_entity(self, entity_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("card"):
            d["card"] = json.loads(d["card"])
        return d

    def list_entities(self, type: str | None = None) -> list[dict]:
        if type:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY updated_at DESC", (type,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities ORDER BY updated_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("card"):
                d["card"] = json.loads(d["card"])
            result.append(d)
        return result

    def delete_entity(self, entity_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            return cur.rowcount > 0

    def update_entity_card(self, entity_id: str, card: dict) -> dict | None:
        with self.transaction() as conn:
            existing = self.get_entity(entity_id)
            if not existing:
                return None
            merged = {**(existing.get("card") or {}), **card}
            conn.execute(
                "UPDATE entities SET card = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(merged), entity_id),
            )
        return self.get_entity(entity_id)

    # ── Conclusion CRUD ──────────────────────────────────────

    def add_conclusion(
        self,
        entity_id: str,
        type: str,
        content: str,
        premises: list[str] | None = None,
        confidence: float = 1.0,
        source_sessions: list[str] | None = None,
        embedding: bytes | None = None,
    ) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO conclusions
                   (entity_id, type, content, premises, confidence, source_sessions, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entity_id,
                    type,
                    content,
                    json.dumps(premises) if premises else None,
                    confidence,
                    json.dumps(source_sessions) if source_sessions else None,
                    embedding,
                ),
            )
            return cur.lastrowid

    def get_conclusions(
        self,
        entity_id: str | None = None,
        type: str | None = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        conditions = []
        params: list[Any] = []
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        if type:
            conditions.append("type = ?")
            params.append(type)
        if active_only:
            conditions.append("superseded_by IS NULL")
        where = " AND ".join(conditions)
        if where:
            where = "WHERE " + where
        rows = self.conn.execute(
            f"SELECT * FROM conclusions {where} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for key in ("premises", "source_sessions"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            d.pop("embedding", None)  # Don't return raw bytes
            result.append(d)
        return result

    def supersede_conclusion(self, old_id: int, new_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE conclusions SET superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )

    # ── Summary CRUD ─────────────────────────────────────────

    def add_summary(
        self,
        session_id: str,
        short_summary: str,
        long_summary: str | None = None,
        key_decisions: list[str] | None = None,
        entities_mentioned: list[str] | None = None,
    ) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO summaries
                   (session_id, short_summary, long_summary, key_decisions, entities_mentioned)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    session_id,
                    short_summary,
                    long_summary,
                    json.dumps(key_decisions) if key_decisions else None,
                    json.dumps(entities_mentioned) if entities_mentioned else None,
                ),
            )
            return cur.lastrowid

    def get_summaries(
        self, session_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM summaries WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for key in ("key_decisions", "entities_mentioned"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            result.append(d)
        return result
