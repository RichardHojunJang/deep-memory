"""Hybrid semantic + FTS5 search for Deep Memory."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single search result with combined score."""
    conclusion_id: int
    entity_id: str
    content: str
    type: str
    confidence: float
    fts_score: float = 0.0
    vec_score: float = 0.0
    combined_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.conclusion_id,
            "entity_id": self.entity_id,
            "content": self.content,
            "type": self.type,
            "confidence": self.confidence,
            "score": round(self.combined_score, 4),
        }


def _has_vec_table(conn: sqlite3.Connection) -> bool:
    """Check if the vec0 virtual table exists."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='conclusions_vec'"
    ).fetchone()
    return row is not None


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    entity_id: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    """Full-text search using FTS5 BM25 ranking."""
    # Tokenize query into OR terms for FTS5
    words = query.split()
    if len(words) == 1:
        safe_query = words[0].replace('"', '""')
        fts_expr = f'"{safe_query}"'
    else:
        # Join with OR so any matching term returns results
        safe_parts = [w.replace('"', '""') for w in words]
        fts_expr = " OR ".join(f'"{p}"' for p in safe_parts)

    sql = """
        SELECT c.id, c.entity_id, c.content, c.type, c.confidence,
               rank AS fts_rank
        FROM conclusions_fts fts
        JOIN conclusions c ON c.id = fts.rowid
        WHERE conclusions_fts MATCH ?
          AND c.superseded_by IS NULL
    """
    params: list[Any] = [fts_expr]

    if entity_id:
        sql += " AND c.entity_id = ?"
        params.append(entity_id)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        r = SearchResult(
            conclusion_id=row[0],
            entity_id=row[1],
            content=row[2],
            type=row[3],
            confidence=row[4],
            fts_score=abs(row[5]),  # BM25 returns negative scores
        )
        results.append(r)
    return results


def vec_search(
    conn: sqlite3.Connection,
    query_embedding: bytes,
    entity_id: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    """Vector similarity search using sqlite-vec."""
    if not _has_vec_table(conn):
        return []

    sql = """
        SELECT v.rowid, v.distance,
               c.entity_id, c.content, c.type, c.confidence
        FROM conclusions_vec v
        JOIN conclusions c ON c.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
          AND c.superseded_by IS NULL
    """
    params: list[Any] = [query_embedding, limit * 2]  # over-fetch for filtering

    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        if entity_id and row[2] != entity_id:
            continue
        r = SearchResult(
            conclusion_id=row[0],
            entity_id=row[2],
            content=row[3],
            type=row[4],
            confidence=row[5],
            vec_score=1.0 - row[1],  # distance → similarity
        )
        results.append(r)
        if len(results) >= limit:
            break
    return results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: bytes | None = None,
    entity_id: str | None = None,
    limit: int = 10,
    fts_weight: float = 0.4,
    vec_weight: float = 0.6,
) -> list[SearchResult]:
    """Combine FTS5 and vector search results with weighted scoring.
    
    If query_embedding is None, falls back to FTS-only search.
    """
    # FTS results
    fts_results = fts_search(conn, query, entity_id=entity_id, limit=limit * 2)

    # Normalize FTS scores
    max_fts = max((r.fts_score for r in fts_results), default=1.0) or 1.0
    for r in fts_results:
        r.fts_score = r.fts_score / max_fts

    # Vector results (if embedding provided)
    vec_results = []
    if query_embedding and _has_vec_table(conn):
        vec_results = vec_search(conn, query_embedding, entity_id=entity_id, limit=limit * 2)

    # Merge by conclusion_id
    merged: dict[int, SearchResult] = {}
    for r in fts_results:
        merged[r.conclusion_id] = r
    for r in vec_results:
        if r.conclusion_id in merged:
            merged[r.conclusion_id].vec_score = r.vec_score
        else:
            merged[r.conclusion_id] = r

    # Compute combined score
    use_vec = bool(vec_results)
    for r in merged.values():
        if use_vec:
            r.combined_score = (fts_weight * r.fts_score) + (vec_weight * r.vec_score)
        else:
            r.combined_score = r.fts_score  # FTS only
        # Boost by confidence
        r.combined_score *= r.confidence

    # Sort and limit
    ranked = sorted(merged.values(), key=lambda r: r.combined_score, reverse=True)
    return ranked[:limit]
