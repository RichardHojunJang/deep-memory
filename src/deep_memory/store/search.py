"""Hybrid search combining FTS5 keyword search and sqlite-vec cosine similarity."""

from __future__ import annotations

import sqlite3
from typing import Any


def _fts_search(
    conn: sqlite3.Connection,
    query: str,
    entity_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Run an FTS5 keyword search on conclusions_fts and return scored results."""
    sql = """
        SELECT c.*, bm25(conclusions_fts) AS fts_score
        FROM conclusions_fts f
        JOIN conclusions c ON c.id = f.rowid
        WHERE conclusions_fts MATCH ?
    """
    params: list[Any] = [query]

    if entity_id is not None:
        sql += " AND c.entity_id = ?"
        params.append(entity_id)

    sql += " AND c.superseded_by IS NULL ORDER BY fts_score LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _vec_search(
    conn: sqlite3.Connection,
    embedding: bytes,
    entity_id: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Run a cosine-similarity search using sqlite-vec on conclusions_vec."""
    sql = """
        SELECT v.conclusion_id, v.distance
        FROM conclusions_vec v
        WHERE embedding MATCH ? AND k = ?
    """
    params: list[Any] = [embedding, limit]

    rows = conn.execute(sql, params).fetchall()
    vec_results = []
    for row in rows:
        cid = row["conclusion_id"]
        distance = row["distance"]
        c = conn.execute("SELECT * FROM conclusions WHERE id = ?", (cid,)).fetchone()
        if c is None:
            continue
        if entity_id is not None and c["entity_id"] != entity_id:
            continue
        if c["superseded_by"] is not None:
            continue
        d = dict(c)
        d["vec_distance"] = distance
        vec_results.append(d)
    return vec_results


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    embedding: bytes | None = None,
    entity_id: str | None = None,
    limit: int = 20,
    fts_weight: float = 0.5,
    vec_weight: float = 0.5,
) -> list[dict]:
    """
    Combine FTS5 keyword search with sqlite-vec cosine similarity.

    Returns a ranked list of conclusion dicts with a combined ``score`` key.
    If no embedding is provided, falls back to FTS-only search.
    """
    results_by_id: dict[int, dict] = {}

    # --- FTS5 leg ---
    fts_results = _fts_search(conn, query, entity_id=entity_id, limit=limit)
    if fts_results:
        # bm25 returns negative scores (lower = more relevant), normalise to 0..1
        min_s = min(r["fts_score"] for r in fts_results)
        max_s = max(r["fts_score"] for r in fts_results)
        spread = max_s - min_s if max_s != min_s else 1.0
        for r in fts_results:
            norm = 1.0 - ((r["fts_score"] - min_s) / spread)  # higher = better
            rid = r["id"]
            entry = dict(r)
            entry.pop("fts_score", None)
            entry["_fts"] = norm
            entry["_vec"] = 0.0
            results_by_id[rid] = entry

    # --- Vector leg ---
    if embedding is not None:
        vec_results = _vec_search(conn, embedding, entity_id=entity_id, limit=limit)
        if vec_results:
            max_d = max(r["vec_distance"] for r in vec_results) or 1.0
            for r in vec_results:
                norm = 1.0 - (r["vec_distance"] / max_d) if max_d else 1.0
                rid = r["id"]
                if rid in results_by_id:
                    results_by_id[rid]["_vec"] = norm
                else:
                    entry = dict(r)
                    entry.pop("vec_distance", None)
                    entry["_fts"] = 0.0
                    entry["_vec"] = norm
                    results_by_id[rid] = entry

    # --- Combine ---
    for entry in results_by_id.values():
        entry["score"] = fts_weight * entry.pop("_fts") + vec_weight * entry.pop("_vec")

    ranked = sorted(results_by_id.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:limit]
