"""Hermes-independent service layer for Deep Memory adapters.

This module exposes a small provider-agnostic API over the existing storage and
search primitives so current tools and future backends can share one stable
contract-oriented entrypoint.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from deep_memory.api.contracts import (
    ConclusionRecord,
    EntityDetailRecord,
    EntityQuery,
    EntityRecord,
    EntityUpdate,
    EntityUpsert,
    LearnRequest,
    RecallRecord,
    RecallRequest,
    VALID_CONCLUSION_TYPES,
)
from deep_memory.runtime import resolve_deep_memory_db_path
from deep_memory.store import DeepMemoryDB, hybrid_search
from deep_memory.store import db as db_module
from deep_memory.store.db import get_embedder


class DeepMemoryService:
    """Adapter-friendly facade over Deep Memory storage and search logic."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = (
            resolve_deep_memory_db_path(db_path)
            if db_path is not None
            else db_module.DEFAULT_DB_PATH
        )
        self._db = DeepMemoryDB(self.db_path)

    @property
    def db(self) -> DeepMemoryDB:
        return self._db

    def close(self) -> None:
        """Close the underlying database connection."""
        self._db.close()

    def upsert_entity(self, payload: EntityUpsert) -> EntityRecord:
        """Create or update an entity profile using the underlying store layer."""
        row = self._db.upsert_entity(
            payload.entity_id,
            payload.name,
            payload.entity_type,
            payload.card,
        )
        return EntityRecord.from_row(row)

    def learn(self, payload: LearnRequest) -> Dict[str, object]:
        """Store a new conclusion, auto-creating the entity when needed."""
        if payload.conclusion_type not in VALID_CONCLUSION_TYPES:
            raise ValueError(
                "Invalid conclusion type %r. Must be one of: %s"
                % (payload.conclusion_type, ", ".join(VALID_CONCLUSION_TYPES))
            )

        entity_id = payload.entity_id or self.slugify(payload.entity)
        existing = self._db.get_entity(entity_id)
        if not existing:
            self._db.upsert_entity(
                entity_id,
                payload.entity,
                payload.entity_type,
                payload.card,
            )
        elif payload.card:
            self._db.update_entity_card(entity_id, payload.card)

        conclusion_id = self._db.add_conclusion(
            entity_id=entity_id,
            type=payload.conclusion_type,
            content=payload.insight,
            premises=payload.premises or None,
            confidence=payload.confidence,
            source_sessions=payload.source_sessions or None,
        )

        return {
            "stored": True,
            "conclusion_id": conclusion_id,
            "entity": self.get_entity(EntityQuery(entity_id=entity_id)).entity.to_dict(),
            "conclusion": ConclusionRecord.from_row(
                self._db.get_conclusions(entity_id=entity_id, limit=1)[0]
            ).to_dict(),
        }

    def recall(self, payload: RecallRequest) -> List[RecallRecord]:
        """Run hybrid recall using the existing embedder and search stack."""
        query_embedding = get_embedder().embed(payload.query)
        results = hybrid_search(
            self._db.conn,
            query=payload.query,
            query_embedding=query_embedding,
            entity_id=payload.entity_id,
            limit=payload.limit,
        )

        entity_names = self._load_entity_names([result.entity_id for result in results])
        return [
            RecallRecord(
                conclusion_id=result.conclusion_id,
                entity_id=result.entity_id,
                entity_name=entity_names.get(result.entity_id),
                conclusion_type=result.type,
                content=result.content,
                confidence=result.confidence,
                score=round(result.combined_score, 4),
            )
            for result in results
        ]

    def list_entities(self, query: Optional[EntityQuery] = None) -> List[EntityRecord]:
        """List tracked entities, optionally filtered by entity type and limit."""
        query = query or EntityQuery()
        rows = self._db.list_entities(type=query.entity_type)
        if query.limit is not None:
            rows = rows[: query.limit]
        return [EntityRecord.from_row(row) for row in rows]

    def get_entity(self, query: EntityQuery) -> EntityDetailRecord:
        """Resolve one entity by explicit id or slugified name."""
        entity_id = query.entity_id or (self.slugify(query.name) if query.name else None)
        if not entity_id:
            raise ValueError("entity_id or name is required")

        row = self._db.get_entity(entity_id)
        if not row:
            raise KeyError("Entity %r not found" % entity_id)

        conclusions = []
        if query.include_conclusions:
            conclusions = [
                ConclusionRecord.from_row(item)
                for item in self._db.get_conclusions(
                    entity_id=entity_id,
                    limit=query.conclusion_limit,
                )
            ]
        return EntityDetailRecord(
            entity=EntityRecord.from_row(row),
            conclusions=conclusions,
        )

    def update_entity(self, payload: EntityUpdate) -> EntityRecord:
        """Merge card fields into an existing entity profile."""
        row = self._db.update_entity_card(payload.entity_id, payload.card)
        if not row:
            raise KeyError("Entity %r not found" % payload.entity_id)
        return EntityRecord.from_row(row)

    def build_context(
        self,
        query: Union[str, RecallRequest],
        entity_id: Optional[str] = None,
        limit: int = 5,
        header: str = "Deep Memory Context",
    ) -> str:
        """Format a compact context block adapters can inject into prompts."""
        request = query if isinstance(query, RecallRequest) else RecallRequest(
            query=query,
            entity_id=entity_id,
            limit=limit,
        )
        records = self.recall(request)
        if not records:
            return "%s\n- No matching insights found." % header

        lines = [header]
        for index, record in enumerate(records, 1):
            label = record.entity_name or record.entity_id
            lines.append(
                "%d. [%s/%s] %s (confidence=%.2f, score=%.4f)"
                % (
                    index,
                    label,
                    record.conclusion_type,
                    record.content,
                    record.confidence,
                    record.score,
                )
            )
        return "\n".join(lines)

    @staticmethod
    def slugify(name: str) -> str:
        """Convert a display name into the canonical entity identifier."""
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    def _load_entity_names(self, entity_ids: Sequence[str]) -> Dict[str, str]:
        names = {}
        seen = set()
        for entity_id in entity_ids:
            if entity_id in seen:
                continue
            seen.add(entity_id)
            row = self._db.get_entity(entity_id)
            if row:
                names[entity_id] = row["name"]
        return names
