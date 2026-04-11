"""Typed, Hermes-independent request and response contracts for Deep Memory.

These payloads provide a stable adapter-facing API layer that can be reused by
Hermes integrations, backend services, or future provider hooks without pulling
in tool-specific code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


VALID_CONCLUSION_TYPES = ("explicit", "deductive", "inductive", "abductive")


@dataclass
class EntityUpsert:
    """Payload for creating or replacing a tracked entity profile."""

    entity_id: str
    name: str
    entity_type: str = "person"
    card: Optional[Dict[str, Any]] = None


@dataclass
class LearnRequest:
    """Payload for storing a new conclusion or learned insight."""

    entity: str
    insight: str
    conclusion_type: str = "explicit"
    entity_id: Optional[str] = None
    entity_type: str = "person"
    card: Optional[Dict[str, Any]] = None
    premises: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_sessions: List[str] = field(default_factory=list)


@dataclass
class RecallRequest:
    """Payload for semantic recall queries."""

    query: str
    entity_id: Optional[str] = None
    limit: int = 10


@dataclass
class EntityQuery:
    """Payload for listing or resolving tracked entities."""

    entity_id: Optional[str] = None
    name: Optional[str] = None
    entity_type: Optional[str] = None
    limit: Optional[int] = None
    include_conclusions: bool = False
    conclusion_limit: int = 20


@dataclass
class EntityUpdate:
    """Payload for merging new card fields into an entity profile."""

    entity_id: str
    card: Dict[str, Any]


@dataclass
class EntityRecord:
    """Adapter-friendly entity record returned by the service layer."""

    entity_id: str
    name: str
    entity_type: str
    card: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "EntityRecord":
        return cls(
            entity_id=row["id"],
            name=row["name"],
            entity_type=row["type"],
            card=row.get("card") or {},
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConclusionRecord:
    """Normalized conclusion record for list/get/build-context responses."""

    conclusion_id: int
    entity_id: str
    conclusion_type: str
    content: str
    confidence: float
    premises: List[str] = field(default_factory=list)
    source_sessions: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    superseded_by: Optional[int] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ConclusionRecord":
        return cls(
            conclusion_id=row["id"],
            entity_id=row["entity_id"],
            conclusion_type=row["type"],
            content=row["content"],
            confidence=row["confidence"],
            premises=list(row.get("premises") or []),
            source_sessions=list(row.get("source_sessions") or []),
            created_at=row.get("created_at"),
            superseded_by=row.get("superseded_by"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecallRecord:
    """Formatted recall result with ranking and optional entity metadata."""

    conclusion_id: int
    entity_id: str
    entity_name: Optional[str]
    conclusion_type: str
    content: str
    confidence: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityDetailRecord:
    """Expanded entity response including recent conclusions when requested."""

    entity: EntityRecord
    conclusions: List[ConclusionRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity.to_dict(),
            "conclusions": [item.to_dict() for item in self.conclusions],
        }
