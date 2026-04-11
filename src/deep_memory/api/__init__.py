"""Public, Hermes-independent API surface for Deep Memory.

Adapters should prefer these contracts and services instead of importing tool
modules directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

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
from deep_memory.api.service import DeepMemoryService

PathLike = Union[str, Path]


def create_service(db_path: Optional[PathLike] = None) -> DeepMemoryService:
    """Create a Deep Memory service instance for adapters or host runtimes."""
    return DeepMemoryService(db_path=db_path)


def get_service(db_path: Optional[PathLike] = None) -> DeepMemoryService:
    """Alias kept for adapter discovery ergonomics."""
    return create_service(db_path=db_path)


__all__ = [
    "ConclusionRecord",
    "DeepMemoryService",
    "EntityDetailRecord",
    "EntityQuery",
    "EntityRecord",
    "EntityUpdate",
    "EntityUpsert",
    "LearnRequest",
    "RecallRecord",
    "RecallRequest",
    "VALID_CONCLUSION_TYPES",
    "create_service",
    "get_service",
]
