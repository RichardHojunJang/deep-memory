"""Public, Hermes-independent API surface for Deep Memory.

Adapters should prefer these contracts and services instead of importing tool
modules directly.
"""

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
]
