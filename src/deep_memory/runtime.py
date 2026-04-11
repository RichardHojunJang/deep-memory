"""Runtime path helpers for Hermes-independent Deep Memory wiring.

These helpers avoid direct Hermes imports while still honoring the same home
and database layout expected by existing adapters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Union


DEFAULT_HERMES_HOME = Path.home() / ".hermes"
DEFAULT_DEEP_MEMORY_DB_NAME = "memory.db"
DEFAULT_DEEP_MEMORY_DIRNAME = "deep_memory"


PathLike = Union[str, Path]


def resolve_hermes_home(env: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve the Hermes home directory from environment without Hermes imports."""
    env = env or os.environ
    configured = (env.get("HERMES_HOME") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_HERMES_HOME.expanduser()


def resolve_deep_memory_home(
    home: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the directory used for Deep Memory state."""
    base_home = Path(home).expanduser() if home is not None else resolve_hermes_home(env)
    return base_home / DEFAULT_DEEP_MEMORY_DIRNAME


def resolve_deep_memory_db_path(
    db_path: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Path:
    """Resolve the Deep Memory SQLite path, preferring explicit configuration."""
    if db_path is not None:
        return Path(db_path).expanduser()

    env = env or os.environ
    configured = (env.get("DEEP_MEMORY_DB_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser()

    return resolve_deep_memory_home(env=env) / DEFAULT_DEEP_MEMORY_DB_NAME
