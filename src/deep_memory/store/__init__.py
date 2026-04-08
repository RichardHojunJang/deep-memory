"""Deep Memory storage layer — SQLite + FTS5 + sqlite-vec."""

from .db import init_db, get_connection
from .search import hybrid_search

__all__ = ["init_db", "get_connection", "hybrid_search"]
