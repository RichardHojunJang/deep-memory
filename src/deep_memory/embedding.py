"""Embedding generation for semantic search.

Supports multiple backends:
- 'local': sentence-transformers (if installed) — no API calls
- 'openai': OpenAI embeddings API
- 'none': skip embeddings, FTS-only search
- 'auto': auto-detect best available backend

Defaults to 'auto' — picks the best available backend automatically.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for an embedding backend."""

    backend: str  # "auto", "none", "local", "openai"
    model: Optional[str] = None
    dimension: Optional[int] = None
    api_key: Optional[str] = None


def _float_list_to_blob(floats: List[float]) -> bytes:
    """Pack a list of floats into a binary blob for sqlite-vec."""
    return struct.pack(f"{len(floats)}f", *floats)


def _blob_to_float_list(blob: bytes) -> List[float]:
    """Unpack a binary blob back to a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _check_sentence_transformers() -> bool:
    """Check if sentence-transformers is importable."""
    try:
        import importlib

        importlib.import_module("sentence_transformers")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _check_openai_key() -> bool:
    """Check if OPENAI_API_KEY environment variable is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def _check_sqlite_vec() -> bool:
    """Check if sqlite-vec is available."""
    try:
        import sqlite_vec  # noqa: F401

        return True
    except ImportError:
        return False


def _read_hermes_config_backend() -> Optional[str]:
    """Read embedding backend from Hermes config.yaml, if available."""
    try:
        from pathlib import Path

        import yaml

        config_path = (
            Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "config.yaml"
        )
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            backend = cfg.get("deep_memory", {}).get("embedding_backend")
            if backend:
                return str(backend)
    except Exception:
        pass
    return None


def auto_detect_backend() -> EmbeddingConfig:
    """Auto-detect the best available embedding backend.

    Priority:
    1. Hermes config.yaml deep_memory.embedding_backend → use that
    2. sentence-transformers importable → local (all-MiniLM-L6-v2, dim=384)
    3. OPENAI_API_KEY set → openai (text-embedding-3-small, dim=1536)
    4. Otherwise → none (FTS-only)
    """
    # 1. Check explicit config
    configured = _read_hermes_config_backend()
    if configured and configured != "auto":
        logger.info("Embedding backend from config: %s", configured)
        if configured == "local":
            return EmbeddingConfig(
                backend="local", model="all-MiniLM-L6-v2", dimension=384
            )
        elif configured == "openai":
            return EmbeddingConfig(
                backend="openai",
                model="text-embedding-3-small",
                dimension=1536,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        elif configured == "none":
            return EmbeddingConfig(backend="none")
        else:
            logger.warning("Unknown configured backend '%s', falling back", configured)

    # 2. Check sentence-transformers
    if _check_sentence_transformers():
        logger.info("Auto-detected local embedding backend (sentence-transformers)")
        return EmbeddingConfig(
            backend="local", model="all-MiniLM-L6-v2", dimension=384
        )

    # 3. Check OpenAI key
    if _check_openai_key():
        logger.info("Auto-detected OpenAI embedding backend")
        return EmbeddingConfig(
            backend="openai",
            model="text-embedding-3-small",
            dimension=1536,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    # 4. Fallback
    logger.info("No embedding backend available, using FTS-only mode")
    return EmbeddingConfig(backend="none")


def diagnose() -> Dict[str, object]:
    """Check which embedding backends are available.

    Returns a dict with availability info for each backend.
    """
    result: Dict[str, object] = {
        "sentence_transformers_available": _check_sentence_transformers(),
        "openai_api_key_set": _check_openai_key(),
        "sqlite_vec_available": _check_sqlite_vec(),
        "configured_backend": _read_hermes_config_backend(),
    }
    auto = auto_detect_backend()
    result["auto_detected"] = auto.backend
    return result


class Embedder:
    """Abstract base for embedding providers."""

    dimension: int = 384

    def embed(self, text: str) -> Optional[bytes]:
        """Return embedding as packed float32 bytes for sqlite-vec."""
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[Optional[bytes]]:
        """Embed multiple texts. Default: sequential single calls."""
        return [self.embed(t) for t in texts]


class NoopEmbedder(Embedder):
    """No-op embedder that returns None. FTS-only mode."""

    def embed(self, text: str) -> Optional[bytes]:
        return None


class LocalEmbedder(Embedder):
    """Uses sentence-transformers for local embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> bytes:
        vec = self._model.encode(text, normalize_embeddings=True)
        return _float_list_to_blob(vec.tolist())

    def embed_batch(self, texts: List[str]) -> List[bytes]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [_float_list_to_blob(v.tolist()) for v in vecs]


class OpenAIEmbedder(Embedder):
    """Uses OpenAI's embedding API.

    Supports text-embedding-3-small with configurable dimensions.
    OpenAI's API accepts a ``dimensions`` parameter to truncate output.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.dimension = dimension

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import urllib.request

        body: Dict[str, object] = {
            "input": texts,
            "model": self._model,
        }
        # Only include dimensions param for models that support it
        if self._model.startswith("text-embedding-3"):
            body["dimensions"] = self.dimension

        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        return [item["embedding"] for item in result["data"]]

    def embed(self, text: str) -> bytes:
        vecs = self._call_api([text])
        return _float_list_to_blob(vecs[0])

    def embed_batch(self, texts: List[str]) -> List[bytes]:
        vecs = self._call_api(texts)
        return [_float_list_to_blob(v) for v in vecs]


def get_embedder(backend: str = "auto", **kwargs) -> Embedder:
    """Factory function to create an embedder.

    Args:
        backend: 'auto', 'none', 'local', or 'openai'
        **kwargs: passed to the embedder constructor

    Returns:
        An Embedder instance
    """
    if backend == "auto":
        config = auto_detect_backend()
        backend = config.backend
        # Pass auto-detected settings as defaults (kwargs override)
        if config.model and "model" not in kwargs and "model_name" not in kwargs:
            if backend == "local":
                kwargs.setdefault("model_name", config.model)
            elif backend == "openai":
                kwargs.setdefault("model", config.model)
        if config.dimension and "dimension" not in kwargs:
            kwargs.setdefault("dimension", config.dimension)
        if config.api_key and "api_key" not in kwargs:
            kwargs.setdefault("api_key", config.api_key)

    if backend == "none":
        return NoopEmbedder()
    elif backend == "local":
        # LocalEmbedder doesn't accept 'dimension' kwarg
        kwargs.pop("dimension", None)
        return LocalEmbedder(**kwargs)
    elif backend == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
