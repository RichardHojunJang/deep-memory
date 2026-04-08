"""Embedding generation for semantic search.

Supports multiple backends:
- 'local': sentence-transformers (if installed) — no API calls
- 'openai': OpenAI embeddings API
- 'none': skip embeddings, FTS-only search

Defaults to 'none' until explicitly configured.
"""

from __future__ import annotations

import json
import os
import struct
from typing import List, Optional


def _float_list_to_blob(floats: List[float]) -> bytes:
    """Pack a list of floats into a binary blob for sqlite-vec."""
    return struct.pack(f"{len(floats)}f", *floats)


def _blob_to_float_list(blob: bytes) -> List[float]:
    """Unpack a binary blob back to a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class Embedder:
    """Abstract base for embedding providers."""

    dimension: int = 384

    def embed(self, text: str) -> bytes:
        """Return embedding as packed float32 bytes for sqlite-vec."""
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[bytes]:
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
    """Uses OpenAI's embedding API."""

    dimension = 1536  # text-embedding-3-small default

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import urllib.request

        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"input": texts, "model": self._model}).encode(),
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


def get_embedder(backend: str = "none", **kwargs) -> Embedder:
    """Factory function to create an embedder.

    Args:
        backend: 'none', 'local', or 'openai'
        **kwargs: passed to the embedder constructor

    Returns:
        An Embedder instance
    """
    if backend == "none":
        return NoopEmbedder()
    elif backend == "local":
        return LocalEmbedder(**kwargs)
    elif backend == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
