"""Tests for Phase 6: Embedding model auto-config."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch, MagicMock

import pytest

from deep_memory.embedding import (
    EmbeddingConfig,
    auto_detect_backend,
    diagnose,
    get_embedder,
    NoopEmbedder,
)


# ── EmbeddingConfig dataclass ────────────────────────────────

class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig(backend="none")
        assert cfg.backend == "none"
        assert cfg.model is None
        assert cfg.dimension is None
        assert cfg.api_key is None

    def test_all_fields(self):
        cfg = EmbeddingConfig(
            backend="openai",
            model="text-embedding-3-small",
            dimension=1536,
            api_key="sk-test",
        )
        assert cfg.backend == "openai"
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimension == 1536
        assert cfg.api_key == "sk-test"

    def test_local_config(self):
        cfg = EmbeddingConfig(backend="local", model="all-MiniLM-L6-v2", dimension=384)
        assert cfg.backend == "local"
        assert cfg.dimension == 384


# ── auto_detect_backend ──────────────────────────────────────

class TestAutoDetectBackend:
    @patch("deep_memory.embedding._read_hermes_config_backend", return_value="local")
    def test_explicit_config_local(self, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "local"
        assert result.model == "all-MiniLM-L6-v2"
        assert result.dimension == 384

    @patch("deep_memory.embedding._read_hermes_config_backend", return_value="openai")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_explicit_config_openai(self, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "openai"
        assert result.model == "text-embedding-3-small"
        assert result.dimension == 1536

    @patch("deep_memory.embedding._read_hermes_config_backend", return_value="none")
    def test_explicit_config_none(self, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "none"

    @patch("deep_memory.embedding._read_hermes_config_backend", return_value=None)
    @patch("deep_memory.embedding._check_sentence_transformers", return_value=True)
    def test_auto_picks_local(self, mock_st, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "local"
        assert result.dimension == 384

    @patch("deep_memory.embedding._read_hermes_config_backend", return_value=None)
    @patch("deep_memory.embedding._check_sentence_transformers", return_value=False)
    @patch("deep_memory.embedding._check_openai_key", return_value=True)
    def test_auto_picks_openai(self, mock_oai, mock_st, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "openai"
        assert result.dimension == 1536

    @patch("deep_memory.embedding._read_hermes_config_backend", return_value=None)
    @patch("deep_memory.embedding._check_sentence_transformers", return_value=False)
    @patch("deep_memory.embedding._check_openai_key", return_value=False)
    def test_auto_falls_to_none(self, mock_oai, mock_st, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "none"

    @patch("deep_memory.embedding._read_hermes_config_backend", return_value="bogus")
    @patch("deep_memory.embedding._check_sentence_transformers", return_value=False)
    @patch("deep_memory.embedding._check_openai_key", return_value=False)
    def test_unknown_config_falls_through(self, mock_oai, mock_st, mock_cfg):
        result = auto_detect_backend()
        assert result.backend == "none"


# ── diagnose ─────────────────────────────────────────────────

class TestDiagnose:
    @patch("deep_memory.embedding._check_sentence_transformers", return_value=False)
    @patch("deep_memory.embedding._check_openai_key", return_value=False)
    @patch("deep_memory.embedding._check_sqlite_vec", return_value=False)
    @patch("deep_memory.embedding._read_hermes_config_backend", return_value=None)
    def test_diagnose_keys(self, mock_cfg, mock_vec, mock_oai, mock_st):
        result = diagnose()
        assert "sentence_transformers_available" in result
        assert "openai_api_key_set" in result
        assert "sqlite_vec_available" in result
        assert "configured_backend" in result
        assert "auto_detected" in result
        assert result["auto_detected"] == "none"

    @patch("deep_memory.embedding._check_sentence_transformers", return_value=True)
    @patch("deep_memory.embedding._check_openai_key", return_value=True)
    @patch("deep_memory.embedding._check_sqlite_vec", return_value=True)
    @patch("deep_memory.embedding._read_hermes_config_backend", return_value=None)
    def test_diagnose_all_available(self, mock_cfg, mock_vec, mock_oai, mock_st):
        result = diagnose()
        assert result["sentence_transformers_available"] is True
        assert result["openai_api_key_set"] is True
        assert result["sqlite_vec_available"] is True
        # local takes priority over openai
        assert result["auto_detected"] == "local"


# ── get_embedder("auto") ─────────────────────────────────────

class TestGetEmbedderAuto:
    @patch("deep_memory.embedding._read_hermes_config_backend", return_value=None)
    @patch("deep_memory.embedding._check_sentence_transformers", return_value=False)
    @patch("deep_memory.embedding._check_openai_key", return_value=False)
    def test_auto_returns_noop(self, mock_oai, mock_st, mock_cfg):
        embedder = get_embedder("auto")
        assert isinstance(embedder, NoopEmbedder)

    def test_none_returns_noop(self):
        embedder = get_embedder("none")
        assert isinstance(embedder, NoopEmbedder)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_embedder("imaginary")


# ── Dimension consistency ─────────────────────────────────────

class TestDimensionConsistency:
    def test_schema_accepts_custom_dimension(self):
        """init_schema should accept embedding_dim parameter."""
        from deep_memory.store.schema import init_schema

        conn = sqlite3.connect(":memory:")
        # Just verify it doesn't error with a custom dimension
        init_schema(conn, embedding_dim=768)
        conn.close()

    def test_noop_embedder_dimension(self):
        embedder = NoopEmbedder()
        assert embedder.dimension == 384

    def test_default_get_embedder_is_auto(self):
        """get_embedder default should be 'auto', not 'none'."""
        import inspect

        sig = inspect.signature(get_embedder)
        assert sig.parameters["backend"].default == "auto"
