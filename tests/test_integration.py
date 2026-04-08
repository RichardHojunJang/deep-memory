"""Tests for Hermes integration, embedding, and session hooks."""

import json
import struct
import pytest

from deep_memory.store.db import DeepMemoryDB
from deep_memory.embedding import (
    NoopEmbedder,
    _float_list_to_blob,
    _blob_to_float_list,
    get_embedder,
)
from deep_memory.session_hook import _format_messages_as_transcript, process_session_async


# ── Embedding tests ──────────────────────────────────────────

class TestEmbedding:
    def test_float_roundtrip(self):
        original = [0.1, 0.2, 0.3, 0.4]
        blob = _float_list_to_blob(original)
        result = _blob_to_float_list(blob)
        assert len(result) == 4
        assert abs(result[0] - 0.1) < 1e-6

    def test_noop_embedder(self):
        embedder = NoopEmbedder()
        assert embedder.embed("hello") is None

    def test_get_embedder_none(self):
        embedder = get_embedder("none")
        assert isinstance(embedder, NoopEmbedder)

    def test_get_embedder_unknown(self):
        with pytest.raises(ValueError):
            get_embedder("unknown_backend")


# ── Transcript formatting ────────────────────────────────────

class TestTranscriptFormatting:
    def test_basic_formatting(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        transcript = _format_messages_as_transcript(messages)
        assert "User: Hello" in transcript
        assert "Assistant: Hi there!" in transcript

    def test_skips_system_and_tool(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": '{"result": true}'},
            {"role": "assistant", "content": "Done!"},
        ]
        transcript = _format_messages_as_transcript(messages)
        assert "system" not in transcript.lower()
        assert "tool" not in transcript.lower()
        assert "User: Hello" in transcript

    def test_truncates_long_messages(self):
        messages = [
            {"role": "user", "content": "x" * 5000},
        ]
        transcript = _format_messages_as_transcript(messages)
        assert "[truncated]" in transcript
        assert len(transcript) < 5000

    def test_empty_messages(self):
        assert _format_messages_as_transcript([]) == ""


# ── Session hook ─────────────────────────────────────────────

class TestSessionHook:
    def test_skips_short_sessions(self):
        # Should not crash on very short sessions
        process_session_async("test-short", [{"role": "user", "content": "hi"}])
        # No exception = pass (it's async/fire-and-forget)

    def test_skips_empty_sessions(self):
        process_session_async("test-empty", [])


# ── Integration context building ─────────────────────────────

class TestContextBuilding:
    def test_build_context_with_entity(self, tmp_path):
        from unittest.mock import patch
        from deep_memory.hermes_integration import build_deep_memory_context

        db = DeepMemoryDB(db_path=tmp_path / "test.db")
        db.upsert_entity("richard", "Richard", card={
            "name": "Richard",
            "role": "Developer",
            "preferences": ["self-hosted", "cost-effective"],
        })
        db.add_conclusion("richard", "explicit", "Richard prefers SQLite over PostgreSQL", confidence=1.0)
        db.add_conclusion("richard", "inductive", "Richard values local-first tooling", confidence=0.85)

        with patch("deep_memory.store.db.DEFAULT_DB_PATH", tmp_path / "test.db"):
            context = build_deep_memory_context("richard")

        assert "DEEP MEMORY" in context
        assert "Richard" in context
        assert "Developer" in context
        assert "SQLite" in context
        db.close()

    def test_build_context_empty(self, tmp_path):
        from unittest.mock import patch
        from deep_memory.hermes_integration import build_deep_memory_context

        with patch("deep_memory.store.db.DEFAULT_DB_PATH", tmp_path / "empty.db"):
            context = build_deep_memory_context("nobody")

        assert context == ""
