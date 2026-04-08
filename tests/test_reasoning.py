"""Tests for the reasoning pipeline."""

import json
import pytest

from deep_memory.store.db import DeepMemoryDB
from deep_memory.reasoning.extractor import (
    _parse_json_response,
    extract_session_summary,
    extract_entities,
    extract_reasoning,
    process_session,
)
from deep_memory.reasoning.consolidator import consolidate_entity


# ── Mock LLM responses ──────────────────────────────────────

MOCK_SUMMARY_RESPONSE = json.dumps({
    "short_summary": "Discussed building a memory system for Hermes Agent.",
    "key_decisions": ["Use SQLite instead of PostgreSQL", "Build as native toolset"],
    "entities_mentioned": ["Richard", "Hermes Agent", "Honcho"],
})

MOCK_ENTITIES_RESPONSE = json.dumps([
    {"name": "Richard", "type": "person", "relevance": "Project lead"},
    {"name": "Hermes Agent", "type": "project", "relevance": "Target application"},
    {"name": "Honcho", "type": "concept", "relevance": "Inspiration for the design"},
])

MOCK_REASONING_RESPONSE = json.dumps({
    "explicit": [
        {"content": "Richard wants memory features built into Hermes", "confidence": 1.0},
        {"content": "Richard finds Honcho's pricing too expensive", "confidence": 0.95},
    ],
    "deductive": [
        {
            "premises": [
                "Richard wants the system embedded in Hermes",
                "Hermes uses SQLite for session storage",
            ],
            "conclusion": "SQLite is the natural choice for the memory store",
            "confidence": 0.9,
        }
    ],
    "inductive": [
        {
            "observations": [
                "Prefers local over hosted solutions",
                "Concerned about pricing of external services",
            ],
            "pattern": "Richard values self-hosted, cost-effective tooling",
            "confidence": 0.8,
        }
    ],
    "abductive": [],
    "contradictions": [],
    "card_updates": {
        "name": "Richard",
        "role": "Developer",
        "preferences": ["self-hosted solutions", "cost-effective tooling"],
    },
})

MOCK_CONSOLIDATION_RESPONSE = json.dumps({
    "redundant_pairs": [
        {"ids": [1, 2], "keep": 2, "reason": "More specific version of the same fact"}
    ],
    "contradictions": [],
    "consolidated": [],
})


SAMPLE_TRANSCRIPT = """
User: honcho와 비슷한 오픈소스 솔루션 있어?
Assistant: Honcho 문서를 분석해봤어. 핵심은 formal logic reasoning...
User: hermes에 memory 관련 기능을 강화하고 싶어. honcho는 유료라 부담이 되네.
Assistant: Hermes에 직접 내장하는 형태로 설계해볼게. Deep Memory 툴셋 제안...
User: 완성까지는 얼마나 걸릴까?
Assistant: 현실적으로 2-3주 예상.
"""


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    d = DeepMemoryDB(db_path=tmp_path / "test.db")
    yield d
    d.close()


# ── JSON parsing tests ───────────────────────────────────────

class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_json_array(self):
        result = _parse_json_response('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _parse_json_response("not json at all")


# ── Extractor tests ──────────────────────────────────────────

class TestExtractSessionSummary:
    def test_summary_extraction(self):
        def mock_llm(prompt):
            return MOCK_SUMMARY_RESPONSE

        result = extract_session_summary(SAMPLE_TRANSCRIPT, mock_llm)
        assert "short_summary" in result
        assert "memory" in result["short_summary"].lower()
        assert len(result["key_decisions"]) == 2

    def test_summary_with_markdown_wrapped_response(self):
        def mock_llm(prompt):
            return f"```json\n{MOCK_SUMMARY_RESPONSE}\n```"

        result = extract_session_summary(SAMPLE_TRANSCRIPT, mock_llm)
        assert "short_summary" in result


class TestExtractEntities:
    def test_entity_extraction(self):
        def mock_llm(prompt):
            return MOCK_ENTITIES_RESPONSE

        result = extract_entities(SAMPLE_TRANSCRIPT, mock_llm)
        assert len(result) == 3
        names = [e["name"] for e in result]
        assert "Richard" in names

    def test_empty_extraction(self):
        def mock_llm(prompt):
            return "[]"

        result = extract_entities("Hello world", mock_llm)
        assert result == []


class TestExtractReasoning:
    def test_reasoning_extraction(self):
        def mock_llm(prompt):
            return MOCK_REASONING_RESPONSE

        result = extract_reasoning(
            "Richard", SAMPLE_TRANSCRIPT, [], mock_llm
        )
        assert len(result["explicit"]) == 2
        assert len(result["deductive"]) == 1
        assert len(result["inductive"]) == 1
        assert result["card_updates"]["name"] == "Richard"

    def test_with_existing_conclusions(self):
        existing = [
            {"type": "explicit", "content": "Richard is a developer", "confidence": 1.0}
        ]

        def mock_llm(prompt):
            assert "Richard is a developer" in prompt
            return MOCK_REASONING_RESPONSE

        result = extract_reasoning("Richard", SAMPLE_TRANSCRIPT, existing, mock_llm)
        assert "explicit" in result


# ── Full pipeline test ───────────────────────────────────────

class TestProcessSession:
    def test_full_pipeline(self, db):
        call_count = {"n": 0}

        def mock_llm(prompt):
            call_count["n"] += 1
            if "Summarize" in prompt:
                return MOCK_SUMMARY_RESPONSE
            elif "identify all entities" in prompt:
                return MOCK_ENTITIES_RESPONSE
            else:
                return MOCK_REASONING_RESPONSE

        stats = process_session(
            session_id="test-session-1",
            transcript=SAMPLE_TRANSCRIPT,
            llm_call=mock_llm,
            db=db,
        )

        assert stats["entities_found"] == 3
        assert stats["conclusions_added"] > 0

        # Verify data was stored
        summaries = db.get_summaries(session_id="test-session-1")
        assert len(summaries) == 1

        richard = db.get_entity("richard")
        assert richard is not None

        conclusions = db.get_conclusions(entity_id="richard")
        assert len(conclusions) > 0

    def test_empty_transcript(self, db):
        def mock_llm(prompt):
            if "Summarize" in prompt:
                return json.dumps({
                    "short_summary": "Empty session",
                    "key_decisions": [],
                    "entities_mentioned": [],
                })
            return "[]"

        stats = process_session("empty-session", "", mock_llm, db=db)
        assert stats["entities_found"] == 0
        assert stats["conclusions_added"] == 0


# ── Consolidation tests ──────────────────────────────────────

class TestConsolidation:
    def test_consolidate_with_redundancy(self, db):
        db.upsert_entity("u1", "Alice")
        for i in range(6):
            db.add_conclusion("u1", "explicit", f"Fact about Alice #{i}")

        def mock_llm(prompt):
            return MOCK_CONSOLIDATION_RESPONSE

        stats = consolidate_entity("u1", mock_llm, db=db, min_conclusions=5)
        assert stats["redundant_removed"] == 1

    def test_skip_if_too_few(self, db):
        db.upsert_entity("u1", "Alice")
        db.add_conclusion("u1", "explicit", "Only one fact")

        def mock_llm(prompt):
            raise AssertionError("Should not be called")

        stats = consolidate_entity("u1", mock_llm, db=db, min_conclusions=5)
        assert stats["redundant_removed"] == 0

    def test_nonexistent_entity(self, db):
        def mock_llm(prompt):
            raise AssertionError("Should not be called")

        stats = consolidate_entity("nobody", mock_llm, db=db)
        assert stats["entity_id"] == "nobody"
