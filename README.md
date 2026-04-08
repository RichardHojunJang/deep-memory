# Deep Memory

Honcho-inspired reasoning memory system for [Hermes Agent](https://github.com/plastic-labs/hermes-agent). Built as a native toolset — no external hosting required.

## Overview

Deep Memory enhances Hermes's memory capabilities by adding:
- **Structured reasoning** over conversations (explicit → deductive → inductive)
- **Entity tracking** with evolving profiles (peer cards)
- **Semantic + keyword hybrid search** via SQLite + FTS5 + sqlite-vec
- **Automatic post-session reasoning** that extracts insights without manual intervention

## Architecture

```
~/.hermes/deep_memory/
└── memory.db              # SQLite (FTS5 + sqlite-vec) — single file, zero hosting

src/deep_memory/
├── store/                 # Data layer (SQLite, embeddings, search)
│   ├── schema.py          # Table definitions + migrations
│   ├── db.py              # Connection management
│   └── search.py          # Hybrid semantic + FTS5 search
├── reasoning/             # LLM reasoning pipeline
│   ├── extractor.py       # Post-session insight extraction
│   ├── consolidator.py    # Contradiction detection + merging
│   └── prompts.py         # Reasoning prompt templates
├── tools/                 # Hermes tool definitions
│   ├── recall.py          # Semantic recall tool
│   ├── learn.py           # Manual insight storage
│   └── entities.py        # Entity profile management
└── __init__.py            # Toolset registration for Hermes
```

## Data Model

| Table | Purpose | Honcho Equivalent |
|-------|---------|-------------------|
| `entities` | People, projects, concepts that persist over time | Peers |
| `conclusions` | Structured insights with embeddings | Representations |
| `summaries` | Compressed session digests | Session summaries |

## Development

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Tests
pytest tests/ -q

# Lint
ruff check src/
```

## Roadmap

- [ ] Phase 1: SQLite store + recall/learn tools
- [ ] Phase 2: Post-session reasoning hook
- [ ] Phase 3: Entity cards + system prompt injection
- [ ] Phase 4: Contradiction detection + conclusion consolidation
- [ ] Phase 5: Tests + polish + Hermes integration

## License

AGPL-3.0 — same as Hermes Agent.
