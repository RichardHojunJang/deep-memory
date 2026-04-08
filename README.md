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

## Installation

### As a Hermes Agent plugin

```bash
# Install into Hermes's venv
source ~/.hermes/hermes-agent/venv/bin/activate
pip install -e /path/to/deep-memory

# Add to Hermes model_tools.py _discover_tools():
#   try:
#       import deep_memory.hermes_integration
#   except ImportError:
#       pass

# Add to Hermes toolsets.py _HERMES_CORE_TOOLS:
#   "recall", "learn", "entities",
```

### Standalone

```bash
pip install -e ".[dev]"
```

### Optional: Embedding backends

```bash
# Local embeddings (no API calls)
pip install sentence-transformers

# Or use OpenAI embeddings (requires OPENAI_API_KEY)
# Configure in ~/.hermes/config.yaml:
#   deep_memory:
#     embedding_backend: openai
```

## Usage

Once installed, three new tools are available in Hermes:

- **`recall`** — Search deep memory for insights: `recall(query="Python preferences", entity="Richard")`
- **`learn`** — Store insights: `learn(entity="Richard", insight="Prefers SQLite", type="explicit")`  
- **`entities`** — Manage profiles: `entities(action="get", name="Richard")`

Post-session reasoning runs automatically via `session_hook.py`, extracting structured conclusions from conversations.

### Embedding Auto-Config

Deep Memory automatically detects the best embedding backend:

1. **Explicit config** — Set `deep_memory.embedding_backend` in `~/.hermes/config.yaml`
2. **Local** — If `sentence-transformers` is installed, uses `all-MiniLM-L6-v2` (dim=384, no API calls)
3. **OpenAI** — If `OPENAI_API_KEY` is set, uses `text-embedding-3-small` (dim=1536)
4. **FTS-only** — Falls back to keyword search if no embedding backend is available

```yaml
# ~/.hermes/config.yaml (optional — auto-detection works without this)
deep_memory:
  embedding_backend: local  # or "openai" or "none"
```

To check what's available:

```python
from deep_memory.embedding import diagnose
print(diagnose())
# {'sentence_transformers_available': True, 'openai_api_key_set': False,
#  'sqlite_vec_available': True, 'configured_backend': None, 'auto_detected': 'local'}
```

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q   # 60 tests
ruff check src/
```

## Status

- [x] Phase 1: SQLite store + recall/learn tools
- [x] Phase 2: Reasoning pipeline (extract/consolidate)
- [x] Phase 3: Entity cards + system prompt injection
- [x] Phase 4: Hermes integration (tool registry + session hook)
- [x] Phase 5: Tests + polish (60 tests passing)
- [x] Phase 6: Embedding model auto-config
- [x] Phase 7: Hermes PR submission

## License

AGPL-3.0 — same as Hermes Agent.
