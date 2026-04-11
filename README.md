# Deep Memory

Deep Memory is a reasoning-oriented memory engine for Hermes Agent. It works **today** as a Hermes toolset/plugin package and is now structured so the same core can later be wired into more native backend/provider hooks when Hermes exposes them.

## What it does

Deep Memory adds four practical capabilities on top of Hermes conversations:

- **Structured memory extraction** from completed sessions
- **Entity-centric profiles** that accumulate durable context over time
- **Recall tools** for semantic/keyword lookup over stored conclusions
- **A thin integration layer** that keeps Hermes-specific wiring separate from the core engine

## Architecture progression

The package is organized around a simple progression.

### 1) Core engine

The core engine is the host-agnostic part of the package.

```text
src/deep_memory/store/       SQLite schema, persistence, search
src/deep_memory/reasoning/   extraction + consolidation pipeline
src/deep_memory/api/         stable contracts + service facade
src/deep_memory/tools/       compatibility wrappers built on the service layer
```

This layer owns the actual memory behavior: entities, conclusions, summaries, reasoning prompts, and recall/search.

### 2) Adapters

The adapter layer is intentionally thin.

```text
src/deep_memory/adapters/
├── hermes_tools.py   # bridge to Hermes tool registration
└── hermes_plugin.py  # bridge to prompt-context / session-hook style integration
```

Adapters do **not** reimplement the engine. They translate between:

- Deep Memory's stable service/contracts
- Hermes's current plugin/toolset integration points
- Future backend/provider service hooks when those become available

### 3) Current Hermes usage: toolset first

Deep Memory is usable right now as a Hermes extension in two ways:

- **Toolset registration** for `recall`, `learn`, and `entities`
- **Plugin/session-hook style integration** for post-session processing and prompt-context injection

So today's integration story is **toolset/plugin first**, not a tightly embedded backend hook.

### 4) Future backend/provider integration

Longer term, Hermes may expose richer backend/provider hooks for memory-aware prompting, native session events, or direct service registration. This repository is now structured so those hooks can be adopted by extending the adapter layer rather than rewriting the engine.

That future integration is **not** required for Deep Memory to be useful today.

## Repository layout

```text
src/deep_memory/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── hermes_plugin.py
│   └── hermes_tools.py
├── api/
│   ├── __init__.py
│   ├── contracts.py
│   └── service.py
├── embedding.py
├── hermes_integration.py     # current direct Hermes bridge
├── reasoning/
├── runtime.py                # host-agnostic path resolution
├── session_hook.py           # current session-hook implementation
├── store/
└── tools/
```

## Installation

### Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Install into a Hermes environment

```bash
pip install -e /path/to/deep-memory
```

After installation, Hermes can expose Deep Memory through the current toolset/plugin integration points.

### Optional: embedding backends

```bash
# Local embeddings (no API calls)
pip install sentence-transformers

# Or use OpenAI embeddings (requires OPENAI_API_KEY)
# Configure in ~/.hermes/config.yaml:
#   deep_memory:
#     embedding_backend: openai
```

## Usage in Hermes today

Once registered with Hermes, the package provides:

- **`recall`** — search stored conclusions and summaries
- **`learn`** — persist a new insight for an entity
- **`entities`** — inspect or update entity cards

Post-session extraction can also be triggered through the plugin/session-hook path so completed conversations are turned into durable memory.

### Adapter API examples

The current public adapter surface is deliberately small and backend-ready.

**Tool registration adapter**

```python
from deep_memory.adapters import register_with_registry

# Hermes-like registry object
register_with_registry(registry)
```

**Prompt/session plugin adapter**

```python
from deep_memory.adapters import DeepMemorySessionPlugin, build_prompt_context

plugin = DeepMemorySessionPlugin()
context_block = build_prompt_context(entity_id="richard")
```

Those helpers keep today's Hermes integration thin while giving future provider hooks a stable seam to target.

### Embedding auto-config

Deep Memory automatically detects the best embedding backend:

1. **Explicit config** — set `deep_memory.embedding_backend` in `~/.hermes/config.yaml`
2. **Local** — if `sentence-transformers` is installed, use `all-MiniLM-L6-v2` (dim=384, no API calls)
3. **OpenAI** — if `OPENAI_API_KEY` is set, use `text-embedding-3-small` (dim=1536)
4. **FTS-only** — fall back to keyword search if no embedding backend is available

```yaml
# ~/.hermes/config.yaml (optional — auto-detection works without this)
deep_memory:
  embedding_backend: local  # or "openai" or "none"
```

To inspect what's available:

```python
from deep_memory.embedding import diagnose
print(diagnose())
```

## Development notes

- Python compatibility target is **3.9+**
- The adapter layer is intentionally lightweight and defensive
- Tool wrappers now route through `deep_memory.api.DeepMemoryService`
- Documentation should stay honest: current value comes from Hermes toolset/plugin integration; deeper native backend hooks are future work

## Tests

Run the focused architecture/service checks:

```bash
pytest tests/test_service_api.py tests/test_adapter_layer.py -q
```

Or the full suite:

```bash
pytest tests/ -q
```

## Status

- [x] Phase 1: SQLite store + recall/learn tools
- [x] Phase 2: Reasoning pipeline (extract/consolidate)
- [x] Phase 3: Entity cards + system prompt injection
- [x] Phase 4: Hermes integration (tool registry + session hook)
- [x] Phase 5: Tests + polish
- [x] Phase 6: Embedding model auto-config
- [x] Phase 7: Hermes PR submission groundwork
- [x] Phase 8: backend-ready service + adapter split

## License

AGPL-3.0.
