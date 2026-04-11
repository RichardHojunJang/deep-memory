# Deep Memory

Deep Memory is a reasoning-oriented memory engine for Hermes Agent. It works **today** as a Hermes toolset/plugin package and is now structured so the same core can later be wired into more native backend/provider hooks when Hermes exposes them.

## What it does

Deep Memory adds four practical capabilities on top of Hermes conversations:

- **Structured memory extraction** from completed sessions
- **Entity-centric profiles** that accumulate durable context over time
- **Recall tools** for semantic/keyword lookup over stored conclusions
- **A thin integration layer** that keeps Hermes-specific wiring separate from the core engine

## Architecture progression

The package is organized around a simple progression:

### 1) Core engine

The core engine is the host-agnostic part of the package.

```text
src/deep_memory/store/       SQLite schema, persistence, search
src/deep_memory/reasoning/   extraction + consolidation pipeline
src/deep_memory/tools/       current callable tools built on the engine
```

This layer owns the actual memory behavior: entities, conclusions, summaries, reasoning prompts, and recall/search.

### 2) Adapters

The new adapter layer is intentionally thin.

```text
src/deep_memory/adapters/
├── hermes_tools.py   # bridge to Hermes tool registration
└── hermes_plugin.py  # bridge to prompt-context / session-hook style integration
```

The adapters do **not** reimplement the engine. They translate between:

- Deep Memory's existing tool/runtime surface
- Hermes's current plugin/toolset integration points
- Future backend/provider service APIs when those become available

If a dedicated service layer is added later, the adapters are the place to plug it in without forcing another rewrite of the store/reasoning code.

### 3) Current Hermes usage: toolset first

Deep Memory is usable right now as a Hermes extension in two ways:

- **Toolset registration** for `recall`, `learn`, and `entities`
- **Plugin/session-hook style integration** for post-session processing and prompt-context injection

In other words: today's integration story is **toolset/plugin first**, not a tightly embedded backend hook.

### 4) Future backend/provider integration

Longer term, Hermes may expose richer backend/provider hooks for memory-aware prompting, native session events, or direct service registration. This repository is now documented and scaffolded so those hooks can be adopted by extending the adapter layer rather than rewriting the engine.

That future integration is **not** required for Deep Memory to be useful today.

## Repository layout

```text
src/deep_memory/
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── hermes_plugin.py
│   └── hermes_tools.py
├── embedding.py
├── hermes_integration.py     # existing direct Hermes bridge
├── reasoning/
├── session_hook.py           # existing session-hook implementation
├── store/
└── tools/
```

## Current integration contract

### Hermes tool registration

The adapter contract for Hermes-facing tools is intentionally small:

- a tool name
- the tool schema
- a callable handler
- an availability check

The current adapter exposes three tools:

- `recall`
- `learn`
- `entities`

Today those handlers delegate to `src/deep_memory/tools/*`. Later they may delegate to a service API instead.

### Prompt/session integration

The plugin adapter offers thin wrappers for:

- building prompt context for an entity/session
- forwarding completed Hermes sessions into background reasoning
- exposing a small plugin object for session-end hooks

These wrappers are defensive on purpose so the package can remain importable even on branches where some integration pieces are incomplete or moving.

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

## Usage today in Hermes

Once registered with Hermes, the package provides:

- **`recall`** — search stored conclusions and summaries
- **`learn`** — persist a new insight for an entity
- **`entities`** — inspect or update entity cards

Post-session extraction can also be triggered through the plugin/session-hook path so completed conversations are turned into durable memory.

## Development notes

- Python compatibility target for this package is **3.9+**
- The adapter layer is intentionally lightweight and defensive
- Documentation should stay honest: current value comes from Hermes toolset/plugin integration; deeper native backend hooks are future work

## Tests

The lightweight architecture tests in this branch focus on adapter contracts rather than end-to-end Hermes embedding:

```bash
pytest tests/test_service_api.py tests/test_adapter_layer.py -q
```

## License

AGPL-3.0.
