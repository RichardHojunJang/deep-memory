# Hermes Minimal Hook Plan for Deep Memory

> **For Hermes:** Use this as the implementation/design note for making Deep Memory a first-class external memory provider without hardwiring Deep Memory into Hermes core.

**Goal:** Let Deep Memory run today as an external package with minimal Hermes changes, then allow tighter native integration later without rewriting either side.

**Architecture:** Hermes already has a usable memory-provider abstraction (`MemoryProvider`, `MemoryManager`, provider tool injection, prefetch, session-end hooks). The missing piece is not a whole new backend system — it is a small expansion so external packages like `deep-memory` can register as memory providers and, optionally, override session recall/memory semantics more cleanly.

**Tech Stack:** Python, Hermes memory provider plugin system, external package loading, Deep Memory service/adapters.

---

## Executive summary

The good news: Hermes is already much closer than expected.

### What Hermes already has

Hermes currently already provides:

- `agent/memory_provider.py` — provider ABC
- `agent/memory_manager.py` — orchestration layer
- `run_agent.py` wiring for:
  - provider init
  - provider tool schema injection
  - `prefetch_all()` before the turn loop
  - `sync_all()` after final response
  - `on_session_end()` at actual session boundary
  - `on_memory_write()` bridge when the built-in `memory` tool writes
- `plugins/memory/` repo-local provider loader
- built-in provider + one external provider model

That means **Deep Memory does not need a brand new hook family**.

### What is still missing

For Deep Memory to work in the cleanest way as an *external package* rather than a Hermes-in-tree plugin, Hermes still needs a few minimal changes:

1. **External memory provider discovery**
   - today memory providers are loaded only from `plugins/memory/<name>/` inside the Hermes repo
   - Deep Memory should be loadable from an installed Python package

2. **Optional session-search backend hook**
   - today `session_search` is still hardwired to the SQLite session DB in `run_agent.py`
   - Deep Memory may want to augment or replace that recall path

3. **Optional built-in memory delegation policy**
   - today `memory` remains a special agent-level tool backed by `MemoryStore`
   - Deep Memory can mirror writes today, but cannot transparently become the primary persistence backend

So the true “minimal hook plan” is:

- **Phase A:** external provider loading
- **Phase B:** optional recall/search providerization
- **Phase C:** optional memory-write backend delegation

---

## Current state in Hermes

### 1) Existing provider lifecycle is already real

Relevant files:

- `agent/memory_provider.py`
- `agent/memory_manager.py`
- `agent/builtin_memory_provider.py`
- `run_agent.py`
- `plugins/memory/__init__.py`

Important existing behaviors already present:

#### Provider init in `run_agent.py`
`run_agent.py` already does roughly this:

```python
self._memory_manager = MemoryManager()
_mp = load_memory_provider(_mem_provider_name)
if _mp and _mp.is_available():
    self._memory_manager.add_provider(_mp)
self._memory_manager.initialize_all(...)
```

#### Provider tools are already injected
Also already present:

```python
for _schema in self._memory_manager.get_all_tool_schemas():
    self.tools.append({"type": "function", "function": _schema})
```

#### Provider-prefetch is already wired
Before the turn loop:

```python
_ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
```

#### Provider session-end hook is already wired
At true session boundaries:

```python
self._memory_manager.on_session_end(messages or [])
self._memory_manager.shutdown_all()
```

#### Built-in memory writes already mirror outward
When Hermes handles the `memory` tool:

```python
self._memory_manager.on_memory_write(action, target, content)
```

So Deep Memory already has most of the lifecycle it needs.

---

## The actual gap: repo-local plugin loading

### Problem
The provider system exists, but the loader is still repo-centric.

Current loader path:
- `plugins/memory/__init__.py`
- scans only `plugins/memory/<name>/`
- imports provider modules from inside Hermes itself

That means Deep Memory cannot be shipped cleanly as:

- `pip install deep-memory`
- `memory.provider: deep_memory`
- done

without also changing Hermes or vendoring a plugin shim into the Hermes repo.

### Why this matters
Richard explicitly wants:

- plugin / optional-dependency path
- external package stays in its own repo
- minimal core changes

So **the most important minimal Hermes hook is external memory-provider discovery**.

---

## Minimal Hermes changes

## Change 1: support external memory provider entrypoints

### Objective
Allow installed packages to register memory providers without living inside `plugins/memory/`.

### Recommended design
Extend `plugins/memory/__init__.py` so provider discovery checks two sources:

1. **existing in-repo providers**
   - `plugins/memory/<name>/`
2. **installed package entrypoints**
   - e.g. `importlib.metadata.entry_points(group="hermes.memory_providers")`

### Example external package contract
Deep Memory package would expose something like:

```toml
[project.entry-points."hermes.memory_providers"]
deep_memory = "deep_memory.hermes_provider:DeepMemoryProvider"
```

Hermes then loads it with something like:

```python
from importlib.metadata import entry_points

for ep in entry_points(group="hermes.memory_providers"):
    provider_cls = ep.load()
    provider = provider_cls()
```

### Why this is the best minimal change
- no Deep Memory code inside Hermes
- no custom import hacks in `run_agent.py`
- keeps current repo-local plugin behavior intact
- matches the existing provider abstraction instead of bypassing it

### Files to modify
- `plugins/memory/__init__.py`

### Backward compatibility
Full backward-compatible.
If no external providers are installed, Hermes behaves exactly as now.

---

## Change 2: add optional provider-owned session recall hook

### Objective
Let an external memory provider augment or replace the current `session_search` behavior.

### Problem today
`session_search` is still handled directly in `run_agent.py` via the SQLite session DB:

```python
elif function_name == "session_search":
    from tools.session_search_tool import session_search as _session_search
    return _session_search(...)
```

This is fine for built-in Hermes memory, but it means Deep Memory cannot become the unified long-term recall layer. At best it lives beside `session_search`, not under it.

### Minimal design
Do **not** remove built-in `session_search`.
Instead, add an optional provider hook such as:

```python
def session_search(self, query: str, role_filter: str | None = None, limit: int = 3, **kwargs) -> str:
    return ""
```

Then in `MemoryManager`:

```python
def session_search(self, query: str, role_filter=None, limit=3, **kwargs) -> str | None:
    for provider in self._providers:
        if hasattr(provider, "session_search"):
            result = provider.session_search(query, role_filter=role_filter, limit=limit, **kwargs)
            if result:
                return result
    return None
```

Then in `run_agent.py`:

```python
elif function_name == "session_search":
    if self._memory_manager:
        provider_result = self._memory_manager.session_search(...)
        if provider_result:
            return provider_result
    return _session_search(...built_in sqlite path...)
```

### Why this is minimal
- built-in behavior remains default
- providers can opt in
- no tool schema changes needed
- Deep Memory can become the recall backend later without breaking existing users

### Files to modify
- `agent/memory_provider.py`
- `agent/memory_manager.py`
- `run_agent.py`

---

## Change 3: add optional primary memory-write delegation

### Objective
Allow an external provider to act as the primary backend for the `memory` tool, not just mirror writes after the built-in store changes.

### Problem today
Current behavior for `memory` is:

1. built-in `MemoryStore` writes to MEMORY/USER
2. external provider receives a best-effort mirror event through `on_memory_write()`

That is useful, but it is **not** a true backend swap.

### Minimal design
Keep the existing default.
Add an optional provider hook such as:

```python
def handle_builtin_memory_write(self, action: str, target: str, content: str | None, old_text: str | None, **kwargs) -> str | None:
    return None
```

And add a config flag:

```yaml
memory:
  provider: deep_memory
  delegate_builtin_memory_tool: true
```

Then in `run_agent.py` memory-tool interception:

```python
elif function_name == "memory":
    if self._memory_manager:
        delegated = self._memory_manager.handle_builtin_memory_write(...)
        if delegated is not None:
            return delegated
    return _memory_tool(...existing builtin path...)
```

### Why this is better than replacing the built-in provider outright
Because Hermes still needs a safe default.
The built-in MEMORY/USER files should remain the fallback path.
Delegation should be explicit, opt-in, and reversible.

### Files to modify
- `agent/memory_provider.py`
- `agent/memory_manager.py`
- `run_agent.py`
- optionally `hermes_cli/config.py` for a documented config key

---

## Change 4: optional prompt-block composition hint

### Objective
Let external providers contribute a richer prompt block without Deep Memory-specific code in Hermes.

### Status today
This is already *mostly solved* via:
- `system_prompt_block()`
- `prefetch()`

So this is optional, not required.

### Only if needed later
If Deep Memory wants a clearer distinction between:
- static profile block
- dynamic recall block
- compression-preservation block

then Hermes could eventually split prompt insertion into named channels.

For example:

```python
provider.system_prompt_block()
provider.dynamic_context_block()
provider.compression_hint_block(messages)
```

But this is not part of the minimal plan.

---

## Recommended implementation order

## Phase A — enough for real external Deep Memory usage

### Task A1: external provider entrypoint loading
**Objective:** make Deep Memory installable as a package-backed memory provider.

**Files:**
- Modify: `plugins/memory/__init__.py`
- Test: `tests/plugins/test_memory_provider_loader.py` or equivalent

**Acceptance criteria:**
- Hermes still discovers in-repo providers
- Hermes also discovers installed entrypoint providers
- `memory.provider: deep_memory` can resolve an external package provider

### Task A2: Deep Memory provider package class
**Objective:** implement `DeepMemoryProvider` against Hermes’s existing `MemoryProvider` ABC.

**Files:**
- Create in Deep Memory repo: `src/deep_memory/hermes_provider.py`

**Acceptance criteria:**
- provider initializes via `initialize()`
- provider exposes current Deep Memory tools through `get_tool_schemas()`
- provider uses `prefetch()` / `on_session_end()` / `on_memory_write()`

This phase alone gets Deep Memory very far.

---

## Phase B — make Deep Memory feel more native

### Task B1: provider-owned `session_search` fallback path
**Objective:** let Deep Memory serve session recall when configured.

**Files:**
- Modify: `agent/memory_provider.py`
- Modify: `agent/memory_manager.py`
- Modify: `run_agent.py`
- Test: `tests/test_run_agent.py`

**Acceptance criteria:**
- built-in `session_search` remains default
- provider can override when it returns a non-empty result
- failure in provider falls back cleanly to built-in session DB search

---

## Phase C — true backend substitution for built-in memory tool

### Task C1: delegated `memory` tool backend
**Objective:** allow Deep Memory to become the primary write backend for `memory`.

**Files:**
- Modify: `agent/memory_provider.py`
- Modify: `agent/memory_manager.py`
- Modify: `run_agent.py`
- Optionally modify: `hermes_cli/config.py`

**Acceptance criteria:**
- default Hermes behavior unchanged
- config can opt into delegation
- Deep Memory can own writes while Hermes retains fallback safety

---

## What Deep Memory should do on its side

Hermes minimal hooks are only half the story. Deep Memory should meet them cleanly.

### Deep Memory side responsibilities

1. **Implement `DeepMemoryProvider`**
   - wrap `DeepMemoryService`
   - expose existing toolset
   - perform `prefetch()` using recall/build-context
   - use `on_session_end()` for extraction/consolidation
   - mirror built-in memory writes with `on_memory_write()`

2. **Publish Hermes memory-provider entrypoint**
   - package metadata should advertise the provider class

3. **Keep provider thin**
   - no business logic in the Hermes adapter
   - all memory logic remains in `deep_memory.api` / `deep_memory.store` / `deep_memory.reasoning`

That preserves the toolset-first → backend-ready evolution path.

---

## My recommendation

If the goal is **minimum upstream Hermes change with maximum future payoff**, do this:

### Do now
1. **Add external memory-provider entrypoint support**
2. **Implement `DeepMemoryProvider` in the Deep Memory repo**

### Do next
3. **Add optional provider-owned `session_search` override**

### Do only if truly needed
4. **Add delegated `memory` backend ownership**

That sequence is the best tradeoff.

It keeps Hermes lean, keeps Deep Memory external, and avoids prematurely entangling the core around one backend.

---

## Bottom line

Hermes already has the beginnings of the backend/provider hook system.
The biggest missing piece is **external provider loading**, not a brand new memory architecture.

So the minimal path is:

- **use the existing `MemoryProvider` / `MemoryManager` infrastructure**
- **teach Hermes to load provider classes from installed packages**
- **optionally providerize `session_search` and `memory` later**

That gets Deep Memory from:

- **today:** toolset/plugin-compatible external package
- **next:** true external Hermes memory provider
- **later:** tighter native recall/write integration

without forcing a rewrite on either side.
