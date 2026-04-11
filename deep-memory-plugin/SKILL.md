---
name: deep-memory-plugin
description: "Manage the deep-memory plugin for Hermes Agent — a custom reasoning-based memory system with entity tracking, insight extraction, and semantic recall. Use when troubleshooting deep-memory tools (recall, learn, entities), checking integration status, or after running `hermes update` which can break the plugin hook."
---

# Deep Memory Plugin — Hermes Integration

## What It Is

Deep-memory is a **custom plugin** (not upstream Hermes) that adds reasoning-based memory to Hermes Agent. It lives at `~/Repository/deep-memory/` and is pip-installed in dev mode into Hermes's venv.

### Components
- **Tools**: `recall` (semantic search), `learn` (store insights), `entities` (manage entity profiles)
- **Reasoning**: Post-session extraction of entities and conclusions from conversation transcripts
- **Embedding**: Supports `local` (sentence-transformers), `openai`, or `none` (FTS-only)
- **Storage**: SQLite + sqlite-vec for vector search

## Key Paths

```text
~/Repository/deep-memory/                    # Source repo
~/Repository/deep-memory/src/deep_memory/    # Package source
  ├── hermes_integration.py                  # Hermes tool registration bridge
  ├── session_hook.py                        # Post-session reasoning hook
  ├── embedding.py                           # Embedding backends (local/openai/none)
  ├── tools/                                 # recall.py, learn.py, entities.py
  ├── store/                                 # db.py, schema.py, search.py
  └── reasoning/                             # consolidator.py, extractor.py, prompts.py

~/Repository/deep-memory/deep-memory-plugin/ # Hermes skill packaging folder
~/.hermes/deep_memory/                       # Runtime data directory
~/.hermes/config.yaml → deep_memory:         # Config section
```

## Config (in ~/.hermes/config.yaml)

```yaml
deep_memory:
  embedding_backend: local    # local | openai | none | auto
```

## How Integration Works

1. `hermes_integration.py` auto-registers tools on import via `register_tools()`
2. It registers 3 tools (`recall`, `learn`, `entities`) under the `deep_memory` toolset
3. Hermes needs to import this module during tool discovery in `model_tools.py`:

```python
# In model_tools.py _discover_tools():
try:
    import deep_memory.hermes_integration  # noqa: F401
except ImportError:
    pass
```

4. The `.pth` file makes the package importable:
   `~/.hermes/hermes-agent/venv/lib/python3.11/site-packages/_deep_memory.pth`
   → points to `~/Repository/deep-memory/src`

## ⚠️ CRITICAL: `hermes update` Breaks Integration

**Problem**: `hermes update` pulls upstream code and overwrites **3 files** with custom deep-memory hooks. The `.pth` file and pip install survive, but all Hermes-side integration code gets wiped. Local changes may be auto-stashed but not auto-restored (especially if conflicts occur).

### Files affected by `hermes update`

| File | What gets wiped |
|---|---|
| `model_tools.py` | `import deep_memory.hermes_integration` in `_discover_tools()` |
| `toolsets.py` | `"recall", "learn", "entities"` in `_HERMES_CORE_TOOLS` list |
| `run_agent.py` | System prompt context injection + post-session reasoning hook |

### After every `hermes update`

**Step 1 — Check if integration survived:**
```bash
cd ~/.hermes/hermes-agent
grep -n "deep_memory" model_tools.py toolsets.py run_agent.py
```
If no results, re-patch all 3 files below.

**Step 2 — Check stash for preserved changes:**
```bash
git stash list
git stash show -p stash@{0} | grep -A5 -B5 "deep_memory"
```
Note: `git stash apply` may conflict on unrelated files (for example `discord.py`). Prefer manual patching over stash restore.

**Step 3 — Re-patch `model_tools.py`** (in `_discover_tools()`):
```python
# Add after the _modules list, before `import importlib`:
    # Deep Memory (external package — optional)
    try:
        import deep_memory.hermes_integration  # noqa: F401
    except ImportError:
        pass
```

**Step 4 — Re-patch `toolsets.py`** (in `_HERMES_CORE_TOOLS`):
```python
# Add after `"todo", "memory",`:
    # Deep memory (reasoning-based insights — optional, requires deep-memory package)
    "recall", "learn", "entities",
```

**Step 5 — Re-patch `run_agent.py`** (two locations):

*Location A — System prompt injection* (find `has_skills_tools = any(`, add block before it):
```python
        # Deep Memory — reasoning-based insights (optional, requires deep-memory package)
        if any(t in self.valid_tool_names for t in ("recall", "learn", "entities")):
            try:
                from deep_memory.hermes_integration import build_deep_memory_context
                _dm_entity = getattr(self, "_deep_memory_entity_id", None)
                _dm_block = build_deep_memory_context(_dm_entity)
                if _dm_block:
                    prompt_parts.append(_dm_block)
            except ImportError:
                pass
            except Exception as _dm_exc:
                logger.debug("Deep memory context injection failed: %s", _dm_exc)
```

*Location B — Post-session reasoning hook* (find `on_session_end hook failed`, add block after its except/logger line, before `return result`):
```python
        # Deep Memory — async post-session reasoning (fire-and-forget)
        if any(t in self.valid_tool_names for t in ("recall", "learn", "entities")):
            try:
                from deep_memory.session_hook import process_session_async
                process_session_async(
                    session_id=self.session_id,
                    messages=messages,
                )
            except ImportError:
                pass
            except Exception as _dm_exc:
                logger.debug("Deep memory post-session reasoning failed: %s", _dm_exc)
```

**Step 6 — Restart gateway:**
```bash
nohup sh -c 'hermes gateway restart >/tmp/hermes-gateway-restart.log 2>&1' >/dev/null 2>&1 &
# Wait a few seconds, then verify:
hermes gateway status
```

### Quick verification after patching
```bash
grep -c "deep_memory" model_tools.py   # Should be >= 1
grep -c "recall" toolsets.py            # Should be >= 1
grep -c "deep_memory" run_agent.py      # Should be >= 3
tail -n 50 ~/.hermes/logs/gateway.log | grep "Deep memory tools registered"
```

## Upstream PR Status

Two separate PRs on NousResearch/hermes-agent (split from original combined PR #6692 which was closed):

- **PR #6694** — `fix/discord-bot-loop` — Discord bot-to-bot reply ping loop prevention (discord.py only)
- **PR #6695** — `feat/deep-memory` — Deep-memory plugin integration (model_tools.py, toolsets.py, run_agent.py)

Check status:
```bash
gh pr view 6694 --repo NousResearch/hermes-agent --json state
gh pr view 6695 --repo NousResearch/hermes-agent --json state
```

If both PRs are merged and you run `hermes update`, verify the patches survived by running the verification commands above. If they're now in upstream, no re-patching is needed.

**Deep-memory plugin repo:** <https://github.com/RichardHojunJang/deep-memory>

## Coupled Patch: Discord Bot-Loop Prevention

The same `hermes update` that wipes deep-memory also wipes the **discord.py bot-to-bot loop prevention patch**. These two patches travel together because they're both local modifications to upstream files.

The discord.py patch adds:
- `_discord_body_mentions_user()` — checks explicit body text `<@id>` instead of `message.mentions` (which includes reply pings)
- `_is_peer_bot_author()` — distinguishes peer bot messages from human messages
- Updated mention checks in `on_message`, `DISCORD_ALLOW_BOTS=mentions` filter, `DISCORD_IGNORE_NO_MENTION` logic, and `_handle_message()`

Without this patch, multiple Hermes bots in the same Discord channel will loop when one bot's reply ping triggers another bot's response.

## Pitfalls

- The `deep_memory` toolset must be listed in enabled toolsets in config.yaml for the tools to appear
- `hermes update` wipes **3 files** — not just `model_tools.py`. Don't forget `toolsets.py` and `run_agent.py`
- `hermes update` auto-stashes local changes, but stash restore often conflicts on unrelated files (for example `discord.py`). Prefer targeted patching over `git stash apply`
- After `hermes update`, the stash contains all local changes, including unrelated ones like discord.py bot-mention patches. Do not blindly apply the full stash — cherry-pick only deep-memory changes
- Stash may not exist at all if it was previously dropped. Always be prepared to re-patch manually from the code blocks above
- The discord.py bot-loop patch and deep-memory patch are coupled — both get wiped by `hermes update`, both need re-applying together
- The `.pth` file (`_deep_memory.pth`) is fragile — if the venv is recreated, re-run `pip install -e .` from `~/Repository/deep-memory/`
- `embedding_backend: local` requires `sentence-transformers` installed in the Hermes venv
- Gateway restart after patching is required — new sessions pick up tool registration, but the existing gateway process uses stale code
