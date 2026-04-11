---
name: deep-memory-plugin
description: "Manage the Deep Memory plugin for Hermes Agent — a reasoning-based memory system with semantic recall, structured insight capture, and entity tracking. Use this skill when installing Deep Memory, verifying tool registration, or repairing Hermes integration after updates."
---

# Deep Memory Plugin

## What It Is

Deep Memory is an add-on for Hermes Agent that provides longer-term, structured memory on top of normal session context.

It adds:
- **`recall`** for semantic + keyword memory search
- **`learn`** for saving explicit insights
- **`entities`** for viewing and managing entity profiles
- **Post-session reasoning** that extracts conclusions from conversations automatically
- **SQLite-based storage** with optional vector search embeddings

Use this skill when you need to install, verify, troubleshoot, or repair a Deep Memory integration.

## When To Use This Skill

Use this skill if:
- Deep Memory tools are missing from Hermes
- `hermes update` or a local Hermes upgrade broke the integration
- You need to confirm Deep Memory is installed correctly
- You want to re-enable automatic post-session reasoning
- You need to check embedding or database setup

## Typical Layout

Paths vary by installation, but a common setup looks like this:

```text
/path/to/deep-memory/                 # Deep Memory source repository
/path/to/deep-memory/src/deep_memory/ # Python package source
~/.hermes/deep_memory/                # Runtime data directory
~/.hermes/config.yaml                 # Hermes configuration
~/.hermes/hermes-agent/               # Hermes checkout / install
```

## Configuration

Optional Hermes config:

```yaml
deep_memory:
  embedding_backend: local   # local | openai | none | auto
```

Notes:
- `local` uses `sentence-transformers` if installed
- `openai` requires a valid API key in the environment
- `none` disables embeddings and uses keyword/FTS search only
- `auto` or no setting lets Deep Memory pick the best available backend

## How Integration Works

A working Deep Memory install usually depends on four things:

1. The `deep_memory` Python package is installed into the same environment Hermes uses
2. Hermes imports `deep_memory.hermes_integration` during tool discovery
3. Hermes exposes the Deep Memory tools in its enabled/core toolsets
4. Hermes runs the optional session hook so post-session reasoning can process transcripts

In practice, Deep Memory is often installed in editable mode:

```bash
source /path/to/hermes/venv/bin/activate
pip install -e /path/to/deep-memory
```

## Minimum Hermes Integration Points

Hermes must be able to import the integration module during tool discovery:

```python
try:
    import deep_memory.hermes_integration  # noqa: F401
except ImportError:
    pass
```

Hermes must also expose the tool names:

```python
"recall", "learn", "entities",
```

If your Hermes integration supports prompt augmentation and post-session hooks, Deep Memory can also:
- inject memory/entity context into the system prompt
- trigger asynchronous post-session reasoning after a conversation ends

## Installation Checklist

1. Clone or place the Deep Memory repo somewhere stable
2. Install it into Hermes's Python environment
3. Ensure Hermes imports `deep_memory.hermes_integration`
4. Ensure `recall`, `learn`, and `entities` are included in the available toolset list
5. Restart Hermes or the Hermes gateway
6. Verify the tools appear and registration logs look healthy

## Verification

### 1) Verify the package imports

Run this in the same Python environment Hermes uses:

```bash
python -c "import deep_memory, deep_memory.hermes_integration; print('ok')"
```

### 2) Verify Hermes-side wiring

From the Hermes codebase, search for Deep Memory references in the files responsible for:
- tool discovery
- toolset registration
- agent runtime / session-end hooks

Example:

```bash
grep -n "deep_memory\|recall\|learn\|entities" model_tools.py toolsets.py run_agent.py
```

If your Hermes layout differs, search the equivalent files in your installation.

### 3) Verify logs after restart

Check Hermes logs for successful registration messages, for example:

```bash
grep -i "deep memory\|deep_memory\|registered" ~/.hermes/logs/*.log
```

### 4) Verify the tools are actually available

Confirm Hermes can see `recall`, `learn`, and `entities` in its active tool list or tool discovery output.

## Why `hermes update` May Break This

Deep Memory commonly depends on small Hermes-side integration edits. A Hermes update can overwrite those local changes even if the Python package itself is still installed.

Common symptoms after an update:
- `deep_memory` still imports, but the tools no longer appear
- `recall`, `learn`, and `entities` are missing from Hermes
- post-session reasoning no longer runs
- Deep Memory logs disappear after restart

Editable installs and `.pth`-based imports often survive updates, but Hermes source-level hooks may not.

## Repair Steps After Hermes Updates

If Deep Memory stops working after an update:

### 1) Confirm the package still exists

```bash
python -c "import deep_memory; print(deep_memory.__file__)"
```

If import fails, reinstall Deep Memory into the Hermes environment:

```bash
source /path/to/hermes/venv/bin/activate
pip install -e /path/to/deep-memory
```

### 2) Re-check Hermes integration points

Make sure the update did not remove:
- the `import deep_memory.hermes_integration` hook
- the `recall`, `learn`, `entities` tool registrations
- any optional prompt/session hook integration you rely on

### 3) Re-apply the missing integration edits

Restore the minimal Deep Memory hooks in the relevant Hermes files for your version.

At minimum, Hermes must:
- import `deep_memory.hermes_integration`
- expose `recall`, `learn`, and `entities`

If your setup supports it, also restore:
- system-prompt context injection
- post-session reasoning hook execution

### 4) Restart Hermes

After restoring the integration, restart Hermes or the Hermes gateway so the updated code is loaded.

### 5) Verify again

Repeat the verification steps above:
- package import
- source-level integration checks
- logs
- visible tool registration

## Common Pitfalls

- Deep Memory was installed into a different Python environment than Hermes uses
- The Hermes update removed local integration edits
- The tool names exist in code but are not enabled in the active toolset/config
- The virtual environment was rebuilt, removing the editable install or `.pth` entry
- `sentence-transformers` is missing while `embedding_backend: local` is configured
- Hermes was not restarted after code or environment changes

## Practical Recovery Order

When troubleshooting, use this order:

1. Confirm the package imports
2. Confirm Hermes imports `deep_memory.hermes_integration`
3. Confirm `recall`, `learn`, and `entities` are registered/exposed
4. Restart Hermes
5. Check logs
6. Test the tools directly

## Success Criteria

Deep Memory is working when all of the following are true:
- `deep_memory` imports in Hermes's Python environment
- Hermes discovers the plugin during startup
- `recall`, `learn`, and `entities` are available to the agent
- logs show successful registration or initialization
- post-session reasoning runs if your Hermes integration enables it
