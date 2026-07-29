---
# Orchestration Guide — Custom Agent Dispatch

## Custom Agent Dispatch — Root Cause and Fix

### Problem

All custom agents (`.github/agents/*.agent.md`) failed with "No response generated" when dispatched via the `task` tool. Built-in agents (`general-purpose`, `task`) worked fine with the same model.

### Root Cause

The `model` field in `.agent.md` frontmatter triggers the CLI's native `agentsResolveCustomAgentModel` function, which:

1. Takes the frontmatter model (e.g., `glm-5.2:cloud`)
2. Resolves it to a provider-specific format by appending a suffix: `glm-5.2:cloud (ollama)`
3. Sends this suffixed name to the Ollama API at `http://127.0.0.1:11434/v1/chat/completions`
4. The Ollama API rejects it with **400 Bad Request** ("invalid model name") because model names cannot contain spaces or parentheses

The `general-purpose` agent works because it has `model: void 0` (undefined) — no frontmatter model. Without a frontmatter model, `agentsResolveCustomAgentModel` returns the session model directly (`glm-5.2:cloud`) without any suffix, and the request goes to the Copilot API (not Ollama).

### Evidence

- `subagent.started` for custom agent: `model: "glm-5.2:cloud (ollama)"` — has suffix
- `subagent.started` for general-purpose: `model: "glm-5.2:cloud"` — no suffix
- Ollama API test: `glm-5.2:cloud (ollama)` → 400 Bad Request
- Ollama API test: `glm-5.2:cloud` → 200 OK
- All API call IDs for general-purpose: `resp_XXXXXX` format (Copilot API)

### Fix Applied

1. Removed the `model:` field from all 65 `.agent.md` frontmatter files using `node scripts/agent-customization/update-agent-models.mjs --remove`
2. Deleted test agent files (`test-tool-star.agent.md`, `test-minimal.agent.md`)
3. Refreshed the routing table with `npm run agents:routing-table`
4. Fixed Tier-4 agent quality issues (wrong structured-v1 field order) using `validate-agent-quality.fix.mjs`

### Why This Works

Without a `model:` field in frontmatter:
- `agentsResolveCustomAgentModel` receives `void 0` for the model parameter
- It returns the session model (`glm-5.2:cloud`) without any provider suffix
- The clean model name is sent to the Copilot API (same path as `general-purpose`)
- Custom agents work exactly like `general-purpose`

### Key Constraint

The user requires the same model string everywhere. With the `model:` field removed, agents inherit the session model automatically. Changing the session model changes ALL agents. No need to update individual agent files when swapping models.

### Tool Alias Resolution

The CLI's `agentsResolveToolAliases` resolves tool names in agent frontmatter:
- `read` → `view` ✅
- `execute` → `powershell` ✅
- `agent` → `task` ✅
- `edit` → `create` + `edit` ✅
- `search` → [] ❌ (unresolved — use `grep` instead)
- `todo` → [] ❌ (unresolved — use SQL `todos` table or `ask_user` instead)
- `run` → [] ❌ (unresolved)
- `bash` → [] ❌ (unresolved)

### Model Override via task Tool

The `task` tool's `model` parameter does NOT reliably override the frontmatter model. Testing showed that even with a model override, the frontmatter model was still used for the API call. Removing the frontmatter model field is the only reliable fix.

### CLI Restart Required

Agent files are loaded at CLI startup. After modifying `.agent.md` files, the user MUST restart the CLI for changes to take effect.

### Bulk Update Script

Use `scripts/agent-customization/update-agent-models.mjs` to manage agent models:
- `--remove` — Remove all frontmatter `model:` fields (recommended)
- `--from=X --to=Y` — Replace specific model string
- `--to=Y` — Replace all model strings with Y
- `--dry-run` — Preview changes

## Orchestration Architecture

### Tier Graph

- Tier 0: Agent Zero (root orchestrator) — routes only
- Tier 1: 8 numbered SDLC orchestrators (00-07) — user-invocable
- Tier 2: 11 named coordinators
- Tier 3: 42 hidden scouts & specialists (incl. 3 Chrome DevTools MCP specialists)
- Tier 4: 4 auxiliaries & one-shot helpers

### Delegation Direction

Strictly downward. No tier may call a higher-numbered tier except via `00.cross-tier-helper`.

### Dispatch Protocol

1. Map task goal to target agent (see goal-to-agent mapping table in `copilot-instructions.md`)
2. Call `neataptic-dispatch-mcp / build_dispatch_packet` with caller tier and target agent name
3. If `dispatch_allowed: true`, use the returned dispatch packet with the `task` tool
4. If `dispatch_allowed: false`, escalate via `00.cross-tier-helper`

### RED → IMPLEMENT → GREEN Loop

For sliced implementation steps:
1. Dispatch `03-red-testing` to create failing tests
2. Dispatch `04-implementing` to implement code
3. Dispatch `05-green-testing` to validate
4. If green returns observations (NOT OK), loop back to a NEW `04-implementing` instance
5. Repeat until green returns OK
6. After all slices pass, dispatch `06-documenting` for docs-quality checks

### Concurrency

BYOK (Ollama) users default to 2 concurrent agents. Set `COPILOT_SUBAGENT_MAX_CONCURRENT` environment variable to increase. With limit 2, use flat sequential dispatch (T1→T2, then T1→T3) instead of nested (T1→T2→T3).

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `COPILOT_SUBAGENT_MAX_CONCURRENT` | 2 (BYOK) | Maximum concurrent sub-agents |
| `COPILOT_SUBAGENT_MAX_DEPTH` | 6 | Maximum delegation chain depth |

To override on Windows PowerShell (persists across restarts):
```powershell
[Environment]::SetEnvironmentVariable("COPILOT_SUBAGENT_MAX_CONCURRENT", "10", "User")
[Environment]::SetEnvironmentVariable("COPILOT_SUBAGENT_MAX_DEPTH", "6", "User")
```
Then close and reopen the terminal.
---