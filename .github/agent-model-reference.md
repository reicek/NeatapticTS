# Canonical Agent Model Reference

> Canonical decision aid for NeatapticTS custom-agent model assignment.
> Use this file before adding a new agent or changing any `model:` frontmatter.

## Repo Contract

NeatapticTS targets **Copilot CLI-compatible agent frontmatter**. For this
repository, the `model` field in `.github/agents/*.agent.md` frontmatter
**must be omitted** (not set). Agents inherit the session model automatically.

### Why no `model:` field?

The Copilot CLI's native `agentsResolveCustomAgentModel` function resolves
the frontmatter `model` value by appending a provider suffix (e.g.
`(ollama)`). The suffixed model name (e.g. `glm-5.3-flash:cloud (ollama)`) is
sent to the Ollama API, which rejects it with **400 Bad Request** because
model names cannot contain spaces or parentheses.

When the `model` field is **omitted**, `agentsResolveCustomAgentModel`
receives `void 0` and returns the session model directly (e.g.
`glm-5.3-flash:cloud`) without any suffix. The clean model name is sent to the
Copilot API — the same path used by `general-purpose` and other built-in
agents.

This means:
- **All agents use the same model** — the session model.
- **Changing the session model changes ALL agents** — no per-agent updates needed.
- **No model string mismatch** is possible between agents.

## Maintenance Workflow

1. **Do NOT add a `model:` field** to any `.agent.md` frontmatter file.
2. To change the model for all agents, change the session model in the
   Copilot CLI (e.g. `/model glm-5.3-flash:cloud`).
3. To bulk-remove existing `model:` fields, run:
   `node scripts/agent-customization/update-agent-models.mjs --remove`
4. After any frontmatter change, run:
   `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`
5. Refresh the routing table after agent changes:
   `npm run agents:routing-table`

> Warning: Adding a `model:` field to agent frontmatter will cause the
> agent to fail with "No response generated" when dispatched via the `task`
> tool. The `agentsResolveCustomAgentModel` native function appends a
> provider suffix that the Ollama API rejects.

## Bulk Update Script

Use `scripts/agent-customization/update-agent-models.mjs` to manage agent
models in bulk:

| Command | Purpose |
|---------|---------|
| `--remove` | Remove all frontmatter `model:` fields (recommended) |
| `--from=X --to=Y` | Replace specific model string |
| `--to=Y` | Replace all model strings with Y |
| `--dry-run` | Preview changes without writing files |
| `--json` | Machine-readable JSON output |

## Validation Checklist

After changing agent frontmatter, run:

1. `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`
2. `node scripts/agent-customization/validate-agent-quality.mjs --json`
3. `node scripts/agent-customization/validate-agent-graph.mjs --json`
4. `npm run agents:routing-table`
5. `npm run agents:routing-table:gate`

## CLI Restart Required

Agent files are loaded at CLI startup. After modifying `.agent.md` files,
the user **MUST restart the CLI** for changes to take effect.