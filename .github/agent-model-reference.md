# Canonical Agent Model Reference

> Canonical decision aid for NeatapticTS custom-agent model assignment.
> Use this file before adding a new agent or changing any `model:` frontmatter.

## Repo Contract

NeatapticTS targets **Copilot CLI-compatible agent frontmatter**. For this
repository, the `model` field in `.github/agents/*.agent.md` must be a **single
qualified model string**, not an array.

This repo previously used inline fallback arrays because VS Code custom agents
support them. That shape is now considered legacy here because the Copilot CLI
rejects array-valued `model` frontmatter.

## Maintenance Workflow

1. Confirm the exact qualified model names available in the active Copilot
   client before any frontmatter edit.
2. Update this file when a model becomes confirmed, deprecated, or when the
   routing rationale changes.
3. Edit `.github/agents/*.agent.md` files only after the assignment decision is
   clear and validated.
4. Write `model` as a single quoted qualified string.
5. For legacy array migrations, preserve the first currently listed model unless
   the user explicitly requests a routing change.
6. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`
   after any frontmatter change.
7. Refresh `.github\agent-skill-routing-table.md` only after agent or skill
   frontmatter changes are complete.

> Warning: unverified model strings must not be written to frontmatter.
> A guessed or stale model string can trigger silent fallback or routing drift.

## Confirmed Qualified Model Strings

These are the qualified model strings currently confirmed as safe for
NeatapticTS custom-agent frontmatter.

| Qualified model string        | Role in fleet                           | Capability note                                                                                                                   |
| ----------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `glm-5.2:cloud (ollama)`      | Elite coding and complex orchestration  | Best for SWE-Bench Pro performance, long-horizon autonomy, complex TypeScript refactoring, and multi-step engineering tasks.    |
| `kimi-k2.7-code:cloud (ollama)`     | Frontend-focused and tool integration   | Optimized for modern TS ecosystems (React, Next.js), visual components, MCP tool calling, and cost-effective agentic workflows. |

Other Copilot models may exist in docs or the client, but they are not
canonical for NeatapticTS agent frontmatter until the exact qualified string is
confirmed in the active client and validated locally.

## Canonical Decision Flow

1. Verify availability first.
   Use the active Copilot model picker or `model-name-auditor` to confirm the
   exact qualified string before touching frontmatter.
2. Classify the agent job.
   Decide whether the agent is primarily backend/infrastructure implementation,
   frontend/UI work, tool integration, or long-horizon orchestration.
3. Choose the appropriate model for the job.
   Use `glm-5.2:cloud (ollama)` for complex backend logic, SWE-Bench style tasks,
   and marathon execution loops. Use `kimi-k2.7-code:cloud (ollama)` for frontend work,
   visual components, rapid iteration, and tool-heavy workflows.
4. Write a single model string.
   Do not use arrays in `model:` frontmatter for this repo.
5. Preserve intent during migrations.
   When converting a legacy array-valued `model`, keep the first listed model
   unless the user explicitly wants a different routing decision.
6. Validate before committing.
   Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`
   
after any frontmatter edit.
7. Refresh the generated routing table only after frontmatter changes land.
   This reference is hand-maintained; `.github/agent-skill-routing-table.md` is
   generated and should be refreshed after the actual agent changes, not before.

## Default Assignment Patterns

| Agent type                                               | Canonical scalar model       | Notes                                                                 |
| -------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------- |
| Main orchestrators (00-07 numbered agents)               | `glm-5.2:cloud (ollama)`     | Long-horizon autonomy, complex SDLC phase coordination, marathon execution loops without strategy fatigue. |
| Specialized executors/scouts (all other agents)          | `kimi-k2.7-code:cloud (ollama)` | Superior tool integration, MCP workflows, 30% fewer reasoning tokens, cost-effective for focused tasks. |


## Migration Rule

When repairing legacy agent frontmatter:

- convert `model: ['A', 'B', ...]` to `model: 'A'`,
- preserve the first listed model as the authoritative scalar value,
- update validator and skill guidance in the same pass,
- regenerate the routing table after the source frontmatter is aligned.

## Validation Checklist

After changing agent model frontmatter, run:

1. `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`
2. `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict`
3. `node scripts/agent-customization/validate-agent-graph.mjs --json` when
   delegation-related fields changed
4. `npm run agents:routing-table`
5. `npm run agents:routing-table:gate`