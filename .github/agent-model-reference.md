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
| `GPT-5.4 (copilot)`           | Full-tier coding and implementation     | Best default for code-heavy implementation, red-test authorship, and specialist work that needs stronger tool-aware reasoning.    |
| `Claude Sonnet 4.6 (copilot)` | Full-tier synthesis and coordination    | Best default for planning, documentation, maintenance, and ambiguity-heavy coordination.                                          |
| `GPT-5.4 mini (copilot)`      | Budget-conscious general-purpose work   | Good default for bounded research, validation, subagent coordination, and specialists that still benefit from solid tool use.     |
| `Claude Haiku 4.6 (copilot)`  | Lightweight checklist and summarization | Good default for narrow audits, summaries, small recon tasks, and one-shot helpers where latency and cost matter more than depth. |

Other Copilot models may exist in docs or the client, but they are not
canonical for NeatapticTS agent frontmatter until the exact qualified string is
confirmed in the active client and validated locally.

## Canonical Decision Flow

1. Verify availability first.
   Use the active Copilot model picker or `model-name-auditor` to confirm the
   exact qualified string before touching frontmatter.
2. Classify the agent job.
   Decide whether the agent is primarily implementation, synthesis,
   research/validation, or lightweight checklist/summarization.
3. Choose the lowest-cost model that still fits the job.
   Use `GPT-5.4` for coding-heavy implementation, `Claude Sonnet 4.6` for
   nuanced synthesis, `GPT-5.4-mini` for bounded research and validation, and
   `Claude Haiku 4.6` for narrow mechanical work.
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

| Agent need                                               | Canonical scalar model       | Notes                                                                 |
| -------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------- |
| Coding-heavy implementation or test synthesis            | `GPT-5.4 (copilot)`          | Prefer when code generation quality and edge-case handling matter.    |
| Ambiguity-heavy planning, maintenance, or docs synthesis | `Claude Sonnet 4.6 (copilot)`| Prefer when the agent must weigh tradeoffs, policy, or prose quality. |
| Bounded research, validation, or scout work              | `GPT-5.4 mini (copilot)`     | Prefer Mini when the agent still uses tools heavily or needs depth.   |
| Lightweight checklist, summaries, or one-shot helpers    | `Claude Haiku 4.6 (copilot)` | Prefer Haiku when the task is narrow, mechanical, and frequent.       |

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
