# Agent Quality Contract

This document defines the canonical structural contract for every `.github/agents/*.agent.md` file in NeatapticTS.

It separates two kinds of requirements:

- Enforced rules: checked by `scripts/agent-customization/validate-agent-quality.mjs` and the `agent-quality` gate.
- Documented expectations: required authoring guidance for maintainers, but not always machine-enforced.

## Purpose

Every agent must be explicit about:

- what problem it owns,
- what it must not do,
- how it routes work or hands off results,
- what to do when blocked,
- and exactly what output shape downstream agents or gates can rely on.

## Enforced Rules

### Frontmatter

Every agent must have YAML frontmatter with a valid `tier:` field.

- Tier `1`: numbered SDLC orchestrators.
- Tier `2`: named coordinators and reusable sub-orchestrators.
- Tier `3`: scouts, auditors, and narrow specialists.
- Tier `4`: auxiliary one-shot helpers.

The structured output block inside the body must set `TIER:` to the same numeric value as the frontmatter tier.

### Mandatory Sections By Tier

The validator enforces these heading sets exactly.

| Tier | Required sections |
| --- | --- |
| 1 | `## Mission`, `## Constraints`, `## Default Flow`, `## If Blocked`, `## Output Format` |
| 2 | `## Mission`, `## Constraints`, `## Required Workflow`, `## If Blocked`, `## Output Format` |
| 3 | `## Mission`, `## Constraints`, `## Approach`, `## If Blocked`, `## Output Format` |
| 4 | `## Mission`, `## Constraints`, `## Default Flow`, `## If Blocked`, `## Output Format` |

### Structured Output Contract

Every agent must define exactly one fenced `structured-v1` block inside `## Output Format`.

The validator enforces all of the following:

- `## Output Format` exists.
- the section contains exactly one fenced `structured-v1` block,
- the first non-blank line in the block is `OUTPUT_CONTRACT: structured-v1`,
- the required field set and field order match the tier contract exactly,
- `ROLE:` matches the agent frontmatter `name`,
- `TIER:` matches the agent frontmatter `tier`.

### Required Field Order By Tier

Tier 1 orchestrator contract:

```text
OUTPUT_CONTRACT
TASK_STATUS
TIER
ROLE
TASK_RECEIVED
FILES_READ
FILES_CHANGED
KEY_FINDINGS
ACTIONS_TAKEN
VALIDATION_EVIDENCE
BLOCKERS
RISKS_OR_GAPS
LEARNING_EVENT_NEEDED
SUGGESTED_NEXT_AGENT
PHASE_COMPLETE
SUB_ORCHESTRATORS_USED
SUMMARY
```

Tier 2 coordinator contract:

```text
OUTPUT_CONTRACT
TASK_STATUS
TIER
ROLE
TASK_RECEIVED
FILES_READ
FILES_CHANGED
KEY_FINDINGS
ACTIONS_TAKEN
VALIDATION_EVIDENCE
SPECIALISTS_USED
HANDOFF
BLOCKERS
RISKS_OR_GAPS
LEARNING_EVENT_NEEDED
SUGGESTED_NEXT_AGENT
SUMMARY
```

Tier 3 scout contract:

```text
OUTPUT_CONTRACT
TASK_STATUS
TIER
ROLE
TASK_RECEIVED
FILES_READ
FILES_CHANGED
KEY_FINDINGS
ACTIONS_TAKEN
VALIDATION_EVIDENCE
HANDOFF
BLOCKERS
RISKS_OR_GAPS
LEARNING_EVENT_NEEDED
SUGGESTED_NEXT_AGENT
SUMMARY
```

Tier 4 auxiliary contract:

```text
OUTPUT_CONTRACT
TASK_STATUS
TIER
ROLE
TASK_RECEIVED
FILES_READ
FILES_CHANGED
KEY_FINDINGS
ACTIONS_TAKEN
BLOCKERS
RISKS_OR_GAPS
LEARNING_EVENT_NEEDED
SUGGESTED_NEXT_AGENT
SUMMARY
```

### Delegation And Handoff Expectations Enforced By Contract Shape

Delegation and handoff are enforced structurally in these ways:

- Tier 1 orchestrators must report downstream routing via `SUGGESTED_NEXT_AGENT` and `SUB_ORCHESTRATORS_USED`.
- Tier 2 coordinators must report reroute intent via `HANDOFF` and `SUGGESTED_NEXT_AGENT`.
- Tier 3 scouts must return a compact `HANDOFF` for the caller.
- Tier 4 auxiliaries may omit `HANDOFF`, but must still identify the safest `SUGGESTED_NEXT_AGENT` when follow-up is needed.

## Documented Expectations

These expectations are part of the quality contract even when the validator does not block on them.

### Mission And Scope

Every agent should state:

- what it owns,
- whether it is read-only or edit-capable,
- and the narrow boundary it should not cross.

### Skills

When a durable skill governs the workflow, the agent should name the exact skill and make clear that policy belongs to the skill rather than the agent body.

### MCP Usage

When an agent depends on MCP tools, MCP gate servers, or MCP-only runtime evidence, the body should document:

- which MCP boundary applies,
- when the agent should use it,
- and what to do when MCP evidence is unavailable or ambiguous.

This is a documented expectation because not every agent needs an MCP-specific section.

### Delegation

When an agent delegates, it should document:

- who it delegates to,
- why those delegates are the smallest correct boundary,
- and whether the delegate is used for research, implementation, testing, or handoff only.

### Blocked Behavior

Every agent must describe its blocked path explicitly. The validator enforces the `## If Blocked` section because blocked behavior is mandatory.

### Handoffs

Agents that routinely reroute work should make the handoff target and stop condition obvious in the workflow text, not only in frontmatter.

## Enforcement Surface

The canonical machine checks are:

- `node scripts/agent-customization/validate-agent-quality.mjs --json`
- `node scripts/agent-customization/gates/agent-quality.gate.mjs --json`

The existing frontmatter validator remains responsible for YAML, model, tool, and delegation metadata. The agent-quality validator owns body structure and structured output contract compliance.