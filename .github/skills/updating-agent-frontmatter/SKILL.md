---
name: updating-agent-frontmatter
description: 'Use when: updating .agent.md YAML frontmatter, names, descriptions, tools, agents allow-lists, model fallback arrays, handoffs, or visibility flags.'
argument-hint: 'Name the agent file, metadata fields to change, visibility target, allowed subagents, model tier, and validation mode.'
user-invocable: false
disable-model-invocation: false
---

# Updating Agent Frontmatter

This skill makes safe, targeted edits to YAML frontmatter in existing `.github/agents/*.agent.md` files. It validates that names remain stable, delegation stays bounded, model strings are qualified, and the updated file passes automated frontmatter validation before the change is recorded.

## When to Use

- Changing the `model` or model fallback array for an existing agent.
- Adding, removing, or reordering entries in an `agents: [...]` allow-list.
- Updating the `description` to improve trigger precision or fix a false-positive pattern.
- Toggling `user-invocable` when an agent's visibility target changes.
- Adding or updating `handoffs` entries.
- Fixing a frontmatter field that caused a silent loading failure in VS Code.

## Task Packet

Include the agent filename, the specific frontmatter fields to change, old and new values, and whether strict validation should pass after the edit.

```text
Use updating-agent-frontmatter for <agent-name>.agent.md.
Fields to change: <list of field names>
Old values: <what they are now>
New values: <what they should become>
Compatibility risk: <none | potential rename impact | delegation change>
Validate with: node scripts/agent-customization/validate-agent-frontmatter.mjs --json
```

## Required Workflow

1. Read the current frontmatter of the target `.agent.md` file before making any changes.
2. Confirm whether the agent is a user-facing orchestrator (must keep `user-invocable: true` and stable name) or a hidden specialist (may have `user-invocable: false`).
3. Apply the targeted field change; do not alter unrelated frontmatter fields in the same edit.
4. If changing `agents`, verify that `agent` is present in the `tools` list when the new `agents` list is non-empty.
5. If changing `model`, use a qualified model string or a validated fallback array; document the rationale.
6. If changing the agent `name` or description, assess whether any parent allow-lists or skill handoffs reference the old name and update them.
7. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after the edit.
8. Record files changed, fields changed, and compatibility risk in the active plan or chat summary.

## Guardrails

- Do not rename a user-facing orchestrator without explicit user approval; it breaks any callers that reference it by name.
- Do not use `agents: '*'` or omit `agents` when bounded delegation is the intent.
- Do not use unqualified model strings; always use strings validated by model-routing-and-budget.
- Do not edit unrelated frontmatter fields in the same change; keep edits targeted and reviewable.
- Do not skip validation after the edit; YAML parse failures in VS Code are silent.
- Do not add `agent` to the `tools` list without also populating `agents: [...]` with at least one entry.

## Expected Final Output

- The target `.agent.md` has the updated frontmatter with the requested field changes only.
- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` passes with no errors.
- Files changed, metadata changed, and compatibility risk are recorded in the active plan or chat summary.
