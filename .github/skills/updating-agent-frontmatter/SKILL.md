---
name: updating-agent-frontmatter
description: 'Use when: updating .agent.md frontmatter, names, tools, or visibility.'
argument-hint: 'Name the agent file, metadata fields to change, visibility target, allowed subagents, model tier, and validation mode.'
user-invocable: false
disable-model-invocation: false
skills:
  - agent-frontmatter-standards
  - model-routing-and-budget
  - skill-frontmatter-standards
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Updating Agent Frontmatter

This skill makes safe, targeted edits to YAML frontmatter in existing `.github/agents/*.agent.md` files. It validates that names remain stable, delegation stays bounded, model strings are qualified, and the updated file passes automated frontmatter validation before the change is recorded.

## When to Use

- Changing the `model` string for an existing agent.
- Adding, removing, or reordering entries in an `agents: [...]` allow-list.
- Updating the `description` to improve trigger precision or fix a false-positive pattern.
- Toggling `user-invocable` when an agent's visibility target changes.
- Adding or updating `handoffs` entries.
- Fixing a frontmatter field that caused a silent loading failure in VS Code.

## When NOT to use

Do NOT use for skill frontmatter updates - use `updating-skill-frontmatter` instead. Do NOT use for frontmatter validation - use `agent-frontmatter-standards` instead.

## Workflow Diagram

```text
Flowchart summary: "Read agent file" → "Identify fields to update"; "Identify fields to update" → "Apply changes"; "Apply changes" → "Validate with agent-frontmatter-standards"; "Validate with agent-frontmatter-standards" → "Pass?"; "Pass?" → "Regenerate routing table" (Yes), "Fix validation errors" (No); "Regenerate routing table" → "Done"; "Fix validation errors" → "Validate with agent-frontmatter-standards"; "Done".
```

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
5. If changing `model`, use a qualified single model string; when repairing a legacy array-valued `model`, preserve the first listed entry unless the user explicitly wants a routing change.
6. If changing the agent `name` or description, assess whether any parent allow-lists or skill handoffs reference the old name and update them.
7. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` after the edit.
8. Record files changed, fields changed, and compatibility risk in the active plan or chat summary.

## Why Each Field Matters

Each frontmatter field controls a specific aspect of agent behavior. The `tier` field determines delegation direction and routing. The `model` field controls which AI model processes the agent. The `tools` field limits what the agent can access. The `agents` allow-list controls sub-delegation. The `skills` field attaches durable procedures. A wrong value in any field can cause silent failures - the agent loads but behaves incorrectly.

## Before/After Frontmatter Examples

**Before (incomplete):**

```yaml
---
name: my-agent
description: Does things
---
```

**After (complete):**

```yaml
---
name: my-agent
description: Use when: validating X for Y boundary. Provides Z.
argument-hint: Name the target file and validation mode.
user-invocable: false
tier: 3
model: anthropic/claude-haiku-3.5
skills:
  - implementation-standards
tools:
  - neataptic-cortex-mcp-search_corpus
---
```

## Decision Tree

```text
Flowchart summary: "Frontmatter field change" → "Field affects routing?"; "Field affects routing?" → "Single-field update + validate" (No (description, argument-hint)), "Name changed?" (Yes (name, agents, model)); "Single-field update + validate" → "Run validate-agent-frontmatter.mjs"; "Name changed?" → "Update field + validate + regenerate routing table" (No), "Update field + check parent allow-lists + regenerate" (Yes); "Run validate-agent-frontmatter.mjs"; "Update field + validate + regenerate routing table" → "Run validate-agent-frontmatter.mjs"; "Update field + check parent allow-lists + regenerate" → "Run validate-agent-frontmatter.mjs".
```

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
