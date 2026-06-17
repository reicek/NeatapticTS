---
name: updating-skill-frontmatter
description: 'Use when: updating SKILL.md YAML frontmatter, names, descriptions, argument hints, user-invocable flags, compatibility notes, or resource links.'
argument-hint: 'Name the skill folder, frontmatter fields to update, desired trigger scope, visibility decision, and validation command.'
user-invocable: false
disable-model-invocation: false
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Updating Skill Frontmatter

This skill makes safe, targeted edits to YAML frontmatter in existing `.github/skills/<skill-name>/SKILL.md` files. It preserves the skill's existing intent and name stability, ensures descriptions remain within the 1024-character limit, and validates the result before recording the change.

## When to Use

- Improving a description that is triggering too broadly or not triggering reliably.
- Adding or refining an `argument-hint` to help callers shape their task packets.
- Toggling `user-invocable` when the skill's intended audience changes.
- Fixing a `name` mismatch between the field value and the folder name.
- Updating compatibility notes or local resource links after a skill refactor.
- Preparing a skill for a trigger eval pass by stabilizing its description first.

## Task Packet

Include the skill folder name, the specific frontmatter fields to change, and whether strict validation is expected to pass after the edit.

```text
Use updating-skill-frontmatter for <skill-folder-name>.
Fields to change: <list of field names>
Old values: <what they are now>
New values: <what they should become>
Trigger scope intent: <what tasks should activate this skill>
Validate with: node scripts/agent-customization/validate-skill-frontmatter.mjs --json --strict
```

## Required Workflow

1. Read the current frontmatter of the target `SKILL.md` before making any changes.
2. Confirm the `name` field matches the folder name exactly (lowercase kebab-case); fix any mismatch first.
3. Preserve the skill's existing intent unless the user explicitly requested a split, rename, or deprecation.
4. Apply the targeted field change; do not alter unrelated frontmatter fields in the same edit.
5. Keep the description under 1024 characters; if the revised description exceeds this, restructure rather than truncate.
6. Add or refine `argument-hint` when the skill is broad enough that callers need guidance on what to include.
7. Run `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --strict` after the edit.
8. If the description changed significantly, flag the skill for a trigger eval pass using `skill-description-evals`.
9. Record skill files changed, trigger-scope decision, validation evidence, and any follow-up eval needs in the active plan.

## Guardrails

- Do not rename a skill without updating every reference to its old name in agent allow-lists, skill handoffs, and CLAUDE.md.
- Do not write a description so broad that it triggers on tasks belonging to sibling skills.
- Do not omit `argument-hint` for skills that have a wide enough scope to benefit from task shaping.
- Do not exceed 1024 characters in the description; this is an Agent Skills hard limit.
- Do not skip validation after the edit; frontmatter errors can be silent.
- Do not edit unrelated frontmatter fields in the same change; keep edits targeted and reviewable.

## Expected Final Output

- The target `SKILL.md` has the updated frontmatter with the requested field changes only.
- `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --strict` passes with no errors.
- Trigger-scope decision, validation evidence, and follow-up eval needs recorded in the active plan.
