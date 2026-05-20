---
name: agent-inventory-audit
description: 'Inventory and audit NeatapticTS agents and skills. Use when mapping customizations, counting user-invocable agents, finding model or handoff drift, comparing before/after architecture, or preparing validation evidence.'
argument-hint: 'Describe whether the audit is baseline, after an edit batch, strict target validation, or registration evidence.'
user-invocable: false
disable-model-invocation: false
---

# Agent Inventory Audit

Use this skill before and after customization changes.

## Workflow

1. Run `node scripts/agent-customization/inventory-customizations.mjs --json`.
2. Run `node scripts/agent-customization/validate-agent-frontmatter.mjs --json`.
3. Run `node scripts/agent-customization/validate-skill-frontmatter.mjs --json`.
4. Run `node scripts/agent-customization/validate-agent-graph.mjs --json`.
5. Use `--strict` only when the final eight-agent SDLC surface should already be true.
6. Summarize counts, errors, warnings, and expected pre-migration drift in the active plan.

## Evidence Contract

Report user-invocable agent count, skill count, graph errors, frontmatter errors, and whether strict mode is expected to pass yet.