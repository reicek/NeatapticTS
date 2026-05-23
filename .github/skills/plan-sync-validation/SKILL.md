---
name: plan-sync-validation
description: 'Validate alignment between NeatapticTS active plans, plans/README.md, and plans/Roadmap.md. Use when registering a plan, changing status markers, adding trigger phrases, or checking that roadmap and index entries agree.'
argument-hint: 'Describe the plan path, expected status, index entry, roadmap placement, and whether this is advisory or blocking.'
user-invocable: false
disable-model-invocation: false
---

# Plan Sync Validation

Use this skill when a plan is created, registered, updated, or closed.

## Workflow

1. Confirm the plan has a top-level `**Status:** [PLANNED|WIP|DONE]` line.
2. Confirm `plans/README.md` has an active guide entry for active plans.
3. Confirm `plans/README.md` has relevant trigger phrases.
4. Confirm `plans/Roadmap.md` has the plan in the correct lane or phase.
5. Keep statuses aligned across all three files.
6. Run `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=<path-to-active-plan>` for this workflow plan. The `--plan` argument is required and must name the active plan path (e.g. `plans/NEATchat.plans.md`), not the completed archive. Omitting it causes the validator to abort with an error.
7. Use `tracker-handoff` for tracker shape and closure rules.

## Sources

- `plans/README.md` defines active plan registration rules.
- `tracker-handoff` owns plan and log structure.