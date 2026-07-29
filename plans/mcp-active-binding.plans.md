# MCP Active Binding

**Status:** [WIP]

## Purpose

This plan is a **permanent MCP server binding**, not a workstream tracker. It exists solely to
give `neataptic-workflow-mcp` and `neataptic-validation-mcp` a stable `loadActivePlanContext`
target that is never archived.

**Do not archive or close this file.** When a workstream plan closes, the `--plan` arg in
both `.mcp.json` and `.vscode/mcp.json` must point here rather than at the closing plan. That
is the only update needed.

**Canonical MCP config locations.** Both `.mcp.json` (Copilot CLI) and `.vscode/mcp.json`
(VS Code) are authoritative for their respective clients. They must remain in sync so the same
six servers are available in either client. Do not remove server registrations from
`.mcp.json`; doing so causes Copilot CLI's `/mcp show` to report no servers.

Replace the `--plan` arg only when migrating to a new binding strategy. Keep Phase 1 Step 01
perpetually [WIP] so both MCP servers can start cleanly regardless of which workstream trackers
are open or archived.

## Current active plan

The active workstream plan is: `plans/orchestration-fixes.plans.md`

Session override: `data/mcp-session-override.json` → `plans/orchestration-fixes.plans.md`

Previous workstream plans (archived/completed):

- `plans/completed/Agentic_Workflow_Architecture.plans.md`
- `plans/Neon_Shooter_NGE_Demo.plans.md`

## Implementation phases

### Phase 1 — Permanent MCP binding [WIP]

```yaml
phase: 1
title: 'Permanent MCP binding'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/mcp-active-binding.plans.md
copy_paste: true
next_phase: null
skills:
  - plan-alignment
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/mcp-active-binding.plans.md'
  - 'node scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'execute skill present on any agent dispatched from this plan boundary.'
placeholder_steps:
  - 'Step 01 — MCP servers operational'
```

#### Step 01 — MCP servers operational [WIP]

The MCP servers are operational and resolving slice context against the active plan via the
session override. No further action needed unless the binding strategy changes.

## Validation gates

- plan-sync: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md`
- step-packet: `node scripts/agent-customization/gates/step-packet.gate.mjs --json`
- plan-slice-quality: `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json`

### Latest validation evidence

Permanent binding — no slice-level validation evidence. This file is perpetually [WIP] by design.

## Handoff query

Permanent MCP binding file. Active plan: `plans/orchestration-fixes.plans.md` via `data/mcp-session-override.json`.
