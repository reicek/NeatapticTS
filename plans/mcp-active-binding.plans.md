# MCP Active Binding

**Status:** [WIP]

## Purpose

This plan is a **permanent MCP server binding**, not a workstream tracker. It exists solely to
give `neataptic-workflow-mcp` and `neataptic-validation-mcp` a stable `loadActivePlanContext`
target that is never archived.

**Do not archive or close this file.** When a workstream plan closes, the `--plan` arg in
`.vscode/mcp.json` must point here rather than at the closing plan. That is the only update
needed.

Replace the `--plan` arg only when migrating to a new binding strategy. Keep Phase 1 Step 01
perpetually [WIP] so both MCP servers can start cleanly regardless of which workstream trackers
are open or archived.

## Implementation phases

### Phase 1 — Permanent MCP binding [WIP]

This phase never closes. It provides a stable `[WIP]` context for MCP server startup so the
workflow and validation servers can call `loadActivePlanContext` without depending on any specific
workstream plan.

#### Step 01 — MCP servers operational [WIP]

```yaml
phase: 1
step: 1
agent: "00-helping"
agent_file: ".github/agents/00-helping.agent.md"
status: "[WIP]"
mode: "perpetual"
source_of_truth: "plans/mcp-active-binding.plans.md"
copy_paste: false
next_step: "null"
skills: "mcp-local-server-workflow"
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md
```

**Step objective:** Provide a perpetual stable binding for MCP server startup. This step does not
advance to [DONE]. Both `neataptic-workflow-mcp` and `neataptic-validation-mcp` reference this
step via the `--plan` arg in `.vscode/mcp.json`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md`

**Stop conditions:** Never — this step is perpetually [WIP] by design.

## Validation gates

No workstream-specific gates. Run the self-check commands above to confirm MCP server startup
health at any time.

### Latest validation evidence

- 2026-05-22: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`, plan status `WIP`).

## Handoff query

Stable perpetual binding — no active workstream. Both MCP servers should start without error
when `.vscode/mcp.json` points `--plan` at this file. Run `--self-check` on either server to
verify startup health.
