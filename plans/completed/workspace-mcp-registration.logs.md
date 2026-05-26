# Workspace MCP Registration Log

**Status:** [DONE]

## Audit scope

- Objective: register the repo-owned direct MCP servers in `.vscode/mcp.json`
  with a fixed plan binding and close the resulting workspace-registration lane
  as an archived baseline.
- Bounded implementation surface: `.vscode/mcp.json`,
  `plans/workspace-mcp-registration.plans.md`, `plans/README.md`,
  `plans/Roadmap.md`, and `scripts/agent-customization/mcp/mcp-utils.mjs`.
- Closure pass surfaces: `.vscode/mcp.json`, `plans/README.md`,
  `plans/Roadmap.md`, `plans/completed/README.md`, and the archived plan or
  log pair.

## Durable milestones

### [DONE] Workspace MCP registration and transport repair

- Registered both direct MCP stdio servers with `command: "node"` and the
  fixed `--plan=plans/workspace-mcp-registration.plans.md` binding during the
  active pass.
- Repaired the shared stdio transport utility in
  `scripts/agent-customization/mcp/mcp-utils.mjs` so newline-delimited JSON-RPC
  and Content-Length framing are both accepted and responses preserve the
  inbound framing style.

### [DONE] Validation and documentation closure

- Captured PASS results for packet validation, plan-sync validation, workflow
  MCP self-check, validation MCP self-check, newline initialize probe, and
  Content-Length initialize probe.
- Confirmed that no additional documentation surface was needed beyond
  `plans/README.md` and `plans/Roadmap.md`; no source JSDoc, generated
  READMEs, or `docs/examples/**` surfaces changed.

### [DONE] Archive close and baseline rebinding

- Archived the plan into `plans/completed/` with a compressed [DONE] tracker
  and matching log.
- Updated `.vscode/mcp.json` to point to
  `plans/completed/workspace-mcp-registration.plans.md` so the workspace
  binding continues to reference an existing plan file after closure.
- Updated `plans/README.md`, `plans/Roadmap.md`, and `plans/completed/README.md`
  to reflect the archived baseline and reopen guidance.

## Controls and evidence

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --plan=plans/completed/workspace-mcp-registration.plans.md --json`
- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/completed/workspace-mcp-registration.plans.md --json`
- The active-pass Step 05 evidence preserved six PASS results for packet
  validation, plan-sync validation, workflow MCP self-check, validation MCP
  self-check, newline initialize probe, and Content-Length initialize probe.
- Close-pass note: the archived plan path remains filesystem-valid, but a
  future reopen is required before workspace MCP clients can rely on a live
  `[WIP]` step allow-list again.

## Reopen triggers

- Workspace MCP clients need a live active-step packet rather than the
  archived registration baseline.
- Registration paths or fixed plan-binding rules change.
- Direct MCP self-checks or client initialization behavior regress.
