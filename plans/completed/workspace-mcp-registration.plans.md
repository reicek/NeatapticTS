# Workspace MCP Registration

**Status:** [DONE]

## Scope

- Register the repo-owned direct MCP stdio servers in `.vscode/mcp.json` with
  a fixed plan binding and preserve the resulting workspace-registration
  baseline as a reopen-only archive.
- Keep the closed baseline scoped to workspace registration, plan-sync
  alignment, and the shared stdio transport repair already captured during the
  bounded implementation pass.

## Final state

- `.vscode/mcp.json` registers both direct MCP stdio servers with
  `command: "node"` and now points to
  `--plan=plans/completed/workspace-mcp-registration.plans.md` so the fixed
  binding remains explicit after archival.
- The bounded implementation baseline closed with registration edits in
  `.vscode/mcp.json`, plan-index updates in `plans/README.md` and
  `plans/Roadmap.md`, and the shared stdio transport repair in
  `scripts/agent-customization/mcp/mcp-utils.mjs`.
- Validation evidence from the active pass is preserved in the matching audit
  log, including the packet validator, plan-sync validator, both direct MCP
  self-checks, and the newline plus Content-Length initialize probes.
- No active workspace-registration work remains in `plans/`; future live
  validation allow-list work must reopen from this archive instead of treating
  the closed baseline as an active packet.

## Audit summary

- The active pass closed with six PASS outcomes: packet validation, plan-sync
  validation, workflow MCP self-check, validation MCP self-check, newline
  initialize probe, and Content-Length initialize probe.
- Documentation closure confirmed that no additional surface was needed beyond
  `plans/README.md` and `plans/Roadmap.md`; no source JSDoc, generated
  READMEs, or `docs/examples/**` surfaces changed.
- The closure pass compressed the tracker, added the same-boundary log,
  updated `.vscode/mcp.json`, `plans/README.md`, `plans/Roadmap.md`, and
  `plans/completed/README.md`, and archived the plan under `plans/completed/`.
- The archived plan path keeps the workspace binding filesystem-valid, but a
  future reopen is still required before workspace MCP clients can rely on a
  live `[WIP]` step allow-list again.

## Reopen conditions

- Workspace MCP clients need a live active-step validation allow-list rather
  than the archived registration baseline.
- Direct MCP registration paths, fixed plan-binding rules, or stdio transport
  framing behavior change.
- A future registration pass needs additional workspace surfaces or a
  different plan-path ownership model.

## Audit log

- Durable completion notes now live in
  [workspace-mcp-registration.logs.md](workspace-mcp-registration.logs.md).
