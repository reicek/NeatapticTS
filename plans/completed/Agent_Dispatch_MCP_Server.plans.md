# Agent Dispatch MCP Server

**Status:** [DONE]

## Scope

Add a repo-owned direct MCP server, `neataptic-dispatch-mcp`, that resolves a
named agent target against `.github/agents/*.agent.md` definitions and returns a
structured dispatch packet only after checking caller-to-target tier and
user-invocable rules. The server does **not** spawn subagents; it only produces
the packet that an orchestrator would hand to a subagent runtime.

Scope is bounded to:

- `scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs` — new server entrypoint.
- `scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs` — red-to-green test using `node:test`, consistent with existing MCP tests.
- `.mcp.json` and `.vscode/mcp.json` — registration of the new server.
- `.github/skills/execute/SKILL.md` — document when orchestrators should consult the new server.

No `src/` library code is touched.

## Final state

Phase 1 completed. All planned artifacts are delivered and validated:

- `scripts/agent-customization/mcp/neataptic-dispatch-mcp.mjs` implements `list_dispatchable_agents`, `build_dispatch_packet`, and `get_dispatch_policy` with self-check.
- `scripts/agent-customization/mcp/__tests__/neataptic-dispatch.red.test.mjs` passes 8/8.
- `.mcp.json` and `.vscode/mcp.json` register the server with matching entrypoint and no `--plan` argument.
- `.github/skills/execute/SKILL.md` documents the dispatch-consultation workflow.
- Detailed milestone evidence moved to `plans/completed/Agent_Dispatch_MCP_Server.logs.md`.
- Plan and log pair archived to `plans/completed/`.

### Phase 1 — Implement and register neataptic-dispatch-mcp [DONE]

[DONE] Phase 1: implemented `neataptic-dispatch-mcp`, registered it in `.mcp.json` / `.vscode/mcp.json`, updated `.github/skills/execute/SKILL.md`, and passed green validation (8/8 red tests, self-check, lint, prettier, tsc). Detailed evidence moved to `plans/completed/Agent_Dispatch_MCP_Server.logs.md`.

#### Step 07 — Session logging and plan closure [DONE]

[DONE] Plan compressed to `plans/Agent_Dispatch_MCP_Server.logs.md`; closed plan and log pair archived to `plans/completed/`.

## Decision Record

No ambiguity decisions required. Open assumptions from the research pass were
resolved by this plan:

- Tool name: `build_dispatch_packet` (instead of `resolve_dispatch_target`) to
  emphasize that the server constructs a ready-to-send packet.
- `caller_tier` is an explicit required argument rather than inferred from the
  caller context.
- No `--plan` argument is needed because the server loads the static agent
  inventory, not a dynamic plan context.
- Test strategy: use `node:test` `.test.mjs` for consistency with existing MCP
  tests and because the Jest `agent-customization-scripts` project excludes `.mjs`.
