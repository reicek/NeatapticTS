# MCP Server Validation and Hardening

**Status:** [DONE]

## Scope

This workstream hardened the three workspace MCP servers after restart-oriented validation exposed plan-binding, JSON-RPC method, and allow-list metadata gaps. The bounded surface stayed within `.vscode/mcp.json`, `scripts/agent-customization/mcp/**`, tracker files, and the learning log. The lane is now closed because the MCP binding blocker was cleared by rebinding the workspace servers to `plans/mcp-active-binding.plans.md`, which makes this tracker safe to archive.

## Final state

- The workflow and validation MCP servers no longer depend on this tracker for startup because `.vscode/mcp.json` now points both servers at `plans/mcp-active-binding.plans.md`.
- `scripts/agent-customization/mcp/mcp-utils.mjs` now returns defined empty results for `resources/list` and `prompts/list`, and the bounded MCP red-test surface remains green.
- The active tracker has been replaced by this compressed archive; future work should reopen from this baseline instead of restoring the deleted active packet.

## Key decisions

- Phase 1: answer the deferred questions up front, confirm the archived-plan binding defect, and packetize the remaining phases before touching MCP behavior.
- Phase 2: reproduce the runtime gaps directly and treat archived `[DONE]` plan binding, unsupported `resources/list` or `prompts/list`, and self-check coverage as the controlling hardening surface.
- Phase 3: add six focused `node:test` contracts to lock the failing MCP behaviors before implementation.
- Phase 4: fix the workspace plan binding and JSON-RPC empty-result behavior, then normalize step-packet `validation:` metadata so the validation MCP self-check can execute against a real allow-list.
- Phase 5: close the lane with a full green checklist, preserve AC 1, 4, and 9 as manual-only client checks, and archive only after the workspace binding moved to `plans/mcp-active-binding.plans.md`.

## Changed files

- [.vscode/mcp.json](../../.vscode/mcp.json)
- [scripts/agent-customization/mcp/mcp-utils.mjs](../../scripts/agent-customization/mcp/mcp-utils.mjs)
- [scripts/agent-customization/mcp/**tests**/mcp.red.test.mjs](../../scripts/agent-customization/mcp/__tests__/mcp.red.test.mjs)
- [.github/ai-learning/learning-log.jsonl](../../.github/ai-learning/learning-log.jsonl)
- [plans/README.md](../README.md)
- [plans/Roadmap.md](../Roadmap.md)
- [plans/completed/README.md](README.md)

## Reopen conditions

- MCP startup, JSON-RPC framing, or tool-surface behavior regresses on any of the three servers.
- The workspace binding model changes again and `plans/mcp-active-binding.plans.md` is no longer the correct stable MCP target.
- A future pass needs new live-client validation beyond the manual-only acceptance-criteria boundaries recorded in the audit log.

## Audit log

- Durable completion notes now live in [MCP_Server_Validation_and_Hardening.logs.md](MCP_Server_Validation_and_Hardening.logs.md).
