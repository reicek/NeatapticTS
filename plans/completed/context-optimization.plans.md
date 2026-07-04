# Context Optimization Plan

**Status:** [DONE]

## Scope

Reduce the Copilot CLI context-window overhead caused by the 19.8k System prompt, 5.2k `task` tool definition, and 3.8k `skill` tool definition. This workstream delivered repo-side optimizations and deferred runtime-dependent spikes.

## Final state

Phase 1 complete. All seven steps are [DONE] and green validation has passed. Detailed execution history is preserved in `plans/completed/context-optimization.logs.md`.

## Audit summary

- `CLAUDE.md` deleted; unique content migrated to `.github/copilot-instructions.md`, `research-methodology`, `implementation-standards`, and `educational-docs` skills.
- `.github/copilot-instructions.md` split into a 1.6 KB always-loaded facade plus a 38 KB reference section containing the full original policy.
- `task` tool catalog compressed to compact per-agent table entries; detail moved into `.agent.md` files and the generated routing table.
- `skill` tool catalog replaced with a pointer to `.github/agent-skill-routing-table.md`; invalid `devtools` skill references repaired to `chrome-devtools-mcp`.
- Long playbooks moved into canonical skills; Mermaid diagrams pruned from always-loaded prompt and agent/skill files, preserved in generated README sources.
- Two runtime-dependent spikes recorded with owner `00-helping`:
- Lazy-load agent/skill definitions.
- Schema-split `task`/`skill` tool catalogs.

## Reopen conditions

Reopen if Copilot CLI exposes lazy-load or schema-split hooks, or if a new always-loaded guidance file grows beyond 2 KB and needs consolidation.

## Audit log

- Phase 1 compressed and archived by `07-logging` .
- See `plans/completed/context-optimization.logs.md` for full Phase 1 history.
