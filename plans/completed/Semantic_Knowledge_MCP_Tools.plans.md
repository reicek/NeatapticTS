# Semantic Knowledge MCP Tools (Repo Cortex - Layer 2)

**Status:** [DONE]

## Scope

Repo Cortex Layer 2 exposes the SQLite corpus index from the archived Layer 1 foundation as a direct MCP server named `neataptic-cortex-mcp`.

The closed scope covers six repo-static tools for the full NeatapticTS corpus: `search_corpus`, `load_chunk`, `load_document`, `freshness_check`, `index_stats`, and `list_families`. The layer is additive to the existing workflow, validation, and gate MCP servers and does not claim live VS Code, Copilot client, selected-agent, tool-picker, or model-selection state.

## Final state

- [DONE] `scripts/mcp-semantic/repo-cortex-mcp.mjs` implements the stdio MCP entrypoint using the shared MCP utility layer.
- [DONE] `scripts/mcp-semantic/tools/` contains the six semantic tool handlers plus shared SQLite helper behavior.
- [DONE] `scripts/agent-customization/gates/cortex-mcp-smoke.mjs` provides the direct smoke gate for index presence, stats, and representative search.
- [DONE] `.vscode/mcp.json` registers `neataptic-cortex-mcp` as a fourth sibling MCP server without changing existing registrations.
- [DONE] `scripts/mcp-semantic/README.md` documents the server purpose, tool schemas, representative calls and outputs, `CORTEX_DB_PATH`, rebuild guidance, and direct-MCP boundaries.
- [DONE] Phase 7 / Step 07 recorded the compressed done state and confirmed that `Semantic_Knowledge_Browser_Snapshot.plans.md` is ready to proceed as the next Repo Cortex layer.

## Audit summary

- `node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json` passed against `data/semantic-index.sqlite` with 831 documents, 27703 chunks, 9 families, and 3 results for `NEAT activation`.
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/mcp-semantic` passed with 1 suite and 12 tests.
- `node scripts/mcp-semantic/repo-cortex-mcp.mjs --help` passed and listed all six tools.
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Semantic_Knowledge_MCP_Tools.plans.md` passed with 0 errors and 0 warnings before archival.
- Local Jest rejects the legacy singular `--testPathPattern` spelling for this boundary; use `--testPathPatterns` unless another environment proves otherwise.

## Reopen conditions

Reopen this layer only for MCP tool contract changes, semantic index schema drift that affects tool payloads, MCP registration drift, or direct-MCP smoke gate failures. Browser snapshot, IndexedDB cache, and demo loader work belongs to `plans/Semantic_Knowledge_Browser_Snapshot.plans.md`.

## Audit log

See `plans/completed/Semantic_Knowledge_MCP_Tools.logs.md`.
