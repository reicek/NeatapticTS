# Semantic Knowledge MCP Tools Log

**Status:** [DONE]

## Repo Cortex Layer 2 closeout

- [DONE] Implemented `neataptic-cortex-mcp` as a repo-static MCP server over the Layer 1 SQLite corpus index, exposing search, chunk/document loading, freshness, stats, and family enumeration tools.
- [DONE] Added the `cortex-mcp-smoke` gate and additive `.vscode/mcp.json` registration without changing existing workflow, validation, or gate MCP servers.
- [DONE] Documented the server in `scripts/mcp-semantic/README.md`, including tool schemas, representative calls and outputs, index configuration, rebuild guidance, and bounded direct-MCP claims.
- [DONE] Validation evidence: smoke gate passed with 831 documents, 27703 chunks, 9 families, and 3 `NEAT activation` results; focused Jest passed with 1 suite / 12 tests; server `--help` passed; plan-sync passed with 0 errors / 0 warnings before archival.
- [DONE] Browser Snapshot readiness: `plans/Semantic_Knowledge_Browser_Snapshot.plans.md` remains [PLANNED] and is ready to start Step 01 as the next Repo Cortex layer now that Layer 1 and Layer 2 are complete.
- [DONE] Session compression confirmation: the completed Repo Cortex Layer 2 session record is filed under `plans/completed/` as the compressed plan/log pair for future audit lookup.

## Residual notes

- The local Jest CLI rejects `--testPathPattern` and requires `--testPathPatterns` for the `scripts/mcp-semantic` focused test boundary.
- Generated browser snapshot output remains out of scope for this layer and must be handled by the next plan without hand-editing `docs/assets/semantic-snapshot.json`.
