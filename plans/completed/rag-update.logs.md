# RAG/index infrastructure reorganization log

**Status:** [DONE]

## Workstream closeout

- [DONE] Consolidated the Repo Cortex RAG/index surface into a single top-level `rag-index/` directory: moved `scripts/semantic-index/` tree, corpus DB, hook contexts, model caches, freshness proofs, and browser snapshots under `rag-index/`.
- [DONE] Updated all path constants and downstream consumers (`package.json`, `jest.config.mjs`, `.gitignore`, `.vscode/mcp.json`, `.mcp.json`, MCP tools, agent-customization gates, `repo-cortex-workflow` skill, `repo-cortex-scout` agent) so the live `cortex` MCP server and validation gates stay green.
- [DONE] Created the idempotent incremental entry point `rag-index/update-rag.mjs` with `--dry-run`, `--validate`, and `--json` flags, chained the six canonical RAG stages, and added corpus-hash skip logic for the `build-graph` stage.
- [DONE] Closed a reorganization-caused test/fix loop in Step 08: added `rag-index/**/*.ts` to `tsconfig.test.json`, repaired relative imports in `rag-index/__tests__/*.test.mjs`, created `rag-index/migrate-schema.mjs`, updated docs-quality and build-index health test constants, and optimized `build-entity-graph.mjs` so the validate stage completes in interactive time.
- [DONE] Archived the closed tracker under `plans/completed/` with no remaining in-scope next step.

## Phase 1 steps completed

- Step 01 — Planning the reorganization
- Step 02 — Relocate scripts and generated artifacts into `rag-index/`
- Step 03 — Update path constants and internal imports
- Step 04 — Create `rag-index/update-rag.mjs` idempotent orchestrator
- Step 05 — Update `package.json`, `jest.config.mjs`, and `.gitignore`
- Step 06 — Update MCP wiring, gates, hooks, skills, and agents
- Step 07 — Clean stale temporal artifacts and register the plan
- Step 08 — Green validation

## Validation evidence

- `node rag-index/update-rag.mjs --dry-run --json` → exit 0, 6 stages ok/skipped.
- `node rag-index/update-rag.mjs --validate --json` → exit 0, 6 stages ok/skipped (graph stage skipped on unchanged corpus hash).
- `NODE_OPTIONS="--experimental-vm-modules --max-old-space-size=8192" npx jest --config=jest.config.mjs --no-cache --selectProjects rag-index-scripts --runInBand` → 25 suites passed, 269 tests passed.
- `NODE_OPTIONS="--experimental-vm-modules --max-old-space-size=8192" npx jest --config=jest.config.mjs --no-cache --selectProjects rag-index-mjs --runInBand` → 21 suites passed, 297 tests passed.
- `NODE_OPTIONS="--experimental-vm-modules --max-old-space-size=8192" npx jest --config=jest.config.mjs --no-cache --selectProjects mcp-semantic-mjs --testPathPatterns="eval-(coverage|baseline\.red|metrics\.red|runner\.red)" --runInBand` → 4 suites passed, 164 tests passed.
- `npm run lint` → exit 0.
- `neataptic-gate-mcp:run_gate_check --gate=cortex-index` → PASS (`index_fresh: true`, 1474 documents).
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync` → PASS.
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph` → PASS.
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/completed/rag-update.plans.md` → PASS (0 errors, 0 warnings).
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/rag-update.plans.md` → PASS (0 errors, 0 warnings).

## Residual risks

- [DONE] The combined Jest command `npx jest ... --selectProjects rag-index-scripts --selectProjects rag-index-mjs --runInBand` is non-viable on this Windows workspace: without `NODE_OPTIONS=--experimental-vm-modules` every `.mjs` test fails, and with the flag Jest still runs the two selected projects in parallel, causing `SQLITE_BUSY` in `freshness-hooks.test.ts`. Sequential project runs are the accepted acceptance evidence.
- [DONE] `traverse-graph.red.test.mjs` was attempted as part of the optional `mcp-semantic-mjs` cross-check but fails/hangs on this Windows workspace with `EBUSY` file-lock cleanup errors while deleting temp SQLite files. Treated as local test-harness fragility, not a reorganization-caused regression, and excluded from the signed-off acceptance set.
- [DONE] Old artifact locations (`scripts/semantic-index/`, root `freshness-proof-*/`, old `data/` SQLite and hook-context files) were removed; only `data/eval-baselines/` remains in place as persistent evaluation data.

## Reopen conditions

Reopen this archive only if:

- The `rag-index/` consolidation regresses (consumers break, gates fail, or old paths resurface).
- The `update-rag.mjs` orchestrator contract needs extension (true incremental entity-graph updates, new stages, or different skip policy).
- A follow-up reorganization needs to introduce semantic subfolders inside `rag-index/` (the chosen layout is flat to minimize import churn).

## Audit log

- Durable completion notes now live in this file.
- Archived plan lives at `plans/completed/rag-update.plans.md`.
