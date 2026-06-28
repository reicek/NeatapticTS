# MCP Lazy-Load Facade — Phase 1 log

**Status:** [DONE]

## Phase 1 — Deploy lightweight lazy-load MCP facades

[DONE] Phase 1 completed and compressed on 2026-06-28.

- Step 01: Chose tool facade over hook/script alternatives; named servers `cortex` and `devtools`; documented contract, adoption map, and stop/reset checkpoint.
- Step 02: Finalized single router-tool contract with `operation`/`args` envelope; created lightweight snapshots.
- Step 03: Wrote red tests in `scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts` (initially failing for the right reason).
- Step 04: Implemented `cortex-facade.mjs`, `devtools-facade.mjs`, and `lazy-facade-core.mjs`; migrated all markdown callers; deleted obsolete snapshot generators; updated `.mcp.json` and `.vscode/mcp.json` to register only the facades (alongside the 4 existing servers); added `cortex` and `devtools` to `knownAgentTools` in `scripts/agent-customization/customization-utils.mjs`; corrected config-authority wording in `README.md` and `plans/mcp-active-binding.plans.md`; created/updated `scripts/agent-customization/gates/devtools-coverage.gate.mjs` and its test; created `.github/skills/devtools/SKILL.md`.
- Step 05: Green validation passed — 51/51 lazy-facade tests with 100% coverage on `lazy-facade-core.mjs`, `cortex-facade.mjs`, and `devtools-facade.mjs`; devtools-coverage gate tests passed; both facade self-checks passed; catalog gates passed after semantic-index rebuild.
- Step 06: Regenerated routing table and semantic index.
- Step 07: Session logging and plan closure; compressed plan to this log and archived pair to `plans/completed/`.

### Validation evidence

- `npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --testPathPatterns=lazy-facade.red.test.ts` → PASS (51/51)
- Coverage guard → 100% statements, 100% branches, 100% functions, 100% lines on `lazy-facade-core.mjs`, `cortex-facade.mjs`, `devtools-facade.mjs`
- `npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --testPathPatterns=scripts/agent-customization/gates/devtools-coverage.gate.test.ts` → PASS (4/4)
- `node scripts/agent-customization/mcp/cortex-facade.mjs --self-check --json` → PASS
- `node scripts/agent-customization/mcp/devtools-facade.mjs --self-check --json` → PASS
- `npx tsc --noEmit -p tsconfig.json` → OK
- `npm run lint` → 0 issues
- `npm run quality:folder -- --folder=scripts/agent-customization/mcp` → OK
- `npm run quality:folder -- --folder=scripts/agent-customization/gates` → OK
- `plan-sync`, `step-packet`, `agent-graph`, `agent-quality`, `routing-table-freshness`, `cortex-index`, `devtools-coverage`, `learning-event`, `tier-enforcement`, `delegate-skill-coverage` gates → PASS

### Files changed

- `.mcp.json` — restored all 6 server registrations (4 untouched + 2 facades).
- `.vscode/mcp.json` — kept in sync with `.mcp.json`.
- `scripts/agent-customization/customization-utils.mjs` — added `cortex` and `devtools` to `knownAgentTools`.
- `README.md` — corrected config-authority wording.
- `plans/mcp-active-binding.plans.md` — corrected config-authority wording.
- Created: `scripts/agent-customization/mcp/cortex-facade.mjs`, `devtools-facade.mjs`, `lazy-facade-core.mjs`, `cortex-tool-snapshot.json`, `devtools-tool-snapshot.json`, `__tests__/lazy-facade.red.test.ts`, `__tests__/mjs-cjs-transformer.cjs`, `gates/devtools-coverage.gate.mjs`, `gates/devtools-coverage.gate.test.ts`, `.github/skills/devtools/SKILL.md`, `plans/MCP_Lazy_Load_Facade.plans.md`.
- `node scripts/semantic-index/build-index.mjs` — semantic corpus index rebuilt.

### Risks / out-of-scope notes

- User must restart Copilot CLI / VS Code so `.mcp.json` / `.vscode/mcp.json` reload and the new facades become active.
- Old heavy server entries are removed; if the restart is skipped, calls to the old keys will fail until the host reloads config.
- Pre-existing `tsconfig.test.json` errors in unrelated areas remain outside this workstream.

### Reopen conditions

- Reopen only if the facade contract, MCP registration, caller migration, or generated artifacts need amendment.
- For new facade targets, prefer a fresh plan that references this archive.
