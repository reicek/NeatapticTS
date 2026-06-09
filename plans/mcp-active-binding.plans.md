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
goal: 'helping'
status: '[WIP]'
mode: 'perpetual'
source_of_truth: 'plans/mcp-active-binding.plans.md'
copy_paste: false
next_step: 'null'
skills: 'mcp-local-server-workflow'
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
- 2026-05-25: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`, plan status `WIP`).
- 2026-05-25: `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json` -> PASS (`ok: true`, `0 errors`, `0 warnings`; `run_gate_check`, `query_tier_graph`, and `query_customization_routing_table` remained healthy after `cortex-first-search` gate registration).
- 2026-05-25: `node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json` -> EXPECTED ENVIRONMENT FAIL (`pass: false`; semantic index stale at `data/semantic-index.sqlite`; actionable `fixHint`: `node scripts/semantic-index/build-index.mjs`; not treated as an MCP wiring defect because `corpus_mcp_alive: true` and the gate honestly reports a prerequisite miss rather than claiming runtime tool-order validation).
- 2026-05-25: `node scripts/semantic-index/session-start-index.mjs --json` -> PASS (`databasePath: C:\NeatapticTS\data\semantic-index.sqlite`, `touched: 1257`, `contentChanged: 0`, `onDiskMissing: 0`, `buildPassRan: true`, `buildPassExitCode: 0`, `fatalError: null`), confirming the session-start refresh succeeds locally and the incremental build path stays green on an unchanged workspace.
- 2026-05-25: `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` -> PASS (`pass: true`; `index_documents: 1257`; `index_fresh: true`; `corpus_mcp_alive: true`; `workflow_mcp_alive: true`; `snapshot_age_seconds: 5209`), confirming the session-start refresh clears the index freshness gate in the current workspace.
- 2026-05-26: targeted lint-cleanup pass for `examples/neatChat`, `examples/shared/semantic`, and `src/architecture/network/onnx` -> focused `npx eslint <touched files...>` PASS after removing dead locals/imports, replacing obsolete `require()` red-test shims with direct or dynamic imports, and replacing unsafe `?.!` test assertions with helper-backed presence checks. Touched `src/` files requiring `/coverage-guard` follow-up: `src/architecture/network/onnx/export/network.onnx.export-build.emit.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export-orchestrators.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export.test.ts`, `src/architecture/network/onnx/import/network.onnx.import-external.utils.ts`, `src/architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts`, `src/architecture/network/onnx/schema/network.onnx.schema.tensor-data.utils.ts`, `src/architecture/network/onnx/validate/network.onnx.validate.ts`. Expected next validation: `npm run lint`.
- 2026-05-26: `npm run lint ; Write-Output "EXIT:$LASTEXITCODE"` -> PASS (`EXIT:0`), confirming the post-fix lint gate stays green for the reported cleanup set.
- 2026-05-26: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`, plan status `WIP`) after recording the latest validation state.
- 2026-05-26: `npx jest --config=jest.config.mjs --no-cache --coverage --runInBand --testPathPatterns=src/architecture/network/onnx` -> PASS. Coverage guard stayed at 100% statements, branches, functions, and lines for runtime files `src/architecture/network/onnx/export/network.onnx.export-build.emit.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export-orchestrators.utils.ts`, `src/architecture/network/onnx/import/network.onnx.import-external.utils.ts`, `src/architecture/network/onnx/schema/network.onnx.schema.tensor-data.utils.ts`, and `src/architecture/network/onnx/validate/network.onnx.validate.ts`. `src/architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts` remained a type-only file with no runtime instrumentation obligation, and `src/architecture/network/onnx/export/network.onnx.export.test.ts` remained a test file rather than a production coverage target.
- 2026-05-26: first `npm run test:silent` attempt hit a transient failure in `scripts/semantic-index/docs-quality/docs-quality.metrics.test.ts` (`1 failed, 426 passed, 427 total`). Focused rerun of `npx jest --config=jest.config.mjs --runInBand scripts/semantic-index/docs-quality/docs-quality.metrics.test.ts` passed (`1 suite`, `4 tests`), and the second `npm run test:silent` rerun also passed cleanly (`427 passed, 427 total`; `4654 passed, 4654 total`). No residual blocker remained after rerun.
- 2026-05-29: targeted Repo Cortex CI hardening updated `scripts/agent-customization/gates/cortex-index.gate.test.ts` to build the browser snapshot before running `cortex-index.gate.mjs`, updated `scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts` to provision the semantic index and browser snapshot in `beforeAll`, and updated `.github/workflows/deploy-pages.yml` to clear `~/.cache/puppeteer` before `npx puppeteer browsers install chrome`. Validation: `npx tsc --noEmit -p tsconfig.test.json` -> PASS (`exit 0`); `npx jest --config=jest.config.mjs --no-cache --testPathPattern="cortex-index.gate.test" --runInBand` -> FAIL (`testPathPattern` option renamed by current Jest CLI); corrective rerun `npx jest --config=jest.config.mjs --no-cache --testPathPatterns="cortex-index.gate.test" --runInBand` -> PASS (`1 suite`, `1 test`); `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`).
- 2026-06-04: runtime-enforcement blocker fix aligned strict-action hooks with the session override plan chain. Validation: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts|scripts/agent-customization/mcp/__tests__/mcp.red.test.mjs"` -> PASS for the runtime enforcement hook suite; `node --test scripts/agent-customization/mcp/__tests__/mcp.red.test.mjs` -> PASS (`8 tests`, `0 failed`), including the session-override precedence regression.
- 2026-06-08: Orchestration_System_Optimization Phase 4 Step 05 validation — `tier-enforcement.gate.mjs --json` -> PASS (`ok: true`, `issueCount: 0`, `byTier: {1: 8, 2: 11, 3: 38, 4: 4}`, `userInvocableTotal: 8`); `validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md` -> PASS (`0 errors`, `0 warnings`, status: `WIP`). Flow specialist references validated with zero tier violations.

## Handoff query

Stable perpetual binding — no active workstream. Both MCP servers should start without error
when `.vscode/mcp.json` points `--plan` at this file. Run `--self-check` on either server to
verify startup health.
