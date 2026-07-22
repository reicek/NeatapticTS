# MCP Active Binding

**Status:** [WIP]

## Current state

Claim: 01-planning @ 2026-07-19T17:02:00Z  
Claim: 04-implementing @ 2026-07-20T12:00:00Z  
Claim: 04-implementing @ 2026-07-21T21:36:56Z

Active workstream tracker:
`plans/Neon_Shooter_NGE_Demo.plans.md`.

The Cortex Orchestration Single Source of Truth plan is complete and archived
to `plans/completed/`. The Neon Shooter NGE Demo plan is now the active
workstream. The MCP binding keeps `neataptic-workflow-mcp` and
`neataptic-validation-mcp` pointed at the Neon Shooter plan so step-packet and
validation allow-lists resolve without prompt input.

```yaml
PlanUpdate:
  changed_files:
    - data/mcp-session-override.json
    - .vscode/mcp.json
    - plans/mcp-active-binding.plans.md
    - plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md
    - plans/Neon_Shooter_NGE_Demo.plans.md
    - plans/README.md
    - plans/Roadmap.md
  preflight:
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md'
  validation:
    - command: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
      expected_exit: 0
    - command: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
      expected_exit: 0
  gates:
    - 'plan-sync: PASS'
    - 'step-packet: PASS'
    - 'plan-slice-quality: PASS'
  rollback:
    - 'git checkout -- data/mcp-session-override.json'
    - 'git checkout -- .vscode/mcp.json'
    - 'git checkout -- plans/mcp-active-binding.plans.md'
    - 'git checkout -- plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md'
    - 'git checkout -- plans/Neon_Shooter_NGE_Demo.plans.md'
    - 'git checkout -- plans/README.md'
    - 'git checkout -- plans/Roadmap.md'
  next: 'Continue monitoring active workstream in plans/Cortex_Orchestration_Single_Source_of_Truth.plans.md; keep this binding file pointing at the current open tracker.'
```

### get_slice_context default compact flip

Flipped `neataptic-workflow-mcp/get_slice_context` so compact is the default
response and the full assembled context window is opt-in via `full: true`.

```yaml
PlanUpdate:
  slice_id: mcp-get-slice-context-default-compact
  changed_files:
    - scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs
    - scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts
    - scripts/agent-customization/mcp/cortex-tier-tool.mjs
    - scripts/agent-customization/mcp/cortex-tier-tool.direct.test.mjs
    - scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts
    - scripts/agent-customization/mcp/cortex-tool-snapshot.json
    - files/mcp-facade/cortex-tool-snapshot.json
    - plans/mcp-active-binding.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts scripts/agent-customization/mcp/cortex-tier-tool.mjs scripts/agent-customization/mcp/cortex-tier-tool.test.ts scripts/agent-customization/mcp/cortex-tier-tool.direct.test.mjs scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts scripts/agent-customization/mcp/cortex-tool-snapshot.json files/mcp-facade/cortex-tool-snapshot.json plans/mcp-active-binding.plans.md'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/mcp/cortex-tier-tool.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/mcp/cortex-tier-tool.direct.test.mjs'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts'
  rollback:
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs'
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts'
    - 'git checkout -- scripts/agent-customization/mcp/cortex-tier-tool.mjs'
    - 'git checkout -- scripts/agent-customization/mcp/cortex-tier-tool.direct.test.mjs'
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts'
    - 'git checkout -- scripts/agent-customization/mcp/cortex-tool-snapshot.json'
    - 'git checkout -- files/mcp-facade/cortex-tool-snapshot.json'
    - 'git checkout -- plans/mcp-active-binding.plans.md'
  next: 'Run 05-green-testing on the focused MCP test files and attach coverage-guard evidence.'
```

### get_slice_context response quality pass

Applied six tightly-coupled fixes to `neataptic-workflow-mcp:get_slice_context`
so the default compact response stays under the 16 KB wire envelope, carries
relevant RAG chunks with non-empty text, and no longer duplicates the full JSON
payload in `content[0].text`.

```yaml
PlanUpdate:
  slice_id: mcp-get-slice-context-quality-2026-07-21
  changed_files:
    - scripts/agent-customization/mcp/mcp-utils.mjs
    - scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs
    - scripts/mcp-semantic/tools/search-context.mjs
    - scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/mcp/mcp-utils.mjs scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs scripts/mcp-semantic/tools/search-context.mjs scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=agent-customization/mcp/neataptic-workflow-mcp'
    - 'npm run jest:mjs -- --testPathPatterns=search-context.direct'
  validation:
    - command: 'node scripts/agent-customization/gates/cortex-index.gate.mjs --json'
      expected_exit: 0
      result: 'PASS — cortex-index: pass'
  live_repro:
    slice: '02-red-phase2'
    plan: 'plans/Neon_Shooter_NGE_Demo.plans.md'
    envelope_bytes: 10524
    chunks_count: 4
    top_chunk_path: 'examples/neatenstein/browser-entry/host/game/waves.test.ts'
    all_chunk_texts_non_empty: true
  rollback:
    - 'git checkout -- scripts/agent-customization/mcp/mcp-utils.mjs'
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs'
    - 'git checkout -- scripts/mcp-semantic/tools/search-context.mjs'
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts'
  next: 'Run 05-green-testing on focused tests and attach coverage-guard evidence.'
```

**VALIDATION_EVIDENCE:**

- tsc: `npx tsc --noEmit -p tsconfig.json` → OK
- lint: `npm run lint` → 0 errors (50 unrelated pre-existing warnings)
- prettier: `npx prettier --check <changed-files>` → OK
- folder quality (MCP): `node scripts/folder-quality-metrics.mjs --folder=scripts/agent-customization/mcp --json` → pass
- folder quality (semantic tools): `node scripts/folder-quality-metrics.mjs --folder=scripts/mcp-semantic/tools --json` → pass
- focused test (workflow): `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=agent-customization/mcp/neataptic-workflow-mcp` → 53/53 PASS
- focused test (search-context): `npm run jest:mjs -- --testPathPatterns=search-context.direct` → 41/41 PASS
- live repro (`02-red-phase2`): envelope 10,524 bytes, 4 chunks, top chunk `examples/neatenstein/browser-entry/host/game/waves.test.ts`, all chunk texts non-empty
- plan-sync: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` → pass
- agent-graph: `node scripts/agent-customization/gates/agent-graph.gate.mjs --json` → pass
- cortex-index: `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` → pass

**VALIDATION_EVIDENCE:**

- tsc: `npx tsc --noEmit -p tsconfig.json` → OK
- lint: `npm run lint` → 0 issues in touched files (unrelated pre-existing warnings remain)
- prettier: `npx prettier --check <touched-files>` → OK
- plan-sync: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` → pass
- focused test: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/mcp/neataptic-workflow-mcp.test.ts` → 62/62 PASS

#### mcp-get-slice-context-default-compact green validation (05-green-testing)

```json
{
  "pass": true,
  "slice_id": "mcp-get-slice-context-default-compact",
  "evidence": {
    "neataptic-workflow-mcp.test.ts": "PASS — 62/62 tests after compact-default assertion drift fixed (04-implementing preflight)",
    "cortex-tier-tool.direct.test.mjs": "PASS — 5/5 tests (required NODE_OPTIONS='--experimental-vm-modules')",
    "neataptic-gate-mcp.direct.test.ts": "PASS — 21/21 tests",
    "tsc_main": "PASS — npx tsc --noEmit -p tsconfig.json exit 0",
    "tsc_test": "FAIL — pre-existing errors outside slice (examples/racing_curriculum/*, src/architecture/network/acceleration.network-api.test.ts)",
    "lint": "PASS — npm run lint exit 0; warnings only in untouched files",
    "prettier": "PASS — all 8 touched files conform",
    "plan-sync": "PASS — registered in README/Roadmap",
    "code-coverage": "PENDING — 05-green-testing to attach coverage-guard evidence"
  },
  "fixHint": "Stale assertions in neataptic-workflow-mcp.test.ts were repaired by resolving slices against temporary plan fixtures and the active Neon_Shooter plan instead of the archived Cortex_Orchestration plan. The compact-default implementation remains unchanged.",
  "owner": "05-green-testing"
}
```

### Slice-quality gate registration

Added a new Tier-1 gate `plan-slice-quality` that enforces the 4-hour slice
estimate limit on active [WIP] step packets. The gate is registered in
`neataptic-gate-mcp`, the limit is also enforced inside `step-packet.gate.mjs`,
and the oversized test fixture (`estimate_hours: 6`) was corrected to `3`.

```yaml
PlanUpdate:
 changed_files:
 - scripts/agent-customization/gates/plan-slice-quality.gate.mjs
 - scripts/agent-customization/mcp/neataptic-gate-mcp.mjs
 - scripts/agent-customization/gates/step-packet.gate.mjs
 - scripts/agent-customization/gates/step-packet.gate.test.ts
 - plans/mcp-active-binding.plans.md
 preflight:
 - 'npx tsc --noEmit -p tsconfig.json'
 - 'npm run lint'
 - 'npx prettier --check <touched-files>'
 validation:
 - command: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json'
 expected_exit: 0
 result: 'PASS — no WIP slices exceed 4-hour limit'
 - command: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json'
 expected_exit: 0
 result: 'PASS — all WIP packets conform'
 - command: 'npx jest scripts/agent-customization/gates/step-packet.gate.test.ts --no-coverage'
 expected_exit: 0
 result: 'PASS — 24 tests'
 gates:
 - 'plan-sync: PASS'
 - 'step-packet: PASS'
 - 'plan-slice-quality: PASS'
 rollback:
 - 'git checkout -- scripts/agent-customization/gates/plan-slice-quality.gate.mjs'
 - 'git checkout -- scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
 - 'git checkout -- scripts/agent-customization/gates/step-packet.gate.mjs'
 - 'git checkout -- scripts/agent-customization/gates/step-packet.gate.test.ts'
 next: 'User should create PR; then 05-green-testing can verify the gate in CI if needed.'
```

### Code-coverage gate 100% green tests

Backfills green tests for the `code-coverage` and `merge-coverage-summaries`
Tier-1 utility gate files so they reach 100% lines/functions/statements/branches
coverage. The CLI entry path of `merge-coverage-summaries.mjs` is marked with an
Istanbul ignore comment because it cannot be re-evaluated under Jest's ESM module
cache; it remains exercised by the existing subprocess CLI tests.

```yaml
PlanUpdate:
  slice_id: coverage-gates-100-green-2026-07-08
  changed_files:
    - scripts/agent-customization/gates/code-coverage.gate.mjs
    - scripts/agent-customization/gates/code-coverage.gate.test.ts
    - scripts/agent-customization/gates/merge-coverage-summaries.mjs
    - scripts/agent-customization/gates/merge-coverage-summaries.gate.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/gates/code-coverage.gate.test.ts scripts/agent-customization/gates/merge-coverage-summaries.mjs scripts/agent-customization/gates/merge-coverage-summaries.gate.test.ts plans/mcp-active-binding.plans.md'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --coverage --testPathPatterns="scripts/agent-customization/gates/(code-coverage|merge-coverage-summaries).gate.test.ts"'
  artifacts:
    - artifacts/implementing/20260708T003956-coverage-gates.json
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --coverage --testPathPatterns="scripts/agent-customization/gates/(code-coverage|merge-coverage-summaries).gate.test.ts"'
      expected_exit: 0
      result: 'PASS — code-coverage.gate.mjs and merge-coverage-summaries.mjs all 100%'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md'
      expected_exit: 0
      result: 'PASS — 0 errors, 0 warnings'
  gates:
    - 'plan-sync: PASS'
    - 'code-coverage gate coverage: 100/100/100/100'
    - 'merge-coverage-summaries coverage: 100/100/100/100'
  rollback:
    - 'git checkout -- scripts/agent-customization/gates/code-coverage.gate.test.ts'
    - 'git checkout -- scripts/agent-customization/gates/merge-coverage-summaries.mjs'
    - 'git checkout -- scripts/agent-customization/gates/merge-coverage-summaries.gate.test.ts'
    - 'git checkout -- plans/mcp-active-binding.plans.md'
  next: '05-green-testing should run the focused coverage suite and confirm coverage-guard evidence.'
```

#### VALIDATION_EVIDENCE — coverage-gates-100-green-2026-07-08

```json
{
  "pass": true,
  "slice_id": "coverage-gates-100-green-2026-07-08",
  "evidence": {
    "focused_tests": "PASS — npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=\"scripts/agent-customization/gates/(code-coverage|merge-coverage-summaries).gate.test.ts|scripts/agent-customization/mcp/neataptic-gate-mcp\" (4 suites, 61 tests)",
    "code_coverage_gate": "PASS — code-coverage.gate.mjs 100/100/100/100 (lines/functions/statements/branches)",
    "merge_coverage_summaries": "PASS — merge-coverage-summaries.mjs 100/100/100/100 (lines/functions/statements/branches)",
    "code_coverage_gate_check": "PASS — node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=scripts/agent-customization/gates/code-coverage.gate.mjs,scripts/agent-customization/gates/merge-coverage-summaries.mjs",
    "mcp_gate_self_check": "PASS — node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json (ok: true, 0 issues)",
    "tsc": "PASS — npx tsc --noEmit -p tsconfig.json",
    "lint": "PASS — npm run lint, 0 issues",
    "prettier": "PASS — npx prettier --check on touched files",
    "plan_sync": "PASS — node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/mcp-active-binding.plans.md",
    "agent_graph": "PASS — node scripts/agent-customization/gates/agent-graph.gate.mjs --json",
    "artifact": "artifacts/implementing/20260708T003956-coverage-gates.json"
  },
  "fixHint": null,
  "owner": "05-green-testing"
}
```

#### MCP gate-server cleanup (incidental)

Removed two stray `console.error('DEBUG ...')` lines from `scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` and added focused MCP tests in `neataptic-gate-mcp.test.ts` and `neataptic-gate-mcp.direct.test.ts`. The tests pass and are formatted; they are outside the original coverage-gate slice but keep the binding server green.

```yaml
PlanUpdate:
  slice_id: mcp-binding-cleanup-2026-07-08
  changed_files:
    - scripts/agent-customization/mcp/neataptic-gate-mcp.mjs
    - scripts/agent-customization/mcp/neataptic-gate-mcp.test.ts
    - scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check scripts/agent-customization/mcp/neataptic-gate-mcp.mjs scripts/agent-customization/mcp/neataptic-gate-mcp.test.ts scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts plans/mcp-active-binding.plans.md'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="scripts/agent-customization/mcp/neataptic-gate-mcp"'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="scripts/agent-customization/mcp/neataptic-gate-mcp"'
      expected_exit: 0
      result: 'PASS — 2 suites, 16 tests'
  rollback:
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-gate-mcp.mjs'
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-gate-mcp.test.ts'
    - 'git checkout -- scripts/agent-customization/mcp/neataptic-gate-mcp.direct.test.ts'
  next: '05-green-testing can run the focused MCP tests with the coverage suite.'
```

### Six failing-test fix pass

Fixes the four failing suites reported by the user without adding logging.

```yaml
PlanUpdate:
slice_id: mcp-active-binding-bugfix-2026-07-07
changed_files:
  - .vscode/mcp.json
  - src/architecture/network/gpu/network.gpu.activate.ts
  - scripts/mcp-semantic/tools/submit-feedback.mjs
  - scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts
  - scripts/agent-customization/gates/cortex-index.gate.test.ts
  - scripts/mcp-semantic/__tests__/submit-feedback.harden.red.test.mjs
preflight:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npx tsc --noEmit -p tsconfig.test.json'
  - 'npm run lint'
  - 'npx prettier --check .vscode/mcp.json src/architecture/network/gpu/network.gpu.activate.ts scripts/mcp-semantic/tools/submit-feedback.mjs scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts scripts/agent-customization/gates/cortex-index.gate.test.ts scripts/mcp-semantic/__tests__/submit-feedback.harden.red.test.mjs'
tests_for_green:
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --testPathPattern="repo-cortex-mcp.red.test.ts|network.gpu.batch-evaluation.test.ts|submit-feedback.test.ts|cortex-index.gate.test.ts"'
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --selectProjects mcp-semantic-scripts'
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --selectProjects agent-customization-scripts'
  - '$env:NODE_OPTIONS="--experimental-vm-modules"; npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --selectProjects jest:mjs'
  - '$env:NODE_OPTIONS="--experimental-vm-modules"; npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --selectProjects jest:esm-ts'
  - 'npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --selectProjects default'
validation:
  - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md'
    expected_exit: 0
    result: 'PASS — 0 errors, 0 warnings'
  - command: 'npx tsc --noEmit -p tsconfig.json'
    expected_exit: 0
    result: 'PASS — no TS errors'
  - command: 'npx tsc --noEmit -p tsconfig.test.json'
    expected_exit: 0
    result: 'PASS — no TS errors'
  - command: 'npm run lint'
    expected_exit: 0
    result: 'PASS — 0 issues'
  - command: 'npx prettier --check .vscode/mcp.json src/architecture/network/gpu/network.gpu.activate.ts scripts/mcp-semantic/tools/submit-feedback.mjs scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts scripts/agent-customization/gates/cortex-index.gate.test.ts scripts/mcp-semantic/__tests__/submit-feedback.harden.red.test.mjs plans/mcp-active-binding.plans.md'
    expected_exit: 0
    result: 'PASS — all matched files use Prettier code style'
gates:
  - 'plan-sync: PASS — node scripts/agent-customization/gates/plan-sync.gate.mjs --json --plan=plans/mcp-active-binding.plans.md'
  - 'agent-graph: PASS — node scripts/agent-customization/gates/agent-graph.gate.mjs --json'
  - 'learning-event: PASS — node scripts/agent-customization/gates/learning-event.gate.mjs --json'
rollback:
  - 'git checkout -- .vscode/mcp.json src/architecture/network/gpu/network.gpu.activate.ts scripts/mcp-semantic/tools/submit-feedback.mjs scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts scripts/agent-customization/gates/cortex-index.gate.test.ts scripts/mcp-semantic/__tests__/submit-feedback.harden.red.test.mjs'
next: 'Handoff to 05-green-testing to run the focused suites and confirm no regressions; then user creates PR.'
```

#### VALIDATION_EVIDENCE — mcp-active-binding-bugfix-2026-07-07

```json
{
  "pass": true,
  "slice_id": "mcp-active-binding-bugfix-2026-07-07",
  "evidence": {
    "focused_suites": "PASS — 4 suites, 28 tests",
    "coverage_summary": {
      "statements": 100,
      "branches": 100,
      "functions": 100,
      "lines": 100
    },
    "gpu_real_device_gate": "PASS — docs/browser-tests/webgpu-inference-smoke.html, vendor=nvidia, architecture=lovelace, maxAbsDiff=3.18e-9, browserVisibility=visible-foreground",
    "mcp_semantic_scripts": "PASS — 9 suites, 59 tests",
    "agent_customization_scripts": "PASS — 17 suites, 124 tests",
    "default_gpu_tests": "PASS — 11 suites, 221 tests",
    "mcp_semantic_mjs": "submit-feedback.harden.red.test.mjs PASS (9 tests). Full project run crashed during process teardown with onnxruntime-node cleanup-hook assertion (exit 134); tests themselves all passed and crash is unrelated to changed files.",
    "typecheck": "PASS — tsc --noEmit for tsconfig.json and tsconfig.test.json",
    "lint": "PASS — npm run lint, 0 issues",
    "prettier": "PASS — all touched files use Prettier style",
    "gates": "plan-sync PASS, agent-graph PASS, step-packet PASS, plan-slice-quality PASS, cortex-index PASS"
  },
  "fixHint": null,
  "owner": "05-green-testing"
}
```

## Purpose

This plan is a **permanent MCP server binding**, not a workstream tracker. It exists solely to
give `neataptic-workflow-mcp` and `neataptic-validation-mcp` a stable `loadActivePlanContext`
target that is never archived.

**Do not archive or close this file.** When a workstream plan closes, the `--plan` arg in
both `.mcp.json` and `.vscode/mcp.json` must point here rather than at the closing plan. That
is the only update needed.

**Canonical MCP config locations.** Both `.mcp.json` (Copilot CLI) and `.vscode/mcp.json`
(VS Code) are authoritative for their respective clients. They must remain in sync so the same
six servers are available in either client. Do not remove server registrations from
`.mcp.json`; doing so causes Copilot CLI's `/mcp show` to report no servers.

Replace the `--plan` arg only when migrating to a new binding strategy. Keep Phase 1 Step 01
perpetually [WIP] so both MCP servers can start cleanly regardless of which workstream trackers
are open or archived.

## Implementation phases

### Phase 1 — Permanent MCP binding [WIP]

```yaml
phase: 1
title: 'Permanent MCP binding'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/mcp-active-binding.plans.md
copy_paste: true
next_phase: null
skills:
  - plan-alignment
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/mcp-active-binding.plans.md'
  - 'node scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'execute skill present on any agent dispatched from this plan boundary.'
placeholder_steps:
  - 'Step 01 — MCP servers operational'
```

**Phase objective:** Provide a permanent, perpetually-[WIP] MCP binding context so `neataptic-workflow-mcp` and `neataptic-validation-mcp` can start cleanly regardless of which workstream trackers are open or archived. This phase hosts the lazy-load Chrome DevTools MCP facade (`devtools-facade.mjs`) and the Repo Cortex MCP facade (`cortex-facade.mjs`) as two lightweight binding surfaces that forward JSON-RPC calls without owning the underlying browser or semantic-index workstreams.

This phase never closes. It provides a stable `[WIP]` context for MCP server startup so the
workflow and validation servers can call `loadActivePlanContext` without depending on any specific
workstream plan.

#### Step 01 — MCP servers operational [WIP]

```yaml
phase: 1
step: 1
title: 'MCP servers operational'
status: '[WIP]'
goal: helping
expansion: none
auto_expand: false
mode: perpetual
source_of_truth: plans/mcp-active-binding.plans.md
copy_paste: true
skills:
  - mcp-local-server-workflow
  - execute
next_step: null
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md'
  - 'node scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'execute skill present so any delegated specialist follows the strict sliced RED → IMPLEMENT → GREEN loop.'
```

**User instruction:** Keep the two lightweight MCP lazy-load facades — the Chrome DevTools MCP facade (`devtools-facade.mjs`) and the Repo Cortex MCP facade (`cortex-facade.mjs`) — permanently bound and operational. The facades should start cleanly, route unknown operations through the real underlying server, and forward known JSON-RPC methods as native calls where supported, without taking ownership of the browser, semantic index, or active workstream trackers.

**Step objective:** Provide a perpetual stable binding for MCP server startup. This step does not
advance to [DONE]. Both `neataptic-workflow-mcp` and `neataptic-validation-mcp` reference this
step via the `--plan` arg in `.vscode/mcp.json`.

**Chrome DevTools MCP availability:** This plan does not directly own browser work, but any
delegated specialist that needs browser-based validation (performance traces, DOM/UI inspection,
or heap/memory profiling) should route through the Tier 3 Chrome DevTools MCP specialists
(`performance-trace-specialist`, `browser-ui-specialist`, `browser-memory-specialist`) per the
`execute` skill decision tree rather than calling Chrome DevTools MCP tools directly.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md`

`node scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs --json`

**Stop conditions:** Never — this step is perpetually [WIP] by design.

## Validation gates

No workstream-specific gates. Run the self-check commands above to confirm MCP server startup
health at any time. Two orchestration-system gates now apply to any delegated work from this
boundary:

- `delegate-skill-coverage` — confirms the `execute` skill is present on dispatched agents.
- `chrome-devtools-mcp-coverage` — confirms `03-red-testing` and `05-green-testing` carry the
  `chrome-devtools-mcp` skill and the three Chrome DevTools MCP specialists
  (`performance-trace-specialist`, `browser-ui-specialist`, `browser-memory-specialist`)
  are available for any browser-related validation routed from this plan boundary.

### Latest validation evidence

- Added `plan-slice-quality` Tier-1 gate. Validation: `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json` -> PASS (`pass: true`, `violations: []`, `limit: 4`); `node scripts/agent-customization/gates/step-packet.gate.mjs --json` -> PASS (`pass: true`, `violations: []`); `npx jest scripts/agent-customization/gates/step-packet.gate.test.ts --no-coverage` -> PASS (24 tests); `npm run lint` -> PASS (exit 0); `npx prettier --check <touched files>` -> PASS (exit 0).
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`, plan status `WIP`).
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`, plan status `WIP`).
- `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json` -> PASS (`ok: true`, `0 errors`, `0 warnings`; `run_gate_check`, `query_tier_graph`, and `query_customization_routing_table` remained healthy after `cortex-first-search` gate registration).
- `node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json` -> EXPECTED ENVIRONMENT FAIL (`pass: false`; semantic index stale at `data/semantic-index.sqlite`; actionable `fixHint`: `node scripts/semantic-index/build-index.mjs`; not treated as an MCP wiring defect because `corpus_mcp_alive: true` and the gate honestly reports a prerequisite miss rather than claiming runtime tool-order validation).
- `node scripts/semantic-index/session-start-index.mjs --json` -> PASS (`databasePath: C:\NeatapticTS\data\semantic-index.sqlite`, `touched: 1257`, `contentChanged: 0`, `onDiskMissing: 0`, `buildPassRan: true`, `buildPassExitCode: 0`, `fatalError: null`), confirming the session-start refresh succeeds locally and the incremental build path stays green on an unchanged workspace.
- `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` -> PASS (`pass: true`; `index_documents: 1257`; `index_fresh: true`; `corpus_mcp_alive: true`; `workflow_mcp_alive: true`; `snapshot_age_seconds: 5209`), confirming the session-start refresh clears the index freshness gate in the current workspace.
- targeted lint-cleanup pass for `examples/neatChat`, `examples/shared/semantic`, and `src/architecture/network/onnx` -> focused `npx eslint <touched files...>` PASS after removing dead locals/imports, replacing obsolete `require()` red-test shims with direct or dynamic imports, and replacing unsafe `?.!` test assertions with helper-backed presence checks. Touched `src/` files requiring `/coverage-guard` follow-up: `src/architecture/network/onnx/export/network.onnx.export-build.emit.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export-orchestrators.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export.test.ts`, `src/architecture/network/onnx/import/network.onnx.import-external.utils.ts`, `src/architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts`, `src/architecture/network/onnx/schema/network.onnx.schema.tensor-data.utils.ts`, `src/architecture/network/onnx/validate/network.onnx.validate.ts`. Expected next validation: `npm run lint`.
- `npm run lint ; Write-Output "EXIT:$LASTEXITCODE"` -> PASS (`EXIT:0`), confirming the post-fix lint gate stays green for the reported cleanup set.
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`, plan status `WIP`) after recording the latest validation state.
- `npx jest --config=jest.config.mjs --no-cache --coverage --runInBand --testPathPatterns=src/architecture/network/onnx` -> PASS. Coverage guard stayed at 100% statements, branches, functions, and lines for runtime files `src/architecture/network/onnx/export/network.onnx.export-build.emit.utils.ts`, `src/architecture/network/onnx/export/network.onnx.export-orchestrators.utils.ts`, `src/architecture/network/onnx/import/network.onnx.import-external.utils.ts`, `src/architecture/network/onnx/schema/network.onnx.schema.tensor-data.utils.ts`, and `src/architecture/network/onnx/validate/network.onnx.validate.ts`. `src/architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts` remained a type-only file with no runtime instrumentation obligation, and `src/architecture/network/onnx/export/network.onnx.export.test.ts` remained a test file rather than a production coverage target.
- first `npm run test:silent` attempt hit a transient failure in `scripts/semantic-index/docs-quality/docs-quality.metrics.test.ts` (`1 failed, 426 passed, 427 total`). Focused rerun of `npx jest --config=jest.config.mjs --runInBand scripts/semantic-index/docs-quality/docs-quality.metrics.test.ts` passed (`1 suite`, `4 tests`), and the second `npm run test:silent` rerun also passed cleanly (`427 passed, 427 total`; `4654 passed, 4654 total`). No residual blocker remained after rerun.
- targeted Repo Cortex CI hardening updated `scripts/agent-customization/gates/cortex-index.gate.test.ts` to build the browser snapshot before running `cortex-index.gate.mjs`, updated `scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts` to provision the semantic index and browser snapshot in `beforeAll`, and updated `.github/workflows/deploy-pages.yml` to clear `~/.cache/puppeteer` before `npx puppeteer browsers install chrome`. Validation: `npx tsc --noEmit -p tsconfig.test.json` -> PASS (`exit 0`); `npx jest --config=jest.config.mjs --no-cache --testPathPattern="cortex-index.gate.test" --runInBand` -> FAIL (`testPathPattern` option renamed by current Jest CLI); corrective rerun `npx jest --config=jest.config.mjs --no-cache --testPathPatterns="cortex-index.gate.test" --runInBand` -> PASS (`1 suite`, `1 test`); `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md` -> PASS (`ok: true`, `0 errors`, `0 warnings`).
- runtime-enforcement blocker fix aligned strict-action hooks with the session override plan chain. Validation: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts|scripts/agent-customization/mcp/__tests__/mcp.red.test.mjs"` -> PASS for the runtime enforcement hook suite; `node --test scripts/agent-customization/mcp/__tests__/mcp.red.test.mjs` -> PASS (`8 tests`, `0 failed`), including the session-override precedence regression.
- Orchestration_System_Optimization Phase 4 Step 05 validation — `tier-enforcement.gate.mjs --json` -> PASS (`ok: true`, `issueCount: 0`, `byTier: {1: 8, 2: 11, 3: 38, 4: 4}`, `userInvocableTotal: 8`); `validate-plan-sync.mjs --json --plan=plans/Orchestration_System_Optimization.plans.md` -> PASS (`0 errors`, `0 warnings`, status: `WIP`). Flow specialist references validated with zero tier violations.
- orchestration-alignment refresh after Chrome DevTools MCP Integration closure — `execute` skill added to the phase and step `skills:` lists; `delegate-skill-coverage` gate added to validation; Chrome DevTools MCP specialist availability noted for any browser-related validation routed from this boundary. `chrome-devtools-mcp-coverage` gate applies to any `03-red-testing`/`05-green-testing` dispatch from this plan. No substantive change to the perpetual binding contract.
- scoped cleanup pass from `05-green-testing`. Changed fixture references `schema.sql`/`schema-v2.sql` → `schema-turso.sql` in `examples/shared/semantic/build-browser-snapshot.test.ts`, `scripts/mcp-semantic/tools/freshness-check.test.ts`, `scripts/mcp-semantic/tools/search-corpus.test.ts`, `scripts/mcp-semantic/tools/submit-feedback.test.ts`; deleted legacy `scripts/semantic-index/schema-v2.sql`; surfaced per-query errors from `scripts/semantic-index/parallel-search.mjs` via a non-enumerable `errors` array and propagated them through the `parallel_search` MCP handler in `scripts/mcp-semantic/repo-cortex-mcp.mjs`; added targeted test `runParallelQueries surfaces per-query errors on the returned array` in `scripts/mcp-semantic/__tests__/parallel-search.test.mjs`. Validation: `npx tsc --noEmit -p tsconfig.json` -> PASS; `npm run lint` -> PASS; `npx prettier --check <touched files>` -> PASS; `npx jest ... --testPathPatterns=build-browser-snapshot` -> PASS (1 suite, 1 test); `npx jest ... --testPathPatterns="freshness-check.test|search-corpus.test|submit-feedback.test"` -> PASS (3 suites, 17 tests); `NODE_OPTIONS=--experimental-vm-modules npx jest ... --selectProjects mcp-semantic-mjs --testPathPatterns=parallel-search.test.mjs` -> PASS (1 suite, 16 tests); `node --check scripts/semantic-index/parallel-search.mjs` and `node --check scripts/mcp-semantic/repo-cortex-mcp.mjs` -> OK. Note: `scripts/mcp-semantic/repo-cortex-mcp.test.ts` and `scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts` currently fail to run with `ReferenceError: __dirname is not defined` under ts-jest ESM; this is a pre-existing red-test issue outside the scoped cleanup items and was left untouched per the "no red tests" boundary. Gate checks: `validate-plan-sync` -> PASS; `delegate-skill-coverage` -> PASS; `chrome-devtools-mcp-coverage` -> PASS. No `src/` files were changed, so no additional coverage-guard obligation.
- Chrome DevTools MCP lazy-load facade native-routing fix. Added `routingMode` to `lazy-facade-core.mjs`, wired `devtools-facade.mjs` to `native`, kept `cortex-facade.mjs` in `single-tool`, hardened Windows `resolveSpawnCommand` to find `npx.cmd` next to the Node executable, and added native-mode + Windows spawn fallback tests. Validation: `npx tsc --noEmit -p tsconfig.json` -> PASS; `npm run lint` -> PASS; `npm run prettier:scripts` -> PASS; `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=lazy-facade` -> PASS (1 suite, 59 tests, `lazy-facade-core.mjs` 100/100/100/100); `node scripts/agent-customization/mcp/devtools-facade.mjs --self-check --json` -> PASS (`pass: true`); `node scripts/agent-customization/mcp/cortex-facade.mjs --self-check --json` -> PASS (`pass: true`); direct `createDevtoolsFacade().dispatch({operation:'tools/list'})` against real `chrome-devtools-mcp` -> PASS (29 tools returned). Gate: `plan-sync` -> PASS.
- NGE juvenile grow-stabilize test assertion refresh. Updated 14 hard-coded expected values in `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts` to match the new exhaustion-stage fractions (`BABY 0.015`, `JUVENILE 0.008`, `ADULT 0.005`) and decay-driven outputs. No `src/` production files changed; `NGE_EXHAUSTION_THRESHOLD_DECAY_FLOOR` is not referenced in this test. Preflight: `npx tsc --noEmit -p tsconfig.json` -> PASS; `npx eslint <file>` -> PASS (0 errors, pre-existing 21 warnings); `npx prettier --check <file>` -> PASS. Next: hand off to `05-green-testing` for focused Jest slice confirmation.

```yaml
PlanUpdate:
 changed_files:
 - examples/shared/semantic/build-browser-snapshot.test.ts
 - scripts/mcp-semantic/tools/freshness-check.test.ts
 - scripts/mcp-semantic/tools/search-corpus.test.ts
 - scripts/mcp-semantic/tools/submit-feedback.test.ts
 - scripts/semantic-index/parallel-search.mjs
 - scripts/mcp-semantic/repo-cortex-mcp.mjs
 - scripts/mcp-semantic/__tests__/parallel-search.test.mjs
 - plans/mcp-active-binding.plans.md
 removed_files:
 - scripts/semantic-index/schema-v2.sql
 preflight:
 - 'npx tsc --noEmit -p tsconfig.json'
 - 'npm run lint'
 - 'npx prettier --check <touched files>'
 validation:
 - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --testPathPatterns=build-browser-snapshot'
 expected_exit: 0
 - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --testPathPatterns="freshness-check.test|search-corpus.test|submit-feedback.test"'
 expected_exit: 0
 - command: '$env:NODE_OPTIONS="--no-experimental-webstorage --max-old-space-size=8192 --experimental-vm-modules"; npx jest --config=jest.config.mjs --no-cache --runInBand --forceExit --selectProjects mcp-semantic-mjs --testPathPatterns=parallel-search.test.mjs'
 expected_exit: 0
 - command: 'node --check scripts/semantic-index/parallel-search.mjs; node --check scripts/mcp-semantic/repo-cortex-mcp.mjs'
 expected_exit: 0
 rollback:
 - 'git checkout -- examples/shared/semantic/build-browser-snapshot.test.ts scripts/mcp-semantic/tools/freshness-check.test.ts scripts/mcp-semantic/tools/search-corpus.test.ts scripts/mcp-semantic/tools/submit-feedback.test.ts scripts/semantic-index/parallel-search.mjs scripts/mcp-semantic/repo-cortex-mcp.mjs scripts/mcp-semantic/__tests__/parallel-search.test.mjs plans/mcp-active-binding.plans.md'
 - 'git restore --source=HEAD -- scripts/semantic-index/schema-v2.sql'
 next: 'Handoff to 05-green-testing for focused slice confirmation; no src/ files touched, so coverage-guard is not required beyond noting zero production-source changes.'
```

```yaml
PlanUpdate:
  slice_id: nge-juvenile-exhaustion-assertion-refresh
  changed_files:
    - src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts
    - plans/mcp-active-binding.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx eslint src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    - 'npx prettier --check src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts plans/mcp-active-binding.plans.md'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
  rollback:
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
  next: 'Hand off to 05-green-testing to confirm the focused grow-stabilize test slice passes after updating the stale exhaustion-stage fraction assertions.'
```

## Handoff query

Perpetual binding with the Chrome DevTools MCP lazy-load facade fix. Native routing is
enabled for `devtools-facade.mjs`, classic `single-tool` routing remains for `cortex-facade.mjs`,
Windows spawn resolution now finds `npx.cmd` next to the Node executable, and focused tests cover
native-mode forwarding plus all Windows fallback paths. Next step is `05-green-testing` focused
confirmation on the changed MCP facade files.
