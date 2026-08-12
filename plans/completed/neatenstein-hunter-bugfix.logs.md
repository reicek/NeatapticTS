# Neatenstein hunter bugfix log

**Status:** [DONE]
**Plan:** `plans/completed/neatenstein-hunter-bugfix.plans.md`

## Phase 1 — Hunter bugfix triad

### Final summary

All three live hunter bugs were fixed within the pragmatic child-plan scope and green validated.

- **Center spin (02-spin):** `display.worker.ts` now requires `hasAliveEnemies(gameState.enemies)` before using the champion network path in `humanMode === 'auto'`; fallback exploration takes over when no enemies exist, producing `lookDelta = 0` and forward movement. `neat-io-config.ts` centralized `NEATENSTEIN_FALLBACK_TURN_RATE`.
- **Wall vision (03-los):** `raycast.ts:252` changed from `return wallDist >= dist;` to `return wallDist > dist;` so an enemy whose center lies exactly on the first wall face is no longer treated as visible. `enemy-navigation.ts` and `display.worker.ts` were also touched for tests/usage.
- **Wave spawn at 14 kills (04-waves):** `host/game/waves.ts` removed the `batchComplete` gate that paused after 8 total spawns while any enemy remained alive, and `host/waves.ts` now resets `spawnCount` across waves so `advanceWave` does not leave the batch gate armed.

### Files changed

- `examples/neatenstein/browser-entry/worker/display.worker.ts` — gated champion-network path on alive enemies, kept dead enemies in arrays for index alignment.
- `examples/neatenstein/browser-entry/harness/neat-io-config.ts` — added `NEATENSTEIN_FALLBACK_TURN_RATE` constant export.
- `examples/neatenstein/browser-entry/harness/neat-io-config.test.ts` — asserts the new fallback turn-rate constant.
- `examples/neatenstein/browser-entry/worker/display.worker.test.ts` — added fallback-exploration test, injected alive enemies into existing tests, added champion-network `catch` fallback test.
- `examples/neatenstein/scripts/enemy-navigation.ts` — usage alignment with LOS fix.
- `examples/neatenstein/scripts/enemy-navigation.test.ts` — wall-vision regression tests.
- `examples/neatenstein/browser-entry/renderer/raycast.ts` — boundary-tie fix in `hasLineOfSight`.
- `examples/neatenstein/browser-entry/host/waves.ts` — `advanceWave` now resets `spawnCount`.
- `examples/neatenstein/browser-entry/host/game/waves.ts` — removed `batchComplete` cross-wave gating.
- `coverage/coverage-summary.json` — regenerated from merged coverage runs.
- `plans/completed/neatenstein-hunter-bugfix.plans.md` — tracked implementation, validation, and closure.

### Validation evidence

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-hunter-bugfix.plans.md` → PASS (0 errors, 2 warnings: `research_artifact` key in Step 03 is unexpected; no [WIP] phase exists because Phase 1 is now [DONE]).
- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=04-waves --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/neatenstein-hunter-bugfix.research.md` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass).
- `slice-validator` review of slice `04-waves-green` → VERDICT: PASS (atomic intent, one changed file, complete step packet, dependency `04-waves` [DONE], AC evidence aligned).
- `cortex-index` gate: initially `pass: false` due to stale semantic index; rebuilt with `node rag-index/build-index.mjs` → index_fresh now true; remaining `workflow_mcp_alive: false` is a tooling/server issue, not a content failure.
- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/README.md,plans/Roadmap.md` → `pass: true` (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass).
- Slice `02-spin` preflight: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK; `npx eslint examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts examples/neatenstein/browser-entry/harness/neat-io-config.ts` → 0 issues; `npx prettier --check` on changed files → OK.
- Slice `02-spin` targeted smoke: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="spin|center|no.*enemy|fallback.*exploration"` → 15 suites passed, 80 tests passed (includes new test: `fallback exploration overrides a champion network spin output when no enemies exist`).
- `node scripts/agent-customization/gates/specialist-review-severity.gate.mjs --json --input=examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/harness/neat-io-config.ts` → severity FULL, 1 specialist required.
- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=03-los --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/neatenstein-hunter-bugfix.research.md` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass).
- Slice 03-los research: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"` → 10 passed, 0 failed; `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK.
- Slice 03-los wall-vision map audit: renderer sprite clipping uses the same module-level `wallMap` as `findNearestVisibleEnemy` inside `display.worker.ts`; no separate renderer/AI grid found. `sprites.ts` clips against the per-column `zBuffer` produced by the wall raycaster, not `wallMap` directly. Latent stale-map risk exists only if `simState` carries a different `mapSeed` than `init`.
- Slice 03-los wall-vision map audit validation: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"` → 3 suites passed, 10 tests passed; `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK.
- Slice 03-los DDA traversal audit validation: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein/browser-entry/renderer/raycast.test.ts` → 1 suite passed, 13 tests passed.
- Slice 03-los boundary-tie reproduction: `tsx` script confirms `hasLineOfSight(map, 8, {x:0.5,y:0.5}, {x:1.0,y:0.5})` with a wall at cell `(1,0)` returns `true` (bug), while target `{x:1.0000001,y:0.5}` returns `false` (correct). Source: `examples/neatenstein/browser-entry/renderer/raycast.ts:252` (`wallDist >= dist`).
- Slice 03-los out-of-bounds reproduction: `tsx` script confirms `castRayDDAFromFlatMap` walks off an all-open 4×4 grid (`mapX = 12`) when no perimeter walls are present; `Uint8Array` silently returns `0` for out-of-bounds reads.
- Slice 03-los DDA audit gate: `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=03-los --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/neatenstein-hunter-bugfix.research.md` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass).
- Specialist review by `api-contract-reviewer` → APPROVE (additive export, no breaking changes).
- Slice `02-spin` behavior fix: `display.worker.ts` now requires `hasAliveEnemies(gameState.enemies)` before using the champion network path in `humanMode === 'auto'`, forcing fallback exploration when no enemies exist.
- Slice `02-spin` updated tests: `AC-060` and `AC-066` in `display.worker.test.ts` now inject an alive enemy so the champion network path remains exercised under the new `hasAliveEnemies` guard.
- Slice `02-spin` added test: `falls back to exploration AI when champion network activation throws` covers the `catch` fallback branch in `display.worker.ts`.
- Slice `02-spin` coverage gate: `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files "examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/harness/neat-io-config.ts"` → PASS (display.worker.ts: 100/100/100/100; neat-io-config.ts: 100/100/100/100).
- Slice `02-spin` slice-advancement gate: `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id 02-spin --changed-files "examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/harness/neat-io-config.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts,examples/neatenstein/browser-entry/harness/neat-io-config.test.ts"` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass).
- Slice `02-spin` targeted smoke (post-fix): `npx jest --config=jest.config.mjs --no-cache --runInBand --coverage --coverageDirectory=coverage/run-02-spin-targeted --testPathPatterns="examples/neatenstein/browser-entry/worker/display.worker.test.ts|examples/neatenstein/browser-entry/harness/neat-io-config.test.ts"` → 2 suites passed, 161 tests passed.
- Slice `02-spin` preflight (final): `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK; `npx eslint` on changed files → 0 issues; `npx prettier --check` on changed files → OK.
- Slice `03-los` code fix: `examples/neatenstein/browser-entry/renderer/raycast.ts:252` changed from `return wallDist >= dist;` to `return wallDist > dist;` so an enemy whose center lies exactly on the first wall face is no longer treated as visible.
- Slice `03-los` preflight: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK; `npx eslint examples/neatenstein/browser-entry/renderer/raycast.ts` → 0 issues; `npx prettier --check examples/neatenstein/browser-entry/renderer/raycast.ts` → OK.
- Slice `03-los` targeted smoke: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="examples/neatenstein/browser-entry/renderer/raycast.test.ts"` → 1 suite passed, 13 tests passed.
- Slice `03-los` specialist review: `performance-reviewer` → APPROVE (no perf / allocation / cache / typed-array regression).
- Slice `03-los` slice-advancement gate: `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id 03-los --changed-files "examples/neatenstein/browser-entry/renderer/raycast.ts,plans/neatenstein-hunter-bugfix.plans.md"` → sub-gates plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass.
- Slice `04-waves` severity gate: `node scripts/agent-customization/gates/specialist-review-severity.gate.mjs --json --input=examples/neatenstein/browser-entry/host/game/waves.ts,examples/neatenstein/browser-entry/host/waves.ts` → severity FULL, 1 specialist required.
- Slice `04-waves` preflight: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK; `npx eslint examples/neatenstein/browser-entry/host/waves.ts examples/neatenstein/browser-entry/host/game/waves.ts` → 0 issues; `npx prettier --check examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/waves.ts` → OK.
- Slice `04-waves` targeted smoke: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn"` → 17 suites passed, 91 tests passed.
- Slice `04-waves` specialist review by `determinism-reviewer` → APPROVE (seed-stable, spawnCount monotonic, no time/entropy seeding).
- Slice `04-waves` coverage refresh: `npx jest --config=jest.config.mjs --no-cache --runInBand --coverage --testPathPatterns=examples/neatenstein` → 34 suites passed; `node scripts/agent-customization/gates/merge-coverage-summaries.mjs` regenerated `coverage/coverage-summary.json`; both changed wave modules now report 100/100/100/100.
- Slice `04-waves` slice-advancement gate: `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id 04-waves --changed-files "examples/neatenstein/browser-entry/host/game/waves.ts,examples/neatenstein/browser-entry/host/waves.ts"` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review all pass).
- Slice `03-los-green` targeted tests: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"` → 3 suites passed, 10 tests passed.
- Slice `03-los-green` raycast regression test: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="examples/neatenstein/browser-entry/renderer/raycast.test.ts"` → 1 suite passed, 13 tests passed.
- Slice `03-los-green` type-check: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK.
- Slice `03-los-green` coverage gate: `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files "examples/neatenstein/scripts/enemy-navigation.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/renderer/raycast.ts"` → PASS (all three files at 100/100/100/100).
- Slice `03-los-green` slice-advancement gate: `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=03-los-green --args.changed-files=examples/neatenstein/scripts/enemy-navigation.test.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass; severity TRIVIAL, no specialist review required).
- Slice `04-waves` implementation was already completed and validated (see evidence above); this turn marks the slice and Step 04 `[DONE]` in the plan so `04-waves-green` can proceed.
- Slice `04-waves-green` targeted tests: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn|kills|fourteen"` → 21 suites passed, 96 tests passed.
- Slice `04-waves-green` type-check: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK.
- Slice `04-waves-green` coverage gate: `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files "examples/neatenstein/browser-entry/host/game/waves.ts,examples/neatenstein/browser-entry/host/waves.ts,examples/neatenstein/browser-entry/worker/display.worker.ts"` → PASS (all three files at 100/100/100/100).
- Slice `04-waves-green` slice-advancement gate: `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=04-waves-green --args.changed-files=examples/neatenstein/browser-entry/worker/display.worker.test.ts` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass; severity TRIVIAL, no specialist review required).
- Step 05 full Neatenstein suite: `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein` → 73 suites passed, 1529 tests passed, 1 skipped.

### Decisions

- Champion-network auto-tick path is now gated by `hasAliveEnemies(gameState.enemies)`; fallback exploration runs when no enemies are visible.
- `hasLineOfSight` boundary tie changed from `wallDist >= dist` to `wallDist > dist` so enemies exactly on the first wall face are hidden.
- Wave spawning switched from per-batch gating to continuous capped spawning; `spawnCount` is reset across waves.

### Risks / residual gaps

- Latent DDA out-of-bounds traversal on all-open maps (mitigated by `buildNeatensteinMap` sealed perimeter in the demo).
- Potential stale `wallMap` only if `simState` carries a different `mapSeed` than initialization.
- Real visible-window browser smoke validation was not performed; Jest + tsc + lint passed but in-browser hunter behavior remains unaudited.

### Next resume point

Workstream complete. Reopen only from a new active tracker referencing the archived `plans/completed/` pair.
