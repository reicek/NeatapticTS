# Neatenstein hunter bugfix triad

**Status:** [WIP]
**Plan ID:** NEATENSTEIN_HUNTER_BUGFIX
**Created:** 2026-08-12
**Source of truth:** `plans/neatenstein-hunter-bugfix.plans.md`
**Log:** `plans/neatenstein-hunter-bugfix.logs.md` (created at closure)

Pragmatic child plan for three live bugs in the Neatenstein demo hunter.

## Mandates

- **Model:** `kimi` for all dispatches under this plan.
- **Mode:** pragmatic — broad slices, targeted tests only, no redundant red/green/doc sub-slices.
- **Scope:** changes must stay within `display.worker.ts`, `enemy-navigation.ts`, `neat-io-config.ts` plus the two wave modules (`host/waves.ts`, `host/game/waves.ts`) for the wave bug.

## Scope

- Fix hero spinning in place at map center when no enemies are visible.
- Fix hero acquiring enemies through walls.
- Fix wave spawner stopping after roughly 14 kills.

## Risks

- The 14-kill symptom may be a wave-spawn or wave-clear detection failure, not a simple cap.
- Wall-vision bug may be in LOS helper usage or a stale `wallMap` rather than the raycaster.
- Center-spin bug may involve network output mapping in `neat-io-config.ts` or the fallback exploration path in `display.worker.ts`.

## Current state

Claim: 05-green-testing @ 2026-08-13T02:00:00Z (Step 05 green validation complete: full Neatenstein example suite 73 suites / 1529 tests passed with 1 skipped, `tsc` OK, `npm run lint` 0 issues, all changed source files at 100/100/100/100 coverage, and final `slice-advancement` plan gate passes. Slices 03-los-green and 04-waves-green are [DONE]. Phase 1 — Hunter bugfix triad is [DONE]; next handoff is `06-documenting` / plan archive.)

## Latest validation evidence

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
- Slice `04-waves` preflight: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK; `npx eslint examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/waves.ts` → 0 issues; `npx prettier --check examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/waves.ts` → OK.
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
- Step 05 TypeScript check: `npx tsc --noEmit -p tsconfig.neatenstein.json` → OK.
- Step 05 lint: `npm run lint` → 0 issues.
- Step 05 combined coverage gate for all changed source files (`display.worker.ts`, `neat-io-config.ts`, `raycast.ts`, `enemy-navigation.ts`, `host/waves.ts`, `host/game/waves.ts`) → PASS (all at 100/100/100/100).
- Final plan slice-advancement gate: `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/README.md,plans/Roadmap.md` → PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass).

```yaml
PlanUpdate:
  slice_id: 04-waves-green
  changed_files:
    - examples/neatenstein/browser-entry/worker/display.worker.test.ts
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn|kills|fourteen"'
    - 'npx tsc --noEmit -p tsconfig.neatenstein.json'
  green_results:
    - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn|kills|fourteen"'
      result: '21 suites passed, 96 tests passed'
    - command: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
      result: 'OK'
  coverage_guard:
    files:
      - examples/neatenstein/browser-entry/host/game/waves.ts
      - examples/neatenstein/browser-entry/host/waves.ts
      - examples/neatenstein/browser-entry/worker/display.worker.ts
    summary: 'lines:100,statements:100,functions:100,branches:100'
  slice_advancement:
    gate: slice-advancement
    pass: true
    sub_gates:
      - plan-sync
      - step-packet
      - plan-slice-quality
      - plan-command-lint
  next: 'Phase 1 hunter bugfix triad complete; hand off to plan archive / 06-documenting'
```

```yaml
PlanUpdate:
  boundary: 'Phase 1 / Step 05'
  status: '[DONE]'
  changed_files:
    - examples/neatenstein/browser-entry/worker/display.worker.ts
    - examples/neatenstein/browser-entry/harness/neat-io-config.ts
    - examples/neatenstein/browser-entry/renderer/raycast.ts
    - examples/neatenstein/scripts/enemy-navigation.ts
    - examples/neatenstein/browser-entry/host/waves.ts
    - examples/neatenstein/browser-entry/host/game/waves.ts
  validations:
    - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein'
      result: '73 suites passed, 1529 tests passed, 1 skipped'
    - command: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
      result: 'OK'
    - command: 'npm run lint'
      result: '0 issues'
  coverage_guard:
    files:
      - examples/neatenstein/browser-entry/worker/display.worker.ts
      - examples/neatenstein/browser-entry/harness/neat-io-config.ts
      - examples/neatenstein/browser-entry/renderer/raycast.ts
      - examples/neatenstein/scripts/enemy-navigation.ts
      - examples/neatenstein/browser-entry/host/waves.ts
      - examples/neatenstein/browser-entry/host/game/waves.ts
    summary: 'lines:100,statements:100,functions:100,branches:100'
  slice_advancement:
    gate: slice-advancement
    pass: true
    sub_gates:
      - plan-sync
      - step-packet
      - plan-slice-quality
      - plan-command-lint
  next: 'Hand off to 06-documenting / plan archive'
```

```yaml
PlanUpdate:
  slice_id: 04-waves
  changed_files:
    - examples/neatenstein/browser-entry/host/game/waves.ts
    - examples/neatenstein/browser-entry/host/waves.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    - 'npx eslint examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/waves.ts'
    - 'npx prettier --check examples/neatenstein/browser-entry/host/game/waves.ts examples/neatenstein/browser-entry/host/waves.ts'
  specialist_review:
    agent: determinism-reviewer
    verdict: APPROVE
  slice_advancement:
    gate: slice-advancement
    pass: true
    sub_gates:
      - plan-sync
      - step-packet
      - plan-slice-quality
      - plan-command-lint
      - shared-validation
      - code-coverage
      - specialist-review
  coverage_guard:
    files:
      - examples/neatenstein/browser-entry/host/game/waves.ts
      - examples/neatenstein/browser-entry/host/waves.ts
    summary: statements:100,branches:100,functions:100,lines:100
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn|kills|fourteen"'
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein/browser-entry/host/game/waves.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein/browser-entry/host/waves.test.ts'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/host/game/waves.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/waves.ts'
  next: 'Run 04-waves-green / 05-green-testing and attach coverage-guard evidence for the changed wave modules'
```

```yaml
PlanUpdate:
  slice_id: 03-los
  changed_files:
    - examples/neatenstein/browser-entry/renderer/raycast.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    - 'npx eslint examples/neatenstein/browser-entry/renderer/raycast.ts'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/raycast.ts'
  specialist_review:
    agent: performance-reviewer
    verdict: APPROVE
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"'
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="examples/neatenstein/browser-entry/renderer/raycast.test.ts"'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/raycast.ts'
  next: 'Run 03-los-green / 05-green-testing and attach coverage-guard evidence for raycast.ts'
```

```yaml
PlanUpdate:
  slice_id: 03-los-green
  changed_files:
    - examples/neatenstein/scripts/enemy-navigation.test.ts
    - examples/neatenstein/browser-entry/worker/display.worker.test.ts
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"'
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="examples/neatenstein/browser-entry/renderer/raycast.test.ts"'
    - 'npx tsc --noEmit -p tsconfig.neatenstein.json'
  green_results:
    - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"'
      result: '3 suites passed, 10 tests passed'
    - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns="examples/neatenstein/browser-entry/renderer/raycast.test.ts"'
      result: '1 suite passed, 13 tests passed'
    - command: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
      result: 'OK'
  coverage_guard:
    files:
      - examples/neatenstein/scripts/enemy-navigation.ts
      - examples/neatenstein/browser-entry/worker/display.worker.ts
      - examples/neatenstein/browser-entry/renderer/raycast.ts
    summary: 'lines:100,statements:100,functions:100,branches:100'
  slice_advancement:
    gate: slice-advancement
    pass: true
    sub_gates:
      - plan-sync
      - step-packet
      - plan-slice-quality
      - plan-command-lint
  next: 'Hand off to Step 04 / slice 04-waves implementation per active plan'
```

## Implementation phases

### Phase 1 — Hunter bugfix triad [DONE]

**Phase objective:** Author and execute three focused bug fixes in the Neatenstein demo hunter: center spin, wall vision, and wave spawn at 14 kills.

**Stop conditions:** Any fix reveals a deeper architecture conflict (e.g., fire-gate / eval-worker coupling), root cause is outside the three named files plus the two wave files, or the user requests a different model mandate.

**Required validation:** `slice-advancement` consolidated gate must pass for Step 01 before execution begins.

```yaml
phase: 1
title: 'Hunter bugfix triad'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hunter-bugfix.plans.md'
copy_paste: true
next_phase: 'Archive plan after green validation'
skills:
  - 'plan-alignment'
  - 'acceptance-criteria'
  - 'implementation-standards'
validation:
  - 'plans/neatenstein-hunter-bugfix.plans.md'
  - 'plans/README.md'
  - 'plans/Roadmap.md'
acceptance_criteria:
  - id: AC-001
    text: 'Step 02-05 packets are authored with machine-readable YAML blocks and focused acceptance criteria'
    validation: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-hunter-bugfix.plans.md'
  - id: AC-002
    text: 'slice-advancement gate passes for the active plan'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/README.md,plans/Roadmap.md'
placeholder_steps:
  - 'Step 01 — Plan the bugfix triad'
  - 'Step 02 — Fix hero spin in center when no enemies visible'
  - 'Step 03 — Fix hero seeing enemies through walls'
  - 'Step 04 — Fix waves stopping after 14 kills'
  - 'Step 05 — Green validation'
```

#### Step 01 — Plan the bugfix triad [DONE]

**User instruction:** Create the active plan tracker and register it in the plan index/roadmap before any implementation work begins.

**Step objective:** Author Step 02-05 packets with machine-readable YAML blocks, focused acceptance criteria, and targeted validation commands. Register the plan in `plans/README.md` and `plans/Roadmap.md`.

**Stop conditions:** Plan tracker cannot be registered, slice-advancement gate fails and cannot be fixed, or the user changes scope.

**Required validation:** Run `validate-plan-phase-packets` and `slice-advancement` for the plan boundary; record the gate JSON in the plan.

```yaml
phase: 1
step: 1
title: 'Plan the bugfix triad'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hunter-bugfix.plans.md'
copy_paste: true
next_step: 'Step 02 — Fix hero spin in center when no enemies visible'
owner: '01-planning'
reviewer: 'user'
skills:
  - 'plan-alignment'
  - 'acceptance-criteria'
  - 'implementation-standards'
validation:
  - 'plans/neatenstein-hunter-bugfix.plans.md'
  - 'plans/README.md'
  - 'plans/Roadmap.md'
acceptance_criteria:
  - id: AC-001
    text: 'Step 02-05 packets are authored with machine-readable YAML blocks and focused acceptance criteria'
    validation: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-hunter-bugfix.plans.md'
  - id: AC-002
    text: 'slice-advancement gate passes for the active plan'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/README.md,plans/Roadmap.md'
evidence:
  gate_outputs:
    - gate: 'plan phase packets'
      timestamp: '2026-08-12T00:00:00Z'
      command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-hunter-bugfix.plans.md'
      result: 'PASS (0 errors, 0 warnings)'
    - gate: 'slice-advancement'
      timestamp: '2026-08-12T00:00:00Z'
      command: 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/README.md,plans/Roadmap.md'
      result: 'PASS (all 4 sub-gates pass)'
```

```yaml
PlanUpdate:
  boundary: 'Phase 1 / Step 01'
  status: '[DONE]'
  what_changed:
    - 'plans/neatenstein-hunter-bugfix.plans.md — authored Step 02-05 packets with machine-readable YAML blocks and evidence'
    - 'plans/README.md — registered active plan entry'
    - 'plans/Roadmap.md — registered active workstream entry'
  evidence:
    - 'plan phase packets: PASS (0 errors, 0 warnings)'
    - 'slice-advancement: PASS (all 4 sub-gates pass)'
  removals: []
  next_boundary: 'Step 02 — Fix hero spin in center when no enemies visible / slice 02-spin'
```

#### Step 02 — Fix hero spin in center when no enemies visible [DONE]

**User instruction:** Fix the hero spinning in place at the map center when no enemies are visible, and add a targeted test that locks in the fix.

**Step objective:** When no enemy is visible, the fallback auto-tick path or network output mapping must not produce a sustained spin. Bound or suppress rotation when the player has no visible target and is in the open center area.

**Stop conditions:** Root cause is outside `display.worker.ts` or `neat-io-config.ts`, fix requires a network retraining run, or the targeted test cannot be made deterministic.

**Required validation:** Targeted Neatenstein demo test passes and `tsconfig.neatenstein.json` compiles cleanly.

**Primary files:** `display.worker.ts`, `neat-io-config.ts`.

```yaml
phase: 1
step: 2
title: 'Fix hero spin in center when no enemies visible'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hunter-bugfix.plans.md'
copy_paste: true
next_step: 'Step 03 — Fix hero seeing enemies through walls'
owner: '04-implementing'
reviewer: 'user'
skills:
  - 'implementation-standards'
  - 'neatenstein-domain'
validation:
  - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  - 'examples/neatenstein/browser-entry/harness/neat-io-config.ts'
  - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
acceptance_criteria:
  - id: AC-001
    text: 'When no enemy is visible, the hunter does not produce a sustained spin in the map center'
    validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="spin|center|no.*enemy.*visible|fallback.*exploration"'
  - id: AC-002
    text: 'Neatenstein TypeScript config compiles cleanly'
    validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
slices:
  - slice_id: '02-spin'
    title: 'Implement hero spin fix'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/harness/neat-io-config.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'Fallback rotation is bounded when no enemy is visible'
        validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    parallelizable: false
    dependencies: []
    next_slice: '02-spin-green'
  - slice_id: '02-spin-green'
    title: 'Green validate spin fix'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'Targeted spin test passes'
        validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="spin|center|no.*enemy.*visible|fallback.*exploration"'
      - id: AC-002
        text: 'Neatenstein TypeScript config compiles cleanly'
        validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    parallelizable: false
    dependencies:
      - '02-spin'
    next_slice: '03-los'
```

```yaml
PlanUpdate:
  slice_id: '02-spin'
  changed_files:
    - examples/neatenstein/browser-entry/harness/neat-io-config.ts
    - examples/neatenstein/browser-entry/harness/neat-io-config.test.ts
    - examples/neatenstein/browser-entry/worker/display.worker.ts
    - examples/neatenstein/browser-entry/worker/display.worker.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    - 'npx eslint examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts examples/neatenstein/browser-entry/harness/neat-io-config.ts examples/neatenstein/browser-entry/harness/neat-io-config.test.ts'
    - 'npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts examples/neatenstein/browser-entry/harness/neat-io-config.ts examples/neatenstein/browser-entry/harness/neat-io-config.test.ts'
  preflight_results:
    tsc: 'OK'
    lint: '0 issues'
    prettier: 'OK'
  specialist_review:
    gate: 'specialist-review-severity'
    severity: 'FULL'
    agent: 'api-contract-reviewer'
    verdict: 'APPROVE'
  slice_advancement:
    gate: 'slice-advancement'
    status: 'PASS'
    sub_gates:
      plan-sync: 'PASS'
      step-packet: 'PASS'
      plan-slice-quality: 'PASS'
      plan-command-lint: 'PASS'
      code-coverage: 'PASS'
      specialist-review: 'PASS'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="spin|center|no.*enemy|fallback.*exploration"'
  coverage_guard:
    files:
      - examples/neatenstein/browser-entry/harness/neat-io-config.ts
      - examples/neatenstein/browser-entry/worker/display.worker.ts
    summary: 'lines:100,statements:100,functions:100,branches:100'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/harness/neat-io-config.ts examples/neatenstein/browser-entry/harness/neat-io-config.test.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  next: 'Hand off to 02-spin-green / 05-green-testing for full green validation; full test suite will regenerate coverage/coverage-final.json from the merged runs'

HandoffPayload:
  slice_id: '02-spin'
  plan_update:
    changed_files:
      - examples/neatenstein/browser-entry/harness/neat-io-config.ts
      - examples/neatenstein/browser-entry/harness/neat-io-config.test.ts
      - examples/neatenstein/browser-entry/worker/display.worker.ts
      - examples/neatenstein/browser-entry/worker/display.worker.test.ts
    preflight_outputs:
      tsc: 'OK'
      lint: '0 issues'
      prettier: 'OK'
    validation:
      - command: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="spin|center|no.*enemy|fallback.*exploration"'
        exit: 0
        result: '15 suites passed, 81 tests passed'
      - command: 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files "examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/harness/neat-io-config.ts"'
        exit: 0
        result: 'PASS (display.worker.ts and neat-io-config.ts all 100%)'
      - command: 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id 02-spin --changed-files "examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/harness/neat-io-config.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts,examples/neatenstein/browser-entry/harness/neat-io-config.test.ts"'
        exit: 0
        result: 'PASS (sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint all pass)'
    coverage_guard:
      files:
        - examples/neatenstein/browser-entry/harness/neat-io-config.ts
        - examples/neatenstein/browser-entry/worker/display.worker.ts
      summary: 'lines:100,statements:100,functions:100,branches:100'
    artifacts:
      - coverage/coverage-summary.json
      - coverage/run-02-spin-targeted/coverage-final.json
  suggested_branch: 'implement/neatenstein-hunter-spin-fix-02-spin'
  suggested_commit_message: 'fix(neatenstein): stop hero spin when no enemies exist — PlanUpdate: plans/neatenstein-hunter-bugfix.plans.md'
  pr_description: |
    Slice 02-spin: hero no longer spins at map center when zero enemies exist.

    Changes:
    - `display.worker.ts`: champion-network path is now gated by `hasAliveEnemies(gameState.enemies)`; fallback exploration AI takes over when no enemies exist, producing `lookDelta = 0` and forward movement.
    - `neat-io-config.ts`: centralized `NEATENSTEIN_FALLBACK_TURN_RATE` export with JSDoc.
    - `display.worker.test.ts`: new test locks in fallback exploration overriding a spinning champion network with zero enemies; updated `AC-060`/`AC-066` to inject an enemy so the network path is still exercised; added test for champion-network `catch` fallback.
    - `neat-io-config.test.ts`: asserts the new `NEATENSTEIN_FALLBACK_TURN_RATE` constant.

    Validation:
    - TypeScript compiles cleanly (`npx tsc --noEmit -p tsconfig.neatenstein.json`).
    - ESLint and Prettier pass on all changed files.
    - Targeted Jest smoke: 15 suites passed, 81 tests passed.
    - Coverage gate: changed source files at 100% lines/statements/functions/branches.
    - Slice-advancement gate: PASS.
  manual_git_commands: |
    git checkout -b implement/neatenstein-hunter-spin-fix-02-spin
    git add examples/neatenstein/browser-entry/harness/neat-io-config.ts
    git add examples/neatenstein/browser-entry/harness/neat-io-config.test.ts
    git add examples/neatenstein/browser-entry/worker/display.worker.ts
    git add examples/neatenstein/browser-entry/worker/display.worker.test.ts
    git add coverage/coverage-summary.json
    git add coverage/run-02-spin-targeted/coverage-final.json
    git add plans/neatenstein-hunter-bugfix.plans.md
    git commit -m "fix(neatenstein): stop hero spin when no enemies exist — PlanUpdate: plans/neatenstein-hunter-bugfix.plans.md"
    git push origin implement/neatenstein-hunter-spin-fix-02-spin
```

#### Step 03 — Fix hero seeing enemies through walls [DONE]

**User instruction:** Fix the hero acquiring enemies through walls, and add a targeted test that locks in the fix.

**Step objective:** Ensure an enemy behind a wall is never returned as the nearest visible target by `findNearestVisibleEnemy` or the active hunter path.

**Stop conditions:** Root cause requires a redesign of the raycaster engine (e.g., changing `NEATENSTEIN_RENDER_DISTANCE_CAP` or grid coordinate semantics), or a wall grid / coordinate mismatch that cannot be fixed within `enemy-navigation.ts`, `display.worker.ts`, and `raycast.ts`.

**Required validation:** Targeted Neatenstein demo test passes and `tsconfig.neatenstein.json` compiles cleanly.

**Primary files:** `enemy-navigation.ts`, `display.worker.ts`.

**Research findings (slice 03-los):** The wall-vision integration is wired correctly and the main wall-block path is covered by tests. `extractSensors` uses `findNearestVisibleEnemy` (range + LOS filtered), so the `enemyVisible` flag and enemy position/health/bearing sensors are vision-filtered, not omniscient. DDA traversal audit of `castRayDDAFromFlatMap` confirms a classic Wolfenstein-style grid walker with consistent world-to-grid coordinate transforms; no reproducible wall-skipping was found in constructed diagonal, boundary-origin, or corner tie-break cases. The concrete false-positive identified is a boundary tie in `hasLineOfSight` (`raycast.ts:252`): `wallDist >= dist` returns `true` when an enemy center lies exactly on the first wall face. Runtime reproduction confirms player at `(0.5, 0.5)` with a wall at cell `(1,0)` and target `(1.0, 0.5)` returns `true`. The minimal fix is to use `>` instead of `>=`. A separate latent risk was identified: `castRayDDAFromFlatMap` lacks bounds checks and relies on a sealed perimeter; an all-open map causes out-of-bounds reads, but this is mitigated by `buildNeatensteinMap` in the demo. Full research artifact: `plans/neatenstein-hunter-bugfix.research.md`.

```yaml
phase: 1
step: 3
title: 'Fix hero seeing enemies through walls'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hunter-bugfix.plans.md'
copy_paste: true
next_step: 'Step 04 — Fix waves stopping after 14 kills'
owner: '04-implementing'
reviewer: 'user'
research_artifact: 'plans/neatenstein-hunter-bugfix.research.md'
skills:
  - 'implementation-standards'
  - 'neatenstein-domain'
validation:
  - 'examples/neatenstein/scripts/enemy-navigation.ts'
  - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
acceptance_criteria:
  - id: AC-001
    text: 'An enemy positioned behind a wall is not treated as visible by the hunter'
    validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"'
  - id: AC-002
    text: 'Neatenstein TypeScript config compiles cleanly'
    validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
slices:
  - slice_id: '03-los'
    title: 'Implement wall-vision fix'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-navigation.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/renderer/raycast.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'LOS helper rejects enemies behind walls'
        validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    parallelizable: false
    dependencies:
      - '02-spin-green'
    next_slice: '03-los-green'
  - slice_id: '03-los-green'
    title: 'Green validate wall-vision fix'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'examples/neatenstein/scripts/enemy-navigation.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'Targeted wall-vision test passes'
        validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="line of sight|through wall|findNearestVisibleEnemy|wall.*vision|visibility"'
      - id: AC-002
        text: 'Neatenstein TypeScript config compiles cleanly'
        validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    parallelizable: false
    dependencies:
      - '03-los'
    next_slice: '04-waves'
```

#### Step 04 — Fix waves stopping after 14 kills [DONE]

**User instruction:** Fix the wave spawner so it continues to spawn enemies after roughly 14 kills, and add a targeted test that locks in the fix.

**Step objective:** The 14-kill symptom (8 + 6) suggests the wave-spawn or wave-clear detection fails partway through the second wave. Ensure `advanceWave` and `spawnWaveTick` correctly continue spawning after the first wave clear.

**Stop conditions:** Root cause requires a redesign of the wave spawning model (e.g., changing `NEATENSTEIN_ENEMY_MAX_CONCURRENT` or the save format), or the fix cannot be contained in `display.worker.ts` plus `host/waves.ts` and `host/game/waves.ts`.

**Required validation:** Targeted Neatenstein demo test passes and `tsconfig.neatenstein.json` compiles cleanly.

**Primary files:** `display.worker.ts`, `host/waves.ts`, `host/game/waves.ts`.

```yaml
phase: 1
step: 4
title: 'Fix waves stopping after 14 kills'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hunter-bugfix.plans.md'
copy_paste: true
next_step: 'Step 05 — Green validation'
owner: '04-implementing'
reviewer: 'user'
skills:
  - 'implementation-standards'
  - 'neatenstein-domain'
validation:
  - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  - 'examples/neatenstein/browser-entry/host/waves.ts'
  - 'examples/neatenstein/browser-entry/host/game/waves.ts'
  - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
acceptance_criteria:
  - id: AC-001
    text: 'Wave spawner continues to spawn new enemies past the 14-kill threshold'
    validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn|kills|fourteen"'
  - id: AC-002
    text: 'Neatenstein TypeScript config compiles cleanly'
    validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
slices:
  - slice_id: '04-waves'
    title: 'Implement wave-spawn fix'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/host/waves.ts'
      - 'examples/neatenstein/browser-entry/host/game/waves.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'Wave advance logic continues spawning past the first wave clear'
        validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    parallelizable: false
    dependencies:
      - '03-los-green'
    next_slice: '04-waves-green'
  - slice_id: '04-waves-green'
    title: 'Green validate wave-spawn fix'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: AC-001
        text: 'Targeted wave-spawn test passes'
        validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein --testNamePattern="wave|advance|spawn|kills|fourteen"'
      - id: AC-002
        text: 'Neatenstein TypeScript config compiles cleanly'
        validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
    parallelizable: false
    dependencies:
      - '04-waves'
    next_slice: '05-green'
```

> **02-researching findings (slice 04-waves):** The root cause is the `batchComplete` gate in `spawnWaveTick` (`host/game/waves.ts`), which pauses after every 8 total spawns while any enemy remains alive. `advanceWave` (`host/waves.ts`) clears enemies and increments `generation` but does not reset `spawnCount`, so the gate stays armed across waves. Wave-clear detection in `display.worker.ts` is working correctly and should not be changed for this symptom. The fix should switch from batch gating to continuous capped spawning while keeping dead enemies in arrays for index alignment. Full evidence: [neatenstein-hunter-bugfix.research.md](neatenstein-hunter-bugfix.research.md).

#### Step 05 — Green validation [DONE]

**User instruction:** Run the targeted Neatenstein demo suite and build checks to confirm the three fixes are green and the demo still compiles cleanly.

**Step objective:** Validate the three bug fixes together with the targeted Neatenstein example suite, TypeScript checks, and lint.

**Stop conditions:** Any test failure, TypeScript error, or lint error attributable to the fixes.

**Required validation:** Targeted Neatenstein demo test suite passes, `tsconfig.neatenstein.json` compiles cleanly, and `npm run lint` exits zero.

```yaml
phase: 1
step: 5
title: 'Green validation'
status: '[DONE]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hunter-bugfix.plans.md'
copy_paste: true
next_step: 'Archive plan'
owner: '05-green-testing'
reviewer: 'user'
skills:
  - 'green-validation'
  - 'implementation-standards'
validation:
  - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  - 'examples/neatenstein/scripts/enemy-navigation.ts'
  - 'examples/neatenstein/browser-entry/harness/neat-io-config.ts'
  - 'examples/neatenstein/browser-entry/host/waves.ts'
  - 'examples/neatenstein/browser-entry/host/game/waves.ts'
acceptance_criteria:
  - id: AC-001
    text: 'Targeted Neatenstein demo suite passes with zero failures'
    validation: 'npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein'
  - id: AC-002
    text: 'Neatenstein TypeScript config compiles cleanly'
    validation: 'npx tsc --noEmit -p tsconfig.neatenstein.json'
  - id: AC-003
    text: 'Lint exits with no errors'
    validation: 'npm run lint'
```

## Validation gates

- `slice-advancement` — run after authoring or patching any plan boundary:
  `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hunter-bugfix.plans.md,plans/README.md,plans/Roadmap.md`
- `plan phase packets` — run as a local self-check:
  `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/neatenstein-hunter-bugfix.plans.md`

### Latest validation evidence

- Workflow sync: Advanced Phase 1 Step 3 → [DONE]; Phase 1 Step 4 → [WIP]
- Workflow sync: Advanced Phase 1 Step 3 → [DONE]; Phase 1 Step 4 → [WIP]
- Workflow sync: Advanced Phase 1 Step 3 → [DONE]; Phase 1 Step 4 → [WIP]
- Workflow sync: Advanced Phase 1 Step 2 → [DONE]; Phase 1 Step 3 → [WIP]
- Workflow sync: Advanced Phase 1 Step 2 → [DONE]; Phase 1 Step 3 → [WIP]
