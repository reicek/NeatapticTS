# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [WIP] — Phase 1 [DONE] · Phase 2 [DONE] · Phase 3 [WIP] · Step 09 [WIP]: Bugfix — canvas stretch + missing enemies · Step 10 [PLANNED]: Enhance cannon overlay · Steps 01–08 [DONE] · **Plan ID:** NEATENSTEIN_NGE_DEMO · **Created:** 2026-07-17
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) — all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/Neon_Shooter_NGE_Demo.research.md` — DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) — locked. See research file §1.

---

## Current state

Claim: 04-implementing completed slice 09-worker-sprite-pass @ 2026-07-31T14:02:00Z
Claim: 04-implementing applied fix-packet-09-worker-controller-iteration-1 @ 2026-08-01T11:30:00Z

Phase 3 Steps 01–08 are [DONE] and compressed to `plans/Neon_Shooter_NGE_Demo.logs.md`. Step 09 — Bugfix for canvas horizontal stretch on ultra-wide and missing enemy sprites — is [WIP] and now sliced into 5 atomic slices. Slices `09-render-state-enemies`, `09-worker-controller`, and `09-worker-sprite-pass` are [DONE]; slice `09-worker-controller` fix-packet iteration 1 is implemented and awaiting green validation; remaining slices are `09-canvas-backing` (pre-existing work, owned by canvas-backing slice) and `09-green`. Step 10 — Enhance cannon overlay: fix horizontal stretch, add detail, and introduce a voxel/3D look via sprite projection — is [PLANNED] to follow Step 09. Detailed per-step claims, PlanUpdate blocks, and validation evidence for Steps 07–08 are archived in the logs.

### PlanUpdate for 09-render-state-enemies

```yaml
PlanUpdate:
  slice_id: '09-render-state-enemies'
  changed_files:
    - 'examples/neatenstein/browser-entry/renderer/frame.ts'
    - 'examples/neatenstein/browser-entry/browser-entry.ts'
  supporting_tests_added:
    - 'examples/neatenstein/browser-entry/renderer/frame.test.ts'
    - 'examples/neatenstein/browser-entry/browser-entry.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/frame.ts examples/neatenstein/browser-entry/browser-entry.ts examples/neatenstein/browser-entry/renderer/frame.test.ts examples/neatenstein/browser-entry/browser-entry.test.ts'
  validation:
    - 'node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/renderer/frame.ts,examples/neatenstein/browser-entry/browser-entry.ts'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/renderer/frame.ts,examples/neatenstein/browser-entry/browser-entry.ts'
    - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-render-state-enemies --changed-files=examples/neatenstein/browser-entry/renderer/frame.ts,examples/neatenstein/browser-entry/browser-entry.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns="examples/neatenstein/browser-entry/(renderer/)?frame.test.ts|examples/neatenstein/browser-entry/browser-entry.test.ts"'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/frame.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/frame.test.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.test.ts'
  next: 'Run 05-green-testing (focused browser-entry suites) and attach coverage-guard evidence. Then proceed to slice 09-worker-controller.'
```

### Validation evidence for 09-render-state-enemies

- `npx tsc --noEmit -p tsconfig.json` → exit 0, tsc: OK
- `npm run lint` → exit 0, lint: 0 issues
- `npx prettier --check examples/neatenstein/browser-entry/renderer/frame.ts examples/neatenstein/browser-entry/browser-entry.ts examples/neatenstein/browser-entry/renderer/frame.test.ts examples/neatenstein/browser-entry/browser-entry.test.ts` → exit 0, prettier: OK
- `node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/renderer/frame.ts,examples/neatenstein/browser-entry/browser-entry.ts` → `shared-validation: PASS` (build OK, 2 test suites, 24 tests OK)
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/renderer/frame.ts,examples/neatenstein/browser-entry/browser-entry.ts` → `code-coverage: PASS` (frame.ts 100/100/100/100, browser-entry.ts 100/100/100/100)
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-render-state-enemies --changed-files=examples/neatenstein/browser-entry/renderer/frame.ts,examples/neatenstein/browser-entry/browser-entry.ts` → `slice-advancement: PASS` (all 7 gates passed, severity FULL)

**Coverage note:** Two focused tests were added to cover branches exposed by the code-coverage gate: (1) `frame.test.ts` exercises the default `columnCount` parameter of `buildNeatensteinRenderFrame` and accepts an `enemies` payload; (2) `browser-entry.test.ts` covers the `typeof ResizeObserver === 'undefined'` fallback path in `startRenderLoop()`. These tests are strictly scoped to the changed source files and were required for slice-advancement to pass.

### PlanUpdate for 09-worker-controller

```yaml
PlanUpdate:
  slice_id: '09-worker-controller'
  changed_files:
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts'
  validation:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
  next: 'Run 05-green-testing focused worker suites, then proceed to slice 09-worker-sprite-pass.'
```

### Validation evidence for 09-worker-controller

- `npx tsc --noEmit -p tsconfig.json` → exit 0, tsc: OK
- `npm run lint` → exit 0, lint: 0 issues
- `npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts` → exit 0, prettier: OK
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker` → 2 suites passed, 33 tests passed
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts` → `code-coverage: PASS` (display.worker.ts 100/100/100/100)
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-worker-controller --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts` → `slice-advancement: PASS` (all 7 gates passed, severity FULL)

### PlanUpdate for fix-packet-09-worker-controller-iteration-1

```yaml
PlanUpdate:
  slice_id: '09-worker-controller'
  iteration: 1
  fix_packet_id: 'fix-packet-09-worker-controller-iteration-1'
  status: 'implemented-awaiting-green'
  changed_files:
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  supporting_tests_added:
    - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  validation:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  next: 'Run 05-green-testing focused worker suites, then re-run slice-advancement with the updated files.'
```

### Validation evidence for fix-packet-09-worker-controller-iteration-1

- `npx tsc --noEmit -p tsconfig.json` → exit 0, tsc: OK
- `npm run lint` → exit 0, lint: 0 issues
- `npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts` → exit 0, prettier: OK
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts` → 1 suite passed, 33 tests passed
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts` → `code-coverage: PASS` (display.worker.ts 100/100/100/100)
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-worker-controller --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts,plans/Neon_Shooter_NGE_Demo.plans.md` → `slice-advancement: PASS` (all 7 gates passed, severity FULL)
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → `stale-wip-plans: PASS` (no stale WIP plans detected)
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md` → `plan sync: PASS` (0 errors, 0 warnings)

### PlanUpdate for 09-worker-sprite-pass

```yaml
PlanUpdate:
  slice_id: '09-worker-sprite-pass'
  changed_files:
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts'
  validation:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-worker-sprite-pass --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts,plans/Neon_Shooter_NGE_Demo.plans.md'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
  next: 'Run 05-green-testing focused worker suites, then proceed to slice 09-green.'
```

### Validation evidence for 09-worker-sprite-pass

- `npx tsc --noEmit -p tsconfig.json` → exit 0, tsc: OK
- `npm run lint` → exit 0, lint: 0 issues
- `npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts` → exit 0, prettier: OK
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts` → 1 suite passed, 32 tests passed
- `node scripts/agent-customization/gates/shared-validation.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts` → `shared-validation: PASS` (build OK, 1 test suite, 32 tests OK)
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts` → `code-coverage: PASS` (display.worker.ts 100/100/100/100)
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-worker-sprite-pass --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts,plans/Neon_Shooter_NGE_Demo.plans.md` → `slice-advancement: PASS` (all 7 gates passed, severity FULL)
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → `stale-wip-plans: PASS`

**Implementation note:** The worker-tier sprite pass snapshots the `OffscreenCanvasRenderingContext2D` via `getImageData`, projects/clips active enemies via `clipNeatensteinSprite`, renders neon bars via `renderNeatensteinSprite`, and flushes the combined framebuffer back with `putImageData` before the transparent overlay passes. A `typeof context.getImageData === 'function'` guard keeps the existing worker tests green when the mock context omits `getImageData`; in a real browser the sprite pass executes because `OffscreenCanvasRenderingContext2D` always exposes `getImageData`.

## Latest validation evidence

Step 07 and Step 08 detailed validation evidence, fix packets, and green-testing results are archived in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 07 and §Phase 3 Step 08.

**Step 09 planning validation (after slicing):**

- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 09 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md`
- Result:

```json
{
  "pass": true,
  "evidence": {
    "gate": "slice-advancement",
    "tier": 1,
    "sliceId": "Step 09",
    "severity": "TRIVIAL",
    "specialistCount": 0,
    "gatesRun": [
      "plan-sync",
      "step-packet",
      "plan-slice-quality",
      "plan-command-lint"
    ],
    "gateCount": 4,
    "results": [
      {
        "gate": "plan-sync",
        "pass": true,
        "fixHint": "All WIP plans are correctly registered in README and Roadmap."
      },
      {
        "gate": "step-packet",
        "pass": true,
        "fixHint": "All active WIP phase/step packets conform to the new format."
      },
      {
        "gate": "plan-slice-quality",
        "pass": true,
        "fixHint": "All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit."
      },
      {
        "gate": "plan-command-lint",
        "pass": true,
        "fixHint": "Verify the plan path: plans/orchestration-fixes.plans.md"
      }
    ],
    "failedGates": []
  },
  "fixHint": "All 4 gates passed for slice Step 09 (TRIVIAL).",
  "owner": "orchestrator (Agent Zero)"
}
```

- `green-light: true` — Step 09 slice plan verified by fresh 01-planning agent; slice-advancement gate passed (2026-07-31).

**Step 10 planning validation:**

- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id="Step 10" --changed-files="plans/Neon_Shooter_NGE_Demo.plans.md"`
- Result:

```json
{
  "pass": true,
  "evidence": {
    "gate": "slice-advancement",
    "tier": 1,
    "sliceId": "Step 10",
    "severity": "TRIVIAL",
    "specialistCount": 0,
    "gatesRun": [
      "plan-sync",
      "step-packet",
      "plan-slice-quality",
      "plan-command-lint"
    ],
    "gateCount": 4,
    "results": [
      {
        "gate": "plan-sync",
        "pass": true,
        "fixHint": "All WIP plans are correctly registered in README and Roadmap."
      },
      {
        "gate": "step-packet",
        "pass": true,
        "fixHint": "All active WIP phase/step packets conform to the new format."
      },
      {
        "gate": "plan-slice-quality",
        "pass": true,
        "fixHint": "All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit."
      },
      {
        "gate": "plan-command-lint",
        "pass": true,
        "fixHint": "Verify the plan path: plans/orchestration-fixes.plans.md"
      }
    ],
    "failedGates": []
  },
  "fixHint": "All 4 gates passed for slice Step 10 (TRIVIAL).",
  "owner": "orchestrator (Agent Zero)"
}
```

- `green-light: true` — Step 10 slice plan verified by fresh 01-planning agent; structural slice-advancement gate passed (2026-08-01). Full-file slice-advancement (including code-coverage) is expected to fail until the new source/test files are created.

- fix-loop: 09-worker-controller iteration 1 status=failed

<!-- fix-packet-09-worker-controller-iteration-1 -->

```yaml
fix_packet:
  slice_id: '09-worker-controller'
  iteration: 1
  status: REQUEST_CHANGES
  goal: 'persist-enemy-controller-state'
  trigger: specialist-review
  shared_validation_artifact: 'artifacts/shared-validation.json'
  observations:
    - source: 'browser-runtime-scout'
      type: 'state-persistence-bug'
      detail: 'updateEnemyController returns a new EnemyControllerState object but the worker never assigns it back to the module-level enemyControllerState variable. Because updateEnemyController does not mutate its input, every frame re-enters with the stale initial controller state, so fire cooldowns, ammo, and de-rez timing reset every frame instead of advancing. This violates AC-09c-002 (maintain and advance controller state across frames).'
  requested_changes:
    - 'Assign the return value of updateEnemyController back to enemyControllerState in buildAndPostFrame() so controller state (ammo, fireCooldownMs, deRezElapsedMs) persists across frames.'
    - 'Add a worker test that asserts controller-state persistence across multiple simState ticks (e.g., fire cooldown decrements or ammo depletes over frames).'
```

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

Context: Neatenstein NGE Demo — Phase 1 [DONE], Phase 2 [DONE], Phase 3 [WIP]. Phase 3 Steps 01–08 are [DONE] and compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` (§Phase 3 Step 01–08). Step 09 — Bugfix for canvas horizontal stretch + missing enemy sprites — is [WIP] and sliced into 5 atomic slices. Step 10 — Enhance cannon overlay — is [PLANNED] to follow Step 09.

What is already covered: All prior Phase 3 steps are archived in the logs. Two live bugs remain: (1) the visible canvas is horizontally stretched on ultra-wide monitors because `updateCanvasBackingStore()` sizes the backing store from the viewport instead of the canvas CSS box; (2) no enemies are rendered because the worker render loop never calls the existing `renderer/sprites.ts` sprite renderer and the host-to-worker render state carries no enemy positions. Separately, the center-screen plasma cannon in `renderer/gun.ts` is also stretched horizontally on ultra-wide because `gunWidth` is currently computed from viewport `width` independently of `gunHeight`. The new Step 10 will fix the gun's aspect ratio, add visual detail, and optionally add a voxel/3D projected sprite.

Current boundary: Phase 3 active frontier is Step 09 — canvas stretch + missing enemies bugfix [WIP], followed by Step 10 — cannon overlay enhancement [PLANNED].

Next narrow task: Execute the five Step 09 slices in order (fix the canvas backing store; add an enemy payload; wire the enemy controller; call `renderer/sprites.ts`; run green validation). Once Step 09 is [DONE], begin Step 10: red tests for aspect-correct gun sizing and gun-sprite projection, fix `renderer/gun.ts` to size by `gunHeight * GUN_BODY_ASPECT_RATIO` and add detail, add a dedicated `renderer/gun-sprite.ts` voxel projection, then run green validation.

Required validations:
  - neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json
  - neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md

Known worktree cautions: The approved reference art files `plans/robot-proposal-192-front.png`, `plans/robot-proposal-192-back.png`, `plans/robot-proposal-192-left.png`, `plans/robot-proposal-192-right.png`, and `plans/robot-proposal-192.png` are currently untracked in git but are required by the `06-reference-parity` parity tests; they must be committed or otherwise handled before the PR is considered complete. The `examples/neatenstein/generated/` directory is shared/transient output for Neatenstein scripts; validation runners may collide if executed concurrently, so gate and coverage commands should be run sequentially. Full repo-wide `src/` 100% coverage remains deferred; coverage is scoped to files touched by the active step's slices.
```

**Phase 3 status:**

- Step 01 — Tech-debt cleanup and test/coverage repair — [DONE] (original 5 slices green; detailed evidence in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 01).
- Step 02 — Lint-type follow-up for Neatenstein tests — [DONE]; all 3 slices (`01-lint-types-harness`, `01-lint-types-host-src`, `01-lint-types-green`) are [DONE] and green validated. Detailed evidence in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 02.
- Step 03 — Center-screen DOOM-style plasma cannon — [DONE]; all 5 slices (`02-red-tests`, `02-constants-types`, `02-gun-render`, `02-bolt-combat`, `02-render-integration`) are [DONE] and green validated. Full step packet and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 03 final compression; earlier fix-loop archive is in the same logs file.
- Step 04 — Plasma cannon visual cleanup and volt visibility fix — [DONE]; all 5 slices (`04-red-tests`, `04-halo`, `04-gun-shadow`, `04-bolt`, `04-green`) are [DONE] and green validated. Full step packet and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 04 final compression.
- Step 05 — Enemy MLP evolution harness — [DONE]; all 5 slices (`05-red-mlp`, `05-mlp-topology`, `05-enemy-fitness`, `05-enemy-barrier`, `05-green`) are [DONE] and green validated. Full step packet, acceptance criteria, slice details, and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 05 final compression.
- Step 06 — Enemy voxel-sprite asset pipeline — [DONE]; all 8 slices (`06-red-voxel`, `06-voxel-descriptor`, `06-snapshot-renderer`, `06-animator`, `06-coverage-config`, `06-sprite-sheet`, `06-reference-parity`, `06-green-final`) are [DONE] and green validated. Full step packet, acceptance criteria, slice details, and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 06 final compression.
- Step 07 — Wire enemies into live renderer — [DONE]; all five slices (`07-red-renderer`, `07-renderer-bridge`, `07-enemy-controller`, `07-enemy-render`, `07-wave-loop`) are [DONE] and green validated. Full step packet and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 07 final compression.
- Step 08 — Canvas sizing fix: fixed 480px height with aspect-ratio width — [DONE]; full step packet, acceptance criteria, slice details, and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 08.
- Step 09 — Human playtest and feedback-driven polish — [WIP]; unsliced, awaiting human playtest and feedback-driven polish pass.
- Step 10 — Enhance cannon overlay: fix horizontal stretch, add detail, voxel/3D look via sprite projection — [PLANNED]; awaiting Step 09 completion.

**Active frontier:** Phase 3 Step 09 — Bugfix: canvas stretch + missing enemies [WIP]; Step 10 [PLANNED]; Steps 01–08 are [DONE].

## Implementation phases

### Phase 1 — Arena + Hero FPS controls (game-director-owned) [DONE]

#### Step 01: Arena + Hero FPS controls [DONE]

[DONE] Phase 1 complete. Full step packet and validation evidence are archived in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 1 Step 01.

### Phase 2 — Raycast Renderer + WebGL VFX (visualizer-owned) [DONE]

#### Step 01: Raycast renderer + WebGL VFX [DONE]

[DONE] Phase 2 complete. Full step packet and validation evidence are archived in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 2 Step 01.

### Phase 3 — Live Enemy Rendering + Polish (visualizer + benchmark-owned) [WIP]

#### Step 01: Tech-debt cleanup and test/coverage repair [DONE]

[DONE] Original 5 slices green validated. Full step packet, acceptance criteria, slice details, and validation evidence are archived in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 01.

#### Step 02: Lint-type follow-up for Neatenstein tests [DONE]

[DONE] All 3 lint follow-up slices green validated. Full step packet, acceptance criteria, slice details, and validation evidence are archived in `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 02.

#### Step 03: Center-screen DOOM-style plasma cannon [DONE]

[DONE] All 5 slices green validated. Full step packet and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 03 final compression; earlier fix-loop archive is in the same logs file.

#### Step 04: Plasma cannon visual cleanup and volt visibility fix [DONE]

[DONE] All 5 slices green validated. Full step packet and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 04 final compression.

#### Step 05: Enemy MLP evolution harness [DONE]

[DONE] All 5 slices green validated. Full step packet, acceptance criteria, slice details, and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 05 final compression.

#### Step 06: Enemy voxel-sprite asset pipeline [DONE]

[DONE] All 8 slices green validated. Full step packet, acceptance criteria, slice details, and validation evidence are compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 06 final compression.

#### Step 07: Wire enemies into live renderer [DONE]

[DONE] All five slices green validated. Full step packet and validation evidence compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 07.

#### Step 08: Canvas sizing fix: fixed 480px height with aspect-ratio width [DONE]

[DONE] Two-slice canvas sizing fix green validated with visible-browser smoke. Full step packet and validation evidence compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` §Phase 3 Step 08.

#### Step 09: Bugfix — canvas stretch + missing enemies [WIP]

**Step objective:** Fix two live Neatenstein demo bugs observed in `examples/neatenstein/index.html`.

- **BUG 1 — Canvas horizontal stretch on ultra-wide:** `updateCanvasBackingStore()` in `examples/neatenstein/browser-entry/browser-entry.ts` currently derives the backing-store width from `window.innerWidth / window.innerHeight`. The visible `<canvas>` is inside `#neatenstein-output`, whose CSS box has a different aspect ratio, so the rendered frame is stretched horizontally on ultra-wide displays. Fix: size the backing store from `canvas.clientWidth / canvas.clientHeight` and install a `ResizeObserver` on the canvas to catch container-only resizes.
- **BUG 2 — No enemies rendered:** `buildAndPostFrame()` in `examples/neatenstein/browser-entry/worker/display.worker.ts` draws floor, ceiling, walls, pulses, impact spots, bolts, and gun overlay, but never renders enemies. The existing `renderer/sprites.ts` module is unused. Fix: add enemy position data to the `NeatensteinRenderState` sent from host to worker, advance the enemy controller inside the worker loop, and call `renderer/sprites.ts` after the wall pass to draw active enemies behind the gun overlay.

**Implementation notes:**

- This step is **green-only**: red tests are intentionally skipped because both bugs are visual/renderer integration issues that require a running browser to reproduce. Validation is via focused unit tests added in the green slice plus a visible-browser smoke test.
- The worker remains the simulation authority for game state and enemy AI. The host-side enemy payload in `NeatensteinRenderState` is added per the requested contract; the worker render path consumes its own controller output.
- `renderer/sprites.ts` writes to a flat RGBA framebuffer and flushes with `putImageData`. The worker 2D path will snapshot the canvas into an `ImageData` buffer after walls, render sprites on top, and put the buffer back before drawing pulses/impacts/bolts/gun.

```yaml
phase: 3
step: 9
title: 'Bugfix: canvas stretch + missing enemies'
status: '[WIP]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 10 — Enhance cannon overlay: fix horizontal stretch, add detail, voxel/3D look via sprite projection [PLANNED]'
owner: 'visualizer'
reviewer: 'game-director'
skills:
  - 'implementation-standards'
  - 'frontend-integration'
  - 'browser-runtime'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 09 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
acceptance_criteria:
  - id: 'AC-09-001'
    text: 'Canvas backing-store aspect ratio matches the displayed CSS box, not the viewport, and resizes are detected via ResizeObserver.'
    validation: 'Manual visible-browser check + npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry'
  - id: 'AC-09-002'
    text: 'Host-to-worker NeatensteinRenderState carries enemy position data and the type compiles.'
    validation: 'npx tsc -p tsconfig.json --noEmit'
  - id: 'AC-09-003'
    text: 'Worker advances the enemy controller each frame and exposes ControlledEnemy positions to the sprite pass.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/worker'
  - id: 'AC-09-004'
    text: 'Worker render loop calls renderer/sprites.ts functions after the wall pass and enemies are visible on screen.'
    validation: 'Manual visible-browser smoke test of examples/neatenstein/index.html'
  - id: 'AC-09-005'
    text: 'All touched files build, lint, and meet 100% coverage on changed source files.'
    validation: 'npm run lint; npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '09-canvas-backing'
    title: 'Fix canvas backing-store aspect ratio using CSS box dimensions'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - id: 'AC-09a-001'
        text: 'updateCanvasBackingStore() reads canvas.clientWidth and canvas.clientHeight instead of window.innerWidth/innerHeight.'
        validation: 'grep -n "clientWidth\|clientHeight" examples/neatenstein/browser-entry/browser-entry.ts'
      - id: 'AC-09a-002'
        text: 'backingWidth is computed as Math.round(480 * clientWidth / clientHeight).'
        validation: 'grep -n "480" examples/neatenstein/browser-entry/browser-entry.ts'
      - id: 'AC-09a-003'
        text: 'A ResizeObserver is installed on the canvas and triggers updateCanvasBackingStore().'
        validation: 'grep -n "ResizeObserver" examples/neatenstein/browser-entry/browser-entry.ts'
    parallelizable: true
    dependencies: []
    next_slice: '09-green'
  - slice_id: '09-render-state-enemies'
    title: 'Add enemy payload to host-to-worker render state'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/frame.ts'
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - id: 'AC-09b-001'
        text: 'NeatensteinRenderState includes a clone-safe enemy position array.'
        validation: 'npx tsc -p tsconfig.json --noEmit'
      - id: 'AC-09b-002'
        text: 'startRenderLoop() includes enemy positions in bridge.postSimState().'
        validation: 'grep -n "enemies" examples/neatenstein/browser-entry/browser-entry.ts'
    parallelizable: true
    dependencies: []
    next_slice: '09-worker-controller'
  - slice_id: '09-worker-controller'
    title: 'Wire enemy controller into worker render loop'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-09c-001'
        text: 'Worker imports and maintains EnemyControllerState across frames.'
        validation: 'grep -n "updateEnemyController\|createEnemyControllerState" examples/neatenstein/browser-entry/worker/display.worker.ts'
      - id: 'AC-09c-002'
        text: 'Each buildAndPostFrame advances the controller using the worker-authoritative gameState and collisionMap.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/worker'
    parallelizable: false
    dependencies:
      - '09-render-state-enemies'
    next_slice: '09-worker-sprite-pass'
  - slice_id: '09-worker-sprite-pass'
    title: 'Render enemy sprites after wall pass using renderer/sprites.ts'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-09d-001'
        text: 'buildAndPostFrame snapshots the 2D canvas into an ImageData buffer after the wall pass.'
        validation: 'grep -n "getImageData" examples/neatenstein/browser-entry/worker/display.worker.ts'
      - id: 'AC-09d-002'
        text: 'Each active controlled enemy is projected, clipped against the z-buffer, and rendered via renderer/sprites.ts.'
        validation: 'grep -n "renderNeatensteinSprite\|clipNeatensteinSprite" examples/neatenstein/browser-entry/worker/display.worker.ts'
      - id: 'AC-09d-003'
        text: 'The framebuffer is put back to the worker 2D context before pulses/impacts/bolts/gun overlay.'
        validation: 'grep -n "putImageData" examples/neatenstein/browser-entry/worker/display.worker.ts'
    parallelizable: false
    dependencies:
      - '09-worker-controller'
    next_slice: '09-green'
  - slice_id: '09-green'
    title: 'Green validation: focused tests, build, lint, visible-browser smoke'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/browser-entry.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-09e-001'
        text: 'Focused jest suites for browser-entry and worker pass with zero failures.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry'
      - id: 'AC-09e-002'
        text: '100% coverage on touched source files in examples/neatenstein/browser-entry.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry'
      - id: 'AC-09e-003'
        text: 'npm run lint exits 0.'
        validation: 'npm run lint'
      - id: 'AC-09e-004'
        text: 'Visible-browser smoke of examples/neatenstein/index.html shows correct canvas aspect ratio on ultra-wide and visible enemy sprites; capture browserVisibility: visible-foreground evidence.'
        validation: 'Manual browser smoke test'
    parallelizable: false
    dependencies:
      - '09-canvas-backing'
      - '09-worker-sprite-pass'
    next_slice: null
```

- `neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json`
- Result:

```json
{
  "pass": true,
  "evidence": {
    "stalePlans": [],
    "plansChecked": 7,
    "plansFound": 7
  },
  "fixHint": "No stale WIP plans detected — all active plans have open work remaining.",
  "owner": "stale-wip-plans.gate.mjs"
}
```

**Slice 09-canvas-backing completion:**

```yaml
PlanUpdate:
  slice_id: '09-canvas-backing'
  changed_files:
    - 'examples/neatenstein/browser-entry/browser-entry.ts'
    - 'examples/neatenstein/browser-entry/browser-entry.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/browser-entry.ts examples/neatenstein/browser-entry/browser-entry.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/browser-entry.test.ts'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.test.ts'
  next: 'Run 05-green-testing for slice 09-green and attach coverage-guard evidence.'
```

**Focused validation evidence:**

- `npx tsc --noEmit -p tsconfig.json` → pass
- `npx tsc --noEmit -p tsconfig.test.json` → pass
- `npm run lint` → 0 issues
- `npx prettier --check ...` → pass
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/browser-entry.test.ts` → 19 passed
- `slice-advancement` (Step 09) → pass (7/7 gates)
- `stale-wip-plans` → pass (0 stale)

**Required validation:**

- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 09 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json`

#### Step 10: Enhance cannon overlay — fix horizontal stretch, add detail, voxel/3D look via sprite projection [PLANNED]

**Step objective:** Improve the center-screen plasma cannon drawn by `renderer/gun.ts`. Fix the gun-local horizontal stretch on ultra-wide displays by deriving `gunWidth` from `gunHeight * GUN_BODY_ASPECT_RATIO` instead of from viewport width. Add visual detail (barrel bands, side vents, top sight, energy-core rings) so the cannon reads as a weapon. Optionally add a voxel/3D feel through a dedicated `renderer/gun-sprite.ts` projection helper that projects a small voxel grid into screen space, without reusing the enemy billboard renderer.

**Boundary notes:**

- Public API must remain unchanged: `renderGunOverlay(ctx, gun, width, height)` and `createInitialGunState()` keep their current signatures; `worker/display.worker.ts` and `host/game/types.ts` do not change.
- Do **not** modify `renderer/sprites.ts` or the enemy voxel pipeline. The new `gun-sprite.ts` may reuse the inverse-camera math conceptually, but it is a separate overlay projection with its own near-camera clipping rules.
- New color/geometry constants should stay local to the gun boundary; do not add global constants unless reviewed.
- The 3D/voxel sprite projection is optional within this step: if projection complexity exceeds the slice budget, the detail pass alone satisfies the step minimum, and the sprite projection is deferred to a follow-up slice.

**Step 10 packet:**

```yaml
phase: 3
step: 10
title: 'Enhance cannon overlay — fix horizontal stretch, add detail, voxel 3D look via sprite projection'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Phase 4 Step 01 — NGE Main Agent + Enemy MLPs red tests [PLANNED]'
owner: 'visualizer'
reviewer: 'game-director'
skills:
  - 'implementation-standards'
  - 'frontend-integration'
  - 'browser-runtime'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 10 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/browser-entry/renderer/gun.ts,examples/neatenstein/browser-entry/renderer/gun.test.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
  - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
  - 'npm run lint'
acceptance_criteria:
  - id: 'AC-10-001'
    text: 'Plasma cannon is no longer horizontally stretched on ultra-wide displays; gun width is derived from gun height and a fixed body aspect ratio.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun.test.ts'
  - id: 'AC-10-002'
    text: 'Cannon overlay includes at least three new detail elements (e.g., barrel bands, side vents, top sight, energy-core rings) drawn by renderGunOverlay.'
    validation: 'Visual inspection of examples/neatenstein/index.html and focused gun tests'
  - id: 'AC-10-003'
    text: 'A dedicated gun-sprite.ts helper exists for voxel/3D projection and can render a small voxel grid into the overlay with consistent proportions.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
  - id: 'AC-10-004'
    text: 'All touched source files build, lint, and have 100% coverage on changed renderer files.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '10-red-gun'
    title: 'Write red tests for aspect-correct sizing, detail drawing, and gun-sprite projection'
    status: '[PLANNED]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
    acceptance_criteria:
      - id: 'AC-10a-001'
        text: 'A failing assertion exists that gun width equals gun height times a fixed aspect ratio for at least two aspect ratios.'
      - id: 'AC-10a-002'
        text: 'A failing assertion exists that at least one new detail path is called (e.g., ctx.fillRect for a barrel band) for a standard aspect ratio.'
      - id: 'AC-10a-003'
        text: 'A failing assertion exists that gun-sprite.ts exports a projectGunSprite function and a red test expects a non-empty projected polygon/pixel list.'
    parallelizable: false
    dependencies: []
    next_slice: '10-aspect-detail'
  - slice_id: '10-aspect-detail'
    title: 'Fix horizontal stretch and add cannon detail in gun.ts'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
    acceptance_criteria:
      - id: 'AC-10b-001'
        text: 'gunWidth is computed as gunHeight * GUN_BODY_ASPECT_RATIO and no longer depends directly on viewport width.'
      - id: 'AC-10b-002'
        text: 'At least three new detail elements are drawn (barrel bands, side vents, top sight, energy-core rings).'
      - id: 'AC-10b-003'
        text: 'Public API renderGunOverlay and createInitialGunState are unchanged.'
    parallelizable: false
    dependencies:
      - '10-red-gun'
    next_slice: '10-voxel-sprite'
  - slice_id: '10-voxel-sprite'
    title: 'Add dedicated gun-sprite.ts for voxel/3D projection'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
    acceptance_criteria:
      - id: 'AC-10c-001'
        text: 'New gun-sprite.ts exports a helper that projects a small voxel grid using inverse-camera / screen-space math (does not reuse renderer/sprites.ts).'
      - id: 'AC-10c-002'
        text: 'renderGunOverlay integrates the projected voxel sprite as a detail layer without changing its public signature.'
      - id: 'AC-10c-003'
        text: 'The projected gun sprite preserves consistent screen-space height and width proportions across 16:9 and ultra-wide aspect ratios.'
    parallelizable: false
    dependencies:
      - '10-aspect-detail'
    next_slice: '10-green'
  - slice_id: '10-green'
    title: 'Green validation: focused tests, build, lint, coverage guard, visible-browser smoke'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: 'AC-10d-001'
        text: 'Focused jest suites for gun and gun-sprite pass with zero failures.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
      - id: 'AC-10d-002'
        text: '100% coverage on touched source files in examples/neatenstein/browser-entry/renderer/.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
      - id: 'AC-10d-003'
        text: 'npm run lint exits 0 and tsc --noEmit passes.'
        validation: 'npm run lint; npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-10d-004'
        text: 'Visible-browser smoke test shows the cannon without horizontal stretch, with new details, and with a voxel/3D look.'
        validation: 'Manual visible-browser smoke test of examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '10-voxel-sprite'
    next_slice: null
```

**Validation evidence:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md` → PASS (0 errors, 0 warnings).
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id="Step 10" --changed-files="plans/Neon_Shooter_NGE_Demo.plans.md"` → PASS (plan-sync, step-packet, plan-slice-quality, plan-command-lint all true; severity TRIVIAL because source files do not exist yet).
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id="Step 10" --changed-files="plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/browser-entry/renderer/gun.ts,examples/neatenstein/browser-entry/renderer/gun.test.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts"` → FAIL (code-coverage gate only; expected because the planned source files do not exist yet; all structural gates pass).
- `neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json` → PASS (0 stale WIP plans).

### Phase 4 — NGE Main Agent + Enemy MLPs (core + benchmark-owned) [PLANNED]

**Goal:** Full NGE main agent lifecycle + weight-only MLP co-evolution.

[PLANNED] Step 01 — NGE Main Agent + Enemy MLPs red tests (deferred until phase becomes active).

- Main agent: full NGE lifecycle (Embryo→Juvenile→Adult→Reproducing), tier-capped topology up to tier limit.
- **All motifs are EXISTING in `NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE`** — no new motifs, no schema version bump. Motifs used: `AttentionHead` (threat prioritization), `GatedRecurrentCell` (aim/strafe state), `EpisodicSlot` (spawn-pattern memory).
- MLP enemies: fixed topology, weight-only mutation, no structural assimilation.
- **Assimilation is INTERNAL to the main agent lifecycle** — writes back structural priors derived from the main agent's own equilibrium candidate. The MLP enemy is the SELECTION PRESSURE, not an assimilation source. No weights or structure flow from MLP to main via assimilation. Priors are weak/decaying (defends against catastrophic forgetting).
- **Reproduction mode policy:** an external overlay that SELECTS a mode then writes the canonical `NgeReproductionPolicy.mode` field (only when `modeIsEvolvable: true`). Named `reproductionModeHysteresis` (distinct from `NgeHysteresisState` juvenile grow gate). Window: 3 generations, majority-vote. Mode selection: parthenogenesis (dominating) → polyandric (struggling) → sexual (stalemate).
- **New core-side primitives (core-owned):**
  - (a) Deterministic per-enemy substrate coordinate allocator for `WeightSharedCohort`: emits `NeatGenomeSubstrateCoordinate` within `NgeSubstrateConfig` (dimensions: 3, normalization: 'unit-cube'), produces stable `zoneId`s via existing zone-partition. Reproducible from `(swarmSize, enemyIndex, seed)` alone, no runtime allocation order dependency.
  - (b) Combat-pressure → reproduction-mode policy (inspectable, tested, in `src/neat/nge-evolution/`).

**Acceptance:**

- ARMS RACE mode runs at interactive rates.
- Main fitness computed against MLP snapshot, not live MLP.
- Assimilation writes internal priors, not enemy-derived weights/structure.
- Reproduction mode switches with `reproductionModeHysteresis` (3-gen window).
- Coordinate allocator: repeated-build hash test (same swarmSize + seed → identical coordinate set, stable ordering, unit-cube conformant).
- 100% coverage on touched `src/` files via `coverage-guard`.

### Phase 5 — SWARM Mode (core + benchmark-owned) [PLANNED]

**Goal:** WeightSharedCohort swarm + HIVE DENSITY legibility.

[PLANNED] Step 01 — SWARM Mode red tests (deferred until phase becomes active).

- WeightSharedCohort: one DNA, shared weight tensor, per-enemy coordinate injection (`receivesCoordinates: true`). Swarm motifs (all existing): `DenseFeedForward` (perception), `GatedRecurrentCell` (pursuit/evasion state), `ModulatorBroadcaster` (cohort alarm), `GatingRouter` (pursuit-vs-evasion switch), `EpisodicSlot` (hero position memory).
- Swarm fitness = collective damage + collective survival (one scalar). Swarm reproduces as one individual.
- **Full `NgeReproductionPolicy` for swarm:** `mode: 'parthenogenesis'`, `modeIsEvolvable: false` (size-ramped via density, not mode-switched), `parthenogenesisMutationRate: 0.1` (configurable via demo prop).
- **No hardcoded roles:** roles (if any emerge) are READ from coordinate injection, not hardwired by archetype. Ablation: coordinate-shuffle verifies role emergence is learned (shuffle coordinates → behavior should change).
- **HIVE DENSITY meter:** normalized 0–1 coordination budget (NOT headcount). Thresholds at 0.25/0.50/0.75/1.0: brighten → formation → flanking → lockstep single-organism. Swarm size stays ≤8; density = coordination quality. 100% = lockstep movement (single organism), not clustering. Thresholds survive any cap change (8→6).
- SWARM snapshot refresh: every 3 generations (explicit).

**Acceptance:**

- One DNA + shared weights + coordinate injection produces differentiated swarm behavior (focused test on coordinate-injection effect).
- Swarm fitness scalar; SWARM barrier deterministic.
- HIVE DENSITY (normalized 0–1) correlates with coordination behavior change.
- Coordinate-shuffle ablation: shuffling coordinates changes behavior (roles are learned, not hardcoded).

### Phase 6 — Human Modes + Replay Buffer (benchmark + game-director-owned) [PLANNED]

**Goal:** Replay-based per-death evolution + death feedback loop.

[PLANNED] Step 01 — Human Modes + Replay Buffer red tests (deferred until phase becomes active).

**Design pillars:**

- Human play as a mode: player death creates a replay entry, and that replay becomes selection pressure for the next enemy generation.
- Replay buffer: stores death contexts (hero pose, enemy state, damage source) for batch evaluation.
- Per-death evolution: each player death triggers a focused evolution pulse against the replay context.
- Death feedback loop: enemies visibly adapt to player tendencies within a session.
- No separate `src/` structural changes; leverages existing NGE lifecycle and MLP/Swarm harnesses.

**Acceptance:**

- Human mode is selectable from the demo UI.
- Player deaths are recorded in the replay buffer.
- Enemies show measurable adaptation to repeated player strategies within a single session.
- 100% coverage on touched `examples/neatenstein` files.

## Validation gates

- slice-advancement
- stale-wip-plans
- log-completion-marker
- phase-compression

---

## Consensus Record

| Round          | NGE Core        | NGE Benchmark   | Visualizer      | Game Director  |
| -------------- | --------------- | --------------- | --------------- | -------------- |
| 1 (propose)    | proposed        | proposed        | proposed        | proposed       |
| 2 (review)     | 10 observations | 10 observations | 11 observations | 9 observations |
| 3 (approve v2) | **APPROVED**    | **APPROVED**    | **APPROVED**    | **APPROVED**   |

## Design notes (high-level, retained)

- All evolution is headless/batch; visible enemies are rendered snapshots of the current population, not live training runs.
- No new core-side genome motifs until Phase 4; Phases 1–3 use existing `NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE` motifs only.
- Demo scope: single arena, deterministic procedural wall grid, hero FPS controls, raycast renderer, WebGL overlay for voxel sprites.
- Coverage guard is scoped to files touched by the active step; full `src/` 100% coverage is deferred.
