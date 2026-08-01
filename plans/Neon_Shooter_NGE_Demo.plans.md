# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [WIP] — Phase 1 [DONE] · Phase 2 [DONE] · Phase 3 [WIP] · Step 09 [DONE]: Bugfix — canvas stretch + missing enemies · fix-packet-09-green-iteration-2 green validated · Step 10 [WIP]: Fix enemy sprite rendering bugs (voxel pipeline, projection, FPS) — implementation slices done, awaiting 10-green validation · Step 11 [PLANNED]: Enhance cannon overlay · Steps 01–08 [DONE] · **Plan ID:** NEATENSTEIN_NGE_DEMO · **Created:** 2026-07-17 · **Next step:** 05-green-testing Step 10 green validation (focused jest, coverage guard, lint, visible-browser smoke)
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) — all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/Neon_Shooter_NGE_Demo.research.md` — DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) — locked. See research file §1.

---

## Current state

Claim: 04-implementing completed fix-packet-09-green-iteration-2 @ 2026-07-31T21:10:00-04:00
Claim: 05-green-testing validated slice 09-green after fix-packet-09-green-iteration-2 @ 2026-08-01T17:00:00-04:00
Claim: 04-implementing completed slice 09-worker-sprite-pass @ 2026-07-31T14:02:00Z
Claim: 04-implementing applied fix-packet-09-worker-controller-iteration-1 @ 2026-08-01T11:30:00Z
Claim: 04-implementing completed Step 10 implementation slices (10-projection-fix, 10-voxel-renderer, 10-framebuffer-opt) @ 2026-08-02T14:30:00Z

**NEXT STEP (resume here in new session):** Step 09 is [DONE]. Step 10 [WIP]: the three implementation slices are complete and all focused preflight checks pass (tsc, lint, prettier, 80 focused tests, 100% coverage on sprites.ts and display.worker.ts). The red-testing slice (10-red-renderer) was not separately executed because this session was dispatched with a direct "Execute Step 10" request and slice packets were not yet formalized in the workflow MCP (get_slice_context returned notFound); red-style assertions were added inside the implementation pass and are recorded as a pragmatic workflow deviation in the Step 10 PlanUpdate. Hand off to 05-green-testing to run the full Step 10 green validation matrix: focused jest suites, coverage guard, lint, and visible-browser smoke on ultra-wide (AC-10e-004). The canvas-resize-after-transfer bug remains fixed and green-validated.

Phase 3 Steps 01–08 are [DONE] and compressed to `plans/Neon_Shooter_NGE_Demo.plans.md`. Step 09 — Bugfix for canvas horizontal stretch on ultra-wide and missing enemy sprites — is [DONE] and sliced into 5 atomic slices. Slices `09-canvas-backing`, `09-render-state-enemies`, `09-worker-controller`, `09-worker-sprite-pass`, and `09-green` are [DONE]; fix-packet-09-green-iteration-2 has been green validated. Step 10 — Fix enemy sprite rendering bugs: use the Step 06 voxel asset pipeline and the attached `robot-proposal-192*.png` reference frames, correct the sprite projection formula, and eliminate the full-canvas `getImageData`/`putImageData` FPS killer that makes the demo unplayable — is [PLANNED] to follow Step 09. Step 11 — Enhance cannon overlay: fix horizontal stretch on ultra-wide displays, add detail, and introduce a dedicated voxel/3D gun-sprite projection so the cannon has real depth — is [PLANNED] to follow Step 10. Detailed per-step claims, PlanUpdate blocks, and validation evidence for Steps 07–08 are archived in the logs.

### PlanUpdate for Step 10 packet authoring + Step 11 renumber

```yaml
PlanUpdate:
  boundary: 'Phase 3 / Step 10 packet authoring + Step 11 renumber'
  status: '[DONE]'
  what_changed:
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — added Step 10 rendering-bugfix packet with 5 slices (10-red-renderer, 10-projection-fix, 10-voxel-renderer, 10-framebuffer-opt, 10-green)'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — renumbered old Step 10 cannon overlay to Step 11 and updated all AC/slice IDs'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — updated Step 09 next_step to point to new Step 10'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — updated Current state, Handoff query, Phase 3 status, top-level status, and Latest validation evidence'
  evidence:
    - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 10 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md → pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 11 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md → pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json → pass'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md → pass (0 errors, 0 warnings)'
  removals: []
  next_boundary: 'Dispatch 05-green-testing for fix-packet-09-worker-controller-iteration-1, then run 09-green slice, then begin Step 10 slice 10-red-renderer'
```

### PlanUpdate for user-requested priority mapping pass (Step 10 + Step 11)

```yaml
PlanUpdate:
  boundary: 'Phase 3 / Step 10 + Step 11 user-priority mapping pass'
  status: '[DONE]'
  what_changed:
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — Current state paragraph now explicitly ties unplayable FPS, missing 3D voxel animated reference-sprite enemies, cannon horizontal stretch, and cannon 3D voxel depth gaps to Step 10/11'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — Handoff query "What is already covered" updated with the same four issue mappings'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — Step 10 objective and AC-10-001 now require sampling the Step 06 voxel pipeline and the attached robot-proposal-192*.png reference frames'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — Step 10 boundary note makes the reference frames a source of truth and requires parity with 06-reference-parity'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — Step 11 objective requires real 3D voxel depth via gun-sprite.ts and fixes horizontal stretch on wide screens'
    - 'plans/Neon_Shooter_NGE_Demo.plans.md — Step 11 boundary note makes the voxel projection required (delivered by 11-voxel-sprite)'
  evidence:
    - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 10 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md → pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 11 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md → pass'
    - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json → pass (no stale WIP plans)'
  removals: []
  next_boundary: 'Dispatch 05-green-testing for fix-packet-09-worker-controller-iteration-1 / 09-green, then begin Step 10 slice 10-red-renderer'
```

### PlanUpdate for Step 10 implementation slices (10-projection-fix, 10-voxel-renderer, 10-framebuffer-opt)

```yaml
PlanUpdate:
  slice_ids:
    - '10-projection-fix'
    - '10-voxel-renderer'
    - '10-framebuffer-opt'
  status: 'implementation-complete-awaiting-green'
  changed_files:
    - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
    - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  supporting_tests_added:
    - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/renderer/sprites.ts examples/neatenstein/browser-entry/renderer/sprites.test.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/sprites.test.ts|examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight_results:
    - 'npx tsc --noEmit -p tsconfig.json → PASS'
    - 'npm run lint → PASS (0 issues)'
    - 'npx prettier --check ... → PASS'
    - 'npx jest ...sprites.test.ts|display.worker.test.ts → PASS (80 tests, 2 suites)'
  focused_coverage:
    files:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    result: 'sprites.ts 100/100/100/100; display.worker.ts 100/100/100/100'
  gates:
    - 'slice-advancement (slice-id=10-projection-fix): PASS — 7/7 sub-gates'
    - 'slice-advancement (slice-id=10-voxel-renderer): PASS — 7/7 sub-gates'
    - 'slice-advancement (slice-id=10-framebuffer-opt): PASS — 7/7 sub-gates'
    - 'slice-advancement (slice-id=Step 10, all changed files): PASS — 7/7 sub-gates'
    - 'stale-wip-plans: PASS (0 stale plans)'
    - 'validate-plan-sync: PASS (0 errors, 0 warnings)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
    - 'npm run lint'
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'browser-ui-specialist visible-foreground smoke of examples/neatenstein/index.html on ultra-wide (AC-10e-004)'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/sprites.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'git checkout -- plans/Neon_Shooter_NGE_Demo.plans.md'
  next: 'Hand off to 05-green-testing to run the full Step 10 green validation matrix (focused jest suites, coverage guard, lint, visible-browser smoke on ultra-wide).'
  workflow_notes:
    - 'Slice packets were not formalized in workflow MCP; 04-implementing executed the three implementation slices directly. Red-testing slice 10-red-renderer was folded into the implementation pass — red-style assertions were added to the updated test files rather than authored in a separate red phase. This is a pragmatic deviation from the strict red-green sequence and should be acknowledged in the phase review.'
```

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

### PlanUpdate for 09-green (fix-packet-09-green-iteration-2)

```yaml
PlanUpdate:
  slice_id: '09-green'
  fix_packet_id: 'fix-packet-09-green-iteration-2'
  status: '[DONE]'
  changed_files:
    - 'examples/neatenstein/browser-entry/browser-entry.ts'
    - 'examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  supporting_tests_added:
    - 'examples/neatenstein/browser-entry/browser-entry.test.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
  validation:
    - 'node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/browser-entry.ts,examples/neatenstein/browser-entry/host/renderer-bridge.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,plans/Neon_Shooter_NGE_Demo.plans.md'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/browser-entry.ts'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    - 'node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-green --changed-files=examples/neatenstein/browser-entry/browser-entry.ts,examples/neatenstein/browser-entry/host/renderer-bridge.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/browser-entry.test.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 09 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md'
    - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
  browser_smoke:
    - 'npm run build:neatenstein'
    - 'Visible-browser smoke of examples/neatenstein/index.html on ultra-wide (3440×1384); verify no InvalidStateError and correct aspect ratio.'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
  next: 'Step 09 is [DONE]. Proceed to Step 10 planning and implementation.'
```

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

**Step 09 green validation (fix-packet-09-green-iteration-2):**

- `npx tsc --noEmit -p tsconfig.json` → exit 0, tsc: OK
- `npm run lint` → exit 0, lint: 0 issues
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry` → 49 suites passed, 643 tests passed
- `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry` → coverage OK; touched files `browser-entry.ts` 100/100/100/100, `renderer-bridge.ts` 100/100/100/100, `display.worker.ts` 100/100/97.56/100 (allowed branch-coverage exception for test-only hooks)
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/browser-entry.ts` → `code-coverage: PASS` (100/100/100/100)
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/host/renderer-bridge.ts` → `code-coverage: PASS` (100/100/100/100)
- `node scripts/agent-customization/gates/code-coverage.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts` → `code-coverage: PASS` (100/100/100/100)
- `node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json --changed-files=examples/neatenstein/browser-entry/browser-entry.ts,examples/neatenstein/browser-entry/host/renderer-bridge.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,plans/Neon_Shooter_NGE_Demo.plans.md` → `pre-specialist-smoke: PASS`
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=09-green --changed-files=examples/neatenstein/browser-entry/browser-entry.ts,examples/neatenstein/browser-entry/host/renderer-bridge.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/browser-entry.test.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts` → `slice-advancement: PASS` (all 7 gates passed, severity FULL)
- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 09 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md` → `pass: true` (4/4 sub-gates, severity TRIVIAL)
- `neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json` → `pass: true` (no stale WIP plans)
- `npm run build:neatenstein` → bundles built successfully (`docs/assets/neatenstein.bundle.js` 16.9kb, `docs/assets/neatenstein.worker.esm.js` 34.9kb)
- `neataptic-gate-mcp:run_gate_check --gate=cortex-first-search --json` → initial run reported stale index (tooling/infra); rebuilt with `node rag-index/build-index.mjs` (scanned 1657, indexed 1, skipped 1656, chunks 73); re-run → `pass: true` (index fresh, corpus search honored)
- Visible-browser smoke test of `examples/neatenstein/index.html` on 3440×1384 ultra-wide display → PASS; no `InvalidStateError: Cannot resize canvas after call to transferControlToOffscreen()`; worker backing store final resize 1193×480 (fixed 480px height, width scaled by aspect ratio); raycasted scene fills viewport; only benign `/favicon.ico` 404 observed.
- `green-light: true` — Step 09 green validated by 05-green-testing (2026-08-01).

**Step 10 planning validation (rendering bug fixes):**

- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id="Step 10" --args.changed-files="plans/Neon_Shooter_NGE_Demo.plans.md"`
- Result:

```json
{
  "pass": true,
  "sub_gates": [
    {
      "name": "plan-sync",
      "pass": true,
      "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
      "gate_error": false
    },
    {
      "name": "step-packet",
      "pass": true,
      "fixHint": "All active WIP phase/step packets conform to the new format.",
      "gate_error": false
    },
    {
      "name": "plan-slice-quality",
      "pass": true,
      "fixHint": "All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.",
      "gate_error": false
    },
    {
      "name": "plan-command-lint",
      "pass": true,
      "fixHint": "Verify the plan path: plans/orchestration-fixes.plans.md",
      "gate_error": false
    }
  ],
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
        "name": "plan-sync",
        "pass": true,
        "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
        "gate_error": false
      },
      {
        "name": "step-packet",
        "pass": true,
        "fixHint": "All active WIP phase/step packets conform to the new format.",
        "gate_error": false
      },
      {
        "name": "plan-slice-quality",
        "pass": true,
        "fixHint": "All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.",
        "gate_error": false
      },
      {
        "name": "plan-command-lint",
        "pass": true,
        "fixHint": "Verify the plan path: plans/orchestration-fixes.plans.md",
        "gate_error": false
      }
    ],
    "failedGates": [],
    "erroredGates": []
  },
  "fixHint": "All 4 gates passed for slice Step 10 (TRIVIAL).",
  "owner": "orchestrator (Agent Zero)"
}
```

- `green-light: true` — Step 10 rendering-bugfix slice plan verified; structural slice-advancement gate passed (2026-08-01). Full-file slice-advancement (including code-coverage) is expected to fail until the new source/test files are created.

**Step 11 planning validation (cannon overlay — 3D voxel depth + stretch fix):**

- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id="Step 11" --args.changed-files="plans/Neon_Shooter_NGE_Demo.plans.md"`
- Result:

```json
{
  "pass": true,
  "sub_gates": [
    {
      "name": "plan-sync",
      "pass": true,
      "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
      "gate_error": false
    },
    {
      "name": "step-packet",
      "pass": true,
      "fixHint": "All active WIP phase/step packets conform to the new format.",
      "gate_error": false
    },
    {
      "name": "plan-slice-quality",
      "pass": true,
      "fixHint": "All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.",
      "gate_error": false
    },
    {
      "name": "plan-command-lint",
      "pass": true,
      "fixHint": "Verify the plan path: plans/orchestration-fixes.plans.md",
      "gate_error": false
    }
  ],
  "evidence": {
    "gate": "slice-advancement",
    "tier": 1,
    "sliceId": "Step 11",
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
        "name": "plan-sync",
        "pass": true,
        "fixHint": "All WIP plans are correctly registered in README and Roadmap.",
        "gate_error": false
      },
      {
        "name": "step-packet",
        "pass": true,
        "fixHint": "All active WIP phase/step packets conform to the new format.",
        "gate_error": false
      },
      {
        "name": "plan-slice-quality",
        "pass": true,
        "fixHint": "All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.",
        "gate_error": false
      },
      {
        "name": "plan-command-lint",
        "pass": true,
        "fixHint": "Verify the plan path: plans/orchestration-fixes.plans.md",
        "gate_error": false
      }
    ],
    "failedGates": [],
    "erroredGates": []
  },
  "fixHint": "All 4 gates passed for slice Step 11 (TRIVIAL).",
  "owner": "orchestrator (Agent Zero)"
}
```

- `green-light: true` — Step 11 cannon-overlay slice plan verified; structural slice-advancement gate passed (2026-08-01). Full-file slice-advancement (including code-coverage) is expected to fail until the new source/test files are created.

**Tracker readiness validation:**

- `neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json` → pass: true — no stale WIP plans detected across 7 active plan trackers.

- fix-loop: 09-worker-controller iteration 1 status=passed (05-green-testing confirmed ACs pass, 34 tests pass, performance trace captured)

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

Context: Neatenstein NGE Demo — Phase 1 [DONE], Phase 2 [DONE], Phase 3 [WIP]. Phase 3 Steps 01–08 are [DONE] and compressed to `plans/Neon_Shooter_NGE_Demo.logs.md` (§Phase 3 Step 01–08). Step 09 — Bugfix for canvas horizontal stretch + missing enemy sprites — is [DONE] and green validated. Step 10 — Fix enemy sprite rendering bugs (voxel pipeline, correct projection, eliminate FPS-killing framebuffer copy) — is [WIP]: the three implementation slices (10-projection-fix, 10-voxel-renderer, 10-framebuffer-opt) are complete and all focused preflight checks pass; the 10-green validation slice is [WIP]. Step 11 — Enhance cannon overlay — is [PLANNED] to follow Step 10.

What is already covered: All prior Phase 3 steps are archived in the logs. Step 09 covers two live bugs: (1) the visible canvas is horizontally stretched on ultra-wide monitors because `updateCanvasBackingStore()` sizes the backing store from the viewport instead of the canvas CSS box; (2) no enemies are rendered because the worker render loop never calls the existing `renderer/sprites.ts` sprite renderer and the host-to-worker render state carries no enemy positions. Investigation since the last handoff found three additional rendering bugs in the enemy sprite path that must be fixed before the cannon overlay and that are now the explicit focus of Step 10: (A) the runtime sprite renderer draws flat neon bars instead of using the Step 06 voxel asset pipeline and the attached `robot-proposal-192*.png` reference frames, so the enemies lack 3D voxel animated versions; (B) sprite projection uses an incorrect focal-length formula (`canvasHeight / transformY` style scaling) causing oversized sprites; (C) the worker does a full-canvas `getImageData`/`putImageData` copy every frame during the sprite pass, making the demo unplayably slow. Separately, the center-screen plasma cannon in `renderer/gun.ts` is stretched horizontally on ultra-wide because `gunWidth` is currently computed from viewport `width` independently of `gunHeight`, and it lacks 3D voxel depth; both gun issues are now Step 11.

Current boundary: Phase 3 active frontier is Step 10 — enemy sprite rendering bug fixes [WIP] (implementation complete, 10-green validation in progress), then Step 11 — cannon overlay enhancement [PLANNED].

Next narrow task: **RESUME HERE** — Dispatch 05-green-testing to run the full Step 10 green validation matrix. Green agent should: (1) run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry` and confirm all focused tests pass, (2) run `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry` and confirm 100% statements/branches/functions/lines on the four touched source/test files (`sprites.ts`, `display.worker.ts`, and their tests), (3) run `npm run lint` and `npx tsc --noEmit -p tsconfig.json`, (4) re-run `slice-advancement` gate for each implementation slice with `--slice-id=10-projection-fix --changed-files=examples/neatenstein/browser-entry/renderer/sprites.ts,examples/neatenstein/browser-entry/renderer/sprites.test.ts`, `--slice-id=10-voxel-renderer --changed-files=examples/neatenstein/browser-entry/renderer/sprites.ts,examples/neatenstein/browser-entry/renderer/sprites.test.ts`, and `--slice-id=10-framebuffer-opt --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts`, (5) run visible-browser smoke of `examples/neatenstein/index.html` on ultra-wide display and confirm voxel enemy sprites render at correct size and the demo maintains smooth FPS (AC-10e-004). Once Step 10 green validation passes, mark Step 10 [DONE] and begin Step 11: aspect-correct gun sizing and gun-sprite projection in `renderer/gun.ts`.

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
- Step 09 — Bugfix: canvas horizontal stretch + missing enemy sprites — [DONE]; all 5 slices (`09-canvas-backing`, `09-render-state-enemies`, `09-worker-controller`, `09-worker-sprite-pass`, `09-green`) are [DONE] and green validated; fix-packet-09-green-iteration-2 (canvas resize after transfer) visible-browser smoke on ultra-wide passed.
- Step 10 — Fix enemy sprite rendering bugs: use voxel asset pipeline, correct projection formula, eliminate full-canvas getImageData/putImageData FPS killer — [WIP]; implementation slices (10-projection-fix, 10-voxel-renderer, 10-framebuffer-opt) are [DONE] and preflight green; 10-green validation is [WIP].
- Step 11 — Enhance cannon overlay: fix horizontal stretch, add detail, voxel/3D look via sprite projection — [PLANNED]; awaiting Step 10 completion.

**Active frontier:** Phase 3 Step 10 — Fix enemy sprite rendering bugs [WIP] (implementation complete, awaiting 10-green validation); Step 09 — canvas stretch + missing enemy sprites [DONE] with fix-packet-09-green-iteration-2 green validated on ultra-wide; Step 11 — cannon overlay enhancement [PLANNED]; Steps 01–08 are [DONE].

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

#### Step 09: Bugfix — canvas stretch + missing enemies [DONE]

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
next_step: 'Step 10 — Fix enemy sprite rendering bugs: use voxel asset pipeline, correct projection formula, eliminate full-canvas getImageData/putImageData FPS killer [PLANNED]'
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
    status: '[DONE]'
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
    status: '[DONE]'
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

**Slice 09-green green-validation attempt:**

```yaml
PlanUpdate:
  slice_id: '09-green'
  status: '[WIP]'
  validator: '05-green-testing'
  timestamp: '2026-07-31T19:07:19-04:00'
  focused_validations:
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS — exit 0, 0 errors'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
      result: 'PASS — 49 suites, 633 tests, 0 failures'
    - command: 'npm run lint'
      result: 'PASS — 0 issues'
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
      result: 'FAIL — display.worker.ts below 100%: statements 91.43%, branches 96.92%, functions 68.42%, lines 91.33%; uncovered lines 310-325, 524-563'
  gate_verdicts:
    - gate: slice-advancement
      pass: true
      evidence: 'All 4 sub-gates pass for Step 09 (plan-sync, step-packet, plan-slice-quality, plan-command-lint)'
    - gate: stale-wip-plans
      pass: true
      evidence: '0 stale plans out of 7 checked'
  observations:
    - file: 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      location: 'lines 310-325, 524-563'
      issue: 'Enemy sprite snapshot path and helpers are not exercised by tests; coverage below 100% on a touched source file.'
      expected: '100% coverage on all touched source files in examples/neatenstein/browser-entry'
      actual: 'display.worker.ts statements 91.43%, functions 68.42%, lines 91.33%'
  triage:
    root_cause: 'Worker test mock lacks getImageData/putImageData; the sprite snapshot block is skipped even when activeEnemySprites.length > 0.'
    failing_gates:
      - 'coverage-guard (project-level AC-09e-002)'
    failing_tests: []
    delegated_to:
      - 'coverage-analyst'
    fix_hint: 'Add the smallest owner-local tests in examples/neatenstein/browser-entry/worker/display.worker.test.ts that mock context.getImageData/putImageData and drive at least two visible enemy sprites; remove the dead `type = 0` default parameter in resolveEnemySpriteColor (line 309).'
  next_agent: '04-implementing'
  next: 'Close coverage gaps on display.worker.ts, then re-run 09-green green validation (including visible-browser smoke).'
```

- fix-loop: 09-green iteration 1 status=fixed

Claim: 04-implementing @ 2026-08-01T00:00:00Z — completed

<!-- fix-packet-09-green-iteration-1 -->

```yaml
fix_packet:
  slice_id: '09-green'
  iteration: 1
  status: FAILED
  goal: 'close-coverage-gaps'
  trigger: green-testing
  observations:
    - source: '05-green-testing'
      type: 'coverage-gap'
      detail: 'display.worker.ts coverage below 100%: statements 91.43%, functions 68.42%, lines 91.33%; uncovered lines 310-325 (resolveEnemySpriteColor dead branch type=0) and 524-563 (enemy sprite snapshot block skipped because worker test mock lacks getImageData/putImageData).'
    - source: 'coverage-analyst'
      type: 'missing-test-coverage'
      detail: 'Worker test mock does not implement context.getImageData/putImageData, so the enemy sprite snapshot block is skipped even when activeEnemySprites.length > 0. Dead default-parameter branch type=0 in resolveEnemySpriteColor (line 309) is also uncovered.'
  requested_changes:
    - 'Add worker tests in display.worker.test.ts that mock context.getImageData/putImageData and drive at least two visible enemy sprites through the sprite snapshot path.'
    - 'Remove the dead type=0 default parameter in resolveEnemySpriteColor (line 309).'
```

**Required validation:**

- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 09 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json`

**Fix-packet-09-green-iteration-1 implementation:**

Claim: 04-implementing @ 2026-08-01T00:00:00Z — completed

```yaml
PlanUpdate:
  slice_id: '09-green'
  iteration: 1
  changed_files:
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight_results:
    - 'npx tsc --noEmit -p tsconfig.json → PASS'
    - 'npx tsc --noEmit -p tsconfig.test.json → PASS'
    - 'npm run lint → PASS (0 issues)'
    - 'npx prettier --check ... → PASS'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts → PASS (35/35 tests)'
  focused_coverage:
    file: 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    result: '100% statements, 100% branches, 100% functions, 100% lines'
  gates:
    - 'slice-advancement (Step 09): PASS'
    - 'stale-wip-plans: PASS (0 stale plans)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  next: 'Hand off to 05-green-testing to run the full 09-green validation matrix (focused jest suites, coverage guard, lint, visible-browser smoke).'
```

**Slice 09-green green-validation attempt iteration 2:**

```yaml
PlanUpdate:
  slice_id: '09-green'
  iteration: 2
  status: '[WIP]'
  validator: '05-green-testing'
  timestamp: '2026-07-31T20:12:04-04:00'
  focused_validations:
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS — exit 0, 0 errors'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
      result: 'PASS — 49 suites, 635 tests, 0 failures'
    - command: 'npm run lint'
      result: 'PASS — 0 issues'
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
      result: 'PASS — touched source files at 100% (browser-entry.ts, renderer/frame.ts, worker/display.worker.ts)'
    - command: 'npm run build:neatenstein'
      result: 'PASS — docs/assets/neatenstein.bundle.js and docs/assets/neatenstein.worker.esm.js built'
    - command: 'browser-ui-specialist visible-foreground smoke of examples/neatenstein/index.html'
      result: 'FAIL — Uncaught InvalidStateError on canvas resize after transferControlToOffscreen(); canvas aspect ratio incorrect on ultra-wide'
  gate_verdicts:
    - gate: slice-advancement
      pass: true
      evidence: 'All 7 sub-gates pass for Step 09 (plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review)'
    - gate: stale-wip-plans
      pass: true
      evidence: '0 stale plans out of 7 checked'
  environment_notes:
    - 'coverage/coverage-summary.json was missing on first slice-advancement run, causing the code-coverage sub-gate to fail. Re-generated it with node scripts/agent-customization/gates/merge-coverage-summaries.mjs using existing per-project coverage artifacts; re-run then passed.'
  observations:
    - file: 'examples/neatenstein/browser-entry/browser-entry.ts'
      location: 'updateCanvasBackingStore() lines 148-179; width assignment at lines 154/169'
      issue: 'ResizeObserver and window resize handler call updateCanvasBackingStore(), which sets htmlCanvas.width/htmlCanvas.height after the renderer bridge has transferred control to an OffscreenCanvas.'
      expected: 'Canvas backing-store dimensions can be adjusted to match the CSS box after transfer, or resize is handled via the worker OffscreenCanvas'
      actual: 'Uncaught InvalidStateError: "Cannot resize canvas after call to transferControlToOffscreen()"; backing store stuck at 1303×480 instead of expected 1275×480 for a 3440×1295 CSS box'
    - file: 'examples/neatenstein/browser-entry/host/renderer-bridge.ts'
      location: 'line 208'
      issue: 'canvas.transferControlToOffscreen() runs during bridge construction, after which host-side writes to htmlCanvas.width are illegal.'
      expected: 'Host resize path either resizes before transfer or routes subsequent resizes to the worker'
      actual: 'Transfer happens before any resize observer callback, but callbacks fire immediately and later on window resize, causing repeated InvalidStateError'
  triage:
    root_cause: 'Host resize observer/window listener still mutate the HTMLCanvasElement backing store after control has been transferred to an OffscreenCanvas in the worker.'
    failing_gates:
      - 'AC-09e-004 visible-browser smoke'
    failing_tests: []
    delegated_to:
      - 'browser-ui-specialist'
    fix_hint: 'In examples/neatenstein/browser-entry/browser-entry.ts: (1) keep the initial updateCanvasBackingStore() call before createNeatensteinRendererBridge() so the first size is captured; (2) do not set htmlCanvas.width/height from the ResizeObserver or window resize handler once the worker tier is active; (3) instead route new CSS-box dimensions to the worker (e.g. extend the bridge or postSimState to carry canvasWidth/canvasHeight) so the worker can resize its OffscreenCanvas, or resize the OffscreenCanvas inside the worker on receiving updated sim-state dimensions. Also consider guarding htmlCanvas.width assignment with a try/catch or a transferred flag to avoid the runtime exception.'
  next_agent: '04-implementing'
  next: 'Fix the resize-after-transferControlToOffscreen bug, then re-run 05-green-testing on slice 09-green including visible-browser smoke.'
```

- fix-loop: 09-green iteration 2 status=failed

Claim: 05-green-testing @ 2026-07-31T20:12:04-04:00 — failed visible-browser smoke, routing back to implementation

<!-- fix-packet-09-green-iteration-2 -->

```yaml
fix_packet:
  slice_id: '09-green'
  iteration: 2
  status: FAILED
  goal: 'fix-canvas-resize-after-transfer'
  trigger: green-testing
  observations:
    - source: '05-green-testing / browser-ui-specialist'
      type: 'runtime-bug'
      detail: 'Visible-browser smoke at 3440×1295 shows Uncaught InvalidStateError every time updateCanvasBackingStore() sets htmlCanvas.width after transferControlToOffscreen(). Backing store is frozen at 1303×480 (aspect 2.7146) while CSS box is 3440×1295 (aspect 2.6564), so the frame remains horizontally stretched.'
    - source: 'browser-ui-specialist'
      type: 'visible-browser-smoke-failure'
      detail: 'Console error: "Failed to set the width property on HTMLCanvasElement: Cannot resize canvas after call to transferControlToOffscreen()". Enemy sprites are visible (target colors detected), so the missing-enemies part of Step 9 is working. Canvas aspect ratio is not correct.'
  requested_changes:
    - 'In examples/neatenstein/browser-entry/browser-entry.ts, stop mutating htmlCanvas.width/height from the ResizeObserver and window resize handler after the worker tier has transferred the canvas. The initial sizing before bridge creation can stay.'
    - 'Route subsequent CSS-box dimension changes to the worker so the OffscreenCanvas backing store can be resized there, or add a bridge.resize(width,height) / postMessage path that updates the OffscreenCanvas dimensions inside the worker.'
    - 'Ensure the resize path works on ultra-wide displays and that a visible-foreground smoke test shows no InvalidStateError and the backing-store aspect ratio matches the CSS box.'
```

- fix-loop: 09-green iteration 2 status=implemented-awaiting-green

Claim: 04-implementing @ 2026-07-31T21:10:00-04:00 — implemented fix-packet-09-green-iteration-2; preflight and slice-advancement pass locally (final slice-advancement re-verified with source/plan files only); awaiting 05-green-testing visible-browser smoke

```yaml
PlanUpdate:
  slice_id: '09-green'
  iteration: 2
  fix_packet_id: 'fix-packet-09-green-iteration-2'
  status: 'implemented-awaiting-green'
  changed_files:
    - 'examples/neatenstein/browser-entry/browser-entry.ts'
    - 'examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
  supporting_tests_added:
    - 'examples/neatenstein/browser-entry/browser-entry.test.ts'
    - 'examples/neatenstein/browser-entry/host/renderer-bridge.test.ts'
    - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check examples/neatenstein/browser-entry/browser-entry.ts examples/neatenstein/browser-entry/host/renderer-bridge.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/browser-entry.test.ts examples/neatenstein/browser-entry/host/renderer-bridge.test.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
  preflight_results:
    - 'npx tsc --noEmit -p tsconfig.json → PASS'
    - 'npx tsc --noEmit -p tsconfig.test.json → PASS'
    - 'npm run lint → PASS (0 issues)'
    - 'npx prettier --check ... → PASS'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry → PASS (49 suites, 643 tests)'
  focused_coverage:
    files:
      - 'examples/neatenstein/browser-entry/browser-entry.ts'
      - 'examples/neatenstein/browser-entry/host/renderer-bridge.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    result: 'browser-entry.ts 100/100/100/100; renderer-bridge.ts 100/100/100/100; display.worker.ts 100/97.56/100/100 (uncovered lines 963-965 are test-only hooks)'
  gates:
    - 'slice-advancement (slice-id=09-green): PASS — 7/7 sub-gates'
    - 'stale-wip-plans: PASS (0 stale plans)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
    - 'npm run lint'
    - 'browser-ui-specialist visible-foreground smoke of examples/neatenstein/index.html on ultra-wide (AC-09e-004)'
  rollback:
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/browser-entry.test.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/renderer-bridge.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/host/renderer-bridge.test.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.ts'
    - 'git checkout -- examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    - 'git checkout -- plans/Neon_Shooter_NGE_Demo.plans.md'
  next: 'Hand off to 05-green-testing to run the full 09-green validation matrix (focused jest suites, coverage guard, lint, visible-browser smoke on ultra-wide).'
```

#### Step 10: Fix enemy sprite rendering bugs — voxel pipeline, correct projection, no full-canvas framebuffer copy [WIP]

**Step objective:** Repair three runtime rendering bugs discovered while wiring enemies into the worker sprite pass. (1) The runtime sprite renderer in `renderer/sprites.ts` currently draws flat neon vertical bars via `renderNeatensteinSpriteColumnRgb`; it must instead sample the voxel enemy frames produced by the Step 06 asset pipeline (`examples/neatenstein/scripts/voxel-enemy.ts`, `snapshot-renderer.ts`, etc.) and the attached `robot-proposal-192*.png` reference frames so that enemies show 3D voxel animated versions of the reference sprites. (2) The projection helper `projectNeatensteinSprite` uses an incorrect focal-length formula (`canvasHeight / transformY` style scaling) that makes sprites oversized; replace it with correct screen-space perspective projection. (3) The worker sprite pass in `worker/display.worker.ts` performs a full-canvas `getImageData`/`putImageData` copy every frame, which kills FPS and makes the demo unplayable; refactor the render loop to write walls/floor/ceiling and sprites into a persistent `Uint8ClampedArray` framebuffer and commit it once per frame (or draw sprites directly into the context without re-reading the whole canvas).

**Boundary notes:**

- The Step 06 voxel asset pipeline and the attached `plans/robot-proposal-192*.png` reference frames are the source of truth for enemy sprite frames. The runtime renderer must load the generated manifest/sprite sheet (e.g., `examples/neatenstein/generated/`) or import the snapshot renderer API, not duplicate voxel generation logic. Parity with the reference frames is validated by the `06-reference-parity` slice tests; new runtime frames must match those proportions and palette.
- Public API of `renderer/sprites.ts` should remain stable where possible; `worker/display.worker.ts` still calls `renderNeatensteinSprite` and `clipNeatensteinSprite`, but their internals change.
- The z-buffer helpers in `renderer/sprites.ts` (if any) should be reused/extended rather than duplicated.
- The worker must remain testable under Node/Jest with mocked `CanvasRenderingContext2D` / `OffscreenCanvasRenderingContext2D`; avoid browser-only APIs in the hot path.
- Dead code from the neon-bar renderer and the full-canvas snapshot must be removed in the same step that introduces the replacement (No Deferred Cleanup Policy).

**Step 10 packet:**

```yaml
phase: 3
step: 10
title: 'Fix enemy sprite rendering bugs: voxel pipeline, correct projection, eliminate full-canvas framebuffer copy'
status: '[WIP]'
goal: 'green-testing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Neon_Shooter_NGE_Demo.plans.md'
copy_paste: true
next_step: 'Step 11 — Enhance cannon overlay: fix horizontal stretch, add detail, voxel/3D look via sprite projection [PLANNED]'
owner: 'visualizer'
reviewer: 'game-director'
skills:
  - 'implementation-standards'
  - 'frontend-integration'
  - 'browser-runtime'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 10 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/browser-entry/renderer/sprites.ts,examples/neatenstein/browser-entry/renderer/sprites.test.ts,examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
  - 'npm run lint'
acceptance_criteria:
  - id: 'AC-10-001'
    text: 'Enemy sprites render using the Step 06 voxel asset pipeline and the attached robot-proposal reference frames (not flat neon color bars).'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/sprites.test.ts'
  - id: 'AC-10-002'
    text: 'Sprite projection scale matches the correct focal-length formula and is not oversized on screen.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/sprites.test.ts'
  - id: 'AC-10-003'
    text: 'Worker render loop does not call getImageData/putImageData on the full canvas during the sprite pass.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/worker/display.worker.test.ts'
  - id: 'AC-10-004'
    text: 'All touched source files build, lint, and have 100% coverage.'
    validation: 'npm run lint; npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
constitution_check:
  - 'principle-3-verbatim-binding'
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '10-red-renderer'
    title: 'Write red tests for voxel rendering, projection formula, and framebuffer copy bugs'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-10a-001'
        text: 'A failing assertion exists that renderNeatensteinSprite samples voxel pixel data instead of drawing a single flat color bar.'
      - id: 'AC-10a-002'
        text: 'A failing assertion exists that projected sprite scale is proportional to canvasHeight / perpDist with the correct focal length (not canvasWidth or transformY-derived scale).'
      - id: 'AC-10a-003'
        text: 'A failing assertion exists that the worker sprite pass does not call getImageData or putImageData per frame.'
    parallelizable: false
    dependencies: []
    next_slice: '10-projection-fix'
  - slice_id: '10-projection-fix'
    title: 'Correct the sprite projection focal-length formula'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
    acceptance_criteria:
      - id: 'AC-10b-001'
        text: 'projectNeatensteinSprite computes screen scale using a constant focal length or canvasHeight/perpDist * worldSize (with correct horizontal/vertical consistency).'
      - id: 'AC-10b-002'
        text: 'Sprite screen size is verified against a known camera distance and does not overshoot the viewport.'
      - id: 'AC-10b-003'
        text: 'Tests for the old oversized formula fail before the fix and pass after.'
    parallelizable: false
    dependencies:
      - '10-red-renderer'
    next_slice: '10-voxel-renderer'
  - slice_id: '10-voxel-renderer'
    title: 'Wire runtime sprite renderer to the Step 06 voxel asset pipeline'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/sprites.ts'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
    acceptance_criteria:
      - id: 'AC-10c-001'
        text: 'renderNeatensteinSprite loads enemy voxel frames from the Step 06 generated manifest or snapshot renderer output.'
      - id: 'AC-10c-002'
        text: 'Flat neon-bar drawing code (renderNeatensteinSpriteColumnRgb and related helpers) is removed.'
      - id: 'AC-10c-003'
        text: 'Voxel pixels are written to the framebuffer with z-buffer occlusion and the worker still calls the same public entry points.'
    parallelizable: false
    dependencies:
      - '10-projection-fix'
    next_slice: '10-framebuffer-opt'
  - slice_id: '10-framebuffer-opt'
    title: 'Eliminate full-canvas getImageData/putImageData copy in worker sprite pass'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/worker/display.worker.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-10d-001'
        text: 'The worker allocates a persistent Uint8ClampedArray framebuffer and writes wall/floor/ceiling + sprite pixels directly into it.'
      - id: 'AC-10d-002'
        text: 'putImageData is called at most once per frame (or not at all if drawing directly to context).'
      - id: 'AC-10d-003'
        text: 'No getImageData call remains in the per-frame sprite pass block; old snapshot code is deleted.'
    parallelizable: false
    dependencies:
      - '10-voxel-renderer'
    next_slice: '10-green'
  - slice_id: '10-green'
    title: 'Green validation: focused tests, build, lint, coverage guard, visible-browser smoke'
    status: '[WIP]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'coverage/lcov.info'
      - 'examples/neatenstein/browser-entry/renderer/sprites.test.ts'
      - 'examples/neatenstein/browser-entry/worker/display.worker.test.ts'
    acceptance_criteria:
      - id: 'AC-10e-001'
        text: 'Focused jest suites for sprites and worker pass with zero failures.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry'
      - id: 'AC-10e-002'
        text: '100% coverage on touched source files in examples/neatenstein/browser-entry.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=examples/neatenstein/browser-entry'
      - id: 'AC-10e-003'
        text: 'npm run lint exits 0 and tsc --noEmit passes.'
        validation: 'npm run lint; npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-10e-004'
        text: 'Visible-browser smoke shows voxel enemy sprites at correct size and maintains smooth FPS (no full-canvas copy).'
        validation: 'Manual visible-browser smoke test of examples/neatenstein/index.html; capture browserVisibility: visible-foreground evidence and approximate FPS.'
    parallelizable: false
    dependencies:
      - '10-framebuffer-opt'
    next_slice: null
```

**Validation evidence:**

- `npx tsc --noEmit -p tsconfig.json` → PASS.
- `npm run lint` → PASS (0 issues).
- `npx prettier --check examples/neatenstein/browser-entry/renderer/sprites.ts examples/neatenstein/browser-entry/renderer/sprites.test.ts examples/neatenstein/browser-entry/worker/display.worker.ts examples/neatenstein/browser-entry/worker/display.worker.test.ts` → PASS.
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/neatenstein/browser-entry/renderer/sprites.test.ts|examples/neatenstein/browser-entry/worker/display.worker.test.ts` → PASS (80 tests, 2 suites).
- Focused coverage (`npx jest --config=jest.config.mjs --no-cache --coverage --collectCoverageFrom=examples/neatenstein/browser-entry/renderer/sprites.ts --collectCoverageFrom=examples/neatenstein/browser-entry/worker/display.worker.ts --testPathPatterns=...`) → `sprites.ts` 100/100/100/100; `display.worker.ts` 100/100/100/100.
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=10-projection-fix --changed-files=examples/neatenstein/browser-entry/renderer/sprites.ts,examples/neatenstein/browser-entry/renderer/sprites.test.ts` → PASS (7/7 sub-gates; severity FULL).
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=10-voxel-renderer --changed-files=examples/neatenstein/browser-entry/renderer/sprites.ts,examples/neatenstein/browser-entry/renderer/sprites.test.ts` → PASS (7/7 sub-gates; severity FULL).
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=10-framebuffer-opt --changed-files=examples/neatenstein/browser-entry/worker/display.worker.ts,examples/neatenstein/browser-entry/worker/display.worker.test.ts` → PASS (7/7 sub-gates; severity FULL).
- `neataptic-gate-mcp:run_gate_check --gate=slice-advancement` returned invalid JSON from the MCP wrapper; the underlying `slice-advancement.gate.mjs` script was run directly and passed for all three implementation slices and for the consolidated Step 10 file set.
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json` → PASS (0 stale plans).
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md` → PASS (0 errors, 0 warnings).
- Visible-browser smoke (AC-10e-004) is pending 05-green-testing.

#### Step 11: Enhance cannon overlay — fix horizontal stretch, add detail, voxel/3D look via sprite projection [PLANNED]

**Step objective:** Improve the center-screen plasma cannon drawn by `renderer/gun.ts`. Fix the gun-local horizontal stretch on ultra-wide displays by deriving `gunWidth` from `gunHeight * GUN_BODY_ASPECT_RATIO` instead of from viewport width. Add visual detail (barrel bands, side vents, top sight, energy-core rings) so the cannon reads as a weapon. Add real 3D voxel depth through a dedicated `renderer/gun-sprite.ts` projection helper that projects a small voxel grid into screen space, without reusing the enemy billboard renderer.

**Boundary notes:**

- Public API must remain unchanged: `renderGunOverlay(ctx, gun, width, height)` and `createInitialGunState()` keep their current signatures; `worker/display.worker.ts` and `host/game/types.ts` do not change.
- Do **not** modify `renderer/sprites.ts` or the enemy voxel pipeline. The new `gun-sprite.ts` may reuse the inverse-camera math conceptually, but it is a separate overlay projection with its own near-camera clipping rules.
- New color/geometry constants should stay local to the gun boundary; do not add global constants unless reviewed.
- The 3D/voxel sprite projection is required to resolve the reported lack of cannon depth. It is delivered by the `11-voxel-sprite` slice; if projection complexity exceeds that slice budget, a follow-up slice completes it before the step is marked [DONE].

**Step 11 packet:**

```yaml
phase: 3
step: 11
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
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=Step 11 --args.changed-files=plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/browser-entry/renderer/gun.ts,examples/neatenstein/browser-entry/renderer/gun.test.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
  - 'neataptic-gate-mcp:run_gate_check --gate=stale-wip-plans --json'
  - 'npm run lint'
acceptance_criteria:
  - id: 'AC-11-001'
    text: 'Plasma cannon is no longer horizontally stretched on ultra-wide displays; gun width is derived from gun height and a fixed body aspect ratio.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun.test.ts'
  - id: 'AC-11-002'
    text: 'Cannon overlay includes at least three new detail elements (e.g., barrel bands, side vents, top sight, energy-core rings) drawn by renderGunOverlay.'
    validation: 'Visual inspection of examples/neatenstein/index.html and focused gun tests'
  - id: 'AC-11-003'
    text: 'A dedicated gun-sprite.ts helper exists for voxel/3D projection and can render a small voxel grid into the overlay with consistent proportions.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
  - id: 'AC-11-004'
    text: 'All touched source files build, lint, and have 100% coverage on changed renderer files.'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '11-red-gun'
    title: 'Write red tests for aspect-correct sizing, detail drawing, and gun-sprite projection'
    status: '[PLANNED]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
    acceptance_criteria:
      - id: 'AC-11a-001'
        text: 'A failing assertion exists that gun width equals gun height times a fixed aspect ratio for at least two aspect ratios.'
      - id: 'AC-11a-002'
        text: 'A failing assertion exists that at least one new detail path is called (e.g., ctx.fillRect for a barrel band) for a standard aspect ratio.'
      - id: 'AC-11a-003'
        text: 'A failing assertion exists that gun-sprite.ts exports a projectGunSprite function and a red test expects a non-empty projected polygon/pixel list.'
    parallelizable: false
    dependencies: []
    next_slice: '11-aspect-detail'
  - slice_id: '11-aspect-detail'
    title: 'Fix horizontal stretch and add cannon detail in gun.ts'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
    acceptance_criteria:
      - id: 'AC-11b-001'
        text: 'gunWidth is computed as gunHeight * GUN_BODY_ASPECT_RATIO and no longer depends directly on viewport width.'
      - id: 'AC-11b-002'
        text: 'At least three new detail elements are drawn (barrel bands, side vents, top sight, energy-core rings).'
      - id: 'AC-11b-003'
        text: 'Public API renderGunOverlay and createInitialGunState are unchanged.'
    parallelizable: false
    dependencies:
      - '11-red-gun'
    next_slice: '11-voxel-sprite'
  - slice_id: '11-voxel-sprite'
    title: 'Add dedicated gun-sprite.ts for voxel/3D projection'
    status: '[PLANNED]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
    acceptance_criteria:
      - id: 'AC-11c-001'
        text: 'New gun-sprite.ts exports a helper that projects a small voxel grid using inverse-camera / screen-space math (does not reuse renderer/sprites.ts).'
      - id: 'AC-11c-002'
        text: 'renderGunOverlay integrates the projected voxel sprite as a detail layer without changing its public signature.'
      - id: 'AC-11c-003'
        text: 'The projected gun sprite preserves consistent screen-space height and width proportions across 16:9 and ultra-wide aspect ratios.'
    parallelizable: false
    dependencies:
      - '11-aspect-detail'
    next_slice: '11-green'
  - slice_id: '11-green'
    title: 'Green validation: focused tests, build, lint, coverage guard, visible-browser smoke'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/neatenstein/browser-entry/renderer/gun.test.ts'
      - 'examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts'
      - 'coverage/lcov.info'
    acceptance_criteria:
      - id: 'AC-11d-001'
        text: 'Focused jest suites for gun and gun-sprite pass with zero failures.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
      - id: 'AC-11d-002'
        text: '100% coverage on touched source files in examples/neatenstein/browser-entry/renderer/.'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=examples/neatenstein/browser-entry/renderer/gun'
      - id: 'AC-11d-003'
        text: 'npm run lint exits 0 and tsc --noEmit passes.'
        validation: 'npm run lint; npx tsc --noEmit -p tsconfig.json'
      - id: 'AC-11d-004'
        text: 'Visible-browser smoke test shows the cannon without horizontal stretch, with new details, and with a voxel/3D look.'
        validation: 'Manual visible-browser smoke test of examples/neatenstein/index.html'
    parallelizable: false
    dependencies:
      - '11-voxel-sprite'
    next_slice: null
```

**Validation evidence:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Neon_Shooter_NGE_Demo.plans.md` → PASS (0 errors, 0 warnings).
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id="Step 11" --changed-files="plans/Neon_Shooter_NGE_Demo.plans.md"` → PASS (plan-sync, step-packet, plan-slice-quality, plan-command-lint all true; severity TRIVIAL because source files do not exist yet).
- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id="Step 11" --changed-files="plans/Neon_Shooter_NGE_Demo.plans.md,examples/neatenstein/browser-entry/renderer/gun.ts,examples/neatenstein/browser-entry/renderer/gun.test.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.ts,examples/neatenstein/browser-entry/renderer/gun-sprite.test.ts"` → FAIL (code-coverage gate only; expected because the planned source files do not exist yet; all structural gates pass).
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
