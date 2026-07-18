# Acceleration Parallelism Props Log

**Status:** [DONE]

**Archive date:** 2026-07-16

## Scope

Make acceleration parallelism configurable end-to-end in the racing-curriculum demo and its supporting library code.

## Phase 1: Acceleration Parallelism Props

**Status:** [DONE]

### Step 01 — Plan the focused parallelism fix

- Authored active tracker at `plans/Acceleration_Parallelism_Props.plans.md`.
- Registered in `plans/README.md` and placed in `plans/Roadmap.md`.
- Defined seven scope items, acceptance criteria AC-001..AC-003, and implementation slices 04-01..04-05.
- `validate-plan-phase-packets`, `validate-plan-sync`, and `plan-slice-quality` gates passed.

### Step 02 — Research skip

- Reconnaissance provided; root-cause research recorded in `docs/research/racing-demo-acceleration-cpu-root-cause.md`.
- Identified sync/async backend-detection mismatch: `LifecycleAccelerationPolicy.evaluate` ignores `batchParallelCount` while `autoEnableAcceleration` honors it, causing the chip to show `cpu` while the evaluator uses GPU/worker.
- No additional scouting required before red tests.

### Step 03 — Red tests for acceleration parallelism props

- Added or updated failing tests in:
  - `src/acceleration/acceleration.config.test.ts` (default count, explicit override)
  - `src/acceleration/acceleration.variants.test.ts` (GPU-first resolution, backend logging, CPU fallback)
  - `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts` (stage variant count overrides, `AccelerationConfig` forwarding)
  - `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts` (`RuntimeAdaptationEngineOptions` accepts `accelerationConfig`)
  - `examples/racing_curriculum/__tests__/browser-entry.test.ts` (256x demo default, chip label includes mode and count)
- Red-test evidence showed 16 failing tests across the five test files; failures encoded the expected new behavior.
- `step-packet` gate passed.

### Step 04 — Implement acceleration parallelism props

**Slices completed:**

- `04-01`: Raised `DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT` from `1` to `16` in `src/acceleration/acceleration.constants.ts`.
- `04-02`: Added optional `stageVariantCounts` parameter to `evaluateNgeWeightVariants` and forwarded `AccelerationConfig` to `evaluateWeightVariantsAsync`; relocated the NGE variants test to a sibling file to satisfy repo sibling-test-file policy.
- `04-03`: Exposed `accelerationConfig` on `RuntimeAdaptationEngineOptions`, stored it in the racing runtime engine, and passed the resolved 256x config from `browser-entry.ts` into per-car adaptation engines; updated chip label to show mode and count.
- `04-05`: Removed the hardcoded `const backend = 'cpu'` in `src/acceleration/acceleration.variants.ts`; wired `evaluateWeightVariantsAsync` to resolve backend through `autoEnableAcceleration` (GPU → worker → CPU), log selections and fallback reasons, and emit backend-change telemetry. Added `createBackendCacheObserver()` in `src/acceleration/acceleration.observer.ts` so `browser-entry.ts` can drive the HUD chip from the actual evaluator backend instead of the sync detection path.
- `04-04`: Green-validated the UI chip as part of the fix cycles.

**Fix cycles:**

- Cycle 1: Adjusted concurrency tests to await one tick for async backend resolution; added CPU short-circuit and sequential-path coverage tests.
- Cycle 2: Replaced sync chip lookup with cached backend observer; removed dead `?? 'auto'` fallback in `acceleration.variants.ts`.
- Cycle 3: Closed coverage gaps in `acceleration.variants.test.ts` and `runtime.adaptation.test.ts`.
- Cycle 4: Removed remaining dead branch and validated 100% coverage on touched `src/` files.
- Cycle 5: Added consistent `[NeatapticTS Acceleration]` console diagnostics in `autoEnableAcceleration` and `evaluateWeightVariantsAsync`; confirmed GPU chip and console log in visible browser window.
- Cycle 6: Added missing branch tests for `acceleration.config.ts` and `acceleration.orchestrator.ts` to close the final coverage gap.

### Step 05 — Green validation

**Final cycle (cycle 6) — PASS:**

- Focused Jest: 7/7 suites, 104/104 tests passed.
  - `src/acceleration/acceleration.config.test.ts` — 25/25
  - `src/acceleration/acceleration.variants.test.ts` — 23/23
  - `src/acceleration/acceleration.observer.test.ts` — 11/11
  - `src/acceleration/acceleration.orchestrator.test.ts` — 8/8
  - `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts` — 8/8
  - `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts` — 15/15
  - `examples/racing_curriculum/__tests__/browser-entry.test.ts` — 9/9
- TypeScript: `npx tsc --noEmit -p tsconfig.json` exit 0.
- Lint: `npm run lint` pass (one pre-existing `@typescript-eslint/no-explicit-any` warning outside slice).
- Coverage (scoped to touched `src/` files): 100% lines/statements/functions/branches for `acceleration.constants.ts`, `acceleration.config.ts`, `acceleration.variants.ts`, `acceleration.observer.ts`, `acceleration.orchestrator.ts`, and `neat.nge-juvenile.variants.ts`.
- Browser validation:
  - `npm run build:racing-curriculum` produced `docs/assets/racing-curriculum.bundle.js` (785.9 KB).
  - Chrome DevTools UI check: chip text `GPU 256g/256x`, console log `[NeatapticTS Acceleration] Backend selected: gpu`, visible foreground.
  - Chrome DevTools performance trace: ~77 effective FPS over 30 s, GPU Process primary bottleneck, memory stable.
- `plan-sync` gate: PASS.
- `step-packet` gate: PASS.
- `code-coverage` gate (scoped to touched files): PASS.
- Folder quality gates failed only on pre-existing WebGPU type diagnostics and unrelated missing-sibling-test-file findings outside the slice.

### Step 06 — Documentation and JSDoc updates

- `npm run docs`: PASS; regenerated `src/acceleration/README.md` and `examples/racing_curriculum/controller/README.md`.
- `npm run docs:quality:metrics`: no new high-complexity findings introduced by touched files.
- JSDoc updated in:
  - `src/acceleration/acceleration.types.ts`
  - `src/acceleration/acceleration.config.ts`
  - `src/acceleration/acceleration.variants.ts`
  - `src/acceleration/acceleration.orchestrator.ts`
  - `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`
  - `examples/racing_curriculum/controller/runtime.adaptation.ts`
  - `examples/racing_curriculum/browser-entry/browser-entry.ts`
- Added local-search citation and Mermaid diagram for parallel variant batching in `acceleration.variants.ts`.
- Added NEAT paper citation and usage examples in `neat.nge-juvenile.variants.ts`.
- `step-packet` gate passed.

### Step 07 — Session log and plan closure

- Compressed Phase 1 detailed evidence to this log.
- Trimmed plan file to compact `[DONE]` coverage markers.
- Archived plan/log pair to `plans/completed/`.
- Updated `plans/Roadmap.md` lane to `[DONE]` and pointed the plan link to `plans/completed/`.
- Updated `plans/README.md` entry to `[DONE]` and pointed the plan link to `plans/completed/`.
- Added archive selection entry to `plans/completed/README.md`.
- Gate results:
  - `phase-compression`: PASS
  - `log-completion-marker`: PASS
  - `stale-wip-plans`: PASS
  - `plan-sync`: PASS
  - `step-packet`: PASS

## Changed files

- `src/acceleration/acceleration.constants.ts`
- `src/acceleration/acceleration.config.ts`
- `src/acceleration/acceleration.config.test.ts`
- `src/acceleration/acceleration.variants.ts`
- `src/acceleration/acceleration.variants.test.ts`
- `src/acceleration/acceleration.observer.ts`
- `src/acceleration/acceleration.observer.test.ts`
- `src/acceleration/acceleration.orchestrator.ts`
- `src/acceleration/acceleration.orchestrator.test.ts`
- `src/acceleration/acceleration.types.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts`
- `examples/racing_curriculum/__tests__/browser-entry.test.ts`
- `docs/research/racing-demo-acceleration-cpu-root-cause.md`
- Generated `src/acceleration/README.md`
- Generated `examples/racing_curriculum/controller/README.md`

## Validation summary

- 104/104 focused tests passed
- 100% coverage on touched `src/` source files
- TypeScript compile and lint clean on touched surfaces
- Browser validation confirmed GPU `256g/256x` chip, console diagnostics, ~77 FPS
- `phase-compression`, `log-completion-marker`, `stale-wip-plans`, `plan-sync`, `step-packet` gates passed
