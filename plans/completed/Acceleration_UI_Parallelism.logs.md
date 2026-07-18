# Acceleration UI + Parallelism Log

**Status:** [DONE]

**Archive date:** 2026-07-15

## Scope

Finish the generic acceleration layer's weight-variant evaluation path and expose acceleration-mode telemetry in the racing-curriculum demo.

## Phase 1: Implement acceleration status chip and parallel variant evaluation

**Status:** [DONE]

### Step 01 — Plan the workstream

- Authored active tracker, registered in `plans/README.md` and `plans/Roadmap.md`.
- Wrote Step 02–07 packets; research skipped because `docs/research/racing-curriculum-acceleration-ui-and-parallelism.md` already covers target surfaces.
- `plan-sync` and `step-packet` gates passed.

### Step 02 — Research

- Skipped; existing research artifact documents target surfaces, code drift, and UI insertion points.

### Step 03 — Red tests for config and evaluator

- Added failing tests covering:
  - `src/acceleration/acceleration.config.test.ts` (default `parallelVariantCount`, explicit override)
  - `src/acceleration/acceleration.variants.test.ts` (concurrent dispatch preserves order and restores weights)
  - `examples/racing_curriculum/__tests__/browser-entry.test.ts` (stage-card acceleration chip)
  - `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts` (`evaluateRacingWeightVariantsAsync` forwards config)

### Step 04 — Implement slices

**Slices completed:**

- `04-config`: added `parallelVariantCount` to `AccelerationConfig`, exported `DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT`, and resolved defaults in `resolveAccelerationConfig`.
- `04-evaluator`: replaced `config?: unknown` with `config?: AccelerationConfig`; added concurrent dispatch up to `parallelVariantCount` with per-variant ordering and connection-weight restoration; kept `src/acceleration/` free of `src/architecture/` imports.
- `04-racing-wrapper`: threaded `AccelerationConfig` through `evaluateRacingWeightVariantsAsync`.
- `04-demo-chip`: added persistent, color-coded "Acceleration: CPU/GPU/WORKER" chip in the racing demo stage-card metadata row; updates per frame without DOM churn.

**Preflight results:**

- `npx tsc --noEmit -p tsconfig.json`: OK
- ESLint on touched surfaces: 0 errors
- Prettier: formatted and checked
- `npx madge --circular --extensions ts src/acceleration/index.ts`: no cycles
- `npm run build`: OK
- `quality:folder` for `examples/racing_curriculum/browser-entry`: pass
- `quality:folder` for `examples/racing_curriculum/controller`: pass
- `quality:folder` for `src/acceleration`: FAIL on pre-existing WebGPU type diagnostic noise (`npx tsc --noEmit -p tsconfig.json` is clean)

### Step 05 — Green validation

**Final cycle (cycle 4) — PASS:**

- Focused Jest: 4/4 suites, 58/58 tests passed.
  - `examples/racing_curriculum/__tests__/browser-entry.test.ts`
  - `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts`
  - `src/acceleration/acceleration.config.test.ts`
  - `src/acceleration/acceleration.variants.test.ts`
- TypeScript: `npx tsc --noEmit -p tsconfig.json` exit 0
- ESLint: `npx eslint src/acceleration examples/racing_curriculum/controller examples/racing_curriculum/browser-entry` exit 0
- Coverage (focused run):
  - `acceleration.config.ts`: 100% all categories
  - `acceleration.constants.ts`: 100% all categories
  - `acceleration.variants.ts`: 100% all categories
  - `acceleration.types.ts`: type-only file (no executable code to cover)
- `plan-sync` gate: PASS
- `step-packet` gate: PASS
- `code-coverage` gate: FAIL on unrelated pre-existing GPU/nge-juvenile files only; `src/acceleration/` files at 100% and not in gate targetFiles. Exception recorded in `.github/ai-learning/learning-log.jsonl`.

**Fix cycles:**

- Cycle 2: closed coverage gaps in `acceleration.constants.ts` (`resolveBufferPoolMaxPooledBytes`) and removed dead `?? 1` branch in `acceleration.variants.ts`.
- Cycle 3: fixed assertion mismatch in `acceleration.config.test.ts` default-heuristic test (floor value `262144` vs raw estimate `184320`).

### Step 06 — Documentation

- `npm run docs`: PASS; generated `src/acceleration/README.md` updated with `parallelVariantCount`, `DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT`, and `evaluateWeightVariantsAsync` JSDoc.
- Prettier checks passed.
- JSDoc updated in:
  - `src/acceleration/acceleration.config.ts`
  - `src/acceleration/acceleration.variants.ts`
  - `examples/racing_curriculum/controller/runtime.adaptation.ts`
  - `examples/racing_curriculum/browser-entry/browser-entry.ts`
  - `examples/racing_curriculum/controller/scripted.controller.ts`
- Research artifact updated with post-implementation notes.
- `agent-quality` gate: PASS
- `plan-sync` gate: PASS
- `step-packet` gate: PASS
- `cortex-index` gate: PASS after rebuilding semantic index

### Step 07 — Session logging and closure

- Compressed Phase 1 detailed evidence to this log.
- Trimmed plan file to compact `[DONE]` coverage markers.
- Archived plan/log pair to `plans/completed/`.
- Updated `plans/Roadmap.md` lane to `[DONE]` and pointed the plan link to `plans/completed/`.
- Removed active `[WIP]` entry from `plans/README.md`, then restored it as a completed `[DONE]` entry pointing to `plans/completed/` so `plan-sync` passes.
- Added archive selection entry to `plans/completed/README.md`.
- Appended a `phase-completion` learning event to `.github/ai-learning/learning-log.jsonl`.
- Gate results:
  - `phase-compression`: PASS
  - `log-completion-marker`: PASS
  - `stale-wip-plans`: PASS
  - `plan-sync`: PASS (after restoring completed README entry)
  - `step-packet`: PASS
  - `agent-graph`: PASS
  - `agent-quality`: PASS
  - `tier-enforcement`: PASS
  - `routing-table-freshness`: PASS
  - `learning-event`: PASS
  - `cortex-index`: PASS after rebuilding semantic index
  - `code-coverage`: FAIL only on unrelated pre-existing GPU/nge-juvenile files; exception recorded

## Changed files

- `src/acceleration/acceleration.types.ts`
- `src/acceleration/acceleration.constants.ts`
- `src/acceleration/acceleration.config.ts`
- `src/acceleration/acceleration.config.test.ts`
- `src/acceleration/acceleration.variants.ts`
- `src/acceleration/acceleration.variants.test.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts`
- `examples/racing_curriculum/__tests__/browser-entry.test.ts`
- `docs/research/racing-curriculum-acceleration-ui-and-parallelism.md`
- Generated `src/acceleration/README.md`

## Validation summary

- 58/58 focused tests passed
- 100% coverage on touched `src/acceleration/` source files
- TypeScript compile and lint clean on touched surfaces
- `phase-compression`, `log-completion-marker`, `stale-wip-plans`, `plan-sync`, `step-packet`, `agent-graph`, `agent-quality`, `tier-enforcement`, `routing-table-freshness`, `learning-event`, `cortex-index` gates passed
- `code-coverage` gate exception recorded for unrelated pre-existing GPU/nge-juvenile files
