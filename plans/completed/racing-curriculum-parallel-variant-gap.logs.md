# Phase 3 � Library-default variant evaluation log

**Status:** [DONE]

Original Phase 3 YAML block with slices:

```yaml
phase: 3
step: 1
title: 'Library-default variant evaluation in NGE grow-stabilize'
status: '[WIP]'
goal: implementing
tdd_sequence: green-only
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
skills:
  - implementation-standards
  - coverage-guard
research_artifact: docs/research/racing-curriculum-parallel-variant-gap.md
next_step: 'Phase 4 Step 01 — Green validation and coverage guard'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
acceptance_criteria:
  - id: AC-301
    text: 'Stabilization phase auto-uses evaluateNgeWeightVariants when parallelVariantCount > 1, otherwise applyPlasticity'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
  - id: AC-302
    text: 'Variant generator honors stageVariantCounts for all lifecycle stages and uses bounded multi-connection perturbations'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
  - id: AC-303
    text: 'Racing demo no longer exports a useVariantEvaluator mode and only forwards accelerationConfig'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum'
  - id: AC-304
    text: '100% coverage on all touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '03A-wire-growth-cycle'
    title: 'Wire evaluateNgeWeightVariants into grow-stabilize stabilization'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.types.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.config.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    acceptance_criteria:
      - id: AC-3A1
        text: 'NgeGrowStabilizeConfig accepts accelerationConfig and resolves safe defaults'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.config.test.ts'
      - id: AC-3A2
        text: 'Stabilization branch calls evaluateNgeWeightVariants when parallelVariantCount > 1'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-3A3
        text: 'When parallelVariantCount <= 1 the existing applyPlasticity path remains unchanged'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    parallelizable: false
    dependencies: []
    next_slice: '03B-green-validate-wiring'
  - slice_id: '03B-green-validate-wiring'
    title: 'Green-validate the library wiring and existing variant suites'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    acceptance_criteria:
      - id: AC-3B1
        text: 'Existing library variant suites still pass after the wiring change'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-3B2
        text: 'Grow-stabilize tests assert the backend-agnostic auto-switch between variant evaluation and plasticity'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    parallelizable: false
    dependencies:
      - '03A-wire-growth-cycle'
    next_slice: '03C-redesign-generator-and-demo'
  - slice_id: '03C-redesign-generator-and-demo'
    title: 'Redesign buildVariants and make the racing demo a thin consumer'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
      - 'examples/racing_curriculum/controller/runtime.adaptation.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - id: AC-3C1
        text: 'buildVariants perturbs multiple connections per variant using per-variant deterministic seeds and bounds deltas to the stage mutation magnitude'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-3C2
        text: 'resolveVariantCountForStage honors stageVariantCounts for baby, juvenile, and adult'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-3C3
        text: 'Racing demo only forwards accelerationConfig; no useVariantEvaluator option, CPU safety cap, or non-baby stageVariantCounts guard remains'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum'
    parallelizable: false
    dependencies:
      - '03B-green-validate-wiring'
    next_slice: '03D-relocate-tests'
  - slice_id: '03D-relocate-tests'
    title: 'Relocate stale demo red contracts to library tests and run coverage guard'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - 'examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      - 'examples/racing_curriculum/__tests__/runtime.adaptation.test.ts'
    acceptance_criteria:
      - id: AC-3D1
        text: 'Stale useVariantEvaluator red contracts are removed from demo tests and relocated to library tests'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum'
      - id: AC-3D2
        text: 'Library tests cover default variant-evaluation path and CPU fallback behavior'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
      - id: AC-3D3
        text: '100% coverage on all touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
    parallelizable: false
    dependencies:
      - '03C-redesign-generator-and-demo'
    next_slice: 'Phase 4 Step 01'
```

---

## Detailed implementation notes, validation evidence, PlanUpdate records, and follow-up fixes

## Current state

Claim: 04-implementing @ 2026-07-16T16:50:00Z

Slice **03D-relocate-tests** implementation is applied. The four authorized
test files were edited, formatted, and preflighted.

- `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts` no longer
  carries stale red contracts for removed demo-level APIs
  (`useVariantEvaluator`, `evaluateRacingWeightVariantsAsync`). The remaining
  block exercises `RuntimeAdaptationEngineOptions.accelerationConfig`
  integration.
- `examples/racing_curriculum/controller/runtime.adaptation.test.ts` now
  `await`s the async `adaptOnTick(...)` calls in the default-racing-evaluator,
  cadence-gating, and innovation-rollback tests.
- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts` now covers
  the default `lifecycleStage` fallback to `'baby'` in the parallel variant
  path.
- `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts` now covers
  `resolveVariantCountForStage` aliasing for `embryo` → `baby` and
  `equilibrium` → `adult`, plus deterministic `buildVariants` output for a
  network with zero connections.

Preflight results:

- `npx tsc --noEmit -p tsconfig.json`: **PASS** (no new errors in changed
  files; pre-existing baseline errors remain outside the slice).
- `npx eslint` on the four changed files: **PASS** (0 errors).
- `npx prettier --check` on the four changed files: **PASS**.
- `npm run quality:folder -- --folder=src/neat/nge-juvenile`: **FAIL**
  (pre-existing `missing-test-file` for
  `src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-policy.ts`, unrelated to
  this slice).
- `npm run quality:folder -- --folder=examples/racing_curriculum`: **FAIL**
  (pre-existing TypeScript errors in
  `examples/racing_curriculum/controller/runtime.adaptation.ts` and
  `examples/racing_curriculum/gpu-enabled-racing.example.ts`, outside this
  slice's files_to_change).
- `plan-sync`: to be run after this update.

PlanUpdate:

```yaml
PlanUpdate:
  slice_id: '03D-relocate-tests'
  changed_files:
    - examples/racing_curriculum/__tests__/runtime.adaptation.test.ts
    - examples/racing_curriculum/controller/runtime.adaptation.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: PASS'
    - 'npx eslint on changed files: PASS (0 errors)'
    - 'npx prettier --check on changed files: PASS'
    - 'npm run quality:folder -- --folder=src/neat/nge-juvenile: FAIL (pre-existing missing-test-file)'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum: FAIL (pre-existing TS errors outside slice)'
    - 'neataptic-gate-mcp:run_gate_check gate=plan-sync: PASS'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/__tests__/runtime.adaptation.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
  rollback:
    - 'git checkout -- examples/racing_curriculum/__tests__/runtime.adaptation.test.ts'
    - 'git checkout -- examples/racing_curriculum/controller/runtime.adaptation.test.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
  next: 'Run plan-sync, then hand off to 05-green-testing for the focused test slices and coverage guard.'
```

## Latest validation evidence

- status: implementation-preflight-pass-with-pre-existing-blockers
- timestamp: 2026-06-16T04:20:00Z
- verifier: 04-implementing
- gates:
  - `npx tsc --noEmit -p tsconfig.json` → **PASS**
  - `npx eslint` on changed files → **PASS**
  - `npx prettier --check` on changed files → **PASS**
  - `npm run quality:folder -- --folder=src/neat/nge-juvenile` → **FAIL**
    (pre-existing `missing-test-file` for `neat.nge-juvenile.lifecycle-policy.ts`)
  - `npm run quality:folder -- --folder=examples/racing_curriculum` → **FAIL**
    (pre-existing TS errors in `controller/runtime.adaptation.ts` and
    `gpu-enabled-racing.example.ts`)
  - `neataptic-gate-mcp:run_gate_check gate=plan-sync` → **PASS**
- notes: |
  Slice 03D test edits are internally consistent. The changed files compile and
  lint cleanly. Quality-folder failures are caused by pre-existing conditions
  outside the slice boundary (a missing sibling test file and TypeScript errors
  in implementation/example files). Do not run full jest; hand off to
  05-green-testing for the focused slices.

### Follow-up fix — Acceleration HUD status and parallel count override [DONE]

After the prior thinning slice, two gaps remained in the racing demo:

1. The HUD did not show the **resolved** acceleration backend, any GPU fallback
   reason, or the actual `parallelVariantCount` being used.
2. The demo's `DEMO_ACCELERATION_CONFIG` still used a safe default
   `parallelVariantCount` of 16 instead of the advertised 1024 variant pool.

This fix lives entirely in the demo's browser entry point.

#### Changed files

- `examples/racing_curriculum/browser-entry/browser-entry.ts`
  - `DEMO_ACCELERATION_CONFIG` now requests `parallelVariantCount: 1024` and
    `stageVariantCounts: { baby: 1024, juvenile: 1024, adult: 1024 }`.
  - Added `autoEnableAcceleration` and `AccelerationStatus` imports from the
    library acceleration surface.
  - After the focused controller network is resolved, calls
    `autoEnableAcceleration({ nodeCount, batchParallelCount, config })` to read
    back the actual chosen backend and fallback reasons.
  - Added `formatAccelerationStatus(status, parallelVariantCount)` helper that
    produces labels such as `GPU (1024 variants)` or
    `CPU · GPU blocked: no WebGPU (1024 variants)`.
  - Extended `TelemetryPanelNodes` with `accelerationValue` and added an
    "Acceleration" row in the runtime telemetry panel, initialized to
    `detecting…` and updated each frame.
  - The right-sidebar network HUD title (`hostHandle.networkHud.titleValue`) now
    shows the resolved acceleration summary once at startup.

#### Files intentionally not changed

- `examples/racing_curriculum/controller/runtime.adaptation.ts`
  - Already forwards `options.accelerationConfig` into the NGE grow-stabilize
    cycle; no additional wiring was required for this HUD/count fix.

#### Preflight results

- `npx tsc --noEmit -p tsconfig.json` → **PASS** (no new errors).
- `npx eslint examples/racing_curriculum/browser-entry/browser-entry.ts` → **PASS** (0 errors).
- `npx prettier --check examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/controller/runtime.adaptation.ts` → **PASS**.
- `npx tsc --noEmit -p tsconfig.test.json` → **FAIL** (pre-existing error in
  `node_modules/devtools-protocol/types/protocol-mapping.d.ts`, unrelated to this
  fix).

PlanUpdate:

```yaml
PlanUpdate:
  slice_id: '03E-acceleration-hud-status'
  changed_files:
    - examples/racing_curriculum/browser-entry/browser-entry.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: PASS'
    - 'npx eslint examples/racing_curriculum/browser-entry/browser-entry.ts: PASS (0 errors)'
    - 'npx prettier --check examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/controller/runtime.adaptation.ts: PASS'
    - 'npx tsc --noEmit -p tsconfig.test.json: FAIL (pre-existing node_modules error, unrelated)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum'
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
  next: 'Run 05-green-testing focused on examples/racing_curriculum to confirm demo contract tests still pass.'
```

### Follow-up fix — Direct blocker remediation for slices 03D/03E [DONE]

Claim: 04-implementing @ 2026-07-16T21:17:02Z

Five blockers were preventing slices **03D** and **03E** from passing green
validation. This follow-up edit addresses all five without expanding the slice
boundary.

1. **Stale cleanup-contract assertions** in
   `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts` still
   expected removed demo-level acceleration imports/evaluators
   (`src/acceleration/acceleration.variants`, `evaluateWeightVariantsAsync`,
   `createBackendCacheObserver`). Removed those three stale `it` blocks; the
   remaining assertions verify no legacy `src/performance/nge` imports, no
   hardcoded `disableGPU/disableWorkers` flags, and `accelerationConfig`
   forwarding.

2. **Incorrect zero-connection representative-delta expectation** in
   `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts`. The test asserted
   every patch representative had `delta === 0.05`, but `buildVariants` scales
   linearly (`0.05 * (index + 1)`) up to the stage magnitude. For baby stage
   (magnitude `0.15`) the three patches produce `0.05`, `0.10`, `0.15`. Updated
   the assertion to expect the full array.

3. **`runNgeLifecycle` stage narrowing** in
   `examples/racing_curriculum/controller/runtime.adaptation.ts`. The
   `lifecycleRunner` callback received a `lifecycleInput` whose `stage` is typed
   as `NgeLifecycleStage`, but `runNgeLifecycle` only accepts
   `NgeJuvenileLifecycleInput | NgeAdultLifecycleInput`. Destructured `stage`,
   added a runtime guard, and spread the rest of the input with a narrowed
   `stage: 'juvenile'` into `runNgeLifecycle`, resolving the TypeScript error.

4. **Branch coverage in `neat.nge-juvenile.variants.ts`** was at 93.18% with
   three missing branches (default cases in
   `resolveVariantCountForStage`/`resolveMutationMagnitudeForStage` and the
   `maxDelta > 0 ? maxDelta : 1` false branch). Added targeted tests in
   `neat.nge-juvenile.variants.test.ts` that exercise an unknown lifecycle stage
   and an empty-connection network.

5. **Branch coverage in `neat.nge-juvenile.grow-stabilize.ts`** was at 97.82%
   with three missing branches (large-network throttle interval false branch and
   the `> 500` node adaptive-hysteresis branch). Added targeted tests in
   `neat.nge-juvenile.grow-stabilize.test.ts`:
   - `computeGrowthThrottle` returns `shouldThrottle: false` when the tick
     lands exactly on the computed interval.
   - `runNgeGrowStabilizeCycle` uses `resolveAdaptiveHysteresis(1200) === 5`
     during growth for networks above the 500-node tier.

#### Changed files

- `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts`

#### Preflight results

- `npx tsc --noEmit -p tsconfig.json` → **PASS** (no new errors in changed
  files).
- `npx eslint` on the four changed files → **PASS** (0 errors).
- `npx prettier --check` on the four changed files → **PASS** (after formatting
  `neat.nge-juvenile.variants.test.ts`).
- `npx tsc --noEmit -p tsconfig.test.json` → **FAIL** (pre-existing parse error
  in `node_modules/devtools-protocol/types/protocol-mapping.d.ts`, unrelated to
  this change).

#### Validation evidence

- `tsc` main config: OK
- `eslint` changed files: 0 issues
- `prettier --check` changed files: OK
- `tsconfig.test.json`: blocked by pre-existing `devtools-protocol` d.ts parse
  error (see `BLOCKERS`).

PlanUpdate:

```yaml
PlanUpdate:
  slice_id: '03D-relocate-tests+03E-blocker-fix'
  changed_files:
    - examples/racing_curriculum/__tests__/runtime.adaptation.test.ts
    - examples/racing_curriculum/controller/runtime.adaptation.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: PASS'
    - 'npx eslint on changed files: PASS (0 errors)'
    - 'npx prettier --check on changed files: PASS'
    - 'npx tsc --noEmit -p tsconfig.test.json: FAIL (pre-existing node_modules devtools-protocol parse error, unrelated)'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/__tests__/runtime.adaptation.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/controller/runtime.adaptation.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
  rollback:
    - 'git checkout -- examples/racing_curriculum/__tests__/runtime.adaptation.test.ts'
    - 'git checkout -- examples/racing_curriculum/controller/runtime.adaptation.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
  next: 'Run plan-sync, then hand off to 05-green-testing for the focused test slices and coverage guard.'
```

#### Remaining juvenile test fixes

Fixed the remaining branch-coverage gaps in the two NGE juvenile test files
without editing implementation source.

- `src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts`
  - Removed the unused `NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE` import.
  - Added a `NaN`-count test that exercises the empty-patch fallback branches
    (`scores[0] ?? Number.NEGATIVE_INFINITY` and `maxDelta > 0 ? maxDelta : 1`).
  - Added a real-network test where a later patch scores higher, covering the
    `bestIndex` / `bestScore` update branch.
  - Kept the existing disappearing-connection restore branch test.
- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts`
  - Added a `jest.doMock` / `jest.resetModules` test that makes
    `evaluateNgeWeightVariants` return `bestIndex: -1`, covering the
    `variantResult.bestIndex >= 0` false branch.
  - Added a zero-connection-network test that covers the
    `connectionCount > 0 ? index % connectionCount : 0` false branch and the
    inner `bestVariant !== undefined && network.connections[bestVariant.weightIndex] !== undefined`
    false path.

##### Preflight results

- `npx tsc --noEmit -p tsconfig.json` → **PASS**.
- `npx eslint` on both changed test files → **PASS** (0 errors).
- `npx prettier --check` on both changed test files → **PASS**.
- Focused Jest run on both test files → **PASS** (60/60 tests).
- Focused coverage for `neat.nge-juvenile.variants.ts` and
  `neat.nge-juvenile.grow-stabilize.ts` → **100%** statements/branches/functions/lines.
- `npm run quality:folder -- --folder=src/neat/nge-juvenile` → **FAIL** due to
  pre-existing deficits in other modules (`neat.nge-juvenile.apply.ts`,
  `neat.nge-juvenile.errors.ts`, `neat.nge-juvenile.grow.ts`,
  `neat.nge-juvenile.utils.ts`) and the missing sibling test file for
  `neat.nge-juvenile.lifecycle-policy.ts`. Touched files are at 100%.

PlanUpdate:

```yaml
PlanUpdate:
  slice_id: 'juvenile-test-branch-coverage-fix'
  changed_files:
    - src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: PASS'
    - 'npx eslint on changed files: PASS (0 errors)'
    - 'npx prettier --check on changed files: PASS'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns="neat.nge-juvenile.variants.test.ts"'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns="neat.nge-juvenile.grow-stabilize.test.ts"'
    - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns="neat.nge-juvenile.variants.test.ts" --testPathPatterns="neat.nge-juvenile.grow-stabilize.test.ts"'
  rollback:
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
  next: 'Run plan-sync and hand off to 05-green-testing for focused validation and coverage-guard.'
```

# Phase 5 — Dynamic delta distribution implementation log

**Status:** [DONE]

### Phase 5 — Dynamic delta distribution implementation [WIP]

**Phase objective:** Implement the Round-3-agreed dynamic delta distribution
action plan inside the NGE juvenile variant generator and grow-stabilize cycle.
Replace hardcoded delta constants with the endpoint-inclusive
`resolveRepresentativeDelta` helper, add `resolveEffectiveMagnitude` scaling,
wire the weight-exhaustion force-growth gate, tighten patch constants, add the
seed-stride factor, and update all contracts/docs.

```yaml
phase: 5
title: 'Dynamic delta distribution implementation'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
next_phase: 'Phase 6 — Documentation compression and tracker closure'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - phase-handoff-workflow
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
acceptance_criteria:
  - id: AC-501
    text: 'Phase 5 step packet and all five slices are authored with machine-readable YAML'
    validation: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - id: AC-502
    text: 'No slice exceeds the 4-hour estimate cap (all slices estimated 2-3 hours)'
    validation: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - id: AC-503
    text: 'Plan-sync gate passes after the Phase 5 addition'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
constitution_check:
  - principle-4-small-slices
  - principle-5-unique-ids
placeholder_steps:
  - 'Step 01 — Author Phase 5 implementation slices'
  - 'Slice 5A — Dynamic delta distribution'
  - 'Slice 5B — Magnitude scaling'
  - 'Slice 5C — Weight-exhaustion gate'
  - 'Slice 5D — Patch constants and seed stride'
  - 'Slice 5E — Contract/README cleanup'
```

#### Step 01 — Author Phase 5 implementation slices [DONE]

```yaml
phase: 5
step: 1
title: 'Author Phase 5 implementation slices'
status: '[DONE]'
goal: planning
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
next_step: 'Slice 5A — Phase 5 red tests for dynamic delta distribution'
skills:
  - implementation-standards
  - coverage-guard
  - execute
  - planning-acceptance-criteria
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
acceptance_criteria:
  - id: AC-511
    text: 'All five Phase 5 slices are authored with unique IDs, estimates <= 4 hours, dependencies, and observable acceptance criteria'
    validation: 'node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - id: AC-512
    text: 'Slice ordering honors the step-packet gate red-implement-green sequence and declared dependencies'
    validation: 'node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - id: AC-513
    text: 'Plan-sync passes after Phase 5 is registered'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
constitution_check:
  - principle-4-small-slices
  - principle-5-unique-ids
slices:
  - slice_id: '5A-dynamic-delta-red'
    title: 'Red tests for dynamic delta distribution'
    status: '[WIP]'
    goal: red-testing
    estimate_hours: 2
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.plasticity.test.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    acceptance_criteria:
      - id: AC-5A01
        text: 'Red tests fail because resolveRepresentativeDelta is not yet exported or does not produce endpoint-inclusive deltas on [-magnitude, +magnitude]'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-5A02
        text: 'Red tests fail because buildVariants and buildWeightVariants still use hardcoded constants instead of the shared helper'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat.nge-juvenile.(variants|grow-stabilize).test.ts'
      - id: AC-5A03
        text: 'Red tests fail for the right reason (missing behavior/removed constants), not syntax or fixture errors'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat.nge-juvenile.(variants|grow-stabilize).test.ts'
    parallelizable: false
    dependencies: []
    next_slice: '5B-dynamic-delta-impl'
    red_evidence:
      - file: 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
        state: 'failing'
        reason: 'Missing exports resolveRepresentativeDelta and resolveEffectiveMagnitude from variants.ts; missing new NGE_VARIANT_* constants from constants.ts'
        command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - file: 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
        state: 'failing'
        reason: 'Missing exports buildWeightVariants, resolveExhaustionForceGrowthThreshold, resolveExhaustionImprovementThreshold, resolveNoiseSigmaFraction, resolveStageFraction from grow-stabilize.ts; missing exports from variants.ts'
        command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
  - slice_id: '5B-dynamic-delta-impl'
    title: 'Dynamic delta distribution implementation'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.plasticity.test.ts'
    acceptance_criteria:
      - id: AC-5B01
        text: 'resolveRepresentativeDelta(index, count, magnitude) is exported from variants.ts and returns endpoint-inclusive deltas; N=1 returns +magnitude'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-5B02
        text: 'buildVariants uses resolveRepresentativeDelta for its representative delta and NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA is removed'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-5B03
        text: 'buildWeightVariants in grow-stabilize.ts uses resolveRepresentativeDelta and the local WEIGHT_VARIANT_DELTA constant is removed'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-5B04
        text: 'Plasticity and variants tests updated in 5A now pass'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.(variants|plasticity).test.ts'
      - id: AC-5B05
        text: '100% coverage on all touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.(variants|grow-stabilize)'
    parallelizable: false
    dependencies:
      - '5A-dynamic-delta-red'
    next_slice: '5C-magnitude-scaling'
  - slice_id: '5C-magnitude-scaling'
    title: 'Magnitude scaling'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
    acceptance_criteria:
      - id: AC-5C01
        text: 'resolveEffectiveMagnitude(stage, variantCount, connectionCount) is exported, uses widthFactor and sizeFactor with correct clamps, and replaces resolveMutationMagnitudeForStage'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-5C02
        text: 'buildVariants calls resolveEffectiveMagnitude instead of resolveMutationMagnitudeForStage and passes connectionCount'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-5C03
        text: 'buildWeightVariants signature is (network, variantCount, stage) and uses resolveEffectiveMagnitude; grow-stabilize call site passes stage'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-5C04
        text: 'New constants NGE_VARIANT_WEIGHT_RANGE, NGE_VARIANT_WIDTH_FACTOR_MAX, NGE_VARIANT_SIZE_FACTOR_FLOOR, and NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS are exported'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
      - id: AC-5C05
        text: 'Fixed-point verification table matches the agreed defaults; 100% coverage on touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.(variants|grow-stabilize)'
    parallelizable: false
    dependencies:
      - '5B-dynamic-delta-impl'
    next_slice: '5D-exhaustion-and-patches'
  - slice_id: '5D-exhaustion-and-patches'
    title: 'Weight-exhaustion gate and patch constants'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.types.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    acceptance_criteria:
      - id: AC-5D01
        text: 'NgeGrowStabilizeInput has optional baselineScore, consecutiveWeightExhaustion, and postGrowthThresholdActive; NgeGrowStabilizeResult has required consecutiveWeightExhaustion and postGrowthThresholdActive'
        validation: 'npx tsc --noEmit --project tsconfig.test.json'
      - id: AC-5D02
        text: 'All new NGE_EXHAUSTION_* constants are exported from neat.nge-juvenile.constants.ts and NGE_GROW_STABILIZE_IMPROVEMENT_THRESHOLD (0.01) is removed'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-5D03
        text: 'resolveExhaustionImprovementThreshold, resolveExhaustionForceGrowthThreshold, resolveStageFraction, and resolveNoiseSigmaFraction are implemented and exported'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-5D04
        text: 'grow-stabilize.ts variant path uses resolveVariantCountForStage for effectiveVariantCount and follows the 10-step control flow from Section 8'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-5D05
        text: 'Exhaustion resets to 0 when a variant improvement commits; increments by 1 on no-improvement stabilization; resets to 0 on growth; post-growth boost doubles the limit (4-16) after bad growth'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
      - id: AC-5D06
        text: 'Patch constants updated: NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT=8, NGE_VARIANT_PATCH_SIZE_FRACTION=0.03; NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR=4 with max(offset, patchSize*4); applyWeightMutations accepts optional magnitude'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.(variants|grow-stabilize).test.ts'
      - id: AC-5D07
        text: '100% coverage on all touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile/neat.nge-juvenile.(variants|grow-stabilize)'
    parallelizable: false
    dependencies:
      - '5C-magnitude-scaling'
    next_slice: '5E-contract-green'
  - slice_id: '5E-contract-green'
    title: 'Contract/README cleanup and final green validation'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 2
    files_to_change:
      - 'src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.variants.ts'
      - 'src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
      - 'README.md'
    acceptance_criteria:
      - id: AC-5E01
        text: 'No source JSDoc or README references NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA or WEIGHT_VARIANT_DELTA'
        validation: 'npm run lint'
      - id: AC-5E02
        text: 'All references to resolveMutationMagnitudeForStage are renamed to resolveEffectiveMagnitude in code and docs'
        validation: 'npm run lint'
      - id: AC-5E03
        text: 'New constants are documented in the constant ledger/README and JSDoc is current for buildVariants, buildWeightVariants, resolveRepresentativeDelta, and resolveEffectiveMagnitude'
        validation: 'npm run lint'
      - id: AC-5E04
        text: 'All NGE juvenile suites and racing-curriculum controller suites remain green'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
      - id: AC-5E05
        text: '100% coverage on all touched src/ files'
        validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
    parallelizable: false
    dependencies:
      - '5B-dynamic-delta-impl'
      - '5C-magnitude-scaling'
      - '5D-exhaustion-and-patches'
    next_slice: null
```

Step 01 authored the formal Phase 5 implementation-slice packets above. The
slices are sequenced as a red→implement→green step packet: 5A writes the red
contracts, 5B-5D implement the four agreed action-plan areas, and 5E runs the
final green validation plus contract/README cleanup verification. All
dependencies are acyclic and every slice is within the 4-hour cap.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active workstream: plans/racing-curriculum-parallel-variant-gap.plans.md Phase 5 — Dynamic delta distribution implementation.
Step 01 (slice authoring) is DONE. The current active frontier is Slice 5A-dynamic-delta-red.

Next narrow task:
  Dispatch 03-red-testing for Slice 5A-dynamic-delta-red. Write red tests in
  src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts,
  src/neat/nge-juvenile/neat.nge-juvenile.plasticity.test.ts, and
  src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts that fail
  because resolveRepresentativeDelta is missing and hardcoded delta constants
  are still in use.

Required validations before the next slice starts:
  - node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md
  - node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md

Known context:
  - Phase 4 is DONE; the Round-3-agreed action plan is in Sections 1-12 above.
  - The five implementation slices cover dynamic delta distribution, magnitude
    scaling, weight-exhaustion gate + patch constants, and final green
    validation / contract cleanup.
```

## Open decisions

1. **Integration strategy** — **Decision: library-default variant evaluation**
   - Variant evaluation is now the default stabilization path inside
     `runNgeGrowStabilizeCycle` whenever `AccelerationConfig.parallelVariantCount`
     is greater than 1. When the config is absent or `parallelVariantCount <= 1`,
     the cycle falls back to the existing `applyPlasticity` weight-mutation path.
   - The racing demo no longer owns an opt-in `useVariantEvaluator` mode; it only
     forwards `options.accelerationConfig` into the grow-stabilize cycle config.

2. **Variant generation redesign**
   - Handled in Slice **03C-redesign-generator-and-demo**. `buildVariants` must
     perturb multiple connections per variant, use per-variant deterministic
     seeds, bound deltas to the configured stage mutation magnitude, and honor
     `stageVariantCounts` for `baby`, `juvenile`, and `adult`.

3. **CPU fallback safety**
   - Treated as a library/acceleration concern. `evaluateWeightVariantsAsync`
     in `src/acceleration/acceleration.variants.ts` already branches to
     sequential CPU evaluation when no backend is active and respects the caller's
     `parallelVariantCount`. Any additional CPU safety cap or refusal belongs in
     the library variant evaluator, not in the demo.

## Risks

- Changing the default stabilization path may alter existing racing-curriculum
  behavior for networks that already pass `AccelerationConfig`; keep a
  manual trend-only fallback path for small or explicitly sequential configs.
- The deterministic single-connection sweep currently in `buildVariants` may
  waste compute and produce no improvement; redesign must prove local-search
  value before becoming default.
- `stageVariantCounts.juvenile` and `stageVariantCounts.adult` are currently
  ignored and must be honored by the redesigned `resolveVariantCountForStage`.
- No existing test covers variant evaluation in the live grow-stabilize loop;
  Slice **03D-relocate-tests** must add that coverage.

## Research artifact

- `docs/research/racing-curriculum-parallel-variant-gap.md`

## Current state

Claim: 04-implementing @ 2026-07-17T20:45:00Z

Phase 3 implementation is complete and green-validated. Phase 4 green
validation was superseded by discovery of the **real issue**: hardcoded
delta constants that don't scale with variant count. Phase 5 slices 5A–5D
are implemented and type/lint clean. This session finished the seven failing
NGE juvenile test repairs required before Slice 5E can be considered green.

- **Phase 3 status:** [DONE] — see `plans/racing-curriculum-parallel-variant-gap.logs.md` for detailed slice evidence.
- **Phase 4 status:** [DONE] — specialist analysis and validation loop completed. See "Real Issue" section below.
- **Phase 5 status:**
  - Slice 5A [DONE] — red tests merged.
  - Slice 5B [DONE] — resolveRepresentativeDelta and buildVariants refactor implemented; NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA removed.
  - Slice 5C [DONE] — resolveEffectiveMagnitude, new scaling constants, and buildWeightVariants parity implemented.
  - Slice 5D [DONE] — exhaustion helpers implemented, variant-path control flow wired to Section 8 spec, tsc and eslint clean.
  - Slice 5E [DONE] — seven failing juvenile suites repaired; focused jest, full folder, and `quality:folder` gates pass.

```yaml
PlanUpdate:
  slice_id: 5E-review-fixes
  changed_files:
    - src/neat/nge-juvenile/neat.nge-juvenile.constants.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.variants.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-policy.test.ts
  preflight:
    - 'git status --porcelain'
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check src/neat/nge-juvenile/'
    - 'npm run quality:folder -- --folder=src/neat/nge-juvenile'
  preflight_results:
    tsc: 'tsc exit: 0'
    lint: 'eslint exit: 0 (21 pre-existing any warnings in grow-stabilize.test.ts)'
    prettier: 'prettier check exit: 0 after --write on new test files'
    quality_folder: 'PASS — 0 TypeScript diagnostics, 0 ESLint errors, 62/62 JSDoc exports, 0 missing sibling tests, 0 coverage deficits'
    git_status: 'multiple unrelated working-tree changes present; slice changes limited to the eight files listed above'
  tests_for_green:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/neat\.nge-juvenile\.(lifecycle-policy|grow|grow-stabilize|variants|morph|lifecycle-stages)\.test\.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=src/neat/nge-juvenile/ --coverage --collectCoverageFrom=src/neat/nge-juvenile/**/*.ts'
  rollback:
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.constants.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.variants.ts'
    - 'git rm --cached src/neat/nge-juvenile/neat.nge-juvenile.grow.test.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.test.ts'
    - 'git checkout -- src/neat/nge-juvenile/neat.nge-juvenile.variants.test.ts'
    - 'git rm --cached src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-policy.test.ts'
  next: 'Run 05-green-testing on the full nge-juvenile folder with coverage; confirm coverage-guard 100% line coverage on all touched src/ files'
  handoff_to: '05-green-testing'
```

## Real Issue — Hardcoded Delta Constants

### Root Cause

`NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA = 0.05` (constants.ts:423) and
`WEIGHT_VARIANT_DELTA = 0.05` (grow-stabilize.ts:577) are hardcoded constants
that don't scale with variant count. With 1024 variants, `delta = 0.05 * (index+1)`
capped at baby magnitude (0.15) means ~1000 of 1024 parallel slots evaluate
identical weights — massive wasted compute.

### Agreed Action Plan (Round 3 — pending validation)

#### Section 1: Constants to Remove

- `NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA = 0.05` (constants.ts:423)
- `WEIGHT_VARIANT_DELTA = 0.05` (grow-stabilize.ts:577)

#### Section 2: Constants to Add

| Constant                                           | Value   | Purpose                                                 |
| -------------------------------------------------- | ------- | ------------------------------------------------------- |
| `NGE_VARIANT_WEIGHT_RANGE`                         | `2`     | Weight exploration range [-1, +1] span                  |
| `NGE_VARIANT_WIDTH_FACTOR_MAX`                     | `2.0`   | Upper clamp for widthFactor                             |
| `NGE_VARIANT_SIZE_FACTOR_FLOOR`                    | `0.1`   | Lower clamp for sizeFactor                              |
| `NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS`          | `267`   | C_ref = ceil(MAX/FRACTION) = ceil(8/0.03)               |
| `NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR`             | `4`     | Multiplier for seed stride floor                        |
| `NGE_EXHAUSTION_SCORE_EPSILON`                     | `1e-6`  | Floor for score-scale and finite-ceiling guard          |
| `NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY`         | `0.003` | Noise-sigma fraction for baby stage                     |
| `NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE`     | `0.002` | Noise-sigma fraction for juvenile stage                 |
| `NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT`        | `0.001` | Noise-sigma fraction for adult stage                    |
| `NGE_EXHAUSTION_STAGE_FRACTION_BABY`               | `0.01`  | Relative-improvement fraction for baby stage            |
| `NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE`           | `0.005` | Relative-improvement fraction for juvenile stage        |
| `NGE_EXHAUSTION_STAGE_FRACTION_ADULT`              | `0.003` | Relative-improvement fraction for adult stage           |
| `NGE_EXHAUSTION_TICK_BUDGET`                       | `48`    | Total tick budget driving exhaustion count              |
| `NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS`             | `1`     | Minimum consecutive exhaustion count                    |
| `NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS`             | `8`     | Maximum consecutive exhaustion count                    |
| `NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS` | `16`    | Post-growth-boost maximum exhaustion count              |
| `NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST`      | `2.0`   | Multiplier applied to exhaustion count after bad growth |
| `NGE_EXHAUSTION_NEURON_BUDGET_FACTOR`              | `0.5`   | Neuron-factor curve steepness                           |

> See Section 8 for the full threshold design and rationale.

#### Section 3: Constants to Change

| Constant                                 | Old  | New  | Rationale                                                                             |
| ---------------------------------------- | ---- | ---- | ------------------------------------------------------------------------------------- |
| `NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT` | 16   | 8    | Mandatory — random-sign perturbations scale as sqrt(K), smaller K loses little signal |
| `NGE_VARIANT_PATCH_SIZE_FRACTION`        | 0.05 | 0.03 | Mandatory — even-distribution deltas give full [-mag,+mag] coverage via variant count |

#### Section 4: Delta Formula (endpoint-inclusive)

**Shared helper** `resolveRepresentativeDelta(index, variantCount, magnitude)` exported from `variants.ts`:

```typescript
export function resolveRepresentativeDelta(
  index: number,
  variantCount: number,
  magnitude: number,
): number {
  const safeCount = Math.max(1, variantCount);
  if (safeCount === 1) {
    return roundDelta(magnitude); // N=1: single positive probe, not zero
  }
  const t = (2 * index) / (safeCount - 1) - 1; // [-1, +1] endpoint-inclusive
  return roundDelta(magnitude * t);
}
```

- N > 1: `delta_i = magnitude * (2*i/(N-1) - 1)` — reaches exact +/-magnitude at i=0 / i=N-1
- N = 1: `delta_0 = +magnitude` (special case, not zero)
- All N slots unique, zero redundancy
- Used by BOTH `buildVariants` (variants.ts) and `buildWeightVariants` (grow-stabilize.ts)

#### Section 5: buildVariants Refactor (variants.ts:243)

The evaluator's `buildVariants` MUST be explicitly refactored:

1. **Replace** `resolveMutationMagnitudeForStage(stage)` call (line 250) with
   `resolveEffectiveMagnitude(stage, count, network.connections.length)`
2. **Replace** representative delta computation (lines 274-278):
   ```typescript
   // OLD: Math.min(NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA * (index + 1), magnitude)
   // NEW:
   const representativeDelta = resolveRepresentativeDelta(
     index,
     count,
     effectiveMagnitude,
   );
   ```
3. **Remove** `NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA` import
4. **Semantic change**: representative delta shifts from monotonically-increasing
   positive (0.05 to 0.80) to symmetric endpoint-inclusive [-M, +M]. This is a real
   behavior change to patch generation.
5. **Test updates needed**: `neat.nge-juvenile.plasticity.test.ts` and variants
   tests that assert specific delta values or monotonically-increasing patterns
   must be updated to expect symmetric endpoint-inclusive deltas.

#### Section 6: Magnitude Scaling — resolveEffectiveMagnitude

**Signature**: `resolveEffectiveMagnitude(stage, variantCount, connectionCount)`

**Note**: Uses `connectionCount` (network.connections.length), NOT `nodeCount`.
This aligns with `C_ref = ceil(MAX/FRACTION) = 267` which is derived from
connection-count semantics (patch max connections / size fraction).

```typescript
export function resolveEffectiveMagnitude(
  stage: NgeLifecycleStage,
  variantCount: number,
  connectionCount: number,
): number {
  const base = resolveStageMagnitude(stage); // baby=0.15, juvenile=0.1, adult=0.05
  const vRef = resolveStageVariantRef(stage); // baby=16, juvenile=8, adult=4
  const cRef = NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS; // 267

  const widthFactor = Math.max(
    1,
    Math.min(Math.sqrt(variantCount / vRef), NGE_VARIANT_WIDTH_FACTOR_MAX),
  );

  const sizeFactor = Math.max(
    NGE_VARIANT_SIZE_FACTOR_FLOOR,
    Math.min(Math.sqrt(cRef / Math.max(1, connectionCount)), 1),
  );

  return base * widthFactor * sizeFactor;
}
```

**Key fix from Round 2**: `sizeFactor` now has an upper clamp at 1.0 via
`Math.min(..., 1)`. Small networks (connectionCount < C_ref) get
`sizeFactor = 1.0` (magnitude stays at base). Large networks get
`sizeFactor < 1.0` (magnitude shrinks). This prevents magnitude inflation
for small networks.

**Fixed-point verification** (at defaults, V=V_ref, C <= C_ref):

- Baby (V=16, C=5): widthFactor=1.0, sizeFactor=min(sqrt(267/5),1)=min(7.3,1)=1.0 -> 0.15
- Baby (V=16, C=100): widthFactor=1.0, sizeFactor=min(sqrt(2.67),1)=min(1.63,1)=1.0 -> 0.15
- Baby (V=16, C=267): widthFactor=1.0, sizeFactor=min(sqrt(1.0),1)=1.0 -> 0.15
- Baby (V=16, C=500): widthFactor=1.0, sizeFactor=min(sqrt(0.534),1)=0.73 -> 0.11
- Baby (V=16, C=1000): widthFactor=1.0, sizeFactor=min(sqrt(0.267),1)=0.52 -> 0.078
- 1024 override (V=1024, C=5): widthFactor=min(sqrt(64),2)=2.0, sizeFactor=1.0 -> 0.30
- 1024 override (V=1024, C=500): widthFactor=2.0, sizeFactor=0.73 -> 0.22

**Embryo**: maps to baby reference values (V_ref=16, M_stage=0.15), matching
existing `resolveVariantCountForStage` and `resolveMutationMagnitudeForStage`
embryo/baby fallthrough.

#### Section 7: buildWeightVariants Refactor (grow-stabilize.ts:590)

**Signature**: `buildWeightVariants(network, variantCount, stage)`

Uses `network.connections.length` internally to call `resolveEffectiveMagnitude`.

```typescript
function buildWeightVariants(
  network: Network,
  variantCount: number,
  stage: NgeLifecycleStage,
): WeightVariantResult[] {
  const connectionCount = network.connections.length;
  const effectiveMagnitude = resolveEffectiveMagnitude(
    stage,
    variantCount,
    connectionCount,
  );
  // ... uses resolveRepresentativeDelta(index, variantCount, effectiveMagnitude)
}
```

**Call site** (grow-stabilize.ts:331): update to pass `stage` (already available
at line 320). `network.connections.length` is accessed internally.

**Parity**: Both `buildVariants` (variants.ts) and `buildWeightVariants`
(grow-stabilize.ts) now use the same `resolveEffectiveMagnitude` and
`resolveRepresentativeDelta` — parity guaranteed.

#### Section 8 — Weight-Exhaustion Gate (Final Agreed Spec)

### Formula 1 — Adaptive improvement threshold

```ts
function resolveExhaustionImprovementThreshold(
  baseline: number,
  bestScore: number,
  variantCount: number,
  stage: NgeLifecycleStage,
  neuronBudget: { current: number; max: number },
  scoreCeiling: number,
): number {
  const epsilon = NGE_EXHAUSTION_SCORE_EPSILON;
  const useMagnitude =
    !Number.isFinite(scoreCeiling) ||
    baseline >= scoreCeiling - epsilon ||
    bestScore >= scoreCeiling - epsilon;
  const scoreScale = useMagnitude
    ? Math.max(Math.abs(baseline), Math.abs(bestScore), epsilon)
    : Math.max(scoreCeiling - baseline, scoreCeiling - bestScore, epsilon);
  const stageFraction = resolveStageFraction(stage);
  const relativeBar = stageFraction * scoreScale;
  const noiseSigmaFraction = resolveNoiseSigmaFraction(stage);
  const noiseSigma = noiseSigmaFraction * scoreScale;
  const noiseUplift =
    variantCount <= 1 ? 0 : noiseSigma * Math.sqrt(2 * Math.log(variantCount));
  const neuronFactor =
    Number.isFinite(neuronBudget.max) && neuronBudget.max > 0
      ? Math.min(
          2.0,
          Math.max(
            0.5,
            1.0 + 0.5 * (1.0 - neuronBudget.current / neuronBudget.max),
          ),
        )
      : 1.0;
  return Math.max(relativeBar, noiseUplift) * neuronFactor;
}
```

### Formula 2 — Adaptive exhaustion count

```ts
function resolveExhaustionForceGrowthThreshold(
  variantCount: number,
  postGrowthBoostActive: boolean,
): number {
  const raw = Math.ceil(NGE_EXHAUSTION_TICK_BUDGET / Math.max(1, variantCount));
  const base = Math.min(Math.max(raw, 1), 8);
  if (!postGrowthBoostActive) return base;
  return Math.max(4, Math.min(base * 2.0, 16));
}
```

### Stage fractions and noise sigma

- Stage mapping: embryo→baby, equilibrium→adult
- Stage fractions: baby=0.01, juvenile=0.005, adult=0.003
- Noise sigma fractions: baby=0.003, juvenile=0.002, adult=0.001

### New constants table

| Constant                                         | Value |
| ------------------------------------------------ | ----- |
| NGE_EXHAUSTION_SCORE_EPSILON                     | 1e-6  |
| NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY         | 0.003 |
| NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE     | 0.002 |
| NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT        | 0.001 |
| NGE_EXHAUSTION_STAGE_FRACTION_BABY               | 0.01  |
| NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE           | 0.005 |
| NGE_EXHAUSTION_STAGE_FRACTION_ADULT              | 0.003 |
| NGE_EXHAUSTION_TICK_BUDGET                       | 48    |
| NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS             | 1     |
| NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS             | 8     |
| NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS | 16    |
| NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST      | 2.0   |
| NGE_EXHAUSTION_NEURON_BUDGET_FACTOR              | 0.5   |

### Post-growth anti-runaway

After bad growth (newScore < baseline - threshold, additive), double exhaustion
count (capped at 16, floored at 4). Reset on weight improvement or after 25
stabilization ticks (not total ticks — only ticks where the stabilization branch ran).

### Neuron-factor rationale

The neuron factor raises the threshold (1.5×) when the network is small relative
to the budget, biasing toward structural growth. As the network approaches the
neuron limit, the factor approaches 1.0, making weight refinement relatively
easier. This implements the user's intent: "grow fast past baby, picky in middle,
slower adult."

### Variant-path control flow

1. Compute `effectiveVariantCount = resolveVariantCountForStage(stage, undefined, accelerationConfig)`
2. `shouldEvaluate = effectiveVariantCount > 1 && hasTrainingData`
3. Exhaustion limit = `resolveExhaustionForceGrowthThreshold(effectiveVariantCount, postGrowthActive)`
4. If `consecutiveWeightExhaustion >= exhaustionLimit` → force structural growth, reset to 0
5. If `!shouldEvaluate` → non-variant path, increment exhaustion
6. Evaluate variants → `actualVariantCount = variantResult.metadata?.variantCount ?? effectiveVariantCount`
7. Consolidated guard: `bestIndex < 0 || !isFinite(bestScore) || bestIndex >= variants.length` → exhaustion
8. `threshold = resolveExhaustionImprovementThreshold(baseline, bestScore, actualVariantCount, ...)`
9. If `bestScore > baseline + threshold` → apply weight, reset exhaustion to 0, postGrowth=false
10. Else → increment exhaustion

### Test reconciliation (Slice 5A)

- Tests expecting `consecutiveWeightExhaustion: 2` → updated to 3 (baby default 16 → ceil(48/16)=3)
- Tests using `parallelVariantCount: 2` → updated to use `stageVariantCounts: { baby: 16 }` for deterministic effective count
- New tests for: noise uplift, neuron factor, post-growth boost, exact-ceiling fallback, N≤1 noise=0

### Validation evidence

- Round 1-3 (delta distribution): All 3 specialists GREEN_LIGHT ✅✅✅
- Round 1-7 (threshold gate): A GREEN (R4), B GREEN (R7), C GREEN (R4) ✅✅✅
- All design issues resolved: headroom-based scoreScale, stage-dependent fractions, noise-aware uplift, neuron budget factor, post-growth anti-runaway, exhaustion count scaling

#### Section 9: Seed Stride

```typescript
const seedStride = Math.max(
  NGE_VARIANT_PATCH_SEED_OFFSET, // 1000
  Math.max(1, patchSize) * NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR, // patchSize * 4
);
```

No `count` multiplier — prevents 32-bit overflow at high variant counts.
For patchSize 1-8, `patchSize*4` ranges 4-32, always < 1000, so stride=1000.

#### Section 10: applyWeightMutations Fallback

Add optional `magnitude` parameter to `applyWeightMutations()`:

```typescript
function applyWeightMutations(
  network: Network,
  rate: number,
  magnitude?: number, // optional override, backward-compatible
): void;
```

Existing 2-arg calls remain backward-compatible.

#### Section 11: Contract/README Cleanup

- Remove `NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA` from constants documentation
- Remove `WEIGHT_VARIANT_DELTA` from grow-stabilize documentation
- Rename `resolveMutationMagnitudeForStage` -> `resolveEffectiveMagnitude` in all refs
- Add new constants to the constant ledger
- Update JSDoc for `buildVariants`, `buildWeightVariants`, `resolveRepresentativeDelta`

#### Section 12: Deferred to Follow-up Phase

- Growth cadence scaling (how often growth phases fire relative to stabilization)
- Further tuning of widthFactor/sizeFactor curves based on empirical results

### Validation Loop Status

- **Round 1 specialists**: 3 dispatched, all completed with detailed proposals
- **Round 1 validators**: 3 dispatched, all returned CHANGES_REQUESTED (16 issues total)
- **Round 1 fixes applied**: All 16 issues addressed
- **Round 2 validators**: 3 dispatched — B gave GREEN_LIGHT, A and C returned CHANGES_REQUESTED (4 issues total)
- **Round 2 fixes applied**: All 4 issues addressed (sizeFactor clamp, connectionCount denominator, top-of-cycle exhaustion override, explicit buildVariants refactor)
- **Round 3 validators**: All 3 gave GREEN_LIGHT ✅✅✅
  - r3-validator-a-delta: GREEN_LIGHT — sizeFactor clamp and connectionCount denominator verified
  - r3-validator-b-patch: GREEN_LIGHT — no regressions, all patch scaling concerns remain resolved
  - r3-validator-c-magnitude: GREEN_LIGHT — top-of-cycle override and buildVariants refactor verified

## Latest validation evidence

green-light: true

```yaml
status: green-light
validated_by: r3-validator-a-delta, r3-validator-b-patch, r3-validator-c-magnitude, threshold-validator-a, threshold-validator-b, threshold-validator-c
validated_at: 2026-07-16T21:25-04:00
round: 7
verdict: all-green
notes: >
  All 3 Round 3 delta-distribution validators and all 3 Round 7 threshold-gate
  validators gave GREEN_LIGHT. The agreed action plan and final adaptive
  weight-exhaustion threshold spec (Section 8) are ready for implementation.
  Phase 4 is DONE. Phase 5 slice structure is verified and active.
```

### Phase 5 independent verification — 2026-07-16

```yaml
status: green-light
verifier: 01-planning-verification
verified_at: 2026-07-16T20:23-04:00
mode: independent
phase: 5
verdict: all-green
notes: >
  Fresh independent verification of Phase 5 slice structure and action-plan
  coverage. All five slices are present, all estimates are within the 4-hour cap
  (2–3 hours each), dependencies form a correct red→implement→green chain,
  acceptance criteria are observable and testable, and files-to-change match the
  agreed Sections 1-12 action plan.
```

### Final threshold-design validation — 2026-07-16

```yaml
status: green-light
validated_by: threshold-validator-a, threshold-validator-b, threshold-validator-c
validated_at: 2026-07-16T21:25-04:00
mode: specialist-review
round: 7
verdict: all-green
notes: >
  All 3 validators gave GREEN_LIGHT after 7 rounds of threshold-gate design.
  Section 8 now records the final agreed adaptive threshold spec:
  headroom-based scoreScale, stage-dependent fractions, noise-aware uplift,
  neuron budget factor, post-growth anti-runaway, and exhaustion count scaling.
  Slice 5D acceptance criteria updated to match the final spec.
```

**Phase 5 slice checklist**

| #   | Slice                  | Estimate | Goal          | Dependencies | Maps to action-plan sections                                                                                                                                                                                                                                                                                                                |
| --- | ---------------------- | -------- | ------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5A  | dynamic-delta-red      | 2h       | red-testing   | none         | Section 1 (remove constants prelude), Section 4 (delta formula contract)                                                                                                                                                                                                                                                                    |
| 5B  | dynamic-delta-impl     | 2h       | [DONE]        | 5A           | Section 1 (remove), Section 4 (resolveRepresentativeDelta), Section 5 (buildVariants refactor)                                                                                                                                                                                                                                              |
| 5C  | magnitude-scaling      | 3h       | [DONE]        | 5B           | Section 2 (new constants), Section 6 (resolveEffectiveMagnitude), Section 7 (buildWeightVariants)                                                                                                                                                                                                                                           |
| 5D  | exhaustion-and-patches | 3h       | [DONE]        | 5C           | Section 2 (NGE_EXHAUSTION_* constants), Section 3 (patch constants), Section 8 (final adaptive threshold spec: resolveExhaustionImprovementThreshold, resolveExhaustionForceGrowthThreshold, stage/noise helpers, variant-path control flow, post-growth anti-runaway), Section 9 (seed stride), Section 10 (applyWeightMutations fallback) |
| 5E  | contract-green         | 2h       | green-testing | 5B, 5C, 5D   | Section 11 (contract/README cleanup)                                                                                                                                                                                                                                                                                                        |

- Section 12 (deferred growth cadence / curve tuning) is intentionally excluded from implementation slices.
- Slice ordering: 5A → 5B → 5C → 5D → 5E. Dependencies are acyclic and match the declared `next_slice` chain.
- All estimates: 2h or 3h, within the 4-hour cap and ideally 2–3 hours.
- All acceptance criteria carry unique `AC-5X##` IDs and map to focused Jest/tsc/lint validations.
- Files-to-change lists are concrete and bounded to `src/neat/nge-juvenile/*` plus `README.md`.

### Verification of Section 8 / Slice 5D — 2026-07-16T21:29-04:00

```yaml
status: green-light
verifier: 01-planning-verification
verified_at: 2026-07-16T21:29-04:00
mode: independent
phase: 5
verdict: all-green
checks:
  - slice_size_cap: pass
  - phase_5_slice_completeness: pass
  - section_8_threshold_spec_completeness: pass
  - slice_5d_ac_spec_alignment: pass
  - latest_validation_evidence_green_light: pass
  - plan-slice-quality gate: pass
  - step-packet gate: pass
notes: >
  Independent verification of the Section 8 weight-exhaustion gate spec and Slice 5D
  acceptance criteria. Slice estimates: 5A=2h, 5B=2h, 5C=3h, 5D=3h, 5E=2h (all ≤ 4h).
  Section 8 contains both formulas, the constants table, stage/noise fractions,
  control flow, post-growth anti-runaway, and test reconciliation. Slice 5D AC-5D01..AC-5D07
  map directly to the final spec. Plan-slice-quality and step-packet gates both PASS.
  No blockers. Phase 5 remains green-lit and ready for execution.
```

## Phase 5 planning validation evidence

- **Plan-sync gate**: `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md` — **PASS**
- **Plan-slice-quality gate**: `node scripts/agent-customization/gates/plan-slice-quality.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md` — **PASS**
- **Step-packet gate**: `node scripts/agent-customization/gates/step-packet.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md` — **PASS**
- **Plan-readiness gate**: `node scripts/agent-customization/gates/plan-readiness.gate.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md` — **PASS**
- **Re-validation after Section 8 / Slice 5D update** (2026-07-16T21:25-04:00 and 2026-07-16T21:29-04:00): all four gates above re-run and still PASS.

All Phase 5 slices are authored, within the 4-hour cap, and conform to the
red→implement→green step-packet sequence required by the repo. The next working
frontier is **Slice 5E** (contract/README cleanup and final green validation).

**Full gate JSON**

```json
{
  "planSync": {
    "name": "plan sync",
    "ok": true,
    "issues": [],
    "counts": {
      "errors": 0,
      "warnings": 0
    },
    "summaryText": "PASS plan sync: 0 errors, 0 warnings (plan: plans/racing-curriculum-parallel-variant-gap.plans.md)",
    "plan": {
      "path": "plans/racing-curriculum-parallel-variant-gap.plans.md",
      "status": "WIP"
    },
    "downstreamTrackers": [
      "plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md",
      "plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md",
      "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
      "plans/mcp-active-binding.plans.md"
    ]
  },
  "planSliceQuality": {
    "pass": true,
    "evidence": {
      "plansChecked": [
        "plans/mcp-active-binding.plans.md",
        "plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md",
        "plans/racing-curriculum-parallel-variant-gap.plans.md"
      ],
      "violations": [],
      "limit": 4
    },
    "fixHint": "All WIP plan slices are within the 4-hour estimate limit.",
    "owner": "plan-slice-quality.gate.mjs"
  },
  "stepPacket": {
    "pass": true,
    "evidence": {
      "blocksChecked": [
        "plans/mcp-active-binding.plans.md:yaml@14718",
        "plans/mcp-active-binding.plans.md:yaml@16171",
        "plans/racing-curriculum-parallel-variant-gap.plans.md:yaml@11987"
      ],
      "violations": [],
      "planReadinessWarnings": [],
      "plansScanned": 3
    },
    "fixHint": "All active WIP phase/step packets conform to the new format.",
    "owner": "step-packet.gate.mjs"
  },
  "planReadiness": {
    "pass": true,
    "evidence": {
      "plan": "plans/racing-curriculum-parallel-variant-gap.plans.md",
      "sectionFound": true,
      "greenLightFound": true,
      "sectionPreview": "green-light: true"
    },
    "fixHint": "Plan has a recorded green light from independent 01-planning verification.",
    "owner": "01-planning"
  },
  "generatedAt": "2026-07-16T21:29-04:00"
}
```
