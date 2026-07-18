# Racing Curriculum Parallel Variant Evaluation Gap

**Status:** [DONE]

## Scope

User-raised investigation: the racing-curriculum demo configures
`parallelVariantCount=1024` and `stageVariantCounts.baby=1024` but networks still
grow slowly. Determine whether those values actually reach the live adaptation
loop and, if not, what change is required to make the advertised parallelism
real.

This plan is downstream of the (completed)
`plans/completed/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` NGE racing
workstream.

## Implementation phases

### Phase 1 — Research [DONE]

```yaml
phase: 1
title: 'Research parallel variant gap'
status: '[DONE]'
goal: research
expansion: none
auto_expand: false
mode: one-shot
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
skills:
  - research-methodology
  - subagent-delegation-patterns
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
acceptance_criteria:
  - 'Trace from browser-entry config through createRuntimeAdaptationEngine to live adaptation loop completed.'
  - 'Research artifact materialized at docs/research/racing-curriculum-parallel-variant-gap.md.'
  - 'Plan registered in plans/README.md and plans/Roadmap.md.'
```

#### Step 01 — Trace config-to-execution pipeline [DONE]

```yaml
phase: 1
step: 1
title: 'Trace config-to-execution pipeline'
status: '[DONE]'
goal: research
expansion: none
auto_expand: false
mode: one-shot
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
skills:
  - research-methodology
  - subagent-delegation-patterns
research_artifact: docs/research/racing-curriculum-parallel-variant-gap.md
next_step: 'Phase 2 Step 01 — Design integration strategy and red tests'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
acceptance_criteria:
  - 'Confirm 1024 values are config/display-only in the live loop.'
  - 'Confirm variant evaluator is defined but unwired.'
  - 'Confirm variant generator is deterministic and narrow.'
```

`02-researching` traced the full config-to-execution pipeline. Key findings:

- The 1024 values are wired into `accelerationConfig` and displayed in the HUD,
  but the default racing adaptation engine (`adaptOnTick`) does **not** call
  `evaluateRacingWeightVariantsAsync` or `evaluateNgeWeightVariants`.
- `stageVariantCounts.baby` is forwarded into `runNgeGrowStabilizeCycle` as
  `babyVariantCount`, where it only affects lifecycle scalar policy (cadence,
  stabilization intensity, mutation magnitude), not forward-pass variant
  volume.
- `parallelVariantCount` only seeds the acceleration backend cache and drives
  the HUD label; it never batches variant evaluation.
- When invoked, the variant generator produces 1024 deterministic
  single-connection perturbations with deltas `0.05 * (index + 1)`, which is a
  poor local search and wastes compute for large indices.

### Phase 2 — Design and red tests [DONE]

```yaml
phase: 2
title: 'Design integration strategy and red tests'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
skills:
  - plan-alignment
  - test-design
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
acceptance_criteria:
  - 'Integration strategy chosen and documented in plan.'
  - 'Red tests fail before implementation and pass after.'
```

#### Step 01 — Choose integration strategy and write red tests [DONE]

```yaml
phase: 2
step: 1
title: 'Choose integration strategy and write red tests'
status: '[DONE]'
goal: planning
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
skills:
  - plan-alignment
  - test-design
  - execute
research_artifact: docs/research/racing-curriculum-parallel-variant-gap.md
next_step: 'Phase 3 Step 01 — Implement chosen wiring or config cleanup'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/racing-curriculum-parallel-variant-gap.plans.md'
  - 'npx jest examples/racing_curriculum/controller/runtime.adaptation.test.ts examples/racing_curriculum/__tests__/runtime.adaptation.test.ts --silent'
acceptance_criteria:
  - 'Decide between wiring variant evaluator, adding opt-in mode, or removing misleading config/HUD.'
  - 'Add regression tests that fail before the fix and pass after.'
```

**Strategy decision: Option B — opt-in `useVariantEvaluator` engine mode.**

- Keep the existing trend evaluator as the default path in `adaptOnTick`. This
  preserves the current rollback/hysteresis behavior and all existing test
  expectations.
- Add a new `useVariantEvaluator?: boolean` field to
  `RuntimeAdaptationEngineOptions`. When enabled, the engine exercises the
  existing async `evaluateRacingWeightVariantsAsync` wrapper.
- The opt-in mode must:
  1. Reject non-`baby` `stageVariantCounts` until lifecycle-specific variant
     generation is implemented (owner-local validation at engine construction).
  2. Enforce a CPU fallback safety cap so a CPU-only backend cannot run 1024
     sequential forward passes on the main thread.

Red tests added:

- `examples/racing_curriculum/controller/runtime.adaptation.test.ts`
  - `exercises evaluateRacingWeightVariantsAsync when useVariantEvaluator is enabled`
  - `rejects stageVariantCounts outside baby when useVariantEvaluator is enabled`
  - `enforces a CPU fallback safety cap on variant count`
- `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts`
  - Source-contract assertions for `useVariantEvaluator` option,
    `stageVariantCounts` validation, and CPU cap presence.

**Focused baseline (before red tests):** 2 suites passed, 90 tests passed.

**Red evidence:**

```text
Command: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/runtime.adaptation.test.ts --testPathPatterns=examples/racing_curriculum/__tests__/runtime.adaptation.test.ts
Result: Test Suites: 2 failed, 2 total; Tests: 6 failed, 90 passed, 96 total
Failing tests:
  - exposes useVariantEvaluator in RuntimeAdaptationEngineOptions
  - validates stageVariantCounts in variant mode
  - contains a CPU fallback safety cap for unsafe variant counts
  - exercises evaluateRacingWeightVariantsAsync when useVariantEvaluator is enabled
  - rejects stageVariantCounts outside baby when useVariantEvaluator is enabled
  - enforces a CPU fallback safety cap on variant count
```

The 6 new red contracts target the exact behaviors the Phase 3 implementation
must add. The original 90 tests remain green, confirming the default trend
path is unchanged.

### Phase 3 � Implementation [DONE]

```yaml
phase: 3
step: 1
title: 'Library-default variant evaluation in NGE grow-stabilize'
status: '[DONE]'
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
next_step: 'Phase 4 Step 01 � Green validation and coverage guard'
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
  - principle-4-small-slices
  - principle-5-unique-ids
```

[DONE] Phase 3 complete. Slices 03A�03E passed green validation: 135/135 tests across 5 suites, 100% coverage on `neat.nge-juvenile.variants.ts` and `neat.nge-juvenile.grow-stabilize.ts`, TypeScript/lint/build/plan-sync passed, browser smoke test showed `GPU (1024 variants)` green chip with no TypeError crashes. Detailed slice logs, PlanUpdate records, and follow-up fix notes moved to `plans/racing-curriculum-parallel-variant-gap.logs.md`.

Implement library-default variant evaluation. The stabilization phase of
`runNgeGrowStabilizeCycle` now calls `evaluateNgeWeightVariants` whenever the
resolved `AccelerationConfig.parallelVariantCount` is greater than 1, and
falls back to the existing `applyPlasticity` path otherwise. The racing demo
stops owning variant-evaluation policy and only forwards its
`AccelerationConfig` into the grow-stabilize cycle.

### Phase 4 — Dynamic delta specialist analysis [DONE]

```yaml
phase: 4
step: 1
title: 'Dynamic delta specialist analysis and validation loop'
status: '[DONE]'
goal: green-testing
tdd_sequence: green-only
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/racing-curriculum-parallel-variant-gap.plans.md
copy_paste: true
skills:
  - coverage-guard
next_step: 'Phase 5 — Dynamic delta distribution implementation'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
acceptance_criteria:
  - id: AC-401
    text: 'All library NGE juvenile suites pass'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-juvenile'
  - id: AC-402
    text: '100% coverage on all touched src/ files'
    validation: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-juvenile'
  - id: AC-403
    text: 'Racing-curriculum controller suites remain green'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum'
constitution_check:
  - 'principle-4-small-slices'
```

Run targeted green suites and the coverage guard over all touched `src/`
files.

[DONE] Phase 4 completed via 3-round specialist validation loop. All 3 Round 3
validators gave GREEN_LIGHT. The agreed action plan (Sections 1-12 in "Real
Issue" section above) forms the basis for Phase 5+ implementation slices.

### Phase 5 — Dynamic delta distribution implementation [DONE]

**Phase objective:** Implement the Round-3-agreed dynamic delta distribution
action plan inside the NGE juvenile variant generator and grow-stabilize cycle.

[DONE] Phase 5 complete. Slices 5A–5E passed green validation: 18 suites, 435
tests, 100% coverage on touched `src/neat/nge-juvenile/` files. Detailed slice
logs, specialist review rounds, threshold-gate spec, and final docs cleanup are
in `plans/completed/racing-curriculum-parallel-variant-gap.logs.md`.

- Slice 5A: Red tests written for dynamic delta, magnitude scaling, and adaptive threshold
- Slice 5B: Dynamic delta distribution implemented (`resolveRepresentativeDelta`, endpoint-inclusive formula)
- Slice 5C: Magnitude scaling implemented (`resolveEffectiveMagnitude` with width/size factors)
- Slice 5D: Weight-exhaustion gate implemented (10-step control flow, post-growth anti-runaway, `preGrowthBaseline`)
- Slice 5E: Test reconciliation + branch coverage (18 suites, 435 tests, 100% coverage)
- 3 specialist review rounds (all `GREEN_LIGHT`)
- 7 specialist-found issues fixed (baseline source, post-growth lifecycle, stale config resolver, test imports, README)
- Docs regenerated, 12 private helpers tagged `@internal`, README constants accurate

### Phase 6 — Documentation compression and tracker closure [DONE]

06-documenting completed docs cleanup. 07-logging compressed Phase 5 history to
`plans/completed/racing-curriculum-parallel-variant-gap.logs.md`. Plan status [DONE].
