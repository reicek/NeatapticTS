# Racing Curriculum Parallel Variant Evaluation Gap

Research artifact for ad-hoc user investigation: "Why do racing-curriculum
networks grow slowly despite configuring `parallelVariantCount=1024` and
`stageVariantCounts.baby=1024`?"

## Question

Is the racing-curriculum demo actually generating and evaluating 1024 weight
variants in parallel per adaptation step, and if not, where does the
"1024 × 1024" configuration value drop out of the execution path?

## Evidence

### 1. Static source authority — config is wired and displayed

- The browser harness builds `accelerationConfig` with
  `parallelVariantCount: 1024` and `stageVariantCounts: { baby: 1024 }`
  (`examples/racing_curriculum/browser-entry/browser-entry.ts:481-484`).
- It passes the config to `createPerCarAdaptationEngines`
  (`examples/racing_curriculum/browser-entry/browser-entry.ts:537-543`).
- `parallelVariantCount=1024` is used for backend-selection cache seeding
  (`examples/racing_curriculum/browser-entry/browser-entry.ts:559-564`) and for
  the HUD chip suffix (`examples/racing_curriculum/browser-entry/browser-entry.ts:2084-2088,
  2137-2156`).

### 2. Static source authority — the value is dropped before evaluation

- `createRuntimeAdaptationEngine` receives the whole `accelerationConfig` but
  only extracts `stageVariantCounts.baby`
  (`examples/racing_curriculum/controller/runtime.adaptation.ts:297-298`).
- It threads that single scalar as `babyVariantCount` into the grow-stabilize
  config
  (`examples/racing_curriculum/controller/runtime.adaptation.ts:385-391,
  403-408`).
- `runNgeGrowStabilizeCycle` stores the lifecycle config
  (`src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:258-270`) and
  uses it for stage, cadence, stabilization intensity, and mutation magnitude
  (`src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-stages.ts`). It does **not**
  use `babyVariantCount` to generate or evaluate weight variants.
- The default racing adaptation loop (`adaptOnTick`) calls the core `adapt()`
  API with a custom evaluator whose `baseline`/`apply`/`candidate` functions
  call `evaluateRacingTrendScore`
  (`examples/racing_curriculum/controller/runtime.adaptation.ts:368-426,
  743-762, 1067-1103`). That function runs a small fixed forward-pass sample; it
  never evaluates weight variants.
- `evaluateRacingWeightVariantsAsync`
  (`examples/racing_curriculum/controller/runtime.adaptation.ts:876-906`) is
  defined and exported but has **zero production call sites** in the racing demo.
- Generic variant evaluation in `src/acceleration/acceleration.variants.ts` is
  likewise unreachable from the live demo path.

### 3. Static source authority — the variant generator is narrow

- When the variant evaluator is invoked, `buildVariants`
  (`src/neat/nge-juvenile/neat.nge-juvenile.variants.ts:174-190`) creates one
  variant per index by cycling a single connection index and applying a linear
  delta: `delta = 0.05 * (index + 1)`.
- For 1024 variants the deltas run from `0.05` to `51.2`. Only one connection is
  perturbed per variant, and the perturbations are deterministic, not stochastic.
- `evaluateRacingWeightVariantsAsync` duplicates the same formula in the racing
  layer (`examples/racing_curriculum/controller/runtime.adaptation.ts:891-895`)
  and hard-codes the seed to `undefined`
  (`examples/racing_curriculum/controller/runtime.adaptation.ts:903`).

### 4. Runtime/validation authority — tests confirm the gap

- Targeted Jest suites pass and confirm the current behavior is not failing tests:
  - `examples/racing_curriculum/controller/nge-e2e-growth.test.ts`
  - `examples/racing_curriculum/controller/runtime.adaptation.test.ts`
  - `examples/racing_curriculum/__tests__/runtime.adaptation.test.ts`
  - 96/96 tests passed.
- No existing test asserts that 1024 variants are evaluated, that variants are
  diverse, or that `parallelVariantCount` affects the number of forward passes
  in the live adaptation loop.

### 5. Documentation authority — intended vs. actual

- `examples/racing_curriculum/controller/README.md:339-374` describes
  `parallelVariantCount` and `stageVariantCounts` as controlling how many weight
  variants are scored in parallel and per lifecycle stage. This is the intended
  semantics, but the live racing engine does not realize them.
- The completed plan `plans/completed/Acceleration_Parallelism_Props.plans.md`
  records the intent to make these properties configurable end-to-end, but the
  racing adaptation engine remains unwired.

## Decision

The racing-curriculum demo is **not** evaluating 1024 weight variants in
parallel on its default adaptation path. The `parallelVariantCount=1024` and
`stageVariantCounts.baby=1024` values are received and used for backend cache
seeding and HUD display, but they are dropped before any variant generation or
batched evaluation. The actual growth speed is governed by the NGE
grow-stabilize cycle (`runNgeGrowStabilizeCycle`) with its own cadence,
cooldowns, stabilization windows, and throttle — none of which use the variant
counts.

To fulfill the user's expectation, the next implementation step must either:

1. **Wire the variant evaluator into the adaptation loop**: make `adaptOnTick`
   (or a new fast-growth mode) call `evaluateRacingWeightVariantsAsync`, select
   the best variant, and commit it as an adaptation step; or
2. **Stop advertising unused parallelism**: remove or deprecate the
   misleading `1024` HUD/config wiring and update the README to explain the real
   growth-speed controls.

Both options must preserve the existing commit/rollback/hysteresis machinery
unless the design intentionally replaces it.

## Risks

- **Unintended replacement vs. coexistence**: The current
  `evaluateRacingTrendScore` path exists for a reason (rolling-score-window
  adaptation with rollback). Replacing it wholesale with variant scoring may
  break the trend/hysteresis behavior. The implementer must decide whether the
  variant evaluator supplements or replaces the trend evaluator.
- **Variant diversity is poor**: If the variant evaluator is wired in as-is, the
  1024 variants will be a deterministic single-connection sweep with deltas up
  to 51.2, likely wasting compute. The implementer should redesign variant
  generation (multi-connection perturbations, per-variant seeds, bounded deltas)
  before exposing it on the hot path.
- **CPU fallback wall**: With `Promise.all` over 1024 CPU forward passes, the
  browser main thread will still stall because JavaScript forward passes are
  serial. Real parallelism only happens when WebGPU or Worker acceleration is
  active. The implementer must ensure the batch is dispatched through the
  acceleration backend and add a graceful CPU cap.
- **Config surface ignored fields**: `stageVariantCounts.juvenile` and
  `stageVariantCounts.adult` are currently silently ignored by the racing engine.
  Any wiring fix should either honor them or document that only `baby` is used.
- **Tests do not catch this**: No existing test will detect whether the variant
  evaluator is wired into the live loop. The implementer must add an integration
  test that asserts the number of forward passes scaled with `parallelVariantCount`
  and that a best variant is committed.

## Validation recommended

- Add an integration test under `examples/racing_curriculum/controller/` that:
  - constructs an engine with `stageVariantCounts.baby=8` and
    `parallelVariantCount=8`;
  - stubs the variant evaluator to count how many variants are evaluated;
  - calls `adaptOnTick` and asserts the variant path is exercised and a winning
    variant is committed.
- Re-run the targeted Jest suites after any wiring change to confirm no
  regression in the trend-score commit/rollback behavior.
