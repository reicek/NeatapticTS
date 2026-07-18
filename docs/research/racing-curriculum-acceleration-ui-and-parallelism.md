# Racing Curriculum Acceleration UI & Parallelism Research

## Question

For the racing-curriculum demo and the generic acceleration layer:

1. How is acceleration status currently shown in the demo UI, and where should a color-coded status label be added?
2. Where is the "16x" parallel variant count defined, how does variant evaluation actually work, and is the evaluation truly parallel?
3. What interfaces/types must change to make the variant/parallel count configurable through the acceleration layer?
4. How does the racing demo currently configure acceleration?

## Evidence

### 1. Demo UI — acceleration status display

Source of truth: static source code in `examples/racing_curriculum/`.

- **HTML shell**: `examples/racing_curriculum/index.html` is plain HTML with an inline `<style>` block (lines 7–59). It has a `#status` paragraph for loading messages and a `<div id="racing-curriculum-output">` where the bundle mounts the demo UI (lines 62–65).
- **No separate CSS files**: there are no `.css`, `.scss`, or `.less` files under `examples/racing_curriculum/`. All demo-specific styling is injected at runtime by `browser-entry.ts`.
- **UI rendering entry**: `examples/racing_curriculum/browser-entry/browser-entry.ts` is the only UI-rendering file. It injects a `<style id="racing-curriculum-styles">` block (lines 921–1396) and builds the canvas stage, telemetry panel, and controls using vanilla DOM APIs (`document.createElement`, `textContent`, `className`). No framework or CSS-in-JS is used.
- **Existing acceleration row**: a telemetry row already exists and is updated every frame:
  - Initialized at `browser-entry.ts:1471` as `document.createTextNode('Acceleration: cpu')`.
  - Appended to the telemetry grid at `browser-entry.ts:1487` via `buildPanelRowWithLiveNode('Acceleration', accelerationValue)`.
  - Updated each frame at `browser-entry.ts:2012–2017` to `Acceleration: ${accelerationMode}`, where `accelerationMode` is the result of `LifecycleAccelerationPolicy.evaluate(...).mode` (`'cpu' | 'gpu' | 'worker'`).
- **Status-chip pattern**: the stage-card metadata row (lines 1426–1432) already uses a `createStatusChip(label, value)` helper. This is the natural place to add a prominent, color-coded acceleration chip.

```ts
// examples/racing_curriculum/browser-entry/browser-entry.ts:1426-1432
const metaElement = document.createElement('div');
metaElement.className = 'racing-stage-card__meta';
metaElement.append(
  createStatusChip('Controller', 'Live NGE controller'),
  createStatusChip('Track', 'Spline-smoothed visual'),
  createStatusChip('Seed', '42 • v1 • medium'),
);
```

### 2. Variant evaluation & the "16x" count

Source of truth: static source code in `src/acceleration/` and `src/neat/nge-juvenile/`.

- **Generic evaluator is sequential, not parallel**: `evaluateWeightVariantsAsync` in `src/acceleration/acceleration.variants.ts:112–160` loops `for (const variant of variants)` and `await`s each `evaluateVariant` call. The current backend is hardcoded to `'cpu'` (line 124), and `variantCount` is only reported in metadata (line 155).

```ts
// src/acceleration/acceleration.variants.ts:127-137
const scores: number[] = [];
for (const variant of variants) {
  const score = await evaluateVariant(
    network,
    variant,
    inputs,
    target,
    scorer,
  );
  scores.push(score);
}
```

- **NGE wrapper**: `evaluateNgeWeightVariants` in `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts:58–76` resolves a stage-based variant count, builds variants, and delegates to `evaluateWeightVariantsAsync`. It currently does not pass a `config` or `observer` to the generic evaluator.

```ts
// src/neat/nge-juvenile/neat.nge-juvenile.variants.ts:58-76
export async function evaluateNgeWeightVariants(
  network: VariantEvaluationNetwork,
  stage: NgeLifecycleStage,
  inputs: number[][],
  target: number[],
  seed?: number,
): Promise<WeightVariantResult> {
  const variantCount = resolveVariantCountForStage(stage);
  const variants = buildVariants(network, variantCount);

  return evaluateWeightVariantsAsync(
    network,
    variants,
    inputs,
    target,
    undefined,
    seed,
  );
}
```

- **"16x" is the baby-stage variant count, not a parallel-dispatch count**: the constant is `NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT = 16` at `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:329`. It is hardcoded and exported. The stage resolver maps:
  - `embryo`/`baby` → 16
  - `juvenile` → 8
  - `adult`/`equilibrium` → 2

```ts
// src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:327-336
/**
 * Contract: NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT=16
 */
export const NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT = 16;

/**
 * Contract: NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT=8
 */
export const NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT = 8;
```

### 3. Configurability — interfaces and types

Source of truth: static source code in `src/acceleration/acceleration.types.ts`, `acceleration.config.ts`, `acceleration.constants.ts`, and `acceleration.manager.ts`.

- **No existing `variantCount` / `parallelCount`**: `AccelerationConfig` (`src/acceleration/acceleration.types.ts:84–129`) only has backend-selection thresholds (`gpuNodeThreshold`, `gpuBatchParallelThreshold`, `maxWorkers`, etc.) and benchmark settings. `AccelerationManagerOptions` (`src/acceleration/acceleration.manager.ts:19–25`) only accepts `config?: Partial<AccelerationConfig>` and `observer`.
- **Nearest existing knobs** (not the same thing):
  - `batchParallelCount` appears in `AutoEnableAccelerationOptions` / auto-enable worker/GPU options (`src/acceleration/acceleration.orchestrator.ts`, `acceleration.workers.ts`). It is used for backend auto-enable thresholds, not for the variant evaluator.
  - `BufferPoolWorkload.variantCount` (`src/acceleration/acceleration.types.ts:181`) is for future GPU buffer sizing and is not wired into the current heuristic.
- **Where to add a configurable count**: the cleanest path is:
  1. Add a new optional field to `AccelerationConfig` (e.g. `variantCount?: number` or `parallelVariantCount?: number`) in `src/acceleration/acceleration.types.ts`.
  2. Add a matching default constant in `src/acceleration/acceleration.constants.ts`.
  3. Resolve it inside `resolveAccelerationConfig` in `src/acceleration/acceleration.config.ts`.
  4. Change the `config?: unknown` parameter of `evaluateWeightVariantsAsync` (`src/acceleration/acceleration.variants.ts:119`) to `AccelerationConfig` and use the new count to drive batching/concurrency.

### 4. Racing demo configuration

Source of truth: static source code in `examples/racing_curriculum/controller/runtime.adaptation.ts`.

- The racing demo does **not** use `AccelerationManager` or `resolveAccelerationConfig`. It imports the generic primitive directly:

```ts
// examples/racing_curriculum/controller/runtime.adaptation.ts:19
import { evaluateWeightVariantsAsync } from '../../../src/acceleration/acceleration.variants';
```

- The demo wrapper `evaluateRacingWeightVariantsAsync` (`runtime.adaptation.ts:826–853`) takes `variantCount` as a caller parameter and builds variants with `Array.from({ length: variantCount }, …)`. It passes those variants to `evaluateWeightVariantsAsync` but does not pass any `config`.

```ts
// examples/racing_curriculum/controller/runtime.adaptation.ts:826-853
export async function evaluateRacingWeightVariantsAsync(
  liveNetwork: Network,
  evidenceWindow: readonly (number | RacingQualitySignal)[],
  variantCount: number,
): Promise<WeightVariantResult> {
  // ... build inputs/target ...
  const connectionCount = liveNetwork.connections.length;
  const variants = Array.from({ length: variantCount }, (_, index) => ({
    weightIndex: connectionCount > 0 ? index % connectionCount : 0,
    delta: 0.05 * (index + 1),
  }));

  return evaluateWeightVariantsAsync(
    liveNetwork,
    variants,
    inputs,
    target,
    undefined,
  );
}
```

## Decision

- The cleanest place for a **color-coded acceleration status label** is a new `createStatusChip('Acceleration', accelerationMode)` in the stage-card metadata row (`browser-entry.ts:1426–1432`), updated each frame next to the existing telemetry update (`browser-entry.ts:2012–2017`). The existing telemetry row can be kept or colorized as a lower-impact alternative.
- The "16x" count is **not** a parallel evaluation count today; it is the hardcoded **baby-stage variant count** in `neat.nge-juvenile.constants.ts`. The actual evaluator is sequential.
- To make variant/parallel evaluation configurable, add a new field to `AccelerationConfig`/`resolveAccelerationConfig`/`acceleration.constants`, type the `config` parameter of `evaluateWeightVariantsAsync` as `AccelerationConfig`, and wire the count through the evaluation loop (and eventually introduce real concurrency if parallel evaluation is intended).
- The racing demo currently controls its own variant count via a caller-supplied parameter; it does not consume an acceleration config. Any new acceleration-level config will need to be threaded into `evaluateRacingWeightVariantsAsync` or surfaced in the demo's UI/tuning controls.

## Post-implementation update

Phase 1 implemented the decisions above as follows:

- `AccelerationConfig.parallelVariantCount` was added in `src/acceleration/acceleration.types.ts`, with default constant `DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT = 1` in `src/acceleration/acceleration.constants.ts`. A value of `1` preserves the original sequential path; values greater than `1` dispatch that many variants concurrently using batched `Promise.all` evaluation, restoring each connection's original weight between batches.
- `resolveAccelerationConfig` in `src/acceleration/acceleration.config.ts` now fills `parallelVariantCount`.
- `evaluateWeightVariantsAsync` in `src/acceleration/acceleration.variants.ts` consumes the resolved `parallelVariantCount`, runs batched concurrent CPU evaluation, and reports the active backend in metadata.
- `evaluateRacingWeightVariantsAsync` in `examples/racing_curriculum/controller/runtime.adaptation.ts` accepts an optional `AccelerationConfig` and forwards it to the generic evaluator, so the racing demo can opt into concurrent variant scoring.
- The racing demo UI gained a persistent, color-coded acceleration status chip in the stage-card metadata row (`examples/racing_curriculum/browser-entry/browser-entry.ts`), with neon red for CPU, neon green for GPU, and neon yellow for WebWorker, updated every frame alongside existing telemetry.

The hardcoded NGE lifecycle variant counts (`NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT = 16`, etc.) remain separate from `parallelVariantCount`: they decide how many candidate perturbations to generate, while `parallelVariantCount` decides how many of those candidates are evaluated at once.

## Research pass: 2026-07-15 — `parallelVariantCount` vs. the hardcoded 16

This pass re-audits the current codebase with exact line numbers after the `parallelVariantCount` feature landed.

### Q1: Where exactly is the hardcoded 16?

- **File**: `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts`
- **Line**: 329
- **Constant**: `NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT = 16`
- Sibling defaults:
  - `NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT = 8` (line 336)
  - `NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT = 2` (line 346)

### Q2: How is the 16 used?

- `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts:93-105` maps lifecycle stages to counts; `embryo`/`baby` → 16.
- `evaluateNgeWeightVariants` (lines 58-76) resolves the stage-based count, builds exactly that many weight-perturbed variants, and calls `evaluateWeightVariantsAsync`.
- The 16 is therefore a **variant-generation count** (how many candidates to try), not a parallel-dispatch count.

### Q3: Relationship to `parallelVariantCount`

- They are currently **independent / orthogonal**:
  - `NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT` controls **how many** variants are created.
  - `AccelerationConfig.parallelVariantCount` (default `1` from `src/acceleration/acceleration.constants.ts:45`) controls **how many** of those variants are scored concurrently inside `evaluateWeightVariantsAsync`.
- The two knobs meet only inside `evaluateWeightVariantsAsync`, but `evaluateNgeWeightVariants` passes `undefined` as the `AccelerationConfig`, so `parallelVariantCount` always falls back to 1 for the NGE juvenile wrapper today.

### Q4: How does `evaluateWeightVariantsAsync` receive its variant list?

- Function signature at `src/acceleration/acceleration.variants.ts:118-127`:
  ```ts
  export async function evaluateWeightVariantsAsync(
    network,
    variants,
    inputs,
    target,
    scoreFn?, seed?, config?, observer?
  )
  ```
- `evaluateNgeWeightVariants` builds the array locally and passes it as the `variants` argument.
- `evaluateRacingWeightVariantsAsync` at `examples/racing_curriculum/controller/runtime.adaptation.ts:838-868` also builds its own variant array and now forwards an optional `AccelerationConfig` to the evaluator (line 866).

### Q5: Where does the racing demo create its acceleration config?

- `examples/racing_curriculum/browser-entry/browser-entry.ts:478`:
  ```ts
  const accelerationConfig: AccelerationConfig = resolveAccelerationConfig({});
  ```
- This yields the default `parallelVariantCount: 1`.
- To set `256`, change the line to:
  ```ts
  const accelerationConfig: AccelerationConfig = resolveAccelerationConfig({ parallelVariantCount: 256 });
  ```

### Q6: Where is the acceleration status chip, and could it show the parallel count?

- Globals declared at `browser-entry.ts:315-317`:
  ```ts
  let accelerationStatusChipElement: HTMLDivElement | null = null;
  let accelerationStatusValue: HTMLElement | null = null;
  ```
- Created in `setupCanvasStage` at `browser-entry.ts:1459-1471` with initial label `CPU`.
- Updated every frame in `updateTelemetryPanelNodes` at `browser-entry.ts:2062-2073`:
  ```ts
  const policy = new LifecycleAccelerationPolicy();
  const accelerationMode = policy.evaluate(accelerationConfig, controllerNetwork.nodes.length).mode;
  nodes.accelerationValue.textContent = `Acceleration: ${accelerationMode}`;
  if (accelerationStatusValue) {
    const { label, className } = resolveAccelerationChipPresentation(accelerationMode);
    accelerationStatusValue.textContent = label;
    accelerationStatusValue.className = className;
  }
  ```
- To show the parallel count, change `resolveAccelerationChipPresentation` at `browser-entry.ts:2113-2126` to accept the resolved `parallelVariantCount` and return labels such as `CPU (256x)`. Then pass `accelerationConfig.parallelVariantCount` from the caller.

### Q7: Full chain from racing demo → NGE juvenile → `evaluateWeightVariantsAsync`

**Live racing adaptation chain (currently active):**
1. `examples/racing_curriculum/browser-entry/browser-entry.ts:start()`
2. `RuntimeAdaptationEngine.adaptOnTick()` (`runtime.adaptation.ts:706-725`)
3. `adapt()` internal wrapper
4. `runNgeGrowStabilizeCycle()` (`src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:247-407`)
5. Stabilization phase calls `applyPlasticity()` (lines 300-310), not the variant evaluator.

**Weight-variant chain (currently inactive in the live demo):**
1. Potential caller: `evaluateRacingWeightVariantsAsync` (`runtime.adaptation.ts:838-868`) forwards to `evaluateWeightVariantsAsync`.
2. NGE wrapper: `evaluateNgeWeightVariants` (`src/neat/nge-juvenile/neat.nge-juvenile.variants.ts:58-76`) → `evaluateWeightVariantsAsync` (`src/acceleration/acceleration.variants.ts:118-127`).

The racing demo's `accelerationConfig` is currently used only for the UI chip; it is **not** passed into the live adaptation engine.

## Risks

| Risk | Owner | Mitigation |
|------|-------|------------|
| `evaluateWeightVariantsAsync` is currently sequential; simply exposing a count does not deliver real parallelism. | Implementation step (`04-implementing`) | If "parallel" is the goal, the loop must be replaced with a batched/concurrent strategy (worker/WebGPU/CPU Promise.all) and the backend must no longer be hardcoded to `'cpu'`. |
| The generic evaluator's `config` parameter is typed `unknown` and unused, so wiring it to `AccelerationConfig` is a small API change. | Implementation step | Update the signature and ensure all callers (NGE, racing demo) pass the resolved config. |
| Naming collision: the racing demo already has a caller-supplied `variantCount` parameter, and `BufferPoolWorkload.variantCount` already exists for GPU sizing. | Implementation / planning | Use a distinct name such as `parallelVariantCount` or `evaluationVariantCount` in `AccelerationConfig`, or scope it under a `variantEvaluation` namespace. |
| The demo UI updates acceleration mode every frame from a fresh `LifecycleAccelerationPolicy` instance; a new chip must hold a persistent reference to avoid DOM churn. | UI implementation step | Store the chip element reference in `TelemetryPanelNodes` or a sibling structure and update its `textContent`/`className` in the existing frame loop. |
