# Racing Demo Shows "CPU" Acceleration — Root Cause Research

## Question

Why does the live racing-curriculum demo at `http://localhost:8080/docs/examples/racing_curriculum/index.html` display `Acceleration: cpu` on the stage-card chip, even though the browser may support WebGPU and the acceleration layer has GPU-first auto-detection?

Specifically:

1. Is the CPU label coming from the actual variant-evaluation backend, or from a separate display-only code path?
2. Does the synchronous chip detection use the same rules as the asynchronous evaluator?
3. Is the small racing network (≈33–104 nodes) below the GPU threshold, and if so, why might the evaluator still choose GPU?
4. What is the smallest safe fix so the chip reflects the backend that is actually used?

## Evidence

### 1. Two independent backend-resolution paths exist

| Path | Entry point | Async? | Considers `nodeCount` | Considers `batchParallelCount` | Actually requests GPU device? | Used by |
| ---- | ----------- | ------ | --------------------- | ------------------------------ | ----------------------------- | ------- |
| **Sync display path** | `LifecycleAccelerationPolicy.evaluate()` | No | Yes | **No** | No | Demo chip (`browser-entry.ts`) |
| **Async evaluation path** | `autoEnableAcceleration()` | Yes | Yes | **Yes** | Yes | `evaluateWeightVariantsAsync()` |

Source files:

- `src/acceleration/acceleration.policy.ts:156-169` — `LifecycleAccelerationPolicy.evaluate(config, nodeCount)` calls `detectAcceleration(config, nodeCount)`.
- `src/acceleration/acceleration.detect.ts:191-219` — `detectAcceleration()` is synchronous; it checks `navigator.gpu` and `nodeCount >= gpuNodeThreshold`, but it does **not** accept or inspect `batchParallelCount`.
- `src/acceleration/acceleration.variants.ts:147-152` — `evaluateWeightVariantsAsync()` calls `await autoEnableAcceleration({ nodeCount, batchParallelCount: variants.length, config })`.
- `src/acceleration/acceleration.orchestrator.ts:58-115` — `autoEnableAcceleration()` probes GPU first, then workers, then CPU.
- `src/acceleration/acceleration.gpu.ts:90-107` — `shouldAutoEnableGpu()` returns true when `nodeCount >= gpuNodeThreshold` **OR** `batchParallelCount >= gpuBatchParallelThreshold`.

### 2. The racing network is small but evaluates enough variants to meet the batch threshold

- `examples/racing_curriculum/browser-entry/browser-entry.ts:1772-1785` builds a deterministic MLP controller with at most ~104 nodes (`TOTAL_TIER4_INPUT_SIZE` + hidden + output).
- `src/acceleration/acceleration.constants.ts:15-21` sets:
  - `DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD = 1_024`
  - `DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD = 8`
- `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:329,336,344` sets lifecycle variant counts:
  - baby = 16
  - juvenile = 8
  - adult/equilibrium = 2

Therefore, for the racing demo's tiny network:

- Sync display path: `nodeCount < 1_024` → chip resolves to `cpu` (ignores the 16-variant batch).
- Async evaluator path: `batchParallelCount = variants.length >= 8` (baby stage = 16) → **GPU or worker becomes eligible**, provided the browser supports it and the device request succeeds.

### 3. The chip is recomputed every frame from the sync path

- `examples/racing_curriculum/browser-entry/browser-entry.ts:1459-1471` creates the chip with initial label `CPU` and CSS class `racing-status-chip--cpu`.
- `examples/racing_curriculum/browser-entry/browser-entry.ts:2062-2073` updates the chip each animation frame:

```ts
const policy = new LifecycleAccelerationPolicy();
const accelerationMode = policy.evaluate(accelerationConfig, controllerNetwork.nodes.length).mode;
nodes.accelerationValue.textContent = `Acceleration: ${accelerationMode}`;
```

This never awaits the real backend resolution and never consumes the backend metadata returned by `evaluateWeightVariantsAsync()`.

### 4. The actual evaluator currently hardcodes the reported backend to `'cpu'`

- `src/acceleration/acceleration.variants.ts:130` contains `const backend = 'cpu';`.
- This is already captured as **BLOCKER-001** in `plans/Acceleration_Parallelism_Props.plans.md`.
- Even after `autoEnableAcceleration()` resolves to GPU, the metadata returned to callers would still claim `cpu` until this constant is removed.

### 5. Live demo configuration does not pass `AccelerationConfig` into the adaptation engine

- `examples/racing_curriculum/browser-entry/browser-entry.ts:478` resolves `accelerationConfig` for display only.
- `examples/racing_curriculum/controller/runtime.adaptation.ts:142-164` defines `RuntimeAdaptationEngineOptions` without an `accelerationConfig` field.
- The live adaptation chain `adaptOnTick()` → `adapt()` → `runNgeGrowStabilizeCycle()` does **not** forward the config into `evaluateNgeWeightVariants()` (which currently passes `undefined` as the config argument).

## Decision

The root cause of the `Acceleration: cpu` label is a **mismatch between the synchronous chip-detection path and the asynchronous evaluator path**:

1. The chip uses `LifecycleAccelerationPolicy.evaluate()` → `detectAcceleration()`, which only looks at `nodeCount` and ignores the number of variants being evaluated.
2. The actual evaluator uses `autoEnableAcceleration()`, which considers both `nodeCount` and `batchParallelCount`.
3. Because the racing network is far below the 1,024-node GPU threshold but evaluates 8–16 variants at once, the chip reports `cpu` while the evaluator may actually select GPU or worker.
4. Additionally, the evaluator currently reports `cpu` in metadata regardless of the real backend (`acceleration.variants.ts:130` hardcoded constant), and the demo does not pass `AccelerationConfig` into the live adaptation engine.

**Recommended fix strategy (smallest safe change):**

1. Remove the hardcoded `const backend = 'cpu'` in `src/acceleration/acceleration.variants.ts:130` and use the backend returned by `autoEnableAcceleration()`.
2. Do **not** make the chip update itself async by calling `autoEnableAcceleration()` every frame — that would repeatedly request a WebGPU device and block the render loop.
3. Instead, drive the chip from the **actual evaluator result**:
   - Cache the last resolved backend in the adaptation engine or in an `AccelerationObserver`.
   - Update the chip text/class from that cached value in the existing per-frame `updateTelemetryPanelNodes` call.
4. Optionally extend `detectAcceleration()` / `LifecycleAccelerationPolicy.evaluate()` with an optional `batchParallelCount` parameter so that sync callers (including the chip and `Network.getAccelerationStatus()`) can show a consistent eligibility estimate. This is a secondary improvement; the authoritative source of truth for the live chip should still be the actual evaluator backend.
5. Thread `AccelerationConfig` through `RuntimeAdaptationEngineOptions` and into `evaluateNgeWeightVariants()` / `evaluateWeightVariantsAsync()` so the live demo honors `parallelVariantCount` and other overrides.

## Risks

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Making the chip update async would request a GPU device every frame and block the animation loop. | High UI jank / repeated device creation. | Keep chip update synchronous; feed it from cached evaluator backend or observer events. |
| Fixing only the hardcoded backend without fixing the chip source will still show `cpu` for small networks. | User-visible label remains wrong. | Update both the evaluator metadata and the chip source of truth. |
| Changing `detectAcceleration()` signature may break existing callers that pass `nodeCount` positionally. | Compile/runtime breakage. | Add `batchParallelCount` as an optional third parameter with a default of `1` or `0`. |
| The worker backend has the same threshold inconsistency as GPU. | Chip may show `cpu` when workers are actually used. | Apply the same optional `batchParallelCount` logic to `detectWorker()` if extending the sync path. |
| The live demo does not currently pass `AccelerationConfig` into the engine, so `parallelVariantCount` is ignored. | Evaluator stays sequential/config-ignorant. | Add `accelerationConfig` to `RuntimeAdaptationEngineOptions` and forward it through the adaptation chain. |

## Related artifacts

- `docs/research/racing-curriculum-acceleration-ui-and-parallelism.md` — prior research on the UI chip location, `parallelVariantCount`, and NGE lifecycle variant counts.
- `plans/Acceleration_Parallelism_Props.plans.md` — active plan tracking `BLOCKER-001` and the parallelism props workstream.
