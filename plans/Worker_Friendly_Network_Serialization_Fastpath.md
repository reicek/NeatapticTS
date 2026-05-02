# Worker Inference Transport — Four Progressive Strategies

**Status:** [PLANNED]

## Purpose

Make evaluating networks in worker threads (Node.js `worker_threads` and browser Web Workers) fast, ergonomic, and progressively optimizable by providing four independent inference transport strategies. Each strategy is a first-class public API; users choose based on their host environment and latency requirements.

The strategies share a common intermediate representation (IR) extracted from the `Network` object, enabling consistent, deterministic results across all four approaches.

Astro Bird (the Flappy Bird flagship demo) serves as the primary end-to-end test surface: each strategy fully replaces the previous one before the next phase begins.

## Strategy overview

| API type                       | Transport mechanism                       | Host requirement                               | Key property                               |
| ------------------------------ | ----------------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| `PortableInferencePayload`     | Plain JSON objects + arrays               | None — works with `structuredClone`            | Universal compatibility                    |
| `TransferableInferencePayload` | TypedArrays (`Int32Array`/`Float64Array`) | Standard `postMessage` Transferable            | Zero-copy buffer handoff                   |
| `InferenceChannel`             | Persistent `MessageChannel` port pair     | Standard Web Worker API                        | Warm predictor, concurrent async inference |
| `SharedInferenceWorker`        | `SharedArrayBuffer` + `Atomics`           | Cross-origin isolation (`COOP`/`COEP` headers) | Lock-free, lowest latency                  |

## Goals

- G1: Define a stable, versioned inference IR shared by all four strategies.
- G2: Provide four independently usable transport strategies with a coherent public API.
- G3: Fully integrate each strategy into Astro Bird before advancing, confirmed by user.
- G4: Reach 100% statement/branch/function/line coverage on all new `src/` boundaries.
- G5: Provide comprehensive educational documentation with Mermaid diagrams per project standards.

## Non-goals

- Backpropagation or training in workers.
- Payload compression (typed mode + f32 precision is sufficient).
- NEAT evolution parallelism (evolution stays generation-by-generation).
- Gradient sharing or federated training.

---

## Module layout

New module: `src/architecture/network/worker-payload/`

| File                                       | Responsibility                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| `network.worker-payload.ts`                | Orchestration facade — exports all public functions                      |
| `network.worker-payload.types.ts`          | All interfaces: IR, payloads, predictor, channel, shared worker, options |
| `network.worker-payload.utils.ts`          | IR extraction, activation ID table, typed encoding/decoding helpers      |
| `network.worker-payload.channel.ts`        | `InferenceChannel` implementation (persistent MessageChannel)            |
| `network.worker-payload.shared.ts`         | `SharedInferenceWorker` implementation (SharedArrayBuffer + Atomics)     |
| `network.worker-payload.channel.worker.ts` | Worker-side receiver for `InferenceChannel`                              |
| `network.worker-payload.shared.worker.ts`  | Worker-side receiver for `SharedInferenceWorker`                         |
| `network.worker-payload.test.ts`           | Owner-local tests (all four strategies)                                  |
| `README.md`                                | Auto-generated from JSDoc via `npm run docs`                             |

## Worker script delivery

Both `InferenceChannel` and `SharedInferenceWorker` default to **inline blob** — built at library build time from the worker receiver files, matching the pattern in `src/multithreading/workers/browser/testworker.ts`. The `workerUrl` option on both option interfaces overrides the blob for strict CSP environments or custom bundler setups.

---

## Key existing boundaries reused

- `src/multithreading/multi.utils.ts` — `ACTIVATION_FUNCTIONS` (16-entry stable registry). Re-exported as `INFERENCE_ACTIVATION_TABLE`.
- `src/architecture/network/standalone/network.standalone.utils.setup.ts` — schedule extraction logic. Reused for `extractNetworkInferenceIR`.
- `src/multithreading/workers/browser/testworker.ts` — existing inline blob pattern. Matched for new worker receivers.

---

## Phase 0 — Shared Inference IR [PLANNED]

> Prerequisite — blocks all other phases.

### Step 0.1 — Define `NetworkInferenceIR` type [IN PROGRESS]

File: `network.worker-payload.types.ts`

```ts
interface NetworkInferenceIR {
  readonly inputCount: number;
  readonly outputCount: number;
  readonly nodes: ReadonlyArray<{
    index: number;
    bias: number;
    activationId: number; // stable index into INFERENCE_ACTIVATION_TABLE
    selfWeight: number;
    selfGaterIndex: number; // -1 if no self-gater
  }>;
  readonly edges: ReadonlyArray<{
    from: number;
    to: number;
    weight: number;
    gaterIndex: number; // -1 if ungated
  }>;
  readonly activationSteps: ReadonlyArray<ReadonlyArray<number>>; // ordered traversal groups
  readonly outputNodeIndices: ReadonlyArray<number>;
}
```

### Step 0.2 — Implement `extractNetworkInferenceIR`

File: `network.worker-payload.utils.ts`

- Reuse schedule extraction from `network.standalone.utils.setup.ts`.
- Map activation function names to IDs via the `ACTIVATION_FUNCTIONS` registry from `multi.utils.ts`.
- Guarantee: same `Network` instance always produces byte-identical IR (determinism invariant).

### Step 0.3 — Export `INFERENCE_ACTIVATION_TABLE`

- Re-export/alias the 16-entry `ACTIVATION_FUNCTIONS` array from `multi.utils.ts`.
- JSDoc: explain each entry index, table stability contract.

### Step 0.4 — Tests

File: `network.worker-payload.test.ts`

- Determinism: `extractNetworkInferenceIR(network)` called twice → `deepEqual` result.
- Activation ID coverage: every `activationId` in IR maps to correct function name via `INFERENCE_ACTIVATION_TABLE`.
- Schedule coverage: every node in the network appears in exactly one traversal step.

### Acceptance criteria

- `extractNetworkInferenceIR` on the same network returns identical objects on repeated calls.
- All Phase 0 tests pass at 100% coverage.

---

## Phase 1 — Portable Inference Payload [PLANNED]

> Depends on: Phase 0 complete.
> Astro Bird migration: full — replaces `network.toJSON()` best-network payload.

### Step 1.1 — Define `PortableInferencePayload`

File: `network.worker-payload.types.ts`

```ts
interface PortableInferencePayload {
  readonly version: 1;
  readonly strategy: 'portable';
  readonly inputCount: number;
  readonly outputCount: number;
  readonly activationSteps: number[][];
  readonly nodes: Array<{ id: number; bias: number; activation: string }>;
  readonly edges: Array<{
    from: number;
    to: number;
    weight: number;
    gater?: number;
  }>;
  readonly activationTable: string[]; // self-describing; guards against version drift
}
```

### Step 1.2 — Implement `exportPortableInferencePayload`

```ts
function exportPortableInferencePayload(
  network: Network,
): PortableInferencePayload;
```

- Calls `extractNetworkInferenceIR(network)`.
- Maps IR → payload, embedding full `activationTable` string array.

### Step 1.3 — Define `InferencePredictor`

```ts
interface InferencePredictor {
  predict(input: ReadonlyArray<number>): number[];
  reset(): void;
  readonly strategy: 'portable' | 'transferable';
}
```

### Step 1.4 — Implement `createInferencePredictor` (portable overload)

```ts
function createInferencePredictor(
  payload: PortableInferencePayload,
): InferencePredictor;
```

- Allocates mutable `Float64Array` activation and state buffers internally.
- `predict()` runs activation steps in IR order; must match `network.activate()` within floating-point tolerance.
- `reset()` zeroes state buffers (recurrent state).

### Step 1.5 — Tests

- Roundtrip: `exportPortableInferencePayload` → `createInferencePredictor` → `predict()` matches `network.activate()`.
- Determinism: identical payload object for same frozen network.
- Recurrent: `reset()` restores state; subsequent `predict()` matches fresh-network baseline.
- Self-description: `payload.activationTable.length` equals IR node activation ID range.

### Step 1.6 — Astro Bird full migration

- Replace `network.toJSON()` in `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.evolution.service.ts` with `exportPortableInferencePayload`.
- Update `examples/flappy_bird/browser-entry/browser-entry.worker-channel.utils.ts` to receive `PortableInferencePayload` and use `createInferencePredictor` for playback activation (no `network.activate()` call on main thread).

### Acceptance criteria

- `predict()` matches `network.activate()` within `Number.EPSILON * 100`.
- All Phase 1 tests pass at 100% coverage.
- **🟡 USER TEST** — Astro Bird evolution runs and playback renders correctly with no regressions before Phase 2 begins.

---

## Phase 2 — Transferable Inference Payload [PLANNED]

> Depends on: Phase 1 user-confirmed.
> Astro Bird migration: full — replaces Portable payload with Transferable throughout.

### Step 2.1 — Define `TransferableInferencePayload`

```ts
interface TransferableInferencePayload {
  readonly version: 1;
  readonly strategy: 'transferable';
  readonly inputCount: number;
  readonly outputCount: number;
  readonly activationStepsIndex: Int32Array; // offsets into activationStepsData
  readonly activationStepsData: Int32Array; // flat node-index sequence
  readonly nodeIds: Int32Array;
  readonly nodeBiases: Float64Array; // Float32Array when numericPrecision: 'f32'
  readonly nodeActivationIds: Int32Array;
  readonly nodeSelfWeights: Float64Array;
  readonly nodeSelfGaterIndices: Int32Array; // -1 = no self-gater
  readonly edgeFrom: Int32Array;
  readonly edgeTo: Int32Array;
  readonly edgeWeights: Float64Array;
  readonly edgeGaterIndices: Int32Array; // -1 = ungated
  readonly outputNodeIndices: Int32Array;
  readonly activationTableLength: number; // implicit correctness check
}
```

### Step 2.2 — Define `TransferableInferencePayloadOptions`

```ts
interface TransferableInferencePayloadOptions {
  numericPrecision?: 'full' | 'f32'; // default: 'full' (Float64Array)
}
```

### Step 2.3 — Implement `exportTransferableInferencePayload`

```ts
function exportTransferableInferencePayload(
  network: Network,
  options?: TransferableInferencePayloadOptions,
): TransferableInferencePayload;
```

- Encodes IR into TypedArrays.
- Flattens `activationSteps` into parallel `activationStepsIndex` + `activationStepsData` arrays.
- Default `numericPrecision: 'full'` = `Float64Array` — no silent precision regression.

### Step 2.4 — Implement `getTransferList`

```ts
function getTransferList(payload: TransferableInferencePayload): ArrayBuffer[];
```

- Returns all `.buffer` refs from every TypedArray field.
- JSDoc warns: buffers are neutered after `postMessage(..., getTransferList(payload))`; do not read payload after transfer.

### Step 2.5 — Implement `createInferencePredictor` (transferable overload)

- Reads TypedArrays directly — no JSON parsing.
- Allocates internal mutable `activationValues` and `stateValues` `Float64Array` buffers.
- `reset()` zeroes state slice.

### Step 2.6 — Tests

- Transferable `predict()` matches Portable `predict()` (within `Number.EPSILON * 100`).
- `getTransferList` returns all TypedArray buffers (correct count, all are `ArrayBuffer`).
- `f32` precision: output within `1e-4` of `f64` baseline.
- Buffer lengths: `nodeIds.length === IR.nodes.length`, `edgeFrom.length === IR.edges.length`, etc.

### Step 2.7 — Astro Bird full migration + benchmark

- Replace Portable payload with Transferable throughout evolution and playback.
- Use `getTransferList` in every relevant `postMessage` call.
- Add `bench-browser/` roundtrip timing comparison: Transferable vs old `toJSON` baseline.

### Acceptance criteria

- Typed `predict()` matches Portable `predict()`.
- All Phase 2 tests pass at 100% coverage.
- **🟡 USER TEST** — Astro Bird: postMessage performance measurably improved, evolution correct, playback correct before Phase 3 begins.

---

## Phase 3 — Dedicated Inference Channel [PLANNED]

> Depends on: Phase 2 user-confirmed.
> Astro Bird migration: full — every live bird agent holds a persistent channel; `network.activate()` removed from evaluation hot paths.

### Step 3.1 — Define `InferenceChannel`

```ts
interface InferenceChannel {
  predict(input: Float32Array | ReadonlyArray<number>): Promise<Float32Array>;
  reset(): Promise<void>;
  close(): void;
  readonly strategy: 'channel';
  readonly isOpen: boolean;
}
```

### Step 3.2 — Define `InferenceChannelOptions`

```ts
interface InferenceChannelOptions {
  maxConcurrentRequests?: number; // default: 8; excess requests queued
  workerUrl?: string; // override inline blob for CSP environments
}
```

### Step 3.3 — Implement `openInferenceChannel`

```ts
function openInferenceChannel(
  payload: TransferableInferencePayload,
  options?: InferenceChannelOptions,
): InferenceChannel;
```

- Spawns a persistent `Worker` from **inline blob** (or `workerUrl` override).
- Bootstrap message: sends Transferable payload with `getTransferList`.
- Uses `MessageChannel`: main thread holds `port1`, worker holds `port2`.
- Hot path: each `predict()` sends `{ id: number; input: Float32Array }` through `port1`.
- Worker replies `{ id: number; output: Float32Array }` through `port2`.
- In-flight promises resolved by `id` map. Requests exceeding `maxConcurrentRequests` are queued.

### Step 3.4 — Worker-side channel receiver

File: `network.worker-payload.channel.worker.ts`

- On bootstrap: `createInferencePredictor(payload)` → store predictor; store `port2`.
- On inference message: `predictor.predict(input)` → `port2.postMessage({ id, output })`.
- On reset message: `predictor.reset()` → `port2.postMessage({ id, done: true })`.
- On close: `self.close()`.

### Step 3.5 — Tests

- Bootstrap + `predict()` matches `network.activate()`.
- 8 concurrent `predict()` calls all resolve with correct outputs.
- `reset()` clears state; subsequent `predict()` matches fresh-network baseline.
- `close()` → `isOpen === false`; subsequent `predict()` rejects with descriptive error.

### Step 3.6 — Astro Bird full migration

- Each live bird agent during evaluation holds a persistent `InferenceChannel` — no more synchronous `network.activate()` in the evaluation hot paths.
- Playback uses `InferenceChannel` per visible bird.
- All direct `network.activate()` calls removed from evaluation loops.

### Acceptance criteria

- All Phase 3 tests pass at 100% coverage.
- **🟡 USER TEST** — Astro Bird with persistent channels: agents activate correctly, no stalls or frame drops before Phase 4 begins.

---

## Phase 4 — Shared Memory Inference Worker + Full Parallel Evaluation [PLANNED]

> Depends on: Phase 3 user-confirmed.
> Astro Bird migration: full — sequential evaluation funnel replaced by parallel `FlappyEvaluationWorkerPool`; dynamic population scaling added.

### Step 4.1 — Shared memory layout

Per-worker slot layout:

```
Int32Array control buffer (2 elements):
  [0]: STATUS_FLAG  — 0=IDLE, 1=INPUT_READY, 2=OUTPUT_READY, 3=RESET_REQUESTED
  [1]: INPUT_COUNT  — constant, set on bootstrap

Float64Array data buffer:
  [0 .. inputCount-1]                       — input values (main thread writes)
  [inputCount .. inputCount+outputCount-1]  — output values (worker writes)
```

### Step 4.2 — Define `SharedInferenceWorker`

```ts
interface SharedInferenceWorker {
  infer(input: ReadonlyArray<number>): Promise<Float32Array>;
  submitInput(input: ReadonlyArray<number>): void;
  awaitOutput(): Promise<Float32Array>;
  reset(): Promise<void>;
  release(): void;
  readonly strategy: 'shared-memory';
  readonly isReady: boolean;
}
```

### Step 4.3 — Define `SharedInferenceWorkerOptions`

```ts
interface SharedInferenceWorkerOptions {
  workerUrl?: string; // override inline blob for CSP environments
}
```

### Step 4.4 — Implement `openSharedInferenceWorker`

```ts
function openSharedInferenceWorker(
  payload: TransferableInferencePayload,
  options?: SharedInferenceWorkerOptions,
): SharedInferenceWorker;
```

File: `network.worker-payload.shared.ts`

- Allocates `SharedArrayBuffer` for control (Int32) and data (Float64) regions.
- Spawns persistent worker from **inline blob** (or `workerUrl` override).
- Bootstrap message: Transferable payload + shared buffer references (not transferred — shared).
- `submitInput()`: writes inputs into shared float buffer → `Atomics.store(controlView, 0, INPUT_READY)` → `Atomics.notify(controlView, 0)`.
- `awaitOutput()`: `Atomics.waitAsync(controlView, 0, OUTPUT_READY)` (browser) / `Atomics.wait` (Node) → read output slice → `Atomics.store(controlView, 0, IDLE)`.

### Step 4.5 — Worker-side shared memory receiver

File: `network.worker-payload.shared.worker.ts`

- On bootstrap: `createInferencePredictor(payload)` + store both shared buffers.
- Activation loop: `Atomics.wait(controlView, 0, IDLE)` until `INPUT_READY` → read inputs from shared float buffer → `predictor.predict(inputs)` → write outputs to shared float buffer → `Atomics.store(controlView, 0, OUTPUT_READY)` → `Atomics.notify(controlView, 0)`.
- On `RESET_REQUESTED`: `predictor.reset()` → restore `IDLE`.

### Step 4.6 — Export `SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION`

```ts
const SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION: true = true;
```

- JSDoc explains `COOP`/`COEP` header requirements.
- Library does NOT configure or enforce these headers.
- Note: Node.js environments do not require cross-origin isolation; `SharedArrayBuffer` is available unconditionally.

### Step 4.7 — Tests (all executable in Node — no headers needed)

- `infer()` matches `network.activate()`.
- N=50 sequential `infer()` calls all correct (stress / recurrent safety).
- `reset()` clears state; subsequent `infer()` matches fresh-network baseline.
- `release()` → `isReady === false`; subsequent `infer()` rejects with descriptive error.

### Step 4.8 — Astro Bird full parallel refactor

**New file: `examples/flappy_bird/evaluation/evaluation.worker-pool.ts`**

```ts
class FlappyEvaluationWorkerPool {
  constructor(workerCount?: number); // default: navigator.hardwareConcurrency ?? 4
  initialize(genomes: FlappyTrainerNetwork[]): Promise<void>;
  evaluateGenomesAcrossSeeds(
    genomes: readonly FlappyTrainerNetwork[],
    sharedSeeds: readonly number[],
    rolloutOptions: FlappyRolloutOptions,
  ): Promise<Map<FlappyTrainerNetwork, FlappySeedBatchEvaluation>>;
  dispose(): void;
}
```

- `initialize`: exports each genome as `TransferableInferencePayload`, opens one `SharedInferenceWorker` per slot (round-robin reuse for population > workerCount).
- `evaluateGenomesAcrossSeeds`: dispatches all genome×seed work items in parallel, replacing the sequential `for (const genome of genomes)` loops in `trainer.evaluation.service.services.ts`.

**Refactor `examples/flappy_bird/trainer/evaluation/trainer.evaluation.service.services.ts`:**

- Add optional `workerPool?: FlappyEvaluationWorkerPool` to the evaluation service dependencies.
- Route through pool when provided; fall back to sequential path when absent (zero regression).

**Refactor `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.evolution.service.ts`:**

- Instantiate `FlappyEvaluationWorkerPool` on worker init; pass to evaluation service.
- Dispose pool on worker close.

**Dynamic population sizing — `examples/flappy_bird/constants/constants.runtime.ts`:**

```ts
/** Hard ceiling on browser population size regardless of core count. */
const FLAPPY_MAX_POPULATION_SIZE = 200;

/**
 * Dynamic population size derived from available hardware concurrency.
 * Scales with CPU cores so heavy architectures (LSTM, GRU, NARX) can
 * run more agents without per-frame main-thread stalls.
 * Example: 4 cores → 40 birds, 8 cores → 80 birds, 16 cores → 160 birds.
 */
const FLAPPY_DYNAMIC_POPULATION_SIZE = Math.min(
  Math.max(globalThis.navigator?.hardwareConcurrency ?? 4, 4) * 10,
  FLAPPY_MAX_POPULATION_SIZE,
);
```

**Dev server COOP/COEP headers — `webpack.config.js`:**

- Add `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` to the Astro Bird webpack-dev-server `headers` config.

### Acceptance criteria

- All Phase 4 tests pass at 100% coverage.
- Sequential fallback path covered by tests.
- **🟡 USER TEST** — Astro Bird with full parallel evaluation pool: multiple agents evaluated in parallel, generation speed visibly improved on multi-core hardware, dynamic population size adapts to `navigator.hardwareConcurrency`, heavy architectures (LSTM, GRU, NARX) run more agents without frame drops.

---

## Public API additions

Exported from `src/neataptic.ts`:

**Functions:**

- `exportPortableInferencePayload`
- `exportTransferableInferencePayload`
- `getTransferList`
- `createInferencePredictor`
- `openInferenceChannel`
- `openSharedInferenceWorker`
- `SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION`
- `INFERENCE_ACTIVATION_TABLE`

**Types:**

- `NetworkInferenceIR`
- `PortableInferencePayload`
- `TransferableInferencePayload`
- `TransferableInferencePayloadOptions`
- `InferencePredictor`
- `InferenceChannel`
- `InferenceChannelOptions`
- `SharedInferenceWorker`
- `SharedInferenceWorkerOptions`

---

## Documentation requirements

Every exported symbol must have:

- JSDoc `@param`/`@returns`.
- Short `@example` fenced ` ```ts ` block.
- Explanation of what the strategy is, when to choose it, and host requirements.

Module-level JSDoc / generated README must include Mermaid diagrams:

1. **Decision flowchart** — "Which strategy should I choose?" (portable → transferable → channel → shared-memory).
2. **Architecture overview** — all four strategies as selection paths from `Network`.
3. **Data flow per strategy** — payload → predictor → inference for each of the four.
4. **Shared memory layout diagram** — `SharedArrayBuffer` control and data buffer slots.

---

## Verification checklist (per phase)

1. `npx tsc --noEmit -p tsconfig.json` passes.
2. `npm run test:silent` green — no regressions in any other module.
3. Coverage guard: `src/architecture/network/worker-payload/**` at 100% statements/branches/functions/lines.
4. Phase 2: `bench-browser/` roundtrip timing: Transferable vs `toJSON` baseline recorded.
5. Phase 3: 8 concurrent `predict()` calls resolve correctly in Node test.
6. Phase 4: N=50 `SharedInferenceWorker` stress test in Node passes; dynamic population constant verified against mocked `hardwareConcurrency`.

---

## Relevant files

| File                                                                                        | Change                                                                         |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `src/architecture/network/worker-payload/`                                                  | New module — all 8 files                                                       |
| `src/multithreading/multi.utils.ts`                                                         | Source of `ACTIVATION_FUNCTIONS` — re-exported as `INFERENCE_ACTIVATION_TABLE` |
| `src/architecture/network/standalone/network.standalone.utils.setup.ts`                     | IR extraction logic — reused                                                   |
| `src/neataptic.ts`                                                                          | Public API export additions                                                    |
| `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.evolution.service.ts` | Phase 1: replace `toJSON` with `exportPortableInferencePayload`                |
| `examples/flappy_bird/browser-entry/browser-entry.worker-channel.utils.ts`                  | Phase 1+: receive payloads on main thread                                      |
| `examples/flappy_bird/trainer/evaluation/trainer.evaluation.service.services.ts`            | Phase 4: optional pool routing                                                 |
| `examples/flappy_bird/evaluation/evaluation.worker-pool.ts`                                 | Phase 4: new `FlappyEvaluationWorkerPool`                                      |
| `examples/flappy_bird/constants/constants.runtime.ts`                                       | Phase 4: `FLAPPY_DYNAMIC_POPULATION_SIZE`, `FLAPPY_MAX_POPULATION_SIZE`        |
| `webpack.config.js`                                                                         | Phase 4: COOP/COEP headers for Astro Bird dev server                           |

---

## Decisions

- `activationTable` embedded in `PortableInferencePayload` — self-describing, guards against activation ID drift between library versions.
- `TransferableInferencePayload` defaults to `Float64Array` — matches existing internal buffer types; no silent precision regression.
- `InferenceChannel` uses `MessageChannel` exclusively — dedicated port per channel instance, no fallback to direct postMessage.
- `SharedInferenceWorker` does not enforce COOP/COEP — library exports `SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION`; host application owns header configuration.
- New module lives in `src/architecture/network/worker-payload/` — it is a network portability feature, not multithreading infrastructure.
- `ACTIVATION_FUNCTIONS` from `multi.utils.ts` stays the canonical registry; aliased as `INFERENCE_ACTIVATION_TABLE` in new module.
- Both `InferenceChannel` and `SharedInferenceWorker` default to inline blob; `workerUrl` overrides for CSP environments.
- Each Astro Bird phase is a **full migration** to the new strategy — no optional toggles retaining the old approach after phase completion.
- Dynamic population sizing in Phase 4 via `FLAPPY_DYNAMIC_POPULATION_SIZE` — removes fixed small-population ceiling for users with capable hardware; important for heavy architectures (LSTM, GRU, NARX).

---

## Handoff query

> Continue from: **Phase 0, Step 0.1 — Define `NetworkInferenceIR` type** in `src/architecture/network/worker-payload/network.worker-payload.types.ts`.
>
> Prior context:
>
> - New module path: `src/architecture/network/worker-payload/`
> - Activation ID source: `ACTIVATION_FUNCTIONS` in `src/multithreading/multi.utils.ts` (16 entries, stable indexed order)
> - Schedule extraction source: `src/architecture/network/standalone/network.standalone.utils.setup.ts`
> - Test convention: one top-level `expect()` per `it()`, AAA structure, nested `describe` blocks
> - Coverage target: 100% statements/branches/functions/lines on all new `src/` files
> - After Phase 0 tests are green, advance to Phase 1 (Portable Inference Payload)
