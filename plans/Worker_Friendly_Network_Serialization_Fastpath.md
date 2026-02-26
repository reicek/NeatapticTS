# Worker-Friendly Network Serialization Fastpath Plan

## Purpose

Make evaluating networks in worker threads (Node `worker_threads` and browser Web Workers) fast and ergonomic by providing a **serialization fastpath** that is:

- deterministic
- cheap to clone
- optionally **Transferable** (TypedArrays) for zero-copy
- compatible with standalone inference and activation scheduling

## Goals

- G1: Define a stable, versioned “inference payload” format suitable for `postMessage`/`structuredClone`.
- G2: Support a fast path that avoids reconstructing full `Network` objects when only inference is needed.
- G3: Provide clear compatibility guarantees and migration strategy for payload versioning.

## Non-goals

- Serializing every training/evolution feature into workers in v1.
- Perfect minimal size; focus on speed + stability.

## Payload formats

### Format A — Clone-friendly JSON payload (baseline)

- Plain objects + arrays of numbers.
- Works everywhere.

### Format B — Transferable typed payload (fast path)

- Use TypedArrays (`Float32Array`, `Int32Array`) for weights, biases, indices.
- Designed to be passed via `postMessage(payload, transferList)`.

Start with Format A for correctness, then add Format B.

## Proposed public API

```ts
export type WorkerPayloadMode = 'json' | 'typed';

export interface ExportWorkerPayloadOptions {
  mode: WorkerPayloadMode;
  numericPrecision?: 'full' | 'f32';
}

export interface WorkerPayloadV1Json {
  version: 1;
  mode: 'json';
  inputNodeIds: number[];
  outputNodeIds: number[];
  activationSteps: number[][];
  nodes: Array<{ id: number; bias: number; activation: string }>;
  edges: Array<{ from: number; to: number; weight: number }>;
}

export interface WorkerPayloadV1Typed {
  version: 1;
  mode: 'typed';
  inputNodeIds: Int32Array;
  outputNodeIds: Int32Array;
  activationStepsIndex: Int32Array;
  activationStepsData: Int32Array;
  nodeIds: Int32Array;
  nodeBiases: Float32Array;
  nodeActivationIds: Int32Array;
  edgeFrom: Int32Array;
  edgeTo: Int32Array;
  edgeWeights: Float32Array;
}

export function exportWorkerPayload(
  network: Network,
  options: ExportWorkerPayloadOptions,
): WorkerPayloadV1Json | WorkerPayloadV1Typed;

export function createWorkerPredictor(
  payload: WorkerPayloadV1Json | WorkerPayloadV1Typed,
): { predict(input: number[]): number[]; reset?(): void };
```

Notes:

- Typed payload uses “indexed steps” to represent `number[][]` without nested arrays.
- `nodeActivationIds` maps to a stable activation table.

## Compatibility guarantees

- `version` is required.
- For `version: 1`, provide strict decoding and helpful errors.
- If future versions expand fields, keep decoders for older versions.

## Implementation steps

### Step 1 — Shared inference IR extraction

- Reuse the same IR extraction used by standalone export.
- Ensure deterministic ordering and schedule embedding.

Acceptance:

- Same network → identical IR.

### Step 2 — JSON payload export + predictor

- Implement `exportWorkerPayload(..., { mode: "json" })`.
- Implement `createWorkerPredictor` for JSON payload.

Acceptance:

- Predictor matches `network.activate()` within tolerance.

### Step 3 — Typed payload export + predictor

- Add typed encoding.
- Provide a `getTransferList(payload)` helper (or document how to build it) to transfer buffers.

Acceptance:

- Payload can be transferred and used in worker.

### Step 4 — Docs and examples

- Provide examples for:
  - browser worker
  - Node worker_threads

Acceptance:

- Copy-paste examples work.

## Testing strategy

- Payload roundtrip tests:
  - `export` → `predictor` matches baseline inference
- Determinism tests:
  - payload is stable for fixed networks
- Transferability tests (where environment supports it):
  - ensure buffers are transferred and not copied

## Risks and mitigations

- Risk: activation function mismatch between main thread and worker.
  - Mitigation: ship a stable activation table keyed by numeric IDs.
- Risk: payload size growth for large graphs.
  - Mitigation: typed mode + optional f32; later add compression if needed.

## Success criteria

- Worker evaluation can run without constructing full `Network` objects.
- Payload format is stable and versioned.
- Typed payload path provides measurable speed improvement in benchmarks.
