# architecture/network/worker-payload

One node snapshot inside the worker-friendly inference IR.

The IR stores runtime inference data only: stable node indexes, activation
lookup ids, recurrent self-loop metadata, and scalar modifiers that change
forward-pass output.

Example:

```ts
const irNode: NetworkInferenceIRNode = {
  index: 3,
  bias: 0.15,
  response: 1,
  mask: 1,
  activationId: 4,
  selfWeight: 0,
  selfGaterIndex: -1,
};
```

## architecture/network/worker-payload/network.worker-payload.types.ts

### InferenceChannel

Persistent worker-backed inference channel.

Channels bootstrap one transferable predictor payload exactly once, then keep
the worker-side predictor warm across repeated predict or reset requests sent
through one dedicated `MessageChannel` port pair.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
const channel = openInferenceChannel(payload);
const outputValues = await channel.predict([0.25, 0.75]);
channel.close();
```

### InferenceChannelOptions

Configuration for one dedicated persistent inference channel backed by a warm worker predictor.
These options tune local request queue pressure and worker delivery strategy so repeated inference calls stay stable under bursty client traffic.

Example:

```ts
const channel = openInferenceChannel(payload, {
  maxConcurrentRequests: 4,
});
```

### InferencePredictor

Runtime predictor created from an exported worker payload.

Predictors intentionally keep mutable activation, recurrent-state, and gain
buffers private so callers can reuse the same instance across many
predictions without re-allocating worker state.

Example:

```ts
const predictor = createInferencePredictor(payload);
const outputValues = predictor.predict([0.25, 0.75]);
predictor.reset();
```

### NetworkInferenceIR

Deterministic, worker-consumable snapshot of one network forward pass.

This IR is the common substrate for portable, transferable, channel, and
shared-memory worker transport strategies. It intentionally stores plain data
only so callers can structured-clone it or re-encode it into typed arrays.

Example:

```ts
const inferenceIr = extractNetworkInferenceIR(network);
console.log(inferenceIr.activationSteps);
```

### NetworkInferenceIREdge

One deterministic non-self connection row inside the worker-friendly inference IR.
This shape keeps forward edges compact and index-addressable so predictor hot paths can traverse weighted inputs without object graph lookups or runtime connection instances.

Example:

```ts
const irEdge: NetworkInferenceIREdge = {
  from: 0,
  to: 3,
  weight: 0.75,
  gaterIndex: -1,
};
```

### NetworkInferenceIRNode

One node snapshot inside the worker-friendly inference IR.

The IR stores runtime inference data only: stable node indexes, activation
lookup ids, recurrent self-loop metadata, and scalar modifiers that change
forward-pass output.

Example:

```ts
const irNode: NetworkInferenceIRNode = {
  index: 3,
  bias: 0.15,
  response: 1,
  mask: 1,
  activationId: 4,
  selfWeight: 0,
  selfGaterIndex: -1,
};
```

### PortableInferencePayload

Structured-clone-safe inference payload for the universal worker fallback.

This payload preserves the exact inference metadata needed to replay a
forward pass without shipping live `Network`, `Node`, or `Connection`
instances across the worker boundary.

Example:

```ts
const payload = exportPortableInferencePayload(network);
console.log(payload.activationTable);
```

### PortableInferencePayloadEdge

One structured-clone-safe edge row used by the portable inference payload strategy.
Portable edges preserve the same deterministic topology as IR edges while staying JSON-like and self-describing for debugging, logs, and cross-version message tracing.

Example:

```ts
const portableEdge: PortableInferencePayloadEdge = {
  from: 0,
  to: 3,
  weight: 0.75,
  gaterIndex: -1,
};
```

### PortableInferencePayloadNode

One portable node record used by the structured-clone payload surface.

Portable payloads keep human-readable activation names instead of numeric ids
so they remain self-describing across worker boundaries and versioned message
logs.

Example:

```ts
const portableNode: PortableInferencePayloadNode = {
  id: 3,
  bias: 0.15,
  response: 1,
  mask: 1,
  activation: 'relu',
  selfWeight: 0,
  selfGaterIndex: -1,
};
```

### SharedInferenceWorker

Persistent shared-memory inference worker.

Shared-memory workers keep one predictor alive inside a dedicated worker and
exchange input and output values through `SharedArrayBuffer` shelves instead
of cloning or transferring a fresh payload for every request.

Example:

```ts
const worker = openSharedInferenceWorker(payload);
const outputValues = await worker.infer([0.25, 0.75]);
await worker.release();
```

### SharedInferenceWorkerOptions

Configuration for one dedicated shared-memory inference worker.

Browser hosts default to the module-relative shared worker emitted beside the
library files. Provide `workerUrl` when bundling moves that worker asset or
when CSP policy disallows the default delivery path.

Example:

```ts
const worker = openSharedInferenceWorker(payload, {
  workerUrl: '/assets/shared-inference.worker.js',
});
```

### TransferableInferencePayload

Typed-array inference payload for lower-copy worker transport.

Transferable payloads keep the same semantic content as portable payloads,
but pack it into flat typed arrays so callers can move ownership across
worker boundaries without deep-cloning large object graphs.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
console.log(payload.nodeIds.length);
```

### TransferableInferencePayloadOptions

Configuration knobs for transferable payload export when balancing precision, size, and transport throughput.
Use these options when a worker boundary prefers lower-copy typed shelves but still needs predictable numeric semantics across browser and Node runtimes.

Example:

```ts
const payload = exportTransferableInferencePayload(network, {
  numericPrecision: 'f32',
});
```

### TransferableNumericArray

Numeric typed-array shelf used by transferable payload fields.

### TransferableNumericPrecision

Numeric precision modes supported by transferable payload export.

## architecture/network/worker-payload/network.worker-payload.ts

Worker-friendly inference payload primitives for `Network` forward passes.

This module starts the transport ladder with the one artifact every later
strategy needs: a deterministic, plain-data inference IR. The IR keeps the
runtime execution order, activation identities, recurrent self-loop metadata,
and output ordering, but it deliberately leaves out training-time objects and
host-specific worker mechanics.

Read this boundary in two passes:

1. `extractNetworkInferenceIR(...)` when you want a stable snapshot of a
   live network forward pass.
2. `INFERENCE_ACTIVATION_TABLE` when you need the exact activation-id shelf
   that worker payloads reference.

Example:

```ts
const inferenceIr = extractNetworkInferenceIR(network);
const outputNodeIndexes = inferenceIr.outputNodeIndices;
```

### AutoInferenceTransport

Automatic transport selection result for the current host.

### BatchEvaluationResult

Ordered result envelope returned by `evaluateInWorkers(...)`.

The helper preserves input order even when individual worker tasks finish out
of order, and it reports stable task ids so callers can correlate results
with their original batch shelf without reconstructing indexes manually.

### BrowserWorkerAssetUrlOptions

Options for resolving a browser worker asset URL.

### createInferencePredictor

```ts
createInferencePredictor(
  payload: PortableInferencePayload,
): InferencePredictor
```

Create an inference predictor from a portable or transferable payload so repeated forward evaluations can run without reconstructing a mutable network instance.

### createNeatParallelPopulationEvaluator

```ts
createNeatParallelPopulationEvaluator(
  options: NeatParallelPopulationEvaluatorOptions<TGenome, TPayload, TWorker, TResult>,
): (population: TGenome[]) => Promise<void>
```

Create a NEAT-compatible population fitness delegate on top of `evaluateInWorkers(...)`.

The returned function matches `fitnessPopulation: true` evaluation mode: it
scores the whole population in one async call and writes ordered results back
onto the genomes in place.

Parameters:
- `options` - Boolean-first population-evaluation options plus advanced worker overrides.

Returns: Population fitness delegate compatible with `fitnessPopulation: true`.

Example:

```ts
const evaluatePopulation = createNeatParallelPopulationEvaluator({
  parallel: true,
  evaluateGenome: async (genome) => genome.score ?? 0,
  openWorker: (payload) => openSharedInferenceWorker(payload),
  evaluateWithWorker: async (worker, genome) => worker.infer(genome.activate([0, 1])),
});
```

### detectInferenceWorkerCapabilities

```ts
detectInferenceWorkerCapabilities(
  options: InferenceWorkerCapabilityOptions,
): InferenceWorkerCapabilities
```

Detect the usable worker-backed inference transport tiers for one host.

The probe stays intentionally lightweight. It does not open a worker or fetch
a script; it only combines runtime facts and delivery availability so callers
can decide whether to use shared-memory, channel, or transferable fallback.

Parameters:
- `options` - Runtime and delivery facts for the current host.

Returns: Capability snapshot describing the usable transport tiers.

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
```

### evaluateInWorkers

```ts
evaluateInWorkers(
  options: EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult>,
): Promise<BatchEvaluationResult<TResult>>
```

Evaluates an ordered batch through workers when available, otherwise locally.

This helper keeps the honest async boundary visible: callers provide worker
opening plus task execution when they want parallelism, and they also provide
a local fallback when they need the same semantics in worker-less hosts.

Parameters:
- `options` - Ordered inputs plus worker and fallback execution hooks.

Returns: Ordered batch results with elapsed time and stable task ids.

Example:

```ts
const batchResult = await evaluateInWorkers({
  inputs: payloads,
  openWorker: (payload) => openSharedInferenceWorker(payload),
  evaluateWithWorker: (worker, _payload) => worker.infer([0.25, 0.75]),
});
```

### EvaluateInWorkersOptions

Generic ordered batch-evaluation options for worker and fallback execution.

Keep the transport substrate caller-owned: this helper schedules and assembles
deterministic results, but the caller still decides how a worker is opened,
how a task is scored, and what the single-thread fallback should do.

### exportPortableInferencePayload

```ts
exportPortableInferencePayload(
  network: default,
): PortableInferencePayload
```

Export a portable inference payload with structured-clone-safe data so browser and node runtimes can persist and reload predictor artifacts reliably.

### exportTransferableInferencePayload

```ts
exportTransferableInferencePayload(
  network: default,
  options: TransferableInferencePayloadOptions,
): TransferableInferencePayload
```

Export a transferable inference payload so typed-array-heavy artifacts can cross worker boundaries with explicit ownership transfer and reduced copy overhead.

### extractNetworkInferenceIR

```ts
extractNetworkInferenceIR(
  network: default,
): NetworkInferenceIR
```

Extract a deterministic inference intermediate representation from a live network so transport and payload-export helpers can share one stable graph contract.

### getTransferList

```ts
getTransferList(
  payload: TransferableInferencePayload,
): ArrayBuffer[]
```

Collect every transferable buffer from one typed-array inference payload.

Transfer these buffers exactly once when posting the payload to a worker.
After the transfer completes, the sending-side typed arrays are neutered and
must not be read again.

Parameters:
- `payload` - Transferable inference payload whose buffers should move.

Returns: ArrayBuffer transfer list aligned to the payload's typed shelves.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
worker.postMessage(payload, getTransferList(payload));
```

### INFERENCE_ACTIVATION_TABLE

Stable activation-function table used by worker inference transport.

Each entry index is part of the public inference payload contract. New worker
payloads should reference activations by these numeric ids rather than by
serializing function bodies or relying on runtime function identity.

Example:

```ts
const activationFunction = INFERENCE_ACTIVATION_TABLE[0];
const outputValue = activationFunction(0.5);
```

### InferenceChannel

Persistent worker-backed inference channel.

Channels bootstrap one transferable predictor payload exactly once, then keep
the worker-side predictor warm across repeated predict or reset requests sent
through one dedicated `MessageChannel` port pair.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
const channel = openInferenceChannel(payload);
const outputValues = await channel.predict([0.25, 0.75]);
channel.close();
```

### InferenceChannelOptions

Channel options that configure request batching, queueing, and lifecycle behavior for asynchronous inference transport endpoints across browser and node worker runtimes.
These options define how inference requests are buffered, dispatched, and finalized over long-lived channel sessions.

### InferencePredictor

Runtime predictor created from an exported worker payload.

Predictors intentionally keep mutable activation, recurrent-state, and gain
buffers private so callers can reuse the same instance across many
predictions without re-allocating worker state.

Example:

```ts
const predictor = createInferencePredictor(payload);
const outputValues = predictor.predict([0.25, 0.75]);
predictor.reset();
```

### InferenceWorkerCapabilities

Capability snapshot for the current worker-backed inference host.

This probe answers a narrow question: which worker transport tiers are
honestly usable in the current runtime before one evaluation pool or batch
helper commits to a transport strategy?

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
const transport = resolveAutoInferenceTransport(capabilities);
```

### InferenceWorkerCapabilityOptions

Inputs used to probe worker-backed inference transport support.

Keep delivery facts explicit here. Step 1 only answers capability and auto
selection; Step 2 will own the generic worker-entry and URL resolution
helpers that feed these booleans.

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  crossOriginIsolated: false,
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
```

### NeatParallelPopulationEvaluatorOptions

Low-friction NEAT population-evaluation options built on top of `evaluateInWorkers(...)`.

The factory keeps the public surface focused on population evaluation rather
than transport details. Callers can opt into parallel execution with
`parallel: true`, keep the local scoring delegate as the honest fallback,
and add worker-specific overrides only when they need them.

### NetworkInferenceIR

Deterministic, worker-consumable snapshot of one network forward pass.

This IR is the common substrate for portable, transferable, channel, and
shared-memory worker transport strategies. It intentionally stores plain data
only so callers can structured-clone it or re-encode it into typed arrays.

Example:

```ts
const inferenceIr = extractNetworkInferenceIR(network);
console.log(inferenceIr.activationSteps);
```

### NetworkInferenceIREdge

Directed edge record in the inference graph representation used by payload export, replay, and worker predictor execution pipelines.
The edge contract preserves stable source-target linkage and weight metadata during transport and reconstruction.

### NetworkInferenceIRNode

One node snapshot inside the worker-friendly inference IR.

The IR stores runtime inference data only: stable node indexes, activation
lookup ids, recurrent self-loop metadata, and scalar modifiers that change
forward-pass output.

Example:

```ts
const irNode: NetworkInferenceIRNode = {
  index: 3,
  bias: 0.15,
  response: 1,
  mask: 1,
  activationId: 4,
  selfWeight: 0,
  selfGaterIndex: -1,
};
```

### openInferenceChannel

```ts
openInferenceChannel(
  payload: TransferableInferencePayload,
  options: InferenceChannelOptions,
): InferenceChannel
```

Open one persistent inference worker channel from a transferable payload.

The channel transfers the predictor payload exactly once during bootstrap,
then reuses the worker-side predictor state for every later `predict()` or
`reset()` request sent over one dedicated `MessageChannel` port pair.

Parameters:
- `payload` - Transferable inference payload used to bootstrap the worker predictor.
- `options` - Channel concurrency and worker delivery options.

Returns: Persistent inference channel.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
const channel = openInferenceChannel(payload);
const outputValues = await channel.predict([0.25, 0.75]);
channel.close();
```

### openSharedInferenceWorker

```ts
openSharedInferenceWorker(
  payload: TransferableInferencePayload,
  options: SharedInferenceWorkerOptions,
): SharedInferenceWorker
```

Open one persistent shared-memory inference worker from a transferable payload.

Shared-memory workers keep one predictor alive inside a dedicated worker and
exchange inputs and outputs through `SharedArrayBuffer` shelves rather than
cloning request payloads for every inference call.

Browser hosts default to a module-relative worker script emitted alongside
the library files. Bundled or CSP-constrained hosts can override that entry
with `workerUrl`.

Parameters:
- `payload` - Transferable inference payload used to bootstrap the shared worker predictor.
- `options` - Shared worker delivery options.

Returns: Persistent shared-memory inference worker.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
const sharedWorker = openSharedInferenceWorker(payload);
const outputValues = await sharedWorker.infer([0.25, 0.75]);
await sharedWorker.release();
```

### ParallelInferencePool

Generic bounded worker pool for ordered parallel inference tasks.

The pool keeps persistent predictors warm, caps concurrent worker usage, and
rebuilds results in the caller's original order even when individual jobs
finish out of order.

Example:

```ts
const pool = new ParallelInferencePool({
  openWorker: (payload) => openSharedInferenceWorker(payload),
  workerCount: 4,
});
await pool.initialize(payloads);
const results = await pool.evaluateOrderedBatch(payloads, async (worker) => {
  return worker.infer([0.25, 0.75]);
});
await pool.dispose();
```

#### dispose

```ts
dispose(): Promise<void>
```

Releases every active worker and clears the slot shelf.

Returns: Nothing.

#### evaluateOrderedBatch

```ts
evaluateOrderedBatch(
  payloads: readonly TPayload[],
  evaluator: (worker: TWorker, payload: TPayload, payloadIndex: number) => Promise<TResult>,
): Promise<TResult[]>
```

Evaluates a payload shelf through the bounded worker slots.

The pool schedules work FIFO, lets each slot consume tasks until the queue
is empty, and returns results in the same order as the input payloads.

Parameters:
- `payloads` - Ordered payload shelf to evaluate.
- `evaluator` - Caller-owned evaluation logic for one warm worker slot.

Returns: Results in the caller's original payload order.

#### initialize

```ts
initialize(
  payloads: readonly TPayload[],
): Promise<void>
```

Prepares empty slot state for the next payload shelf.

Existing workers are released before the new shelf becomes active so slot
reuse stays deterministic across generations or evaluation batches.

Parameters:
- `payloads` - Ordered payload shelf that may be evaluated next.

Returns: Nothing.

### ParallelInferencePoolOptions

Configuration for one bounded parallel inference pool.

The opener stays caller-owned so the pool can schedule shared-memory
workers, inference channels, or future worker-backed predictors without
coupling this boundary to one transport strategy.

### ParallelInferenceWorkerLike

Minimal worker resource contract required by the generic inference pool.

The pool owns worker lifecycle, not transport internals. Any persistent
predictor or worker can participate as long as it can release its resources
deterministically.

### PortableInferencePayload

Structured-clone-safe inference payload for the universal worker fallback.

This payload preserves the exact inference metadata needed to replay a
forward pass without shipping live `Network`, `Node`, or `Connection`
instances across the worker boundary.

Example:

```ts
const payload = exportPortableInferencePayload(network);
console.log(payload.activationTable);
```

### PortableInferencePayloadEdge

Portable payload edge schema preserving source-target linkage and connection metadata in runtime-agnostic JSON-style artifacts.
This shape is intentionally serialization-friendly so offline tooling can inspect and replay predictor graphs deterministically.

### PortableInferencePayloadNode

One portable node record used by the structured-clone payload surface.

Portable payloads keep human-readable activation names instead of numeric ids
so they remain self-describing across worker boundaries and versioned message
logs.

Example:

```ts
const portableNode: PortableInferencePayloadNode = {
  id: 3,
  bias: 0.15,
  response: 1,
  mask: 1,
  activation: 'relu',
  selfWeight: 0,
  selfGaterIndex: -1,
};
```

### resolveAutoInferenceTransport

```ts
resolveAutoInferenceTransport(
  capabilities: InferenceWorkerCapabilities,
): AutoInferenceTransport
```

Resolve the honest automatic transport choice for one capability snapshot.

The priority order matches the current transport ladder: shared-memory first,
then persistent channels, then transferable payload fallback.

Parameters:
- `capabilities` - Capability snapshot returned by the probe.

Returns: Best automatic transport choice for the current host.

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  hasChannelWorker: true,
  runtime: 'browser',
});
const transport = resolveAutoInferenceTransport(capabilities);
```

### resolveBrowserWorkerAssetUrl

```ts
resolveBrowserWorkerAssetUrl(
  workerAssetPath: string,
  options: BrowserWorkerAssetUrlOptions,
): string | undefined
```

Resolves a browser worker asset beside the current script or page URL.

This helper keeps the bundler boundary explicit: callers still name the
emitted worker asset they expect, while the library handles the common URL
math for browser demos, nested workers, and side-by-side bundle delivery.

Parameters:
- `workerAssetPath` - Relative worker asset path emitted by the bundler.
- `options` - Optional explicit base URL override.

Returns: Absolute worker asset URL when a browser base URL is available.

Example:

```ts
const sharedWorkerUrl = resolveBrowserWorkerAssetUrl(
  'flappy-shared-inference.worker.bundle.js',
);
```

### SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION

Browser `SharedArrayBuffer` inference requires cross-origin isolation.

Node.js workers can use the shared-memory path without `COOP` or `COEP`, but
browsers only expose `SharedArrayBuffer` reliably when the host page is
cross-origin isolated.

Example:

```ts
if (SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION) {
  console.log('Configure COOP/COEP before enabling shared-memory inference.');
}
```

### SharedInferenceWorker

Persistent shared-memory inference worker.

Shared-memory workers keep one predictor alive inside a dedicated worker and
exchange input and output values through `SharedArrayBuffer` shelves instead
of cloning or transferring a fresh payload for every request.

Example:

```ts
const worker = openSharedInferenceWorker(payload);
const outputValues = await worker.infer([0.25, 0.75]);
await worker.release();
```

### SharedInferenceWorkerOptions

Configuration for one dedicated shared-memory inference worker.

Browser hosts default to the module-relative shared worker emitted beside the
library files. Provide `workerUrl` when bundling moves that worker asset or
when CSP policy disallows the default delivery path.

Example:

```ts
const worker = openSharedInferenceWorker(payload, {
  workerUrl: '/assets/shared-inference.worker.js',
});
```

### TransferableInferencePayload

Typed-array inference payload for lower-copy worker transport.

Transferable payloads keep the same semantic content as portable payloads,
but pack it into flat typed arrays so callers can move ownership across
worker boundaries without deep-cloning large object graphs.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
console.log(payload.nodeIds.length);
```

### TransferableInferencePayloadOptions

Transfer options that control buffer packing and transfer-list behavior for worker-friendly inference payload publication.
They allow callers to tune ownership and copy semantics for high-throughput cross-thread prediction workloads.

## architecture/network/worker-payload/network.worker-payload.neat.ts

### createNeatParallelPopulationEvaluator

```ts
createNeatParallelPopulationEvaluator(
  options: NeatParallelPopulationEvaluatorOptions<TGenome, TPayload, TWorker, TResult>,
): (population: TGenome[]) => Promise<void>
```

Create a NEAT-compatible population fitness delegate on top of `evaluateInWorkers(...)`.

The returned function matches `fitnessPopulation: true` evaluation mode: it
scores the whole population in one async call and writes ordered results back
onto the genomes in place.

Parameters:
- `options` - Boolean-first population-evaluation options plus advanced worker overrides.

Returns: Population fitness delegate compatible with `fitnessPopulation: true`.

Example:

```ts
const evaluatePopulation = createNeatParallelPopulationEvaluator({
  parallel: true,
  evaluateGenome: async (genome) => genome.score ?? 0,
  openWorker: (payload) => openSharedInferenceWorker(payload),
  evaluateWithWorker: async (worker, genome) => worker.infer(genome.activate([0, 1])),
});
```

### NeatParallelPopulationEvaluatorOptions

Low-friction NEAT population-evaluation options built on top of `evaluateInWorkers(...)`.

The factory keeps the public surface focused on population evaluation rather
than transport details. Callers can opt into parallel execution with
`parallel: true`, keep the local scoring delegate as the honest fallback,
and add worker-specific overrides only when they need them.

## architecture/network/worker-payload/network.worker-payload.pool.ts

Minimal worker resource contract required by the generic inference pool.

The pool owns worker lifecycle, not transport internals. Any persistent
predictor or worker can participate as long as it can release its resources
deterministically.

### ParallelInferencePool

Generic bounded worker pool for ordered parallel inference tasks.

The pool keeps persistent predictors warm, caps concurrent worker usage, and
rebuilds results in the caller's original order even when individual jobs
finish out of order.

Example:

```ts
const pool = new ParallelInferencePool({
  openWorker: (payload) => openSharedInferenceWorker(payload),
  workerCount: 4,
});
await pool.initialize(payloads);
const results = await pool.evaluateOrderedBatch(payloads, async (worker) => {
  return worker.infer([0.25, 0.75]);
});
await pool.dispose();
```

#### dispose

```ts
dispose(): Promise<void>
```

Releases every active worker and clears the slot shelf.

Returns: Nothing.

#### evaluateOrderedBatch

```ts
evaluateOrderedBatch(
  payloads: readonly TPayload[],
  evaluator: (worker: TWorker, payload: TPayload, payloadIndex: number) => Promise<TResult>,
): Promise<TResult[]>
```

Evaluates a payload shelf through the bounded worker slots.

The pool schedules work FIFO, lets each slot consume tasks until the queue
is empty, and returns results in the same order as the input payloads.

Parameters:
- `payloads` - Ordered payload shelf to evaluate.
- `evaluator` - Caller-owned evaluation logic for one warm worker slot.

Returns: Results in the caller's original payload order.

#### initialize

```ts
initialize(
  payloads: readonly TPayload[],
): Promise<void>
```

Prepares empty slot state for the next payload shelf.

Existing workers are released before the new shelf becomes active so slot
reuse stays deterministic across generations or evaluation batches.

Parameters:
- `payloads` - Ordered payload shelf that may be evaluated next.

Returns: Nothing.

### ParallelInferencePoolOptions

Configuration for one bounded parallel inference pool.

The opener stays caller-owned so the pool can schedule shared-memory
workers, inference channels, or future worker-backed predictors without
coupling this boundary to one transport strategy.

### ParallelInferenceWorkerLike

Minimal worker resource contract required by the generic inference pool.

The pool owns worker lifecycle, not transport internals. Any persistent
predictor or worker can participate as long as it can release its resources
deterministically.

## architecture/network/worker-payload/network.worker-payload.batch.ts

### BatchEvaluationResult

Ordered result envelope returned by `evaluateInWorkers(...)`.

The helper preserves input order even when individual worker tasks finish out
of order, and it reports stable task ids so callers can correlate results
with their original batch shelf without reconstructing indexes manually.

### evaluateInWorkers

```ts
evaluateInWorkers(
  options: EvaluateInWorkersOptions<TInput, TPayload, TWorker, TResult>,
): Promise<BatchEvaluationResult<TResult>>
```

Evaluates an ordered batch through workers when available, otherwise locally.

This helper keeps the honest async boundary visible: callers provide worker
opening plus task execution when they want parallelism, and they also provide
a local fallback when they need the same semantics in worker-less hosts.

Parameters:
- `options` - Ordered inputs plus worker and fallback execution hooks.

Returns: Ordered batch results with elapsed time and stable task ids.

Example:

```ts
const batchResult = await evaluateInWorkers({
  inputs: payloads,
  openWorker: (payload) => openSharedInferenceWorker(payload),
  evaluateWithWorker: (worker, _payload) => worker.infer([0.25, 0.75]),
});
```

### EvaluateInWorkersOptions

Generic ordered batch-evaluation options for worker and fallback execution.

Keep the transport substrate caller-owned: this helper schedules and assembles
deterministic results, but the caller still decides how a worker is opened,
how a task is scored, and what the single-thread fallback should do.

## architecture/network/worker-payload/network.worker-payload.shared.ts

### copyInputValuesIntoSharedBuffer

```ts
copyInputValuesIntoSharedBuffer(
  dataView: Float64Array<ArrayBufferLike>,
  inputValues: readonly number[],
  inputOffset: number,
  useChunkedConversion: boolean,
): Promise<void>
```

Copy one submitted input vector into the shared numeric shelf.

Browser-backed hosts can chunk very large numeric copies because the shared
worker API already exposes an async boundary. Node keeps the native bulk copy
path because there is no UI thread to protect.

Parameters:
- `dataView` - Shared numeric shelf covering inputs and outputs.
- `inputValues` - Caller-provided numeric input vector.
- `inputOffset` - Start index of the shared input shelf.
- `useChunkedConversion` - True when the caller wants browser-style chunking for large copies.

Returns: Promise resolved after the shared input shelf mirrors the caller input.

### copySharedNumericValuesInChunks

```ts
copySharedNumericValuesInChunks(
  valueCount: number,
  copyChunk: (startIndex: number, endIndex: number) => void,
): Promise<void>
```

Copy one large numeric conversion in browser-friendly chunks.

Parameters:
- `valueCount` - Total numeric slots to copy.
- `copyChunk` - Callback that copies one contiguous chunk range.

Returns: Promise resolved after every chunk has been copied.

### copySharedOutputValues

```ts
copySharedOutputValues(
  dataView: Float64Array<ArrayBufferLike>,
  outputOffset: number,
  outputCount: number,
  useChunkedConversion: boolean,
): Promise<Float64Array<ArrayBufferLike>>
```

Detach one shared-memory output shelf into a standalone typed array.

Parameters:
- `dataView` - Shared numeric shelf covering inputs and outputs.
- `outputOffset` - Start index of the output shelf.
- `outputCount` - Number of output values to copy.
- `useChunkedConversion` - True when the caller wants browser-style chunking for large copies.

Returns: Detached output values safe for the caller to retain.

### openSharedInferenceWorker

```ts
openSharedInferenceWorker(
  payload: TransferableInferencePayload,
  options: SharedInferenceWorkerOptions,
): SharedInferenceWorker
```

Open one persistent shared-memory inference worker from a transferable payload.

Shared-memory workers keep one predictor alive inside a dedicated worker and
exchange inputs and outputs through `SharedArrayBuffer` shelves rather than
cloning request payloads for every inference call.

Browser hosts default to a module-relative worker script emitted alongside
the library files. Bundled or CSP-constrained hosts can override that entry
with `workerUrl`.

Parameters:
- `payload` - Transferable inference payload used to bootstrap the shared worker predictor.
- `options` - Shared worker delivery options.

Returns: Persistent shared-memory inference worker.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
const sharedWorker = openSharedInferenceWorker(payload);
const outputValues = await sharedWorker.infer([0.25, 0.75]);
await sharedWorker.release();
```

### SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION

Browser `SharedArrayBuffer` inference requires cross-origin isolation.

Node.js workers can use the shared-memory path without `COOP` or `COEP`, but
browsers only expose `SharedArrayBuffer` reliably when the host page is
cross-origin isolated.

Example:

```ts
if (SHARED_INFERENCE_REQUIRES_CROSS_ORIGIN_ISOLATION) {
  console.log('Configure COOP/COEP before enabling shared-memory inference.');
}
```

### shouldChunkSharedNumericConversion

```ts
shouldChunkSharedNumericConversion(
  valueCount: number,
  useChunkedConversion: boolean,
): boolean
```

Decide whether one shared numeric copy should yield between browser-sized chunks.

Parameters:
- `valueCount` - Number of numeric slots involved in the conversion.
- `useChunkedConversion` - True when the current call path wants cooperative chunking.

Returns: True when the current host is browser-backed and the conversion is large.

### yieldSharedConversionMacrotask

```ts
yieldSharedConversionMacrotask(): Promise<void>
```

Yield one timer turn between browser conversion chunks.

Returns: Promise resolved on the next macrotask turn.

## architecture/network/worker-payload/network.worker-payload.channel.ts

### openInferenceChannel

```ts
openInferenceChannel(
  payload: TransferableInferencePayload,
  options: InferenceChannelOptions,
): InferenceChannel
```

Open one persistent inference worker channel from a transferable payload.

The channel transfers the predictor payload exactly once during bootstrap,
then reuses the worker-side predictor state for every later `predict()` or
`reset()` request sent over one dedicated `MessageChannel` port pair.

Parameters:
- `payload` - Transferable inference payload used to bootstrap the worker predictor.
- `options` - Channel concurrency and worker delivery options.

Returns: Persistent inference channel.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
const channel = openInferenceChannel(payload);
const outputValues = await channel.predict([0.25, 0.75]);
channel.close();
```

## architecture/network/worker-payload/network.worker-payload.browser-url.ts

### BrowserWorkerAssetUrlOptions

Options for resolving a browser worker asset URL.

### resolveBrowserWorkerAssetUrl

```ts
resolveBrowserWorkerAssetUrl(
  workerAssetPath: string,
  options: BrowserWorkerAssetUrlOptions,
): string | undefined
```

Resolves a browser worker asset beside the current script or page URL.

This helper keeps the bundler boundary explicit: callers still name the
emitted worker asset they expect, while the library handles the common URL
math for browser demos, nested workers, and side-by-side bundle delivery.

Parameters:
- `workerAssetPath` - Relative worker asset path emitted by the bundler.
- `options` - Optional explicit base URL override.

Returns: Absolute worker asset URL when a browser base URL is available.

Example:

```ts
const sharedWorkerUrl = resolveBrowserWorkerAssetUrl(
  'flappy-shared-inference.worker.bundle.js',
);
```

## architecture/network/worker-payload/network.worker-payload.capabilities.ts

Capability snapshot for the current worker-backed inference host.

This probe answers a narrow question: which worker transport tiers are
honestly usable in the current runtime before one evaluation pool or batch
helper commits to a transport strategy?

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
const transport = resolveAutoInferenceTransport(capabilities);
```

### AutoInferenceTransport

Automatic transport selection result for the current host.

### detectInferenceWorkerCapabilities

```ts
detectInferenceWorkerCapabilities(
  options: InferenceWorkerCapabilityOptions,
): InferenceWorkerCapabilities
```

Detect the usable worker-backed inference transport tiers for one host.

The probe stays intentionally lightweight. It does not open a worker or fetch
a script; it only combines runtime facts and delivery availability so callers
can decide whether to use shared-memory, channel, or transferable fallback.

Parameters:
- `options` - Runtime and delivery facts for the current host.

Returns: Capability snapshot describing the usable transport tiers.

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
```

### InferenceWorkerCapabilities

Capability snapshot for the current worker-backed inference host.

This probe answers a narrow question: which worker transport tiers are
honestly usable in the current runtime before one evaluation pool or batch
helper commits to a transport strategy?

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
const transport = resolveAutoInferenceTransport(capabilities);
```

### InferenceWorkerCapabilityOptions

Inputs used to probe worker-backed inference transport support.

Keep delivery facts explicit here. Step 1 only answers capability and auto
selection; Step 2 will own the generic worker-entry and URL resolution
helpers that feed these booleans.

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  crossOriginIsolated: false,
  hasChannelWorker: true,
  hasSharedWorker: true,
  runtime: 'browser',
});
```

### resolveAutoInferenceTransport

```ts
resolveAutoInferenceTransport(
  capabilities: InferenceWorkerCapabilities,
): AutoInferenceTransport
```

Resolve the honest automatic transport choice for one capability snapshot.

The priority order matches the current transport ladder: shared-memory first,
then persistent channels, then transferable payload fallback.

Parameters:
- `capabilities` - Capability snapshot returned by the probe.

Returns: Best automatic transport choice for the current host.

Example:

```ts
const capabilities = detectInferenceWorkerCapabilities({
  hasChannelWorker: true,
  runtime: 'browser',
});
const transport = resolveAutoInferenceTransport(capabilities);
```

## architecture/network/worker-payload/network.worker-payload.shared.worker.ts

### registerSharedInferenceWorkerRuntime

```ts
registerSharedInferenceWorkerRuntime(): void
```

Register the worker-side shared-memory inference runtime.

The worker rebuilds one predictor from the transferred payload during
bootstrap, then services inference requests by reading and writing the
shared control and data shelves coordinated through `Atomics`.

Returns: Nothing.

## architecture/network/worker-payload/network.worker-payload.channel.worker.ts

### registerInferenceChannelWorkerRuntime

```ts
registerInferenceChannelWorkerRuntime(): void
```

Register the worker-side dedicated inference-channel runtime.

The worker receives one transferable predictor payload during bootstrap,
rebuilds the local predictor once, then services predict and reset requests
over the transferred `MessagePort` instead of the default worker channel.

Returns: Nothing.

## architecture/network/worker-payload/network.worker-payload.utils.ts

### activatePortableNode

```ts
activatePortableNode(
  activationContext: PortableActivationContext,
): void
```

Activate one portable node using no-trace runtime semantics.

Parameters:
- `activationContext` - Predictor context for this activation step.

Returns: Nothing.

### activateTransferableNode

```ts
activateTransferableNode(
  activationContext: TransferableActivationContext,
): void
```

Activate one transferable node using no-trace runtime semantics.

Parameters:
- `activationContext` - Predictor context for this activation step.

Returns: Nothing.

### buildInferenceIrEdge

```ts
buildInferenceIrEdge(
  connectionReference: default,
): { from: number; to: number; weight: number; gaterIndex: number; }
```

Build one deterministic non-self edge snapshot for the inference IR.

Parameters:
- `connectionReference` - Runtime connection.

Returns: Edge IR snapshot.

### buildInferenceIrNode

```ts
buildInferenceIrNode(
  nodeReference: default,
): { index: number; bias: number; response: number; mask: number; activationId: number; selfWeight: number; selfGaterIndex: number; }
```

Build one deterministic node snapshot for the inference IR.

Parameters:
- `nodeReference` - Runtime node.

Returns: Node IR snapshot.

### buildPortableInferencePayloadEdge

```ts
buildPortableInferencePayloadEdge(
  inferenceEdge: NetworkInferenceIREdge,
): PortableInferencePayloadEdge
```

Build one portable edge payload record from the inference IR.

Parameters:
- `inferenceEdge` - Inference IR edge snapshot.

Returns: Structured-clone-safe edge payload record.

### buildPortableInferencePayloadNode

```ts
buildPortableInferencePayloadNode(
  inferenceNode: NetworkInferenceIRNode,
): PortableInferencePayloadNode
```

Build one portable node payload record from the inference IR.

Parameters:
- `inferenceNode` - Inference IR node snapshot.

Returns: Structured-clone-safe node payload record.

### createEdgeIndexesByGaterIndex

```ts
createEdgeIndexesByGaterIndex(
  portableEdges: readonly PortableInferencePayloadEdge[],
): Map<number, readonly number[]>
```

Build gated forward-edge indexes keyed by gater node index.

Parameters:
- `portableEdges` - Portable payload edges.

Returns: Gated edge indexes keyed by gater node index.

### createIncomingEdgesByTargetIndex

```ts
createIncomingEdgesByTargetIndex(
  portableEdges: readonly PortableInferencePayloadEdge[],
  nodeCount: number,
): readonly (readonly number[])[]
```

Build incoming-edge indexes keyed by target node index.

Parameters:
- `portableEdges` - Portable payload edges.
- `nodeCount` - Portable node count.

Returns: Incoming edge indexes keyed by target node index.

### createIncomingTransferableEdgesByTargetIndex

```ts
createIncomingTransferableEdgesByTargetIndex(
  edgeTo: Int32Array<ArrayBufferLike>,
  nodeCount: number,
): readonly (readonly number[])[]
```

Build incoming transferable-edge indexes keyed by target node index.

Parameters:
- `edgeTo` - Transferable edge target-node shelf.
- `nodeCount` - Transferable node count.

Returns: Incoming edge indexes keyed by target node index.

### createInferenceActivationIdsByName

```ts
createInferenceActivationIdsByName(): Map<string, number>
```

Build the stable activation-name lookup used by inference payloads.

Returns: Activation-id lookup by canonical key or function name.

### createInferencePredictor

```ts
createInferencePredictor(
  payload: PortableInferencePayload,
): InferencePredictor
```

Create a reusable local predictor from either portable or transferable inference payload contracts.
The factory normalizes both transport strategies into one inference interface so callers can benchmark or run fallback execution paths without special-case runtime branching.

Parameters:
- `payload` - Portable or transferable inference payload.

Returns: Predictor that mirrors runtime no-trace activation semantics.

Example:

```ts
const payload = exportPortableInferencePayload(network);
const predictor = createInferencePredictor(payload);
const outputValues = predictor.predict([0.25, 0.75]);
```

### createNodesByGeneId

```ts
createNodesByGeneId(
  nodes: readonly default[],
): Map<number, default>
```

Build a stable node lookup keyed by runtime gene id.

Parameters:
- `nodes` - Runtime nodes in deterministic index order.

Returns: Node lookup keyed by gene id.

### createPortableInferencePredictor

```ts
createPortableInferencePredictor(
  payload: PortableInferencePayload,
): InferencePredictor
```

Create a reusable local predictor from a portable inference payload.

Parameters:
- `payload` - Portable inference payload.

Returns: Predictor that mirrors runtime no-trace activation semantics.

### createSelfNodeIndexesByGaterIndex

```ts
createSelfNodeIndexesByGaterIndex(
  portableNodes: readonly PortableInferencePayloadNode[],
): Map<number, readonly number[]>
```

Build gated self-connection indexes keyed by gater node index.

Parameters:
- `portableNodes` - Portable payload nodes.

Returns: Self-connection node indexes keyed by gater node index.

### createTransferableEdgeIndexesByGaterIndex

```ts
createTransferableEdgeIndexesByGaterIndex(
  edgeGaterIndices: Int32Array<ArrayBufferLike>,
): Map<number, readonly number[]>
```

Build gated transferable forward-edge indexes keyed by gater node index.

Parameters:
- `edgeGaterIndices` - Transferable edge gater shelf.

Returns: Gated edge indexes keyed by gater node index.

### createTransferableInferencePredictor

```ts
createTransferableInferencePredictor(
  payload: TransferableInferencePayload,
): InferencePredictor
```

Create a reusable local predictor from a transferable inference payload.

Parameters:
- `payload` - Transferable inference payload.

Returns: Predictor that mirrors runtime no-trace activation semantics.

### createTransferableNumericArray

```ts
createTransferableNumericArray(
  values: readonly number[],
  numericPrecision: TransferableNumericPrecision,
): Float32Array<ArrayBufferLike> | Float64Array<ArrayBufferLike>
```

Create one numeric typed array using the requested transferable precision.

Parameters:
- `values` - Numeric values to pack.
- `numericPrecision` - Requested precision mode.

Returns: Float32Array or Float64Array depending on the selected mode.

### createTransferableSelfNodeIndexesByGaterIndex

```ts
createTransferableSelfNodeIndexesByGaterIndex(
  nodeSelfGaterIndices: Int32Array<ArrayBufferLike>,
): Map<number, readonly number[]>
```

Build gated transferable self-connection indexes keyed by gater node index.

Parameters:
- `nodeSelfGaterIndices` - Transferable self-gater shelf.

Returns: Self-connection node indexes keyed by gater node index.

### exportPortableInferencePayload

```ts
exportPortableInferencePayload(
  network: default,
): PortableInferencePayload
```

Export one structured-clone-safe inference payload from a live runtime network.
The resulting payload keeps deterministic activation ordering, stable node or edge indexing, and canonical activation names so a worker can replay inference semantics without shipping live graph objects.

Parameters:
- `network` - Runtime network to serialize for worker transport.

Returns: Portable inference payload.

Example:

```ts
const payload = exportPortableInferencePayload(network);
console.log(payload.nodes[0]?.activation);
```

### exportTransferableInferencePayload

```ts
exportTransferableInferencePayload(
  network: default,
  options: TransferableInferencePayloadOptions,
): TransferableInferencePayload
```

Export one typed-array inference payload optimized for low-copy worker transport.
This variant preserves the same deterministic IR semantics as the portable payload while flattening fields into transfer-friendly typed shelves that can be moved across worker boundaries efficiently.

Parameters:
- `network` - Runtime network to serialize for worker transport.
- `options` - Transferable export configuration.

Returns: Transferable inference payload.

Example:

```ts
const payload = exportTransferableInferencePayload(network, {
  numericPrecision: 'f32',
});
console.log(payload.edgeWeights.length);
```

### extractNetworkInferenceIR

```ts
extractNetworkInferenceIR(
  network: default,
): NetworkInferenceIR
```

Extract a deterministic inference IR from one live network snapshot.
This extraction pass captures exactly the runtime data required for worker-side forward execution, including node scalars, filtered forward edges, grouped activation steps, and stable output indexing.

Parameters:
- `network` - Runtime network to snapshot.

Returns: Worker-friendly inference IR.

Example:

```ts
const inferenceIr = extractNetworkInferenceIR(network);
console.log(inferenceIr.outputNodeIndices);
```

### flattenActivationStepsForTransferablePayload

```ts
flattenActivationStepsForTransferablePayload(
  activationSteps: readonly (readonly number[])[],
): { activationStepsData: number[]; activationStepsIndex: number[]; }
```

Build the transferable activation-step packing from grouped IR steps.

Parameters:
- `activationSteps` - Grouped activation steps from the inference IR.

Returns: Flat activation-step data plus per-step start offsets.

### getTransferList

```ts
getTransferList(
  payload: TransferableInferencePayload,
): ArrayBuffer[]
```

Collect every transferable buffer from one typed-array inference payload.

Transfer these buffers exactly once when posting the payload to a worker.
After the transfer completes, the sending-side typed arrays are neutered and
must not be read again.

Parameters:
- `payload` - Transferable inference payload whose buffers should move.

Returns: ArrayBuffer transfer list aligned to the payload's typed shelves.

Example:

```ts
const payload = exportTransferableInferencePayload(network);
worker.postMessage(payload, getTransferList(payload));
```

### INFERENCE_ACTIVATION_TABLE

Stable activation-function table used by worker inference transport.

Each entry index is part of the public inference payload contract. New worker
payloads should reference activations by these numeric ids rather than by
serializing function bodies or relying on runtime function identity.

Example:

```ts
const activationFunction = INFERENCE_ACTIVATION_TABLE[0];
const outputValue = activationFunction(0.5);
```

### resolveActivationSteps

```ts
resolveActivationSteps(
  network: default,
  generationContext: StandaloneGenerationContext,
  nodesByGeneId: ReadonlyMap<number, default>,
): readonly (readonly number[])[]
```

Resolve grouped activation steps from the compiled runtime schedule when available.

Parameters:
- `network` - Runtime network.
- `generationContext` - Standalone generation context with fallback traversal order.
- `nodesByGeneId` - Stable node lookup by gene id.

Returns: Grouped activation steps.

### resolveActiveSelfConnection

```ts
resolveActiveSelfConnection(
  nodeReference: default,
): default | undefined
```

Resolve the active self-connection for one runtime node.

Parameters:
- `nodeReference` - Runtime node.

Returns: Active self-connection or undefined.

### resolveCompiledActivationSteps

```ts
resolveCompiledActivationSteps(
  network: default,
  nodesByGeneId: ReadonlyMap<number, default>,
): readonly (readonly number[])[] | null
```

Resolve grouped activation steps from the compiled activation schedule.

Parameters:
- `network` - Runtime network.
- `nodesByGeneId` - Stable node lookup by gene id.

Returns: Grouped activation steps or null when the schedule is unavailable.

### resolveInferenceActivationId

```ts
resolveInferenceActivationId(
  nodeReference: default,
): number
```

Resolve the supported activation id for one runtime node.

Parameters:
- `nodeReference` - Runtime node.

Returns: Stable activation id.

### resolveNodeIndexOrThrow

```ts
resolveNodeIndexOrThrow(
  nodeReference: default,
): number
```

Resolve a required node index from the seeded standalone surface.

Parameters:
- `nodeReference` - Runtime node.

Returns: Stable node index.

### resolveOptionalNodeIndex

```ts
resolveOptionalNodeIndex(
  nodeReference: default | null,
): number
```

Resolve an optional node index for gater references.

Parameters:
- `nodeReference` - Optional runtime node.

Returns: Stable node index or `-1` when absent.

### resolvePortableActivationFunction

```ts
resolvePortableActivationFunction(
  portableNode: PortableInferencePayloadNode,
  activationTable: readonly string[],
): ActivationFn
```

Resolve the activation function for one portable node.

Parameters:
- `portableNode` - Portable node payload record.
- `activationTable` - Portable activation table.

Returns: Activation function from the stable inference registry.

### resolvePortableNodesByIndex

```ts
resolvePortableNodesByIndex(
  payload: PortableInferencePayload,
): PortableInferencePayloadNode[]
```

Resolve portable nodes in strict node-index order.

Parameters:
- `payload` - Portable inference payload.

Returns: Portable nodes aligned to index order.

### resolveTransferableActivationFunctions

```ts
resolveTransferableActivationFunctions(
  payload: TransferableInferencePayload,
): ActivationFn[]
```

Resolve activation functions directly from the transferable activation-id shelf.

Parameters:
- `payload` - Transferable inference payload.

Returns: Activation functions aligned to the transferable node order.

### resolveTransferableActivationStepNodeIndex

```ts
resolveTransferableActivationStepNodeIndex(
  activationStepsData: Int32Array<ArrayBufferLike>,
  activationDataIndex: number,
): number
```

Resolve one transferable activation-step node index or throw when the step data is truncated.

Parameters:
- `activationStepsData` - Flat activation-step node shelf.
- `activationDataIndex` - Index inside the flat activation-step shelf.

Returns: Transferable node index for one activation step entry.

### seedPredictorInputActivations

```ts
seedPredictorInputActivations(
  activationValues: Float64Array<ArrayBufferLike>,
  inputValues: readonly number[],
  inputCount: number,
): void
```

Seed the public input activations for one prediction pass.

Parameters:
- `activationValues` - Mutable activation buffer.
- `inputValues` - Caller-provided input vector.
- `inputCount` - Public input width.

Returns: Nothing.

### shouldIncludeConnection

```ts
shouldIncludeConnection(
  connectionReference: default,
): boolean
```

Determine whether a connection contributes to inference regardless of shape.

Parameters:
- `connectionReference` - Runtime connection.

Returns: True when the connection is enabled for inference.

### shouldIncludeForwardEdge

```ts
shouldIncludeForwardEdge(
  connectionReference: default,
): boolean
```

Determine whether a connection contributes to inference.

Parameters:
- `connectionReference` - Runtime connection.

Returns: True when the connection should appear in the IR.

### shouldIncludeSelfConnection

```ts
shouldIncludeSelfConnection(
  connectionReference: default,
): boolean
```

Determine whether a self-connection contributes to inference.

Parameters:
- `connectionReference` - Runtime self-connection.

Returns: True when the self-connection should appear in the IR node snapshot.

### validatePredictorInputSize

```ts
validatePredictorInputSize(
  inputValues: readonly number[],
  expectedInputCount: number,
): void
```

Validate input vector width for one prediction call.

Parameters:
- `inputValues` - Caller-provided input vector.
- `expectedInputCount` - Required input width.

Returns: Nothing.

### validateTransferableActivationTableLength

```ts
validateTransferableActivationTableLength(
  activationTableLength: number,
): void
```

Validate that the transferable activation shelf matches the runtime registry.

Parameters:
- `activationTableLength` - Activation shelf length encoded in the payload.

Returns: Nothing.

### validateTransferableFieldLength

```ts
validateTransferableFieldLength(
  fieldName: string,
  actualLength: number,
  expectedLength: number,
): void
```

Validate one transferable shelf length against the expected payload count.

Parameters:
- `fieldName` - Payload field name.
- `actualLength` - Actual typed-array length.
- `expectedLength` - Expected aligned length.

Returns: Nothing.

### validateTransferableFieldLengths

```ts
validateTransferableFieldLengths(
  payload: TransferableInferencePayload,
): void
```

Validate that every transferable typed shelf stays aligned to the payload contract.

Parameters:
- `payload` - Transferable inference payload.

Returns: Nothing.

### validateTransferableNodeIds

```ts
validateTransferableNodeIds(
  nodeIds: Int32Array<ArrayBufferLike>,
): void
```

Validate that transferable node ids remain aligned to typed-shelf order.

Parameters:
- `nodeIds` - Transferable node-id shelf.

Returns: Nothing.
