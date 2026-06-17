# architecture/network/evaluation-pack

Deterministic evaluation-pack contracts (Layer 2 — Worker Transport
Normalization).

Provides transport-neutral generic contracts for deterministic pack
creation, transfer-list resolution, and schema versioning.  These
contracts bridge the gap between benchmark-specific transport stacks
(racing, predator/prey, ant-hive) and the library's worker-payload
infrastructure by offering a reusable normalization layer that any
benchmark can adopt.

Contracts frozen by Step 01 / confirmed by Step 02:
- `createDeterministicEvaluationPack(seed, inputs)` — transport-neutral
  generic pack creation.  Same seed + same inputs → identical pack (Level 2
  ordered-deterministic).
- `resolveTransferList(pack)` — collects every distinct `ArrayBuffer`
  backing a typed-array field in the pack, deduplicating shared buffers.
- `assertSchemaVersion(pack, expectedVersion)` — generic schema-version
  rejection; throws `RangeError` on mismatch.

## architecture/network/evaluation-pack/network.evaluation-pack.ts

### assertSchemaVersion

```ts
assertSchemaVersion(
  pack: { schemaVersion: unknown; },
  expectedVersion: string,
): void
```

Asserts that the `schemaVersion` field of a pack matches the expected
version string.  Consumers must call this before reading any typed-array
field so that a version mismatch is caught at the boundary rather than
silently misinterpreting the packed bytes.

This is the generic version of the racing-specific
`assertRacingSchemaVersion` — the expected version is passed as a parameter
so any benchmark can use the same boundary guard.

Parameters:
- `pack` - Object with a `schemaVersion` field.
- `expectedVersion` - Expected schema-version sentinel.

Example:

```ts
assertSchemaVersion(receivedPack, 'eval-pack-v1');
// now safe to read typed-array fields
```

### buildPackArrays

```ts
buildPackArrays(
  prng: () => number,
  agentCount: number,
): ArrayBufferView<ArrayBufferLike>[]
```

Builds the typed arrays for an evaluation pack from a PRNG and agent count.

Each array is filled with deterministic values drawn from the PRNG in a
fixed order: agent states first, then agent weights, then agent active
flags.  This ordering guarantees that the same `(seed, agentCount)` tuple
always produces byte-identical arrays.

Parameters:
- `prng` - Deterministic PRNG closure (from `createPackPRNG`).
- `agentCount` - Number of agent slots (determines each array's length).

Returns: Array of typed arrays ready for the pack's `arrays` field.

### collectDistinctBuffers

```ts
collectDistinctBuffers(
  arrays: readonly ArrayBufferView<ArrayBufferLike>[],
): ArrayBuffer[]
```

Collects distinct `ArrayBuffer` references from a collection of typed
arrays, deduplicating shared buffers.

Two typed arrays that view the same underlying `ArrayBuffer` (e.g. offset
views into a shared slab) produce exactly one buffer entry, not two.
This matches the transfer-list contract required by `postMessage` zero-
copy transfer.

Parameters:
- `arrays` - Typed-array views whose backing buffers will transfer.

Returns: Ordered list of distinct `ArrayBuffer` references.

### createDeterministicEvaluationPack

```ts
createDeterministicEvaluationPack(
  seed: number,
  inputs: EvaluationPackInputs,
): DeterministicEvaluationPack
```

Constructs a deterministic evaluation pack from a seed and transport-neutral
inputs.  Identical `(seed, inputs)` → identical pack on the same runtime
(Level 2 ordered-deterministic).

The pack's typed arrays are filled by a self-contained xorshift32 PRNG
seeded from `seed`.  The PRNG algorithm matches the same family used in
`src/neat/rng/core/` but is duplicated here to avoid a cross-layer
dependency — Layer 2 must not import NEAT core internals.

Parameters:
- `seed` - Deterministic pack seed (non-negative integer; zero falls
back to a non-zero constant because xorshift32 cannot advance from zero).
- `inputs` - Transport-neutral inputs (agent count, schema version).

Returns: A `DeterministicEvaluationPack` whose typed arrays are
deterministic functions of `(seed, inputs)`.

Example:

```ts
const pack = createDeterministicEvaluationPack(42, {
  agentCount: 4,
  schemaVersion: 'eval-pack-v1',
});
// pack.arrays are byte-identical on every call with the same (seed, inputs)
```

### createPackPRNG

```ts
createPackPRNG(
  seed: number,
): () => number
```

Creates a deterministic xorshift32 PRNG closure from a seed.

Given the same seed, the returned function always produces the same
sequence of pseudo-random floats in `[0, 1)`.  The algorithm matches the
xorshift32 family used in `src/neat/rng/core/` but is self-contained here
to avoid a cross-layer dependency from Layer 2 to NEAT core.

Parameters:
- `seed` - Non-negative integer seed (zero falls back to a non-zero
constant).

Returns: A function that returns the next deterministic float in `[0, 1)`.

### DeterministicEvaluationPack

Transport-neutral deterministic evaluation pack.

A pack is the complete deterministic initial state for one evaluation
episode.  Given the same `seed` and the same `EvaluationPackInputs`, the
pack's typed arrays are byte-identical within a single runtime (Level 2
ordered-deterministic).

The `arrays` field holds every typed array that participates in zero-copy
transfer; `resolveTransferList` walks this collection to build the
postMessage transfer list.

### EvaluationPackInputs

Transport-neutral inputs consumed by `createDeterministicEvaluationPack`.

Captures the reproducibility tuple components that are NOT the seed:
`agentCount` (determines typed-array sizes) and `schemaVersion` (forward-
compatibility sentinel).  Benchmark-specific inputs (opponent snapshots,
track physics, etc.) are injected by the benchmark's own wrapper, not by
this generic type.

### resolveTransferList

```ts
resolveTransferList(
  pack: DeterministicEvaluationPack,
): ArrayBuffer[]
```

Collects every distinct `ArrayBuffer` backing a typed-array field in `pack`
into a transfer list for zero-copy `postMessage` transfer.

Rules:
- Every typed-array field contributes exactly one buffer entry.
- Shared buffers are deduplicated (listed only once).

Parameters:
- `pack` - The deterministic evaluation pack whose buffers will transfer.

Returns: Ordered list of `ArrayBuffer` references for postMessage transfer.

Example:

```ts
const pack = createDeterministicEvaluationPack(42, inputs);
const transferList = resolveTransferList(pack);
worker.postMessage({ type: 'eval', pack }, transferList);
```
