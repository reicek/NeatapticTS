/**
 * Deterministic evaluation-pack contracts (Layer 2 — Worker Transport
 * Normalization).
 *
 * Provides transport-neutral generic contracts for deterministic pack
 * creation, transfer-list resolution, and schema versioning.  These
 * contracts bridge the gap between benchmark-specific transport stacks
 * (racing, predator/prey, ant-hive) and the library's worker-payload
 * infrastructure by offering a reusable normalization layer that any
 * benchmark can adopt.
 *
 * Contracts frozen by Step 01 / confirmed by Step 02:
 * - `createDeterministicEvaluationPack(seed, inputs)` — transport-neutral
 *   generic pack creation.  Same seed + same inputs → identical pack (Level 2
 *   ordered-deterministic).
 * - `resolveTransferList(pack)` — collects every distinct `ArrayBuffer`
 *   backing a typed-array field in the pack, deduplicating shared buffers.
 * - `assertSchemaVersion(pack, expectedVersion)` — generic schema-version
 *   rejection; throws `RangeError` on mismatch.
 *
 * @module network.evaluation-pack
 */

/**
 * Transport-neutral deterministic evaluation pack.
 *
 * A pack is the complete deterministic initial state for one evaluation
 * episode.  Given the same `seed` and the same `EvaluationPackInputs`, the
 * pack's typed arrays are byte-identical within a single runtime (Level 2
 * ordered-deterministic).
 *
 * The `arrays` field holds every typed array that participates in zero-copy
 * transfer; `resolveTransferList` walks this collection to build the
 * postMessage transfer list.
 */
export type DeterministicEvaluationPack = {
  /** Generic schema-version sentinel for forward-compatibility rejection. */
  readonly schemaVersion: string;
  /** PRNG seed used to generate the pack. */
  readonly seed: number;
  /** Typed arrays held by the pack, available for transfer-list resolution. */
  readonly arrays: readonly ArrayBufferView[];
};

/**
 * Transport-neutral inputs consumed by `createDeterministicEvaluationPack`.
 *
 * Captures the reproducibility tuple components that are NOT the seed:
 * `agentCount` (determines typed-array sizes) and `schemaVersion` (forward-
 * compatibility sentinel).  Benchmark-specific inputs (opponent snapshots,
 * track physics, etc.) are injected by the benchmark's own wrapper, not by
 * this generic type.
 */
export type EvaluationPackInputs = {
  /** Number of agent slots (determines typed-array lengths). */
  readonly agentCount: number;
  /** Schema-version sentinel stamped onto the produced pack. */
  readonly schemaVersion: string;
};

/**
 * Constructs a deterministic evaluation pack from a seed and transport-neutral
 * inputs.  Identical `(seed, inputs)` → identical pack on the same runtime
 * (Level 2 ordered-deterministic).
 *
 * The pack's typed arrays are filled by a self-contained xorshift32 PRNG
 * seeded from `seed`.  The PRNG algorithm matches the same family used in
 * `src/neat/rng/core/` but is duplicated here to avoid a cross-layer
 * dependency — Layer 2 must not import NEAT core internals.
 *
 * @param seed - Deterministic pack seed (non-negative integer; zero falls
 * back to a non-zero constant because xorshift32 cannot advance from zero).
 * @param inputs - Transport-neutral inputs (agent count, schema version).
 * @returns A `DeterministicEvaluationPack` whose typed arrays are
 * deterministic functions of `(seed, inputs)`.
 *
 * @example
 * ```ts
 * const pack = createDeterministicEvaluationPack(42, {
 *   agentCount: 4,
 *   schemaVersion: 'eval-pack-v1',
 * });
 * // pack.arrays are byte-identical on every call with the same (seed, inputs)
 * ```
 */
export function createDeterministicEvaluationPack(
  seed: number,
  inputs: EvaluationPackInputs,
): DeterministicEvaluationPack {
  // Step 1: Seed the deterministic PRNG from the caller-provided seed
  const prng = createPackPRNG(seed);

  // Step 2: Build typed arrays whose contents are deterministic functions
  // of (seed, agentCount)
  const arrays = buildPackArrays(prng, inputs.agentCount);

  // Step 3: Assemble the pack with the schema-version sentinel
  return {
    schemaVersion: inputs.schemaVersion,
    seed,
    arrays,
  };
}

/**
 * Collects every distinct `ArrayBuffer` backing a typed-array field in `pack`
 * into a transfer list for zero-copy `postMessage` transfer.
 *
 * Rules:
 * - Every typed-array field contributes exactly one buffer entry.
 * - Shared buffers are deduplicated (listed only once).
 *
 * @param pack - The deterministic evaluation pack whose buffers will transfer.
 * @returns Ordered list of `ArrayBuffer` references for postMessage transfer.
 *
 * @example
 * ```ts
 * const pack = createDeterministicEvaluationPack(42, inputs);
 * const transferList = resolveTransferList(pack);
 * worker.postMessage({ type: 'eval', pack }, transferList);
 * ```
 */
export function resolveTransferList(
  pack: DeterministicEvaluationPack,
): ArrayBuffer[] {
  return collectDistinctBuffers(pack.arrays);
}

/**
 * Asserts that the `schemaVersion` field of a pack matches the expected
 * version string.  Consumers must call this before reading any typed-array
 * field so that a version mismatch is caught at the boundary rather than
 * silently misinterpreting the packed bytes.
 *
 * This is the generic version of the racing-specific
 * `assertRacingSchemaVersion` — the expected version is passed as a parameter
 * so any benchmark can use the same boundary guard.
 *
 * @param pack - Object with a `schemaVersion` field.
 * @param expectedVersion - Expected schema-version sentinel.
 * @throws {RangeError} When `schemaVersion` does not match `expectedVersion`.
 *
 * @example
 * ```ts
 * assertSchemaVersion(receivedPack, 'eval-pack-v1');
 * // now safe to read typed-array fields
 * ```
 */
export function assertSchemaVersion(
  pack: { schemaVersion: unknown },
  expectedVersion: string,
): void {
  if (pack.schemaVersion !== expectedVersion) {
    throw new RangeError(
      `Unsupported evaluation-pack schemaVersion: expected '${expectedVersion}', got '${String(pack.schemaVersion)}'`,
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers (below the fold)
// ---------------------------------------------------------------------------

/** Left-shift amount for the first xorshift32 state-mixing step. */
const PACK_PRNG_SHIFT_LEFT_PRIMARY = 13;

/** Right-shift amount for the middle xorshift32 state-mixing step. */
const PACK_PRNG_SHIFT_RIGHT_PRIMARY = 17;

/** Left-shift amount for the final xorshift32 state-mixing step. */
const PACK_PRNG_SHIFT_LEFT_SECONDARY = 5;

/**
 * Divisor-plus-one used to normalize the 32-bit integer PRNG state into the
 * `[0, 1)` range.  Stored as the divisor boundary so the normalization step
 * reads as an explicit integer-to-float conversion.
 */
const PACK_PRNG_NORMALIZATION_DIVISOR = 0xffff_ffff;

/**
 * Fallback seed when the caller-provided seed is zero.  Xorshift32 cannot
 * advance from a zero state, so this non-zero constant is the guarded escape
 * hatch that keeps initialization valid.
 */
const PACK_PRNG_FALLBACK_SEED = 0x1a2b3c4d;

/**
 * Scaling factor applied to agent weight values so they span a meaningful
 * range rather than collapsing toward [0, 1).
 */
const PACK_WEIGHT_SCALE = 1_000;

/** Maximum value (exclusive) for `Uint8Array` agent-active flags. */
const PACK_U8_RANGE = 256;

/**
 * Creates a deterministic xorshift32 PRNG closure from a seed.
 *
 * Given the same seed, the returned function always produces the same
 * sequence of pseudo-random floats in `[0, 1)`.  The algorithm matches the
 * xorshift32 family used in `src/neat/rng/core/` but is self-contained here
 * to avoid a cross-layer dependency from Layer 2 to NEAT core.
 *
 * @param seed - Non-negative integer seed (zero falls back to a non-zero
 * constant).
 * @returns A function that returns the next deterministic float in `[0, 1)`.
 */
function createPackPRNG(seed: number): () => number {
  let state = seed === 0 ? PACK_PRNG_FALLBACK_SEED : seed >>> 0;

  return function next(): number {
    // Xorshift32 state-mixing: three shift-XOR steps
    state ^= state << PACK_PRNG_SHIFT_LEFT_PRIMARY;
    state ^= state >>> PACK_PRNG_SHIFT_RIGHT_PRIMARY;
    state ^= state << PACK_PRNG_SHIFT_LEFT_SECONDARY;

    // Normalize the unsigned 32-bit state into [0, 1)
    return (state >>> 0) / (PACK_PRNG_NORMALIZATION_DIVISOR + 1);
  };
}

/**
 * Builds the typed arrays for an evaluation pack from a PRNG and agent count.
 *
 * Each array is filled with deterministic values drawn from the PRNG in a
 * fixed order: agent states first, then agent weights, then agent active
 * flags.  This ordering guarantees that the same `(seed, agentCount)` tuple
 * always produces byte-identical arrays.
 *
 * @param prng - Deterministic PRNG closure (from `createPackPRNG`).
 * @param agentCount - Number of agent slots (determines each array's length).
 * @returns Array of typed arrays ready for the pack's `arrays` field.
 */
function buildPackArrays(
  prng: () => number,
  agentCount: number,
): ArrayBufferView[] {
  const agentStates = new Float32Array(agentCount);
  const agentWeights = new Float64Array(agentCount);
  const agentActive = new Uint8Array(agentCount);

  for (let index = 0; index < agentCount; index++) {
    agentStates[index] = prng();
    agentWeights[index] = prng() * PACK_WEIGHT_SCALE;
    agentActive[index] = Math.floor(prng() * PACK_U8_RANGE);
  }

  return [agentStates, agentWeights, agentActive];
}

/**
 * Collects distinct `ArrayBuffer` references from a collection of typed
 * arrays, deduplicating shared buffers.
 *
 * Two typed arrays that view the same underlying `ArrayBuffer` (e.g. offset
 * views into a shared slab) produce exactly one buffer entry, not two.
 * This matches the transfer-list contract required by `postMessage` zero-
 * copy transfer.
 *
 * @param arrays - Typed-array views whose backing buffers will transfer.
 * @returns Ordered list of distinct `ArrayBuffer` references.
 */
function collectDistinctBuffers(
  arrays: readonly ArrayBufferView[],
): ArrayBuffer[] {
  const transferList: ArrayBuffer[] = [];
  const seenBuffers = new Set<ArrayBuffer>();

  for (const view of arrays) {
    const buffer = view.buffer as ArrayBuffer;

    if (seenBuffers.has(buffer)) {
      continue;
    }

    seenBuffers.add(buffer);
    transferList.push(buffer);
  }

  return transferList;
}
