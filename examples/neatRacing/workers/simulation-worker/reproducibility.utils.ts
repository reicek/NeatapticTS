/**
 * Reproducibility tuple utilities for the neatRacing simulation worker
 * (Phase 2, Step 01, slice P2-S1-C).
 *
 * Educational note — why this file exists:
 * The spec's determinism contract (§5.3, §9.7) pins AC-026: two runs booted
 * from the SAME reproducibility tuple `(seed, agentCount, packSchemaVersion,
 * opponentSnapshot, trackId, featureFlags)` must produce byte-identical
 * frames across `NEATRACING_REPLAY_ADAPTATION_TICKS = 32` adaptation ticks
 * on the same runtime. The recycled runtime adaptation engine defaults to
 * `Math.random`, which would leave that contract numerically untestable, so
 * (D1) ALL sim-side randomness instead derives from the tuple seed through
 * the named §9.7.2 child-seed derivation table.
 *
 * What this boundary owns:
 * - the §9.7.2 mix (`combineNeatRacingChildSeed`), mirroring the recycled
 *   track generator's combined-seed hash pattern (golden-ratio xor +
 *   FNV-style `Math.imul(..., 16777619)` folding, `>>> 0`);
 * - the seeded **xorshift32** stream every sim-side consumer must draw from
 *   (the `random:` source `createRuntimeAdaptationEngine` requires per §9.7.1);
 * - the five named derivation entries (`track`, `car-i network`,
 *   `adaptation-i`, `variant stream`, `rival-swap mutation stream`);
 * - the tuple's canonical serialized shape (static field order, sorted
 *   object keys, `null` for absent fields) plus a compact fingerprint;
 * - the packed-frame byte digest that formalizes the AC-026 parity claim.
 *
 * Determinism rung (D1/D3, reproducibility-contracts language):
 * because every sim-side stream derives deterministically from the tuple
 * seed via the named table, the seed IS the RNG-state component of the
 * tuple — no separate RNG-state capture is needed at this boundary. The
 * parity claim is **same-runtime byte-exact** (Level 3 within the tuple
 * replay boundary); it is NOT cross-environment exact (floating-point
 * accumulation may differ across Node/browser runtimes). Visual-only
 * surfaces ride the separate `NEATRACING_VISUAL_SEED`, which sits OUTSIDE
 * the frame contract and is exempt from byte parity (§9.7.2 exemption).
 */

import {
  NEATRACING_PACKED_MAX_CAR_CAPACITY,
  NEATRACING_PACKED_MIN_CARS,
  assertNeatRacingPackedFrameSchemaVersion,
  type NeatRacingPackedFrame,
} from './frame.pack.service';

/**
 * The byte-parity replay depth (TP-21, §9.7.1 item 4).
 *
 * AC-026 asserts byte-identical frames across AT LEAST this many adaptation
 * ticks so the adaptation-engine RNG path stays inside the replayed surface.
 * At the present boundary each released fixed-step sim tick IS the
 * adaptation-tick unit; the adaptation cadence (Step 03) consumes the same
 * constant without redefining it.
 */
export const NEATRACING_REPLAY_ADAPTATION_TICKS = 32;

/**
 * The visual-only seed base (TP-20, §9.7.2 exemption).
 *
 * Starfield layers, pulses, and scenery placement derive from this base.
 * They never touch sim state, so they sit OUTSIDE the frame contract and
 * are exempt from AC-026 byte parity — the frame channels are the parity
 * surface, not the visual-channel seeds.
 */
export const NEATRACING_VISUAL_SEED = 1337;

/**
 * Golden-ratio xor salt mixed into every child-seed derivation, mirroring
 * the recycled track generator's combined-seed pattern (`0x9e3779b9`).
 */
export const NEATRACING_CHILD_SEED_MIX_GOLDEN_RATIO = 0x9e3779b9;

/**
 * FNV-style folding multiplier applied to every character/byte of a mixed
 * value, mirroring the recycled track generator's `Math.imul(..., 16777619)`
 * folding. Also drives the byte-digest and tuple fingerprint folds so one
 * hashing family covers the whole reproducibility boundary.
 */
export const NEATRACING_CHILD_SEED_MIX_FNV_PRIME = 16777619;

/**
 * Zero-state escape hatch for the xorshift32 stream.
 *
 * Xorshift32 cannot advance from a zero state, so this non-zero constant
 * (the recycled track generator's zero-guard value) keeps a zero child seed
 * a valid, deterministic stream start instead of a dead stream.
 */
export const NEATRACING_XORSHIFT32_ZERO_SEED_FALLBACK = 0x6d2b79f5;

/**
 * FNV-1a offset basis starting every digest/fingerprint fold.
 */
export const NEATRACING_BYTE_DIGEST_FNV_OFFSET_BASIS = 0x811c9dc5;

/**
 * Recycled NGE per-variant seed offset (D2, `NEATRACING_VARIANT_PATCH_SEED_OFFSET`
 * base in `src/neat/nge-juvenile`): the minimum per-variant stream spacing.
 */
export const NEATRACING_VARIANT_PATCH_SEED_OFFSET = 1_000;

/**
 * Recycled NGE per-variant seed stride factor (D2): each variant patch slot
 * widens the stride by this factor.
 */
export const NEATRACING_VARIANT_PATCH_SEED_STRIDE_FACTOR = 4;

/**
 * Per-event stride for the rival-swap mutation stream (§9.7.2, E4).
 *
 * No recycled constant pins this stride; the value mirrors the NGE
 * variant base-slot spacing convention so every swap event owns a disjoint
 * mutation-stream slot. The rival propagation operator serializes the
 * swap-event seeds into `opponentSnapshot`, keeping them inside the tuple
 * contract.
 */
export const NEATRACING_RIVAL_SWAP_SEED_STRIDE = 1_000;

/**
 * Named §9.7.2 surface tags (the derivation vocabulary, single table).
 *
 * Every stochastic surface's tag lives here so a tag rename is one edit
 * and the derivation table stays contract-pinned.
 */
export const NEATRACING_TRACK_SURFACE_TAG = 'track';
/** Prefix tag for the per-car network seed: `car-<carIndex>`. */
export const NEATRACING_CAR_NETWORK_SURFACE_TAG_PREFIX = 'car-';
/** Prefix tag for the per-car adaptation seed: `adaptation-<carIndex>`. */
export const NEATRACING_ADAPTATION_SURFACE_TAG_PREFIX = 'adaptation-';
/** Tag for the shared variant stream base. */
export const NEATRACING_VARIANT_STREAM_SURFACE_TAG = 'variant';
/** Tag for the shared rival-swap mutation stream base. */
export const NEATRACING_RIVAL_SWAP_SURFACE_TAG = 'rival-swap';

/** A seeded xorshift32 draw function returning floats in `[0, 1)`. */
export type NeatRacingXorShift32Stream = () => number;

/**
 * The AC-026 reproducibility tuple carried by the sim-worker init payload.
 *
 * The field order of the interface mirrors the AC-026 wording; the canonical
 * serialized shape is pinned by `serializeNeatRacingReproducibilityTuple`.
 * `featureFlags` uses the worker payload's carried shape (an open boolean
 * record) and stays verbatim; `opponentSnapshot` is opaque at this boundary
 * (the NGE network-controller slice gives it its shape).
 */
export interface NeatRacingReproducibilityTuple {
  /** Deterministic run seed driving every per-car child seed. */
  seed: number;
  /** Fleet size (car 0 plus `agentCount - 1` agent cars). */
  agentCount: number;
  /** Packed-frame schema version string (the wire layout the tuple replays). */
  packSchemaVersion: string;
  /** Opponent snapshot carried verbatim (opaque until the NGE slice). */
  opponentSnapshot: unknown;
  /** Reproducibility track identifier (naming the tuple's owned geometry). */
  trackId: string;
  /** Feature flags carried verbatim (open boolean record; absent = undefined). */
  featureFlags: Readonly<Record<string, boolean>> | undefined;
}

/**
 * Folds one unsigned char/byte value into an FNV-1a state.
 *
 * Shared fold step for the child-seed mix (characters), the tuple
 * fingerprint (serialization characters), and the packed-frame digest
 * (channel bytes) — one hashing family across the boundary.
 *
 * @param digest - Current fold state (unsigned 32-bit).
 * @param value - One character code or byte to fold in.
 * @returns The advanced unsigned 32-bit fold state.
 */
function foldUint8IntoFnv1a(digest: number, value: number): number {
  return Math.imul(digest ^ value, NEATRACING_CHILD_SEED_MIX_FNV_PRIME);
}

/**
 * Folds a whole string (UTF-16 code units, in order) into an FNV-1a state.
 *
 * @param digest - Current fold state (unsigned 32-bit).
 * @param text - The string to fold in, character by character.
 * @returns The advanced unsigned 32-bit fold state.
 */
function foldTextIntoFnv1a(digest: number, text: string): number {
  let folded = digest;
  for (let index = 0; index < text.length; index++) {
    folded = foldUint8IntoFnv1a(folded, text.charCodeAt(index));
  }
  return folded;
}

/**
 * Folds a typed array's raw bytes into an FNV-1a state.
 *
 * @param digest - Current fold state (unsigned 32-bit).
 * @param view - Typed array whose bytes fold in, in order.
 * @returns The advanced unsigned 32-bit fold state.
 */
function foldBytesIntoFnv1a(digest: number, view: ArrayBufferView): number {
  const bytes = new Uint8Array(
    view.buffer as ArrayBuffer,
    view.byteOffset,
    view.byteLength,
  );
  let folded = digest;
  for (let index = 0; index < bytes.length; index++) {
    folded = foldUint8IntoFnv1a(folded, bytes[index] as number);
  }
  return folded;
}

/**
 * Throws when a value is not an unsigned-safe non-negative integer.
 *
 * @param value - Candidate index/stride value.
 * @param fieldName - Static field name for the rejection message.
 * @throws {RangeError} When the value is not a non-negative integer.
 */
function assertNonNegativeInteger(value: number, fieldName: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(
      `neatRacing reproducibility derivation failed: ${fieldName} must be a non-negative integer`,
    );
  }
}

/**
 * Combines the tuple seed and a named surface tag into one 32-bit child
 * seed (the §9.7.2 mix).
 *
 * Mirrors the recycled track generator's combined-seed hash pattern: a
 * golden-ratio xor salts the tuple seed, then each tag character folds in
 * through `Math.imul(state ^ charCode, 16777619)` before the unsigned
 * `>>> 0` normalization. Same inputs always return the same unsigned value
 * on the same runtime (and on any IEEE 754 engine — integer-only math).
 *
 * @param tupleSeed - The AC-026 tuple seed (any finite number; normalized
 *   through the 32-bit fold chain).
 * @param surfaceTag - Named surface tag from the §9.7.2 table vocabulary.
 * @returns The unsigned 32-bit child seed.
 * @throws {RangeError} When `tupleSeed` is not a finite number or
 *   `surfaceTag` is not a non-empty string.
 *
 * @example
 * ```ts
 * const childSeed = combineNeatRacingChildSeed(tuple.seed, 'adaptation-2');
 * const stream = createNeatRacingXorShift32Stream(childSeed);
 * ```
 */
export function combineNeatRacingChildSeed(
  tupleSeed: number,
  surfaceTag: string,
): number {
  // Step 1: Validate the mix inputs (loud, static rejections).
  if (typeof tupleSeed !== 'number' || !Number.isFinite(tupleSeed)) {
    throw new RangeError(
      'neatRacing reproducibility derivation failed: tupleSeed must be a finite number',
    );
  }
  if (typeof surfaceTag !== 'string' || surfaceTag.length === 0) {
    throw new RangeError(
      'neatRacing reproducibility derivation failed: surfaceTag must be a non-empty string',
    );
  }

  // Step 2: Mirror the recycled combined-seed pattern (golden xor + FNV fold).
  let combinedSeed = tupleSeed ^ NEATRACING_CHILD_SEED_MIX_GOLDEN_RATIO;
  combinedSeed = foldTextIntoFnv1a(combinedSeed, surfaceTag);
  return combinedSeed >>> 0;
}

/**
 * Creates a seeded xorshift32 stream (the D1 sim-side RNG source).
 *
 * The stream is the ONLY permitted ambient-free random source sim-side
 * (§9.7.1): consumers pass it as `createRuntimeAdaptationEngine({ random })`
 * and draw car-network RNG through their derived seeds, so no sim-side code
 * path ever touches `Math.random`. The transition mirrors the recycled
 * track generator's xorshift32 (`<< 13`, `>>> 17`, `<< 5`); the zero-state
 * guard substitutes the named fallback so a zero seed stays deterministic.
 *
 * @param childSeed - Unsigned-safe child seed (a §9.7.2 derivation result).
 * @returns A draw function returning floats in `[0, 1)`; identical seeds
 *   produce identical sequences on the same runtime.
 * @throws {RangeError} When `childSeed` is not a finite number.
 *
 * @example
 * ```ts
 * const random = createNeatRacingXorShift32Stream(
 *   deriveNeatRacingAdaptationChildSeed(tuple.seed, carIndex),
 * );
 * createRuntimeAdaptationEngine({ random });
 * ```
 */
export function createNeatRacingXorShift32Stream(
  childSeed: number,
): NeatRacingXorShift32Stream {
  if (typeof childSeed !== 'number' || !Number.isFinite(childSeed)) {
    throw new RangeError(
      'neatRacing xorshift32 stream failed: childSeed must be a finite number',
    );
  }
  let state =
    childSeed === 0
      ? NEATRACING_XORSHIFT32_ZERO_SEED_FALLBACK
      : childSeed >>> 0;

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

/**
 * Derives the `track` surface child seed (§9.7.2 row 1).
 *
 * @param tupleSeed - The AC-026 tuple seed.
 * @returns The unsigned 32-bit child seed for the track generator.
 */
export function deriveNeatRacingTrackChildSeed(tupleSeed: number): number {
  return combineNeatRacingChildSeed(tupleSeed, NEATRACING_TRACK_SURFACE_TAG);
}

/**
 * Derives the `car-<carIndex>` network child seed (§9.7.2 row 2).
 *
 * Consumers `setSeed()` every car network with this seed (the recycled
 * coevolution precedent), keeping network-internal draws inside the tuple.
 *
 * @param tupleSeed - The AC-026 tuple seed.
 * @param carIndex - Fleet car index (non-negative integer).
 * @returns The unsigned 32-bit child seed for the car's network.
 * @throws {RangeError} When `carIndex` is not a non-negative integer.
 */
export function deriveNeatRacingCarNetworkChildSeed(
  tupleSeed: number,
  carIndex: number,
): number {
  assertNonNegativeInteger(carIndex, 'carIndex');
  return combineNeatRacingChildSeed(
    tupleSeed,
    `${NEATRACING_CAR_NETWORK_SURFACE_TAG_PREFIX}${carIndex}`,
  );
}

/**
 * Derives the `adaptation-<carIndex>` child seed (§9.7.2 row 3).
 *
 * Consumers build each car's runtime adaptation engine with an explicit
 * seeded stream from this seed (`createNeatRacingXorShift32Stream`), so the
 * adaptation-RNG path stays inside the replayed surface (D1 item 4).
 *
 * @param tupleSeed - The AC-026 tuple seed.
 * @param carIndex - Fleet car index (non-negative integer).
 * @returns The unsigned 32-bit child seed for the car's adaptation engine.
 * @throws {RangeError} When `carIndex` is not a non-negative integer.
 */
export function deriveNeatRacingAdaptationChildSeed(
  tupleSeed: number,
  carIndex: number,
): number {
  assertNonNegativeInteger(carIndex, 'carIndex');
  return combineNeatRacingChildSeed(
    tupleSeed,
    `${NEATRACING_ADAPTATION_SURFACE_TAG_PREFIX}${carIndex}`,
  );
}

/**
 * Derives the `variant` stream child seed for one variant slot (§9.7.2 row 4).
 *
 * Mirrors the recycled NGE composition (`base + variantIndex × stride`,
 * stride = `max(NGE_VARIANT_PATCH_SEED_OFFSET, max(1, patchSize) ×
 * NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR)`); the eval-worker slice passes the
 * known patch size, while the default (`0`) falls back to the recycled
 * offset base.
 *
 * @param tupleSeed - The AC-026 tuple seed.
 * @param variantIndex - Variant slot index (non-negative integer).
 * @param patchSize - Patch size the slot evaluates (defaults to 0 when the
 *   slot's patch size is not yet known).
 * @returns The unsigned 32-bit seed for the variant's RNG stream.
 * @throws {RangeError} When `variantIndex` or `patchSize` is not a
 *   non-negative integer.
 */
export function deriveNeatRacingVariantStreamChildSeed(
  tupleSeed: number,
  variantIndex: number,
  patchSize = 0,
): number {
  assertNonNegativeInteger(variantIndex, 'variantIndex');
  assertNonNegativeInteger(patchSize, 'patchSize');
  const streamBase = combineNeatRacingChildSeed(
    tupleSeed,
    NEATRACING_VARIANT_STREAM_SURFACE_TAG,
  );
  const perVariantSeedStride = Math.max(
    NEATRACING_VARIANT_PATCH_SEED_OFFSET,
    Math.max(1, patchSize) * NEATRACING_VARIANT_PATCH_SEED_STRIDE_FACTOR,
  );
  return (streamBase + variantIndex * perVariantSeedStride) >>> 0;
}

/**
 * Derives the `rival-swap` mutation-stream child seed for one swap event
 * (§9.7.2 row 5, E4).
 *
 * Each swap event draws a disjoint mutation-stream slot (named
 * `NEATRACING_RIVAL_SWAP_SEED_STRIDE` spacing) and serializes its seed
 * into `opponentSnapshot`, so the reproducibility tuple replays swap
 * events exactly.
 *
 * @param tupleSeed - The AC-026 tuple seed.
 * @param eventIndex - Swap-event index (non-negative integer).
 * @returns The unsigned 32-bit seed for the event's mutation stream.
 * @throws {RangeError} When `eventIndex` is not a non-negative integer.
 */
export function deriveNeatRacingRivalSwapChildSeed(
  tupleSeed: number,
  eventIndex: number,
): number {
  assertNonNegativeInteger(eventIndex, 'eventIndex');
  const streamBase = combineNeatRacingChildSeed(
    tupleSeed,
    NEATRACING_RIVAL_SWAP_SURFACE_TAG,
  );
  return (streamBase + eventIndex * NEATRACING_RIVAL_SWAP_SEED_STRIDE) >>> 0;
}

/**
 * Throws when a value is not JSON-serializable material for the tuple.
 *
 * @param value - The rejected value (for the static message surface).
 * @throws {RangeError} Always — one static rejection message.
 */
function throwNotJsonSerializable(value: unknown): never {
  throw new RangeError(
    `neatRacing reproducibility tuple serialization failed: value must be JSON-serializable (received ${typeof value})`,
  );
}

/**
 * Canonicalizes one JSON-safe scalar (`undefined`/`null` collapse to `null`,
 * booleans/strings pass, non-finite numbers fail loudly).
 *
 * @param value - The scalar to canonicalize.
 * @returns The canonical scalar value.
 * @throws {RangeError} When the value is a non-finite number or is not
 *   JSON-serializable material.
 */
function canonicalizeReproducibilityScalar(value: unknown): unknown {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === 'boolean' || typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new RangeError(
        'neatRacing reproducibility tuple serialization failed: numbers must be finite',
      );
    }
    return value;
  }
  throwNotJsonSerializable(value);
}

/**
 * Reports whether a value behaves as a plain JSON object (prototype is the
 * object prototype or null — built-ins like Date/Map fail).
 *
 * @param value - The candidate value.
 * @returns True when the value is a plain object usable in JSON.
 */
function isPlainJsonObject(value: unknown): value is Record<string, unknown> {
  // Step 1: Reject every non-object (and null) before introspecting the
  // prototype chain — `Object.getPrototypeOf(null)` throws.
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

/**
 * Canonicalizes one plain object into key-sorted order (undefined values
 * drop, matching JSON semantics), so reconstruction order can never change
 * the serialized shape.
 *
 * @param value - The plain object to canonicalize.
 * @returns The canonical key-sorted object.
 * @throws {RangeError} When a property value is not JSON-serializable.
 */
function canonicalizeReproducibilityObject(
  value: Record<string, unknown>,
): Record<string, unknown> {
  const canonical: Record<string, unknown> = {};
  const keys = Object.keys(value).toSorted();
  for (const key of keys) {
    const canonicalValue = canonicalizeReproducibilityJsonValue(value[key]);
    if (canonicalValue === undefined) {
      continue;
    }
    canonical[key] = canonicalValue;
  }
  return canonical;
}

/**
 * Canonicalizes an arbitrary value into a JSON-safe, key-stable form.
 *
 * Arrays stay in order (they carry positional semantics), plain objects
 * rebuild with sorted keys, and anything outside JSON material (functions,
 * symbols, BigInts, class instances) fails loudly instead of silently
 * collapsing the serialized shape.
 *
 * @param value - The value to canonicalize.
 * @returns The canonical JSON-safe value.
 * @throws {RangeError} When the value is not JSON-serializable.
 */
function canonicalizeReproducibilityJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((entry) => canonicalizeReproducibilityJsonValue(entry));
  }
  if (isPlainJsonObject(value)) {
    return canonicalizeReproducibilityObject(value);
  }
  return canonicalizeReproducibilityScalar(value);
}

/**
 * Validates the numeric identity fields of a tuple (seed finiteness and the
 * agentCount fleet bounds).
 *
 * Reuses the frame packer's named fleet-range constants because the tuple
 * replays exactly the packed fleet size those constants bound.
 *
 * @param tuple - The tuple whose numeric fields validate.
 * @throws {RangeError} When `seed` is not a finite number or `agentCount`
 *   is outside the packed fleet bounds.
 */
function assertTupleNumericFields(tuple: NeatRacingReproducibilityTuple): void {
  if (typeof tuple.seed !== 'number' || !Number.isFinite(tuple.seed)) {
    throw new RangeError(
      'neatRacing reproducibility tuple serialization failed: seed must be a finite number',
    );
  }
  if (
    typeof tuple.agentCount !== 'number' ||
    !Number.isInteger(tuple.agentCount) ||
    tuple.agentCount < NEATRACING_PACKED_MIN_CARS ||
    tuple.agentCount > NEATRACING_PACKED_MAX_CAR_CAPACITY
  ) {
    throw new RangeError(
      `neatRacing reproducibility tuple serialization failed: agentCount must be an integer in [${NEATRACING_PACKED_MIN_CARS}, ${NEATRACING_PACKED_MAX_CAR_CAPACITY}]`,
    );
  }
}

/**
 * Validates the string identity fields of a tuple (both must be non-empty).
 *
 * @param tuple - The tuple whose string fields validate.
 * @throws {RangeError} When `packSchemaVersion` or `trackId` is not a
 *   non-empty string.
 */
function assertTupleStringFields(tuple: NeatRacingReproducibilityTuple): void {
  for (const [fieldName, fieldValue] of [
    ['packSchemaVersion', tuple.packSchemaVersion],
    ['trackId', tuple.trackId],
  ] as const) {
    if (typeof fieldValue !== 'string' || fieldValue.length === 0) {
      throw new RangeError(
        `neatRacing reproducibility tuple serialization failed: ${fieldName} must be a non-empty string`,
      );
    }
  }
}

/**
 * Serializes the tuple into its canonical string shape (the replay
 * identity anchor).
 *
 * The top-level field order is PINNED to the AC-026 ordering
 * (`seed, agentCount, packSchemaVersion, opponentSnapshot, trackId,
 * featureFlags`), nested objects rebuild key-sorted (insertion order can
 * never change the serial), and absent fields serialize as explicit `null`
 * so the shape is unambiguous at the boundary.
 *
 * @param tuple - The tuple to serialize.
 * @returns The canonical JSON serial of the tuple.
 * @throws {RangeError} When a tuple field violates its contract or any
 *   carried value is not JSON-serializable.
 *
 * @example
 * ```ts
 * const canonical = serializeNeatRacingReproducibilityTuple(tuple);
 * expect(canonical === serializeNeatRacingReproducibilityTuple(tuple)).toBe(true);
 * ```
 */
export function serializeNeatRacingReproducibilityTuple(
  tuple: NeatRacingReproducibilityTuple,
): string {
  // Step 1: Validate the tuple's own identity fields.
  assertTupleNumericFields(tuple);
  assertTupleStringFields(tuple);

  // Step 2: Build the PINNED field-order literal and canonicalize values.
  return JSON.stringify({
    seed: tuple.seed,
    agentCount: tuple.agentCount,
    packSchemaVersion: tuple.packSchemaVersion,
    opponentSnapshot: canonicalizeReproducibilityJsonValue(
      tuple.opponentSnapshot,
    ),
    trackId: tuple.trackId,
    featureFlags: canonicalizeReproducibilityJsonValue(
      tuple.featureFlags ?? null,
    ),
  });
}

/**
 * Folds the tuple's canonical serial into a compact unsigned 32-bit
 * fingerprint.
 *
 * A convenience identity for logging/dedupe later slices; the canonical
 * STRING (not the fingerprint) stays the durable identity anchor. The fold
 * is the same FNV-1a family as every other digest here.
 *
 * @param tuple - The tuple to fingerprint.
 * @returns The unsigned 32-bit fingerprint.
 * @throws {RangeError} When the tuple violates its serialization contract.
 *
 * @example
 * ```ts
 * expect(computeNeatRacingReproducibilityTupleFingerprint(tupleA)).toBe(
 *   computeNeatRacingReproducibilityTupleFingerprint(tupleB),
 * ); // same tuple → same fingerprint
 * ```
 */
export function computeNeatRacingReproducibilityTupleFingerprint(
  tuple: NeatRacingReproducibilityTuple,
): number {
  const serialized = serializeNeatRacingReproducibilityTuple(tuple);
  return (
    foldTextIntoFnv1a(NEATRACING_BYTE_DIGEST_FNV_OFFSET_BASIS, serialized) >>> 0
  );
}

/**
 * Folds a packed frame's header scalars into the byte digest.
 *
 * The header scalar order is PINNED (fromTick, toTick, agentCount,
 * featureFlags, level, lapsCompleted, raceStatusCode, nextRaceLevel) with
 * `;` separators so no concatenation ambiguity can shift fold boundaries.
 *
 * @param digest - Current fold state.
 * @param frame - The packed frame whose header folds in.
 * @returns The advanced fold state.
 * @throws {RangeError} When a header scalar is not a finite number.
 */
function foldPackedFrameHeaderIntoDigest(
  digest: number,
  frame: NeatRacingPackedFrame,
): number {
  const headerScalars = [
    frame.fromTick,
    frame.toTick,
    frame.agentCount,
    frame.featureFlags,
    frame.level,
    frame.lapsCompleted,
    frame.raceStatusCode,
    frame.nextRaceLevel,
  ];
  let folded = foldTextIntoFnv1a(digest, frame.schemaVersion);
  for (const scalar of headerScalars) {
    if (typeof scalar !== 'number' || !Number.isFinite(scalar)) {
      throw new RangeError(
        'neatRacing packed frame digest failed: header scalars must be finite numbers',
      );
    }
    folded = foldTextIntoFnv1a(folded, `;${scalar}`);
  }
  return folded;
}

/**
 * Validates one packed-frame channel view and folds its raw bytes into the
 * digest.
 *
 * @param digest - Current fold state.
 * @param channelName - Static channel name for the rejection message.
 * @param view - The channel view to walk.
 * @returns The advanced fold state.
 * @throws {RangeError} When the channel is not a Float32Array view.
 */
function foldPackedFrameChannelIntoDigest(
  digest: number,
  channelName: string,
  view: unknown,
): number {
  if (!(view instanceof Float32Array)) {
    throw new RangeError(
      `neatRacing packed frame digest failed: channel ${channelName} must be a Float32Array view`,
    );
  }
  return foldBytesIntoFnv1a(digest, view);
}

/**
 * Folds every packed-frame channel view (rules span first, then the fixed
 * car SoA channel order) into the byte digest.
 *
 * The view list mirrors the packer's fixed channel table declaration order;
 * a channel added to the wire shape must extend this walk in the same step.
 *
 * @param digest - Current fold state.
 * @param frame - The packed frame whose channels fold in.
 * @returns The advanced fold state.
 * @throws {RangeError} When a channel is not a Float32Array view.
 */
function foldPackedFrameChannelsIntoDigest(
  digest: number,
  frame: NeatRacingPackedFrame,
): number {
  let folded = foldPackedFrameChannelIntoDigest(
    digest,
    'rulesChannel',
    frame.rulesChannel,
  );
  const carChannelViews = [
    frame.carX,
    frame.carY,
    frame.carHeading,
    frame.carSpeed,
    frame.carYawRate,
    frame.carSlipAngle,
    frame.carTravelHeadingRad,
    frame.carTractionSeverity,
    frame.carLateralVelocity,
    frame.carDriftCommittedSlip,
    frame.carDriftCommittedTurnSign,
    frame.carDriftCommittedReArmed,
    frame.carDriftPosePose,
    frame.carDriftPoseRate,
    frame.carDriftPoseArticulation,
    frame.carTick,
  ];
  const carChannelNames = [
    'carX',
    'carY',
    'carHeading',
    'carSpeed',
    'carYawRate',
    'carSlipAngle',
    'carTravelHeadingRad',
    'carTractionSeverity',
    'carLateralVelocity',
    'carDriftCommittedSlip',
    'carDriftCommittedTurnSign',
    'carDriftCommittedReArmed',
    'carDriftPosePose',
    'carDriftPoseRate',
    'carDriftPoseArticulation',
    'carTick',
  ];
  for (let index = 0; index < carChannelViews.length; index++) {
    folded = foldPackedFrameChannelIntoDigest(
      folded,
      carChannelNames[index] as string,
      carChannelViews[index],
    );
  }
  return folded;
}

/**
 * Computes the byte-parity witness digest of one packed frame (AC-026).
 *
 * Two frames with EQUAL digests are treated as byte-identical by the parity
 * suite: the fold walks the pinned header scalars, then every channel's RAW
 * BYTES (Float32 storage bytes, not rounded values), so any byte-level
 * drift in the packed wire contract flips the digest. The digest is a
 * 32-bit FNV-1a witness, not a cryptographic identity — a collision would
 * surface as a false PARITY, never a false mismatch. The schema sentinel is
 * guarded first so misread bytes fail loudly at the boundary.
 *
 * @param frame - The packed frame to digest.
 * @returns An 8-character lowercase hex digest string.
 * @throws {RangeError} When the schema sentinel, header scalars, or channel
 *   views violate the packed frame contract.
 *
 * @example
 * ```ts
 * const digestA = computeNeatRacingPackedFrameByteDigest(packFrameA);
 * const digestB = computeNeatRacingPackedFrameByteDigest(packFrameB);
 * expect(digestB).toBe(digestA); // AC-026 byte parity
 * ```
 */
export function computeNeatRacingPackedFrameByteDigest(
  frame: NeatRacingPackedFrame,
): string {
  // Step 1: Guard the schema sentinel before reading any bytes.
  assertNeatRacingPackedFrameSchemaVersion(frame);

  // Step 2: Fold the header, then every channel, in the pinned order.
  const headerFolded = foldPackedFrameHeaderIntoDigest(
    NEATRACING_BYTE_DIGEST_FNV_OFFSET_BASIS,
    frame,
  );
  const channelsFolded = foldPackedFrameChannelsIntoDigest(headerFolded, frame);
  return (channelsFolded >>> 0).toString(16).padStart(8, '0');
}
