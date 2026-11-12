/**
 * NeatRacing rival propagation (slice `P2-S3-D`, AC-032).
 *
 * THE LAW (spec section 9.6, E4): when a rival crosses the finish line, the
 * OTHER rivals' minds are replaced by WEIGHT-NUDGE MUTATIONS of the
 * crossing rival's serialized weights — one bounded MOD_WEIGHT round with a
 * SMALL delta range, seeded per swap event. The operator is a PURE
 * SERIALIZED-WEIGHTS PLANE:
 *
 * - RECYCLED COEVOLUTION PROTOCOL (the racing curriculum's
 *   `createCarGenome`): the mind is `CarGenome.serialize()`'s Float32Array
 *   of connection weights, and the mutation mirrors the recycled MOD_WEIGHT
 *   semantics — per weight, one uniform delta sample added
 *   (`min + u * (max - min)`), here over the SMALL named range
 *   `[NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN, NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX]`.
 *   The src MOD_WEIGHT default range (plus/minus 1) is intentionally NOT
 *   reused: rival-to-rival propagation is a NUDGE (a TP tuning point, not
 *   a full re-roll), and the mutation RNG is the seeded xorshift32 stream
 *   (section 9.7.1), never `Math.random`.
 * - STRUCTURAL GROWTH STAYS WITH EACH RIVAL'S OWN ADAPTATION ENGINE: the
 *   operator touches WEIGHTS ONLY — its input and output are flat Float32
 *   arrays, it never imports `Network`, and it never adds, removes or
 *   rewires anything. NGE growth (nodes, topology) remains the per-rival
 *   adaptation engine's lane; this operator only re-seeds the weights.
 * - LAP/LINE-BOUNDARY SWAPS ONLY: the operator refuses every call whose
 *   `atLapLineBoundary` flag is not `true` — a mid-lap swap is a
 *   programming error, never a silent one (spec E4: swaps are applied at
 *   LAP/line boundaries, never mid-lap).
 * - THE MAIN AGENT IS STRUCTURALLY OUT OF REACH (the Blocked stop
 *   condition is a GUARD, not a promise): car 0 is the main agent's slot at
 *   EVERY level team layout (`[0]` / `[0, 1]` / `[0, 1, 1, 1, 1]` — AC-033
 *   "human takes car 0"), and the operator REJECTS any rival index at or
 *   below `NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX`. Replacement minds
 *   are derived ONLY from the rival set minus the crossing rival, so no
 *   code path can return a main-agent mind. The main agent is NEVER
 *   touched and NEVER fully reset by rival success.
 * - LVL-2 IS A DOCUMENTED NO-OP: a single-rival session has nobody to
 *   propagate to — the crossing rival keeps its mind, the result carries
 *   ZERO replacements, and the swap payload records an empty propagated
 *   list. The lvl-3 case (four rivals) propagates to the THREE others.
 *
 * THE SWAP PAYLOAD IS THE SECTION 9.7 REPLAY ANCHOR: the crossing-network
 * snapshot (carIndex + serialized weights) and the event's mutation RNG
 * seed are serialized into the reproducibility tuple's `opponentSnapshot`
 * AT EACH SWAP, so the tuple replays swap events exactly. The payload shape
 * is JSON-SAFE BY CONSTRUCTION (hex weights, plain numbers, arrays) — the
 * one shape the base tuple canonicalizer digests — and this module owns the
 * canonicalization for its own NEW payload: byte-exact hex, key-sorted
 * canonical serial (the D4 law from `promotion.rules.ts`, never a
 * `toJSON`). The replay rung is ORDERED-DETERMINISTIC LEVEL 2
 * (same-runtime): given the same delta-range constants and the same
 * underlying float32 math, the payload alone rebuilds the post-swap minds
 * (`restoreNeatRacingRivalSwapReplacementMinds`), with the draw order
 * pinned — targets ascending by car index, then weight index ascending.
 *
 * REALM LAW: pure/headless — no ambient RNG, no wall clock, no DOM. The
 * only random source is the seeded xorshift32 stream.
 *
 * @module rival.propagation
 */

import {
  createNeatRacingXorShift32Stream,
  deriveNeatRacingRivalSwapChildSeed,
  type NeatRacingXorShift32Stream,
} from '../workers/simulation-worker/reproducibility.utils';

/**
 * The rival-swap nudge delta range bounds (E4: "one bounded MOD_WEIGHT
 * round" with a SMALL delta range — a TP tuning point, NOT the library
 * default).
 *
 * The recycled coevolution protocol mutates with the MOD_WEIGHT method's
 * `{ min, max }` shape; the src default (plus/minus 1) is a full-weight-
 * magnitude rewrite, wrong for rival-to-rival propagation where the
 * winner's mind is a starting point to be NUDGED, not re-rolled. The
 * symmetric range keeps every swap a bounded refinement. The sample mirrors
 * the method's contract: `min + u * (max - min)` with `u` drawn from the
 * event's seeded stream.
 */
export const NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN = -0.25;

/**
 * The rival-swap nudge delta range upper bound (see the lower bound — the
 * range is symmetric around zero).
 */
export const NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX = 0.25;

/**
 * The main agent's fleet car index (car 0 — AC-033; every level team layout
 * names it team 0's slot: `NEATRACING_TEAM_LAYOUTS` = `[0]` / `[0, 1]` /
 * `[0, 1, 1, 1, 1]`). The propagation operator never owns this slot: rival
 * car indices must be STRICTLY greater, so no replacement mind and no
 * crossing anchor can ever address the main agent. The Blocked stop
 * condition ("propagation touches the main agent state") is enforced in
 * code, not in prose.
 */
export const NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX = 0;

/**
 * The swap payload's kind discriminator inside `opponentSnapshot`. The
 * reproducibility tuple's opponent snapshot is opaque at that boundary, so
 * the payload self-describes: a payload whose `kind` is not
 * `NEATRACING_RIVAL_SWAP_PAYLOAD_KIND` is rejected (no silent nulls).
 */
export const NEATRACING_RIVAL_SWAP_PAYLOAD_KIND = 'rival-swap';

/** The error-voice prefix for the propagation operator (E4 lane). */
const PROPAGATION_ERROR_PREFIX = 'neatRacing rival propagation failed';

/** The error-voice prefix for the swap payload (the tuple-record lane). */
const PAYLOAD_ERROR_PREFIX = 'neatRacing rival swap payload failed';

/** The unsigned 32-bit seed ceiling (the derivation table's `>>> 0` shape). */
const UNSIGNED_32_BIT_SEED_MAX = 0xffffffff;

/**
 * A possibly-malformed record handed to the shape guards (the base tuple
 * carries `opponentSnapshot` as `unknown` at the boundary). Every field is
 * validated at runtime instead of trusted — the guard view types carry
 * nothing but the object-ness itself.
 */
type SwapRecordGuardView = {
  eventIndex?: unknown;
  kind?: unknown;
  mutationSeed?: unknown;
  crossing?: { carIndex?: unknown; weights?: unknown } | undefined;
  propagatedCarIndices?: unknown;
};

/**
 * One replacement rival mind: the target rival's fleet car index plus its
 * post-swap serialized weights (the crossing mind with the bounded nudges
 * applied). The runtime applies the weights to the rival's network through
 * the Step 05 host wiring; this module stays on the flat serialized plane.
 */
export interface NeatRacingRivalSwapReplacementMind {
  /** The replaced rival's fleet car index (never the main agent's slot). */
  readonly carIndex: number;
  /** The post-swap serialized weights (a NEW array; never aliased input). */
  readonly weights: Float32Array;
}

/**
 * The crossing-network snapshot: WHICH rival crossed and the mind it
 * crossed with. The snapshot owns its bytes — the weights are copied from
 * the caller's array so later runtime mutation can never change them.
 */
export interface NeatRacingRivalSwapCrossingSnapshot {
  /** The crossing rival's fleet car index. */
  readonly carIndex: number;
  /** The crossing rival's serialized weights (the propagation base). */
  readonly weights: Float32Array;
}

/**
 * The runtime-form swap snapshot for one rival swap event (the typed twin
 * of the tuple payload). Converted into the JSON-safe
 * `NeatRacingRivalSwapPayload` for the reproducibility tuple's
 * `opponentSnapshot` record.
 */
export interface NeatRacingRivalSwapSnapshot {
  /** The swap-event index (the per-event stride slot in the swap stream). */
  readonly eventIndex: number;
  /** The event's derived mutation seed (the recorded RNG-state component). */
  readonly mutationSeed: number;
  /** The crossing network snapshot (who crossed, and with which weights). */
  readonly crossing: NeatRacingRivalSwapCrossingSnapshot;
  /** The replaced rivals' car indices, ascending (the crossing one absent). */
  readonly propagatedCarIndices: readonly number[];
}

/**
 * The tuple-carried swap record — the JSON-SAFE shape written into the AC-026
 * reproducibility tuple's `opponentSnapshot` at each swap (base-tuple
 * canonicalizer parity: plain objects, finite numbers, hex strings — no
 * typed arrays, whose canonical position in the tuple is the D4
 * `seedNetwork.weights` field only).
 */
export interface NeatRacingRivalSwapPayload {
  /** The self-describing kind discriminator (`NEATRACING_RIVAL_SWAP_PAYLOAD_KIND`). */
  readonly kind: typeof NEATRACING_RIVAL_SWAP_PAYLOAD_KIND;
  /** The swap-event index (the per-event stride slot in the swap stream). */
  readonly eventIndex: number;
  /** The event's derived mutation seed (the recorded RNG-state component). */
  readonly mutationSeed: number;
  /** The crossing network snapshot (index + byte-exact hex weights). */
  readonly crossing: {
    /** The crossing rival's fleet car index. */
    readonly carIndex: number;
    /**
     * The crossing weights encoded as byte-exact lowercase hex (NATIVE
     * byte order — same-runtime Level 2 byte parity; see
     * `encodeFloat32ArrayAsHex`).
     */
    readonly weights: string;
  };
  /** The replaced rivals' car indices, ascending (the crossing one absent). */
  readonly propagatedCarIndices: readonly number[];
}

/** The operator call contract for one lap/line-boundary rival swap. */
export interface NeatRacingRivalPropagationOptions {
  /** The session's rival car indices (positive integers, unique). */
  readonly rivalCarIndices: readonly number[];
  /** The crossing rival's car index (must be one of the rivals). */
  readonly crossingCarIndex: number;
  /** The crossing rival's serialized weights (the propagation base). */
  readonly winnerWeights: Float32Array;
  /** The AC-026 tuple seed (the rival-swap row of the derivation table). */
  readonly tupleSeed: number;
  /** The swap-event index (each rival crossing consumes the next slot). */
  readonly eventIndex: number;
  /** True ONLY at a forward finish-line crossing; a mid-lap call throws. */
  readonly atLapLineBoundary: boolean;
}

/** The operator result: the runtime snapshot plus the replacement minds. */
export interface NeatRacingRivalPropagationResult {
  /** The swap snapshot (typed arrays) — convert via `toNeatRacingRivalSwapPayload`. */
  readonly snapshot: NeatRacingRivalSwapSnapshot;
  /** One replacement mind per non-crossing rival, in ascending car order. */
  readonly replacements: readonly NeatRacingRivalSwapReplacementMind[];
}

/**
 * Replaces every NON-crossing rival's mind with a bounded seeded
 * weight-nudge mutation of the crossing rival's serialized weights.
 *
 * THE E4 OPERATOR: at a lap/line boundary a rival crossed the finish line
 * (the crossing rival). Its serialized mind — the recycled
 * `CarGenome.serialize()` Float32Array — becomes the base for ALL other
 * rivals: each target gets a NEW weights array holding the crossing weights
 * with ONE bounded MOD_WEIGHT-style round applied (per weight, one uniform
 * delta from the event's seeded stream, drawn in a pinned order: targets
 * ascending by car index, then weight index ascending). Structural growth
 * is untouched by construction — the plane is flat weights only.
 *
 * The event's mutation seed derives from the section 9.7.2 rival-swap row
 * (`deriveNeatRacingRivalSwapChildSeed` — the named per-event stride keeps
 * every swap event's stream disjoint), and the result carries the snapshot
 * needed to serialize the seed + crossing weights into `opponentSnapshot`
 * so the reproducibility tuple replays the swap exactly.
 *
 * @param options - The swap call: the session's rival car indices (never a
 *   main-agent index), the crossing rival's car index, the crossing
 *   rival's serialized weights, the AC-026 tuple seed, the swap-event
 *   index, and the LAP/line-boundary flag (a mid-lap call is rejected).
 * @returns The swap result: the runtime snapshot plus one replacement mind
 *   per non-crossing rival in ascending car order. The lvl-2 no-op (a
 *   single rival) returns ZERO replacements and an empty propagated list.
 * @throws {RangeError} When `atLapLineBoundary` is not `true`, the rival
 *   set is empty or carries duplicate, fractional or main-agent indices,
 *   the crossing index is not one of the rivals, the crossing weights are
 *   empty or non-finite, or the tuple-seed derivation rejects its inputs.
 *
 * @example
 * ```ts
 * const { snapshot, replacements } = propagateNeatRacingRivalWinnerWeights({
 *   rivalCarIndices: [1, 2, 3, 4],
 *   crossingCarIndex: 2,
 *   winnerWeights: crossing.carGenome.serialize(),
 *   tupleSeed: tuple.seed,
 *   eventIndex: nextRivalSwapEventIndex(session),
 *   atLapLineBoundary: true,
 * });
 * ```
 */
export function propagateNeatRacingRivalWinnerWeights(
  options: NeatRacingRivalPropagationOptions,
): NeatRacingRivalPropagationResult {
  // Step 1: Guard the LAP/line boundary timing (a mid-lap call is a
  // programming error, never a silent swap).
  if (options.atLapLineBoundary !== true) {
    throw new RangeError(
      `${PROPAGATION_ERROR_PREFIX}: swaps are applied at lap/line boundaries only (forward finish-line crossings), never mid-lap`,
    );
  }

  // Step 2: Guard the rivalry set (at least one rival; positive integers; unique).
  const rivalCarIndices = [...options.rivalCarIndices];
  if (rivalCarIndices.length === 0) {
    throw new RangeError(
      `${PROPAGATION_ERROR_PREFIX}: the rivalry needs at least one rival car index`,
    );
  }
  for (const carIndex of rivalCarIndices) {
    if (!Number.isInteger(carIndex)) {
      throw new RangeError(
        `${PROPAGATION_ERROR_PREFIX}: rival car indices must be integers`,
      );
    }
    if (carIndex <= NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX) {
      throw new RangeError(
        `${PROPAGATION_ERROR_PREFIX}: rival car indices start after car 0 — the main agent's slot (team layouts, AC-033)`,
      );
    }
  }
  if (new Set(rivalCarIndices).size !== rivalCarIndices.length) {
    throw new RangeError(
      `${PROPAGATION_ERROR_PREFIX}: rival car indices must be unique`,
    );
  }

  // Step 3: Anchor the crossing rival inside the rival set (the crossing
  // rival contributes the propagated mind; it is never replaced itself).
  if (!rivalCarIndices.includes(options.crossingCarIndex)) {
    throw new RangeError(
      `${PROPAGATION_ERROR_PREFIX}: the crossing rival must be one of the session rivals`,
    );
  }

  // Step 4: Guard the crossing network's serialized weights (the recycled
  // serialize() plane — non-empty and finite only).
  assertCrossingWeights(options.winnerWeights, PROPAGATION_ERROR_PREFIX);

  // Step 5: Derive THE event's mutation seed from the section 9.7.2
  // rival-swap row (the named per-event stride; the derivation table owns
  // the tuple-seed and event-index rejections — no second derivation path).
  const mutationSeed = deriveNeatRacingRivalSwapChildSeed(
    options.tupleSeed,
    options.eventIndex,
  );

  // Step 6: Resolve the propagation targets — every rival EXCEPT the
  // crossing one, in ascending car order (the pinned Level 2 draw order).
  const targetCarIndices = options.rivalCarIndices
    .toSorted((first, second) => first - second)
    .filter((carIndex) => carIndex !== options.crossingCarIndex);

  // Step 7: Build one replacement mind per target (bounded seeded nudges).
  const replacements = buildReplacementMinds(
    options.winnerWeights,
    mutationSeed,
    targetCarIndices,
  );

  // Step 8: Snapshot the crossing network (the weights are COPIED so the
  // snapshot owns its bytes) and hand back the swap record.
  return {
    snapshot: {
      eventIndex: options.eventIndex,
      mutationSeed,
      crossing: {
        carIndex: options.crossingCarIndex,
        weights: Float32Array.from(options.winnerWeights),
      },
      propagatedCarIndices: targetCarIndices,
    },
    replacements,
  };
}

/**
 * Encodes the runtime swap snapshot into the JSON-safe tuple payload — the
 * ONE shape that may be written into the AC-026 reproducibility tuple's
 * `opponentSnapshot` record at a swap (the base canonicalizer digests plain
 * objects, finite numbers and strings; typed arrays belong in the D4
 * `seedNetwork.weights` slot only, so the crossing weights ride as
 * byte-exact hex here).
 *
 * @param snapshot - The runtime snapshot (the operator result's `snapshot`).
 * @returns The tuple-carried swap payload (hex weights, plain numbers).
 * @throws {RangeError} When the snapshot carries a fractional/negative
 *   event index, a mutation seed outside the unsigned 32-bit range, a
 *   main-agent-slot crossing index, empty or non-finite crossing weights,
 *   or propagated indices that are not strictly ascending positive rivals
 *   distinct from the crossing one.
 *
 * @example
 * ```ts
 * nextTuple.opponentSnapshot = toNeatRacingRivalSwapPayload(swap.snapshot);
 * ```
 */
export function toNeatRacingRivalSwapPayload(
  snapshot: NeatRacingRivalSwapSnapshot,
): NeatRacingRivalSwapPayload {
  // Step 1: Guard the runtime snapshot's shape (no silent nulls).
  assertSnapshotShape(snapshot);

  // Step 2: Encode (hex weights; the payload's arrays own their entries).
  return {
    kind: NEATRACING_RIVAL_SWAP_PAYLOAD_KIND,
    eventIndex: snapshot.eventIndex,
    mutationSeed: snapshot.mutationSeed,
    crossing: {
      carIndex: snapshot.crossing.carIndex,
      weights: encodeFloat32ArrayAsHex(snapshot.crossing.weights),
    },
    propagatedCarIndices: [...snapshot.propagatedCarIndices],
  };
}

/**
 * Rebuilds the post-swap replacement minds from the SERIALIZED PAYLOAD
 * ALONE (the section 9.7 replay path — "the reproducibility tuple replays
 * swap events exactly").
 *
 * The payload carries the crossing network's hex weights, the event's
 * mutation seed and the propagated car indices; those three facts plus the
 * module's fixed delta range and the pinned draw order (targets ascending
 * by car index, then weight index ascending) fully determine every
 * replacement mind. A replay needs neither the original options, the tuple
 * seed nor any runtime state — only the payload that lived inside
 * `opponentSnapshot`. Same-runtime exactness: the math is the same float32
 * plane the operator wrote.
 *
 * @param payload - The swap payload (as stored in the tuple's
 *   `opponentSnapshot` record; key order does not matter).
 * @returns One replacement mind per propagated rival, in ascending car
 *   order — byte-identical to the operator's own replacements for the same
 *   payload.
 * @throws {RangeError} When the payload is not an object, misses the
 *   `rival-swap` kind discriminator or its crossing snapshot record, and
 *   when it carries a fractional/negative event index, a mutation seed
 *   outside the unsigned 32-bit range, a main-agent-slot crossing index, a
 *   malformed hex weights string (non-lowercase hex, wrong float32
 *   alignment or empty), weights that decode to non-finite values, or
 *   propagated indices that are not strictly ascending positive rivals
 *   distinct from the crossing one.
 *
 * @example
 * ```ts
 * const replacements = restoreNeatRacingRivalSwapReplacementMinds(
 *   tuple.opponentSnapshot as NeatRacingRivalSwapPayload,
 * );
 * ```
 */
export function restoreNeatRacingRivalSwapReplacementMinds(
  payload: NeatRacingRivalSwapPayload,
): readonly NeatRacingRivalSwapReplacementMind[] {
  // Step 1: Guard the payload shape (no silent nulls).
  assertPayloadShape(payload);

  // Step 2: Decode the crossing weights and re-derive the nudged minds
  // (the finite-weights guard re-runs on the decoded plane).
  const crossingWeights = decodeHexAsFloat32Array(payload.crossing.weights);
  assertCrossingWeights(crossingWeights, PAYLOAD_ERROR_PREFIX);
  return buildReplacementMinds(crossingWeights, payload.mutationSeed, [
    ...payload.propagatedCarIndices,
  ]);
}

/**
 * Serializes the swap payload into a canonical byte-exact string — the
 * replay anchor of one swap event.
 *
 * THE ONE SERIALIZATION PATH for this module's payload: the canonical
 * object is rebuilt with keys in sorted order (`crossing`, `eventIndex`,
 * `kind`, `mutationSeed`, `propagatedCarIndices`; `crossing` sorts its own
 * two keys) so the serial can never depend on insertion order, and the
 * crossing weights ride as BYTE-EXACT HEX (never a `toJSON`). The payload
 * is validated first: a hand-rolled payload carrying non-finite numbers, a
 * typed array or a malformed hex string is rejected instead of silently
 * collapsing (the promotion module's strictness law, mirrored for this NEW
 * payload).
 *
 * @param payload - The swap payload (any key order).
 * @returns The canonical JSON string (sorted keys, hex weights embedded).
 * @throws {RangeError} With the same rejections as
 *   `restoreNeatRacingRivalSwapReplacementMinds` (the shared shape guard).
 *
 * @example
 * ```ts
 * const replayAnchor = serializeNeatRacingRivalSwapPayload(
 *   toNeatRacingRivalSwapPayload(swap.snapshot),
 * );
 * ```
 */
export function serializeNeatRacingRivalSwapPayload(
  payload: NeatRacingRivalSwapPayload,
): string {
  // Step 1: Guard the payload shape (the shared strict guard).
  assertPayloadShape(payload);

  // Step 2: Rebuild the payload in canonical (sorted) key order and emit
  // the ONE canonical serial (key-order invariant, byte-stable).
  const canonical = {
    crossing: {
      carIndex: payload.crossing.carIndex,
      weights: payload.crossing.weights,
    },
    eventIndex: payload.eventIndex,
    kind: payload.kind,
    mutationSeed: payload.mutationSeed,
    propagatedCarIndices: [...payload.propagatedCarIndices],
  };
  return JSON.stringify(canonical);
}

// ── Helpers below the fold ───────────────────────────────────────────────────

/**
 * Samples ONE bounded nudge delta for the swap round (the recycled
 * MOD_WEIGHT uniform-sample contract on the small named range).
 *
 * @param stream - The event's seeded xorshift32 stream.
 * @returns The delta (in the closed-ish `[MIN, MAX)` uniform range).
 */
function sampleSwapDelta(stream: NeatRacingXorShift32Stream): number {
  return (
    NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN +
    stream() *
      (NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX -
        NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN)
  );
}

/**
 * Builds one replacement mind per target rival (the ONE nudge round).
 *
 * THE PINNED DRAW ORDER (Level 2 ordered-deterministic): targets ascend by
 * car index; within a target, weight indices ascend; each weight consumes
 * exactly one stream draw. Both the operator and the replay path funnel
 * through this helper, so the replay is the operator's own math.
 *
 * @param crossingWeights - The crossing rival's serialized weights.
 * @param mutationSeed - The event's derived mutation seed.
 * @param targetCarIndices - The targets, ascending (the caller guards).
 * @returns One replacement mind per target (NEW arrays; input never
 *   mutated, never aliased).
 */
function buildReplacementMinds(
  crossingWeights: Float32Array,
  mutationSeed: number,
  targetCarIndices: readonly number[],
): NeatRacingRivalSwapReplacementMind[] {
  const stream = createNeatRacingXorShift32Stream(mutationSeed);
  return targetCarIndices.map((carIndex) => {
    const weights = new Float32Array(crossingWeights.length);
    for (let weightIndex = 0; weightIndex < weights.length; weightIndex += 1) {
      weights[weightIndex] =
        crossingWeights[weightIndex] + sampleSwapDelta(stream);
    }
    return { carIndex, weights };
  });
}

/**
 * Encodes a typed array as a byte-exact hex string.
 *
 * The `promotion.rules.ts` encoder is private to ITS payload; this module
 * owns the canonicalization for its OWN payload, so the D4 encoder is
 * mirrored here (one serialization path per payload — never a shared dual
 * path). Lowercase hex, two characters per byte, no separators — the exact
 * byte pattern `Float32Array` holds, never a `toJSON`.
 *
 * @param weights - The typed array to encode.
 * @returns The hex string of the array's raw bytes in NATIVE byte order
 *   (little-endian on the supported runtimes — the D4 law's own convention,
 *   pinned by `promotion.rules.test.ts` as `0000803f` for 1.0). NOT
 *   endian-stable: the payload's byte parity is same-runtime (Level 2,
 *   ordered-deterministic), never cross-environment exact.
 */
function encodeFloat32ArrayAsHex(weights: Float32Array): string {
  const bytes = new Uint8Array(
    weights.buffer as ArrayBuffer,
    weights.byteOffset,
    weights.byteLength,
  );
  let hex = '';
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, '0');
  }
  return hex;
}

/**
 * Decodes a byte-exact hex string back into its Float32Array.
 *
 * The caller's shape guard (`assertHexWeights`) rejects non-lowercase,
 * non-float32-aligned or empty strings BEFORE the decode, so the decode
 * itself stays total on guarded input.
 *
 * @param hex - The guarded hex string (lowercase, length a multiple of 8).
 * @returns The decoded weights.
 */
function decodeHexAsFloat32Array(hex: string): Float32Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let byteIndex = 0; byteIndex < bytes.length; byteIndex += 1) {
    bytes[byteIndex] = parseInt(
      hex.slice(byteIndex * 2, byteIndex * 2 + 2),
      16,
    );
  }
  return new Float32Array(bytes.buffer);
}

/**
 * Guards the crossing network's serialized weights (empty and non-finite
 * are loud rejections, mirroring the D4 finite-weights guard).
 *
 * @param weights - The weights to guard.
 * @param prefix - The error-voice prefix of the calling lane.
 * @throws {RangeError} When the weights are empty or carry a non-finite
 *   value (NaN or an infinity has no replay-stable float32 meaning here).
 */
function assertCrossingWeights(weights: unknown, prefix: string): void {
  // fix-packet-P2-S3-D-iteration-1 FIX 3 companion: a non-typed-array is a
  // loud rejection (the lanes guard unknown material at the tuple
  // boundary), never a TypeError.
  if (!(weights instanceof Float32Array)) {
    throw new RangeError(
      `${prefix}: the crossing network's serialized weights must be float32 typed-array bytes`,
    );
  }
  if (weights.length === 0) {
    throw new RangeError(
      `${prefix}: the crossing network's serialized weights must not be empty`,
    );
  }
  for (const weight of weights) {
    if (!Number.isFinite(weight)) {
      throw new RangeError(
        `${prefix}: the crossing network's serialized weights must be finite`,
      );
    }
  }
}

/**
 * Guards an unsigned 32-bit seed value (the derivation table's integer
 * shape — a seed is a stored RNG-state component, not a free number).
 *
 * @param value - The seed to guard.
 * @param label - The seed's name (for the static message surface).
 * @throws {RangeError} When the value is not a non-negative integer or
 *   exceeds the unsigned 32-bit ceiling.
 */
function assertUnsigned32Seed(value: unknown, label: string): void {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: ${label} must be a non-negative integer`,
    );
  }
  if ((value as number) > UNSIGNED_32_BIT_SEED_MAX) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: ${label} must fit an unsigned 32-bit seed`,
    );
  }
}

/**
 * Guards the propagated rival car indices: positive rivals (never the main
 * agent's slot, never the crossing rival), strictly ascending (the pinned
 * Level 2 draw order's record must be canonical).
 *
 * @param indices - The propagated car indices.
 * @param crossingCarIndex - The crossing rival's car index.
 * @throws {RangeError} When an index is a fraction, touches the main
 *   agent's slot, is not strictly ascending, or repeats the crossing index.
 */
function assertPropagatedCarIndices(
  indices: unknown,
  crossingCarIndex: number,
): void {
  if (!Array.isArray(indices)) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the propagated rival car indices must be an array`,
    );
  }
  let previousCarIndex = Number.NEGATIVE_INFINITY;
  for (const carIndex of indices) {
    if (
      !Number.isInteger(carIndex) ||
      carIndex <= NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX
    ) {
      throw new RangeError(
        `${PAYLOAD_ERROR_PREFIX}: propagated rival car indices must be integers strictly after car 0 — the main agent's slot`,
      );
    }
    if (carIndex <= previousCarIndex) {
      throw new RangeError(
        `${PAYLOAD_ERROR_PREFIX}: propagated rival car indices must be strictly ascending`,
      );
    }
    if (carIndex === crossingCarIndex) {
      throw new RangeError(
        `${PAYLOAD_ERROR_PREFIX}: the crossing rival is never among the propagated rivals`,
      );
    }
    previousCarIndex = carIndex;
  }
}

/**
 * Guards the hex crossing-weights field of a payload — STRUCTURE and the
 * DECODED PLANE both (fix-packet-P2-S3-D-iteration-1 FIX 2).
 *
 * The structure check alone would let a hex string that decodes to a
 * non-finite float32 (NaN or an infinity) reach the canonical serial — a
 * certified replay anchor the restore path would then reject.
 * Strict-rejection parity: the serializer rejects exactly what the restore
 * rejects.
 *
 * @param hex - The candidate hex string.
 * @throws {RangeError} When the value is not a non-empty lowercase hex
 *   string whose length decodes to whole float32 values (8 hex characters
 *   per weight), or when the decoded plane is not finite float32 material.
 */
function assertHexWeights(hex: unknown): void {
  if (
    typeof hex !== 'string' ||
    hex.length === 0 ||
    hex.length % 8 !== 0 ||
    !/^[0-9a-f]+$/.test(hex)
  ) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the crossing network's serialized weights must be a lowercase hex string of one or more float32 values`,
    );
  }
  assertCrossingWeights(decodeHexAsFloat32Array(hex), PAYLOAD_ERROR_PREFIX);
}

/**
 * Guards the runtime snapshot's shape (the payload encoder's entry guard).
 *
 * @param snapshot - The runtime snapshot to guard.
 * @throws {RangeError} When the snapshot is not an object, misses its
 *   crossing snapshot record, or the event index, mutation seed, crossing
 *   anchor, crossing weights or propagated indices fail their contracts.
 */
function assertSnapshotShape(snapshot: NeatRacingRivalSwapSnapshot): void {
  // fix-packet-P2-S3-D-iteration-1 FIX 3: guard the record's existence
  // before any dereference — a malformed record must fail with the module's
  // RangeError voice at the unknown-typed tuple boundary, never a TypeError.
  if (typeof snapshot !== 'object' || snapshot === null) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the swap snapshot must be an object`,
    );
  }
  const candidate = snapshot as SwapRecordGuardView;
  if (
    !Number.isInteger(candidate.eventIndex) ||
    (candidate.eventIndex as number) < 0
  ) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the eventIndex must be a non-negative integer`,
    );
  }
  assertUnsigned32Seed(candidate.mutationSeed, 'the mutationSeed');
  if (typeof candidate.crossing !== 'object' || candidate.crossing === null) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the swap snapshot must carry the crossing network's snapshot`,
    );
  }
  assertCrossingCarIndex(candidate.crossing.carIndex);
  assertCrossingWeights(candidate.crossing.weights, PAYLOAD_ERROR_PREFIX);
  assertPropagatedCarIndices(
    candidate.propagatedCarIndices,
    candidate.crossing.carIndex as number,
  );
}

/**
 * Guards the payload's shape (the restore and serializer's entry guard).
 *
 * @param payload - The payload to guard.
 * @throws {RangeError} When the payload is not an object, misses the
 *   `rival-swap` kind discriminator, carries no crossing snapshot record,
 *   or fails the event/seed/anchor/hex/propagated-index contracts.
 */
function assertPayloadShape(payload: NeatRacingRivalSwapPayload): void {
  if (typeof payload !== 'object' || payload === null) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the payload must be an object`,
    );
  }
  const candidate = payload as SwapRecordGuardView;
  if (candidate.kind !== NEATRACING_RIVAL_SWAP_PAYLOAD_KIND) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the payload must carry the rival-swap kind discriminator`,
    );
  }
  if (
    !Number.isInteger(candidate.eventIndex) ||
    (candidate.eventIndex as number) < 0
  ) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the eventIndex must be a non-negative integer`,
    );
  }
  assertUnsigned32Seed(candidate.mutationSeed, 'the mutationSeed');
  // fix-packet-P2-S3-D-iteration-1 FIX 3: the crossing record must EXIST
  // before its fields are dereferenced (the tuple boundary types this as
  // unknown — malformed records fail with the RangeError voice).
  if (typeof candidate.crossing !== 'object' || candidate.crossing === null) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the payload must carry the crossing network's snapshot`,
    );
  }
  assertCrossingCarIndex(candidate.crossing.carIndex);
  assertHexWeights(candidate.crossing.weights);
  assertPropagatedCarIndices(
    candidate.propagatedCarIndices,
    candidate.crossing.carIndex as number,
  );
}

/**
 * Guards a crossing rival car index (a positive integer — never the main
 * agent's slot).
 *
 * @param carIndex - The crossing car index to guard.
 * @throws {RangeError} When the index is a fraction or touches the main
 *   agent's slot.
 */
function assertCrossingCarIndex(carIndex: unknown): void {
  if (
    !Number.isInteger(carIndex) ||
    (carIndex as number) <= NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX
  ) {
    throw new RangeError(
      `${PAYLOAD_ERROR_PREFIX}: the crossing car index must be an integer strictly after car 0 — the main agent's slot`,
    );
  }
}
