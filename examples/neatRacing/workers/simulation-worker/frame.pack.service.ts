/**
 * Frame pack service for the neatRacing simulation worker (Phase 2, Step 01,
 * slice P2-S1-B).
 *
 * Educational note — why this file exists:
 * The P2-S1-A worker currently replies to every paced frame with structured
 * clones (plain `PhysicsCarState[]` objects). Structured cloning allocates a
 * fresh object graph per frame and pays the clone tax on the host<->worker
 * boundary every round trip. The Phase-2 learning-document contract moves the
 * per-frame transfer budget onto **preallocated buffers with explicit
 * offsets** (AC-042 partial): the worker packs each fleet snapshot into a
 * reusable slot buffer laid out as structure-of-arrays (SoA) channels, the
 * host reads (or unpacks) the same layout, and the only buffers that ever
 * cross `postMessage` are the slots preallocated once per session.
 *
 * Wire contract (AC-025):
 * - The schema sentinel is `'neatracing-packed-v1'` (the NeatRacing analogue
 *   of the racing-curriculum `'racing-packed-v1'` sentinel); consumers must
 *   reject any other value via `assertNeatRacingPackedFrameSchemaVersion`
 *   so a future schema drift is caught at the message boundary instead of
 *   silently misreading packed bytes.
 * - EVERY packed frame channel is a `Float32Array` (D8a; the racing
 *   precedent packs every channel the same way). Float32 keeps the packing
 *   deterministic — pack/unpack round-trips are Float32-lossy but exactly
 *   reproducible, which the byte-parity slice (P2-S1-C) formalizes.
 * - The feature-flag bit table is the recycled D8b assignment: bit 0 =
 *   `tiresEnabled`, bit 1 = `radioEnabled`, bit 2 = `pitsEnabled`. NeatRacing
 *   pins tires = 0 and pits = 0 (no tire wear, no pit stops) and keeps
 *   radio = 1 (the observation tails survive), so the default bitfield
 *   encodes `{ tires: false, radio: true, pits: false }`.
 *
 * Preallocated-buffer contract (AC-042 partial):
 * - `createNeatRacingFramePackPool` allocates ONE `ArrayBuffer` per slot,
 *   sized once from the packer's `carCapacity`, and partitions it into a
 *   rules span followed by `NEATRACING_PACKED_CAR_CHANNEL_COUNT` SoA channel
 *   spans, all at computed byte offsets. Views (`Float32Array` subarrays)
 *   are created exactly once per slot; packing writes into them in place.
 * - `packFrame` allocates NOTHING: it round-robins over the slots, mutates
 *   the slot's reused frame object, and fills the channels from a single
 *   fixed channel table. The double-buffered default (`poolSlots: 2`) keeps
 *   the previously packed frame readable while the next one packs because a
 *   packed slot stays valid until `poolSlots` further packs overwrite it.
 * - `resolveNeatRacingPackedFrameTransferList` collects the distinct buffers
 *   backing a packed frame so `postMessage(message, transferList)` transfers
 *   them zero-copy; with the single-buffer layout the list is always exactly
 *   one entry (`NEATRACING_PACKED_TRANSFER_BUFFER_COUNT`).
 *
 * Recycling note (spec §11): the SoA channel pattern, the transfer-list
 * resolver, and the schema-sentinel guard are recycled from
 * `examples/racing_curriculum/workers/simulation-worker/`
 * (`simulation-worker.race-pack.service.ts` + `simulation-worker.snapshot.utils.ts`),
 * reshaped for the NeatRacing fleet + race-rules state and the single
 * preallocated slot buffer the AC-042 budget prefers.
 *
 * Determinism discipline (D8a / AC-026): no `Math.random`, no clock reads;
 * packing is a pure state->bytes function of its input.
 */

import type { PhysicsCarState } from '../../physics/physics.step.service';
import type { RaceRulesState, RaceStatus } from '../../physics/race-rules';

/** Schema sentinel for forward-compatibility rejection on packed frames. */
export const NEATRACING_PACK_SCHEMA_VERSION = 'neatracing-packed-v1';

/**
 * The D8b feature-flag bit assignments, carried from the racing precedent:
 * bit 0 = `tiresEnabled`, bit 1 = `radioEnabled`, bit 2 = `pitsEnabled`.
 */
export const NEATRACING_FEATURE_FLAG_BITS = Object.freeze({
  tiresEnabled: 1,
  radioEnabled: 2,
  pitsEnabled: 4,
});

/** The full D8b bit mask covering all three recycled feature-flag bits. */
export const NEATRACING_FEATURE_FLAG_MASK = 7;

/**
 * The feature flags NeatRacing pins by default: tires = 0 (no tire wear),
 * pits = 0 (no pit stops), radio = 1 (observation tails survive).
 */
export const NEATRACING_DEFAULT_FEATURE_FLAGS: Readonly<{
  tiresEnabled: boolean;
  radioEnabled: boolean;
  pitsEnabled: boolean;
}> = Object.freeze({
  tiresEnabled: false,
  radioEnabled: true,
  pitsEnabled: false,
});

/** Numeric form of `NEATRACING_DEFAULT_FEATURE_FLAGS` under the D8b table. */
export const NEATRACING_DEFAULT_FEATURE_FLAG_BITFIELD = 2;

/**
 * Numeric sentinel written for a null `nextRaceLevel` in the packed rules
 * header (a real level is a non-negative integer, so -1 can never collide).
 */
export const NEATRACING_PACK_NULL_LEVEL_SENTINEL = -1;

/** Float32 words the packed rules span occupies at the head of every slot. */
export const NEATRACING_PACKED_RULES_FLOAT_COUNT = 2;

/** Byte size of one Float32 word in the packed slot layout. */
export const NEATRACING_PACKED_WORD_BYTES = Float32Array.BYTES_PER_ELEMENT;

/** Smallest legal packed fleet size (the single human car). */
export const NEATRACING_PACKED_MIN_CARS = 1;

/**
 * Largest legal packed fleet capacity.
 *
 * Mirrors the sim worker's fleet bounds (`NEATRACING_SIM_WORKER_MAX_AGENTS`
 * in `./simulation-worker.ts`); duplicated as a local constant on purpose —
 * importing the worker module would drag its import-time `self.onmessage`
 * adapter into every packer consumer.
 */
export const NEATRACING_PACKED_MAX_CAR_CAPACITY = 32;

/** Default double-buffering slot count for `createNeatRacingFramePackPool`. */
export const NEATRACING_FRAME_PACK_POOL_SLOTS = 2;

/** Smallest legal pool slot count (single reusable slot). */
export const NEATRACING_FRAME_PACK_MIN_POOL_SLOTS = 1;

/** Largest legal pool slot count (bounded fan of preallocated slots). */
export const NEATRACING_FRAME_PACK_MAX_POOL_SLOTS = 4;

/**
 * Buffers a packed frame contributes to a transfer list.
 *
 * The slot layout packs every channel into ONE preallocated buffer, so the
 * resolved transfer list is always exactly a single entry — the AC-042
 * "single packed frame ArrayBuffer with offsets" story.
 */
export const NEATRACING_PACKED_TRANSFER_BUFFER_COUNT = 1;

/**
 * The recycled race-status code table: the fixed enum mapping packed into
 * the rules header (a plain number, exact across the boundary).
 */
export const NEATRACING_PACKED_RACE_STATUS_CODES = Object.freeze({
  running: 0,
  raceComplete: 1,
  outOfTime: 2,
});

/** Race-status names indexed by their packed code (single derived table). */
const NEATRACING_PACKED_RACE_STATUS_BY_CODE: Readonly<
  Record<number, RaceStatus>
> = Object.freeze(
  Object.fromEntries(
    Object.entries(NEATRACING_PACKED_RACE_STATUS_CODES).map(
      ([statusName, statusCode]) => [statusCode, statusName],
    ),
  ),
) as Readonly<Record<number, RaceStatus>>;

/**
 * The SINGLE fixed channel mapping for the packed SoA layout.
 *
 * Educational note: the channel list is the one source of truth for the
 * packed frame — the declaration order fixes the byte offsets, the field
 * names fix the wire shape, and each `reader` picks the physics field that
 * fills the channel. Packing, channel allocation, and finiteness validation
 * all walk this table so a new channel can never be added to the wire shape
 * without declaring its physics reader alongside it.
 */
export const NEATRACING_PACKED_CAR_CHANNELS: Readonly<
  Readonly<{ name: string; reader: (car: PhysicsCarState) => number }>[]
> = Object.freeze([
  { name: 'carX', reader: (car) => car.x },
  { name: 'carY', reader: (car) => car.y },
  { name: 'carHeading', reader: (car) => car.heading },
  { name: 'carSpeed', reader: (car) => car.speed },
  { name: 'carYawRate', reader: (car) => car.yawRate },
  { name: 'carSlipAngle', reader: (car) => car.slipAngle },
  { name: 'carTravelHeadingRad', reader: (car) => car.travelHeadingRad },
  { name: 'carTractionSeverity', reader: (car) => car.tractionSeverity },
  { name: 'carLateralVelocity', reader: (car) => car.lateralVelocity },
  { name: 'carDriftCommittedSlip', reader: (car) => car.drift.committed.slip },
  {
    name: 'carDriftCommittedTurnSign',
    reader: (car) => car.drift.committed.turnSign,
  },
  {
    name: 'carDriftCommittedReArmed',
    // Booleans ride as exact 0/1 floats in the D8a channel space.
    reader: (car) => (car.drift.committed.reArmed ? 1 : 0),
  },
  { name: 'carDriftPosePose', reader: (car) => car.drift.pose.pose },
  { name: 'carDriftPoseRate', reader: (car) => car.drift.pose.poseRate },
  {
    name: 'carDriftPoseArticulation',
    reader: (car) => car.drift.pose.articulation,
  },
  { name: 'carTick', reader: (car) => car.tick },
]);

/** SoA channel span count derived from the fixed channel table. */
export const NEATRACING_PACKED_CAR_CHANNEL_COUNT =
  NEATRACING_PACKED_CAR_CHANNELS.length;

/** A packed feature-flag triple under the recycled D8b bit table. */
export interface NeatRacingFeatureFlagState {
  /** Whether tire wear simulation is enabled (NeatRacing pins false). */
  tiresEnabled: boolean;
  /** Whether the team radio observation tails are enabled (pinned true). */
  radioEnabled: boolean;
  /** Whether pit stops are enabled (NeatRacing pins false). */
  pitsEnabled: boolean;
}

/** A rules span holding the continuous race-rules floats per packed frame. */
export interface NeatRacingPackedFrameRulesChannel {
  /** Remaining checkpoint budget seconds (float32, lossy-but-deterministic). */
  checkpointBudgetS: number;
  /** Total raced seconds (float32, lossy-but-deterministic). */
  elapsedS: number;
}

/**
 * A packed render frame on the wire (AC-025).
 *
 * The discrete scalars ride as plain numbers (exact through structured
 * clone and transfer), while EVERY bulk channel is a `Float32Array` view
 * into the slot's single preallocated buffer at its computed byte offset.
 */
export interface NeatRacingPackedFrame {
  /** Schema sentinel; must be `'neatracing-packed-v1'` (guarded on read). */
  schemaVersion: typeof NEATRACING_PACK_SCHEMA_VERSION;
  /** Total-tick counter before this frame's released ticks. */
  fromTick: number;
  /** Total-tick counter after this frame's released ticks. */
  toTick: number;
  /** Fleet size packed into this frame's channels (one entry per car). */
  agentCount: number;
  /** D8b feature-flag bitfield (bit 0 tires, bit 1 radio, bit 2 pits). */
  featureFlags: number;
  /** Packed discrete rules scalars (exact integers / code table entries). */
  level: number;
  /** Packed laps-completed scalar (exact integer). */
  lapsCompleted: number;
  /** Packed race-status code from `NEATRACING_PACKED_RACE_STATUS_CODES`. */
  raceStatusCode: number;
  /** Next level scalar, or `NEATRACING_PACK_NULL_LEVEL_SENTINEL` for null. */
  nextRaceLevel: number;
  /** Continuous rules channel `[checkpointBudgetS, elapsedS]` (float32). */
  rulesChannel: Float32Array;
  // --- Per-car SoA channels (declaration order == channel table order) ---
  /** Car X positions in meters. Length = agentCount. */
  carX: Float32Array;
  /** Car Y positions in meters. Length = agentCount. */
  carY: Float32Array;
  /** Car headings in radians. Length = agentCount. */
  carHeading: Float32Array;
  /** Longitudinal speeds in m/s. Length = agentCount. */
  carSpeed: Float32Array;
  /** Yaw responses in rad/tick. Length = agentCount. */
  carYawRate: Float32Array;
  /** Slip angles in radians. Length = agentCount. */
  carSlipAngle: Float32Array;
  /** Travel-direction headings in radians. Length = agentCount. */
  carTravelHeadingRad: Float32Array;
  /** Traction severities in [0, 1]. Length = agentCount. */
  carTractionSeverity: Float32Array;
  /** Lateral velocities in m/s. Length = agentCount. */
  carLateralVelocity: Float32Array;
  /** Committed drift slip angles in radians. Length = agentCount. */
  carDriftCommittedSlip: Float32Array;
  /** Committed turn signs (+1 left / -1 right / 0 uncommitted). Length = agentCount. */
  carDriftCommittedTurnSign: Float32Array;
  /** Committed re-arm tokens as 0/1 floats. Length = agentCount. */
  carDriftCommittedReArmed: Float32Array;
  /** Visible drift body yaw in radians. Length = agentCount. */
  carDriftPosePose: Float32Array;
  /** Visible drift yaw rates in rad/s. Length = agentCount. */
  carDriftPoseRate: Float32Array;
  /** Steer-yaw articulations in radians. Length = agentCount. */
  carDriftPoseArticulation: Float32Array;
  /** Completed-tick counters per car. Length = agentCount. */
  carTick: Float32Array;
}

/** The input a frame packing needs: the P2-S1-A frames payload + flags. */
export interface NeatRacingPackedFrameInput {
  /** Total-tick counter before this frame's released ticks. */
  fromTick: number;
  /** Total-tick counter after this frame's released ticks. */
  toTick: number;
  /** Post-frame fleet snapshot in tick order (car 0 first). */
  cars: readonly PhysicsCarState[];
  /** Post-frame race-rules snapshot. */
  rules: RaceRulesState;
  /**
   * Feature flags to stamp; omitted input packs the NeatRacing pinned
   * defaults (`NEATRACING_DEFAULT_FEATURE_FLAGS`).
   */
  featureFlags?: NeatRacingFeatureFlagState;
}

/** The unpacked plain-object reconstruction of one packed frame. */
export interface NeatRacingPackedFrameSnapshot {
  /** Total-tick counter before this frame's released ticks. */
  fromTick: number;
  /** Total-tick counter after this frame's released ticks. */
  toTick: number;
  /** Fleet size packed into the frame's channels. */
  agentCount: number;
  /** D8b feature-flag bitfield carried verbatim. */
  featureFlags: number;
  /** Reconstructed race-rules state (Float32-lossy continuous fields). */
  rules: RaceRulesState;
  /** Reconstructed fleet in channel order (Float32-lossy physics fields). */
  cars: PhysicsCarState[];
}

/** Options accepted by `createNeatRacingFramePackPool`. */
export interface NeatRacingFramePackPoolOptions {
  /** Max fleet size the slots will pack (validated against the 32-car cap). */
  carCapacity: number;
  /** Preallocated slot count for double-buffering (default 2; range [1, 4]). */
  poolSlots?: number;
}

/** A preallocated pack pool owning one slot buffer per configured slot. */
export interface NeatRacingFramePackPool {
  /** Max fleet size the slots were allocated for. */
  readonly carCapacity: number;
  /** Slot count the pool round-robins over. */
  readonly poolSlots: number;
  /** Diagnostic: byte length of every slot buffer (all slots are equal). */
  readonly slotByteLength: number;
  /**
   * Packs one fleet snapshot into the next slot with ZERO allocation.
   *
   * The returned frame object is one of the pool's reused slot frames: it
   * stays valid until `poolSlots` further packs overwrite its slot, which
   * is exactly the double-buffering contract the host-side consumer needs
   * (read slot N while slot N+1 is being packed).
   *
   * @param input - The frames payload to pack (validated before writing).
   * @returns The packed frame (a reused slot object; treat as read-only).
   * @throws {RangeError} When the input violates the pack contract.
   */
  packFrame(input: NeatRacingPackedFrameInput): NeatRacingPackedFrame;
  /**
   * Reconstructs a plain-object snapshot from a packed frame.
   *
   * Educational note: unpacking allocates (the plain objects mirror the
   * P2-S1-A structured payload). It is the bridge/test seam for consumers
   * that want typed access, NOT the hot path — the hot path reads channel
   * views directly.
   *
   * @param frame - A packed frame (schema-sentinel checked first).
   * @returns The reconstructed snapshot (Float32-lossy on packing).
   * @throws {RangeError} When the schema sentinel or packed codes are invalid.
   */
  unpackFrame(frame: NeatRacingPackedFrame): NeatRacingPackedFrameSnapshot;
}

/** First Float32 word of a SoA channel span (rules span comes first). */
export function resolveNeatRacingPackedChannelFirstFloatWord(
  channelIndex: number,
  carCapacity: number,
): number {
  return NEATRACING_PACKED_RULES_FLOAT_COUNT + channelIndex * carCapacity;
}

/** Total Float32 words one packed slot buffer reserves for a car capacity. */
export function resolveNeatRacingPackedFrameFloatWordCount(
  carCapacity: number,
): number {
  return (
    NEATRACING_PACKED_RULES_FLOAT_COUNT +
    NEATRACING_PACKED_CAR_CHANNEL_COUNT * carCapacity
  );
}

/** Total byte length of one packed slot buffer for a car capacity. */
export function resolveNeatRacingPackedSlotByteLength(
  carCapacity: number,
): number {
  return (
    resolveNeatRacingPackedFrameFloatWordCount(carCapacity) *
    NEATRACING_PACKED_WORD_BYTES
  );
}

/**
 * Encodes a feature-flag triple into the D8b bitfield.
 *
 * @param flags - The flag triple to encode (booleans; no coercion).
 * @returns The bitfield (bit 0 tires, bit 1 radio, bit 2 pits).
 *
 * @example
 * ```ts
 * encodeNeatRacingFeatureFlags({
 *   tiresEnabled: false,
 *   radioEnabled: true,
 *   pitsEnabled: false,
 * }); // 2 — the NeatRacing pinned default
 * ```
 */
export function encodeNeatRacingFeatureFlags(
  flags: NeatRacingFeatureFlagState,
): number {
  return (
    (flags.tiresEnabled ? NEATRACING_FEATURE_FLAG_BITS.tiresEnabled : 0) |
    (flags.radioEnabled ? NEATRACING_FEATURE_FLAG_BITS.radioEnabled : 0) |
    (flags.pitsEnabled ? NEATRACING_FEATURE_FLAG_BITS.pitsEnabled : 0)
  );
}

/**
 * Decodes a D8b bitfield into a feature-flag triple (masked to the table).
 *
 * @param bitfield - Arbitrary number; masked to the known three bits.
 * @returns The decoded flag triple.
 *
 * @example
 * ```ts
 * decodeNeatRacingFeatureFlags(0b011); // { tiresEnabled: true, radioEnabled: true, pitsEnabled: false }
 * ```
 */
export function decodeNeatRacingFeatureFlags(
  bitfield: number,
): NeatRacingFeatureFlagState {
  const masked = bitfield & NEATRACING_FEATURE_FLAG_MASK;
  return {
    tiresEnabled: (masked & NEATRACING_FEATURE_FLAG_BITS.tiresEnabled) !== 0,
    radioEnabled: (masked & NEATRACING_FEATURE_FLAG_BITS.radioEnabled) !== 0,
    pitsEnabled: (masked & NEATRACING_FEATURE_FLAG_BITS.pitsEnabled) !== 0,
  };
}

/**
 * Throws a `RangeError` when a received frame does not carry the
 * `'neatracing-packed-v1'` sentinel.
 *
 * Consumers must call this before reading any channel so a schema mismatch
 * fails loudly at the boundary instead of silently misreading packed bytes.
 *
 * @param frame - The received frame-like object.
 * @throws {RangeError} When `schemaVersion` is not the packed sentinel.
 *
 * @example
 * ```ts
 * assertNeatRacingPackedFrameSchemaVersion(message.payload);
 * // now safe to read the channel views
 * ```
 */
export function assertNeatRacingPackedFrameSchemaVersion(frame: {
  schemaVersion: unknown;
}): void {
  if (frame.schemaVersion !== NEATRACING_PACK_SCHEMA_VERSION) {
    throw new RangeError(
      `Unsupported NeatRacingPackedFrame schemaVersion: ${String(frame.schemaVersion)}`,
    );
  }
}

/**
 * Collects the distinct `ArrayBuffer`s backing a packed frame's channels
 * into a zero-copy `postMessage` transfer list.
 *
 * With the single preallocated slot buffer every channel view shares, the
 * list is always exactly `NEATRACING_PACKED_TRANSFER_BUFFER_COUNT` entries;
 * the dedupe keeps the resolver correct even if a future layout splits the
 * channels across multiple slot buffers.
 *
 * @param frame - The packed frame whose buffers will be transferred.
 * @returns Ordered, deduped `ArrayBuffer` list for `postMessage`.
 *
 * @example
 * ```ts
 * worker.postMessage(
 *   { type: 'frames', payload: packedFrame },
 *   resolveNeatRacingPackedFrameTransferList(packedFrame),
 * );
 * ```
 */
export function resolveNeatRacingPackedFrameTransferList(
  frame: NeatRacingPackedFrame,
): ArrayBuffer[] {
  const seenBuffers = new Set<ArrayBuffer>();
  const transferList: ArrayBuffer[] = [];
  const channelViews = [
    frame.rulesChannel,
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

  // Step 1: Walk every channel and register unseen buffers once.
  for (const channelView of channelViews) {
    const buffer = channelView.buffer as ArrayBuffer;
    if (seenBuffers.has(buffer)) {
      continue;
    }
    seenBuffers.add(buffer);
    transferList.push(buffer);
  }
  return transferList;
}

/** Static validation error carrier for pack-side rejections. */
type FramePackValidationError = { error: string };

/**
 * Validates the fleet part of a packing input against the packer bounds and
 * the channel table's finiteness contract.
 *
 * @param cars - The fleet snapshot to validate.
 * @param carCapacity - The packer's preallocated fleet capacity.
 * @returns A static rejection reason, or null when the fleet may pack.
 */
function validatePackedFrameCars(
  cars: readonly PhysicsCarState[],
  carCapacity: number,
): FramePackValidationError | null {
  // Step 1: Enforce the fleet-size bounds (no empty and no over-capacity packs).
  if (cars.length < NEATRACING_PACKED_MIN_CARS) {
    return { error: 'frame pack failed: at least one car is required' };
  }
  if (cars.length > carCapacity) {
    return {
      error: 'frame pack failed: car count exceeds the packer carCapacity',
    };
  }

  // Step 2: Enforce finiteness per channel reader (deterministic float32 fill).
  for (const car of cars) {
    for (const channel of NEATRACING_PACKED_CAR_CHANNELS) {
      if (!Number.isFinite(channel.reader(car))) {
        return {
          error: `frame pack failed: car channel ${channel.name} must be a finite number`,
        };
      }
    }
  }
  return null;
}

/**
 * Validates the rules part of a packing input (status code membership and
 * finiteness of every packed field).
 *
 * @param rules - The race-rules snapshot to validate.
 * @returns A static rejection reason, or null when the rules may pack.
 */
function validatePackedFrameRules(
  rules: RaceRulesState,
): FramePackValidationError | null {
  // Step 1: The status must map through the fixed code table (plain keyed
  // lookup — no per-frame allocation in the validation path).
  const statusCode =
    NEATRACING_PACKED_RACE_STATUS_CODES[
      rules.raceStatus as keyof typeof NEATRACING_PACKED_RACE_STATUS_CODES
    ];
  if (statusCode === undefined) {
    return { error: 'frame pack failed: raceStatus is not a packed code' };
  }

  // Step 2: Every packed numeric field must be finite.
  const finiteFields: readonly [string, number][] = [
    ['level', rules.level],
    ['lapsCompleted', rules.lapsCompleted],
    ['checkpointBudgetS', rules.checkpointBudgetS],
    ['elapsedS', rules.elapsedS],
  ];
  for (const [fieldName, fieldValue] of finiteFields) {
    if (!Number.isFinite(fieldValue)) {
      return {
        error: `frame pack failed: rules ${fieldName} must be a finite number`,
      };
    }
  }

  // Step 3: nextRaceLevel must be null or finite.
  if (rules.nextRaceLevel !== null && !Number.isFinite(rules.nextRaceLevel)) {
    return {
      error: 'frame pack failed: rules nextRaceLevel must be null or finite',
    };
  }
  return null;
}

/**
 * One preallocated pool slot: the slot buffer plus its once-created views
 * and the reused frame object that carries them on the wire.
 */
interface PoolSlot {
  /** The slot's single preallocated backing buffer. */
  buffer: ArrayBuffer;
  /** Reused packed frame object (mutated in place by every pack). */
  frame: NeatRacingPackedFrame;
  /** SoA channel views in channel-table order. */
  carChannels: Float32Array[];
}

/**
 * Allocates one pool slot: a rules span at word 0 plus one span per channel
 * table entry, every view created exactly once.
 *
 * @param carCapacity - Fleet capacity to size the spans for.
 * @returns The slot bundle (buffer + views + reusable frame object).
 */
function createPoolSlot(carCapacity: number): PoolSlot {
  const buffer = new ArrayBuffer(
    resolveNeatRacingPackedSlotByteLength(carCapacity),
  );
  const rulesChannel = new Float32Array(
    buffer,
    0,
    NEATRACING_PACKED_RULES_FLOAT_COUNT,
  );
  // Channel views are created in the fixed channel-table declaration order.
  const carChannels = NEATRACING_PACKED_CAR_CHANNELS.map(
    (channel, channelIndex) =>
      new Float32Array(
        buffer,
        resolveNeatRacingPackedChannelFirstFloatWord(
          channelIndex,
          carCapacity,
        ) * NEATRACING_PACKED_WORD_BYTES,
        carCapacity,
      ),
  );
  // The frame object is built once and reused for every pack of this slot;
  // the literal field order mirrors the channel-table order above.
  const frame: NeatRacingPackedFrame = {
    schemaVersion: NEATRACING_PACK_SCHEMA_VERSION,
    fromTick: 0,
    toTick: 0,
    agentCount: 0,
    featureFlags: 0,
    level: 0,
    lapsCompleted: 0,
    raceStatusCode: NEATRACING_PACKED_RACE_STATUS_CODES.running,
    nextRaceLevel: NEATRACING_PACK_NULL_LEVEL_SENTINEL,
    rulesChannel,
    carX: carChannels[0],
    carY: carChannels[1],
    carHeading: carChannels[2],
    carSpeed: carChannels[3],
    carYawRate: carChannels[4],
    carSlipAngle: carChannels[5],
    carTravelHeadingRad: carChannels[6],
    carTractionSeverity: carChannels[7],
    carLateralVelocity: carChannels[8],
    carDriftCommittedSlip: carChannels[9],
    carDriftCommittedTurnSign: carChannels[10],
    carDriftCommittedReArmed: carChannels[11],
    carDriftPosePose: carChannels[12],
    carDriftPoseRate: carChannels[13],
    carDriftPoseArticulation: carChannels[14],
    carTick: carChannels[15],
  };
  return { buffer, frame, carChannels };
}

/**
 * Writes the frame header + discrete rules scalars of a packed frame.
 *
 * @param frame - The slot frame to mutate in place.
 * @param input - The validated packing input.
 */
function writePackedFrameHeader(
  frame: NeatRacingPackedFrame,
  input: NeatRacingPackedFrameInput,
): void {
  frame.fromTick = input.fromTick;
  frame.toTick = input.toTick;
  frame.agentCount = input.cars.length;
  frame.featureFlags = encodeNeatRacingFeatureFlags(
    input.featureFlags ?? NEATRACING_DEFAULT_FEATURE_FLAGS,
  );
  // Discrete rules scalars: exact numbers (never float32-rounded).
  frame.level = input.rules.level;
  frame.lapsCompleted = input.rules.lapsCompleted;
  frame.raceStatusCode =
    NEATRACING_PACKED_RACE_STATUS_CODES[
      input.rules.raceStatus as keyof typeof NEATRACING_PACKED_RACE_STATUS_CODES
    ];
  frame.nextRaceLevel =
    input.rules.nextRaceLevel === null
      ? NEATRACING_PACK_NULL_LEVEL_SENTINEL
      : input.rules.nextRaceLevel;
}

/**
 * Fills the continuous rules span of a packed frame.
 *
 * @param frame - The slot frame to mutate in place.
 * @param input - The validated packing input.
 */
function fillPackedFrameRulesChannel(
  frame: NeatRacingPackedFrame,
  input: NeatRacingPackedFrameInput,
): void {
  frame.rulesChannel[0] = input.rules.checkpointBudgetS;
  frame.rulesChannel[1] = input.rules.elapsedS;
}

/**
 * Fills every per-car SoA channel from the validated fleet, walking the
 * fixed channel table (zero allocation, deterministic float32 writes).
 *
 * @param frame - The slot frame whose channel views get filled.
 * @param cars - The validated fleet snapshot (car 0 first).
 */
function fillPackedFrameCarChannels(
  frame: NeatRacingPackedFrame,
  cars: readonly PhysicsCarState[],
): void {
  for (let carIndex = 0; carIndex < cars.length; carIndex++) {
    const car = cars[carIndex];
    for (
      let channelIndex = 0;
      channelIndex < NEATRACING_PACKED_CAR_CHANNEL_COUNT;
      channelIndex++
    ) {
      const channel = NEATRACING_PACKED_CAR_CHANNELS[channelIndex];
      // The slot's channel views line up with the table by construction.
      const channelView = frame[
        channel.name as keyof NeatRacingPackedFrame
      ] as Float32Array;
      channelView[carIndex] = channel.reader(car);
    }
  }
}

/**
 * Reconstructs the race-rules state from a packed frame.
 *
 * @param frame - The packed frame to read (sentinel already checked).
 * @returns The reconstructed rules (Float32-lossy continuous fields).
 * @throws {RangeError} When the packed status code is not in the table.
 */
function reconstructPackedFrameRules(
  frame: NeatRacingPackedFrame,
): RaceRulesState {
  const raceStatus =
    NEATRACING_PACKED_RACE_STATUS_BY_CODE[frame.raceStatusCode];
  if (raceStatus === undefined) {
    throw new RangeError(
      `Unsupported NeatRacingPackedFrame raceStatusCode: ${String(frame.raceStatusCode)}`,
    );
  }
  return {
    level: frame.level,
    lapsCompleted: frame.lapsCompleted,
    checkpointBudgetS: frame.rulesChannel[0],
    elapsedS: frame.rulesChannel[1],
    raceStatus,
    nextRaceLevel:
      frame.nextRaceLevel === NEATRACING_PACK_NULL_LEVEL_SENTINEL
        ? null
        : frame.nextRaceLevel,
  };
}

/**
 * Reconstructs one physics car state from a packed frame's channels.
 *
 * @param frame - The packed frame to read (sentinel already checked).
 * @param carIndex - Fleet index to reconstruct (below `agentCount`).
 * @returns The reconstructed car (Float32-lossy physics fields).
 */
function reconstructPackedFrameCar(
  frame: NeatRacingPackedFrame,
  carIndex: number,
): PhysicsCarState {
  return {
    x: frame.carX[carIndex],
    y: frame.carY[carIndex],
    heading: frame.carHeading[carIndex],
    speed: frame.carSpeed[carIndex],
    yawRate: frame.carYawRate[carIndex],
    slipAngle: frame.carSlipAngle[carIndex],
    travelHeadingRad: frame.carTravelHeadingRad[carIndex],
    tractionSeverity: frame.carTractionSeverity[carIndex],
    lateralVelocity: frame.carLateralVelocity[carIndex],
    drift: {
      committed: {
        slip: frame.carDriftCommittedSlip[carIndex],
        turnSign: frame.carDriftCommittedTurnSign[carIndex],
        reArmed: frame.carDriftCommittedReArmed[carIndex] !== 0,
      },
      pose: {
        pose: frame.carDriftPosePose[carIndex],
        poseRate: frame.carDriftPoseRate[carIndex],
        articulation: frame.carDriftPoseArticulation[carIndex],
      },
    },
    tick: frame.carTick[carIndex],
  };
}

/**
 * Creates a preallocated frame pack pool.
 *
 * Educational note: the pool is the AC-042 unit — one `ArrayBuffer` per
 * slot allocated up front, partitioned into a rules span plus per-car SoA
 * channel spans at computed byte offsets. Every later `packFrame` is a
 * pure in-place fill: no view construction, no buffer churn, no per-frame
 * allocation in the worker's 60 Hz budget.
 *
 * @param options - Capacity and slot count; defaults documented on the type.
 * @returns The pool handle (pack / unpack + diagnostics).
 * @throws {RangeError} When capacity or slots violate their bounds.
 *
 * @example
 * ```ts
 * const pool = createNeatRacingFramePackPool({ carCapacity: 3 });
 * const packed = pool.packFrame({ fromTick: 0, toTick: 2, cars, rules });
 * worker.postMessage(
 *   { type: 'frames', payload: packed },
 *   resolveNeatRacingPackedFrameTransferList(packed),
 * );
 * const snapshot = pool.unpackFrame(packed);
 * ```
 */
export function createNeatRacingFramePackPool(
  options: NeatRacingFramePackPoolOptions,
): NeatRacingFramePackPool {
  // Step 1: Validate capacity against the mirrored sim-worker fleet cap.
  if (
    !Number.isInteger(options.carCapacity) ||
    options.carCapacity < NEATRACING_PACKED_MIN_CARS ||
    options.carCapacity > NEATRACING_PACKED_MAX_CAR_CAPACITY
  ) {
    throw new RangeError(
      'frame pack pool failed: carCapacity must be an integer in [1, 32]',
    );
  }
  // Step 2: Validate slot count.
  const poolSlots = options.poolSlots ?? NEATRACING_FRAME_PACK_POOL_SLOTS;
  if (
    !Number.isInteger(poolSlots) ||
    poolSlots < NEATRACING_FRAME_PACK_MIN_POOL_SLOTS ||
    poolSlots > NEATRACING_FRAME_PACK_MAX_POOL_SLOTS
  ) {
    throw new RangeError(
      'frame pack pool failed: poolSlots must be an integer in [1, 4]',
    );
  }

  // Step 3: Allocate every slot exactly once (the only allocation the pool does).
  const slots: PoolSlot[] = [];
  for (let slotIndex = 0; slotIndex < poolSlots; slotIndex++) {
    slots.push(createPoolSlot(options.carCapacity));
  }
  let slotCursor = 0;

  return {
    carCapacity: options.carCapacity,
    poolSlots,
    slotByteLength: resolveNeatRacingPackedSlotByteLength(options.carCapacity),
    packFrame(input: NeatRacingPackedFrameInput): NeatRacingPackedFrame {
      // Step 1: Round-robin to the next slot (modular cursor, no allocation).
      const slot = slots[slotCursor];
      slotCursor = (slotCursor + 1) % poolSlots;
      // Step 2: Validate the fleet and rules before touching any slot bytes.
      const carsError = validatePackedFrameCars(
        input.cars,
        options.carCapacity,
      );
      if (carsError !== null) {
        throw new RangeError(carsError.error);
      }
      const rulesError = validatePackedFrameRules(input.rules);
      if (rulesError !== null) {
        throw new RangeError(rulesError.error);
      }
      // Step 3: Guard the released-tick envelope ordering.
      if (input.toTick < input.fromTick) {
        throw new RangeError(
          'frame pack failed: toTick must not precede fromTick',
        );
      }
      // Step 4: Fill header, rules span and car channels in place.
      writePackedFrameHeader(slot.frame, input);
      fillPackedFrameRulesChannel(slot.frame, input);
      fillPackedFrameCarChannels(slot.frame, input.cars);
      return slot.frame;
    },
    unpackFrame(frame: NeatRacingPackedFrame): NeatRacingPackedFrameSnapshot {
      // Step 1: Guard the schema sentinel before reading any bytes.
      assertNeatRacingPackedFrameSchemaVersion(frame);
      // Step 2: Reconstruct rules + fleet (plain objects; the bridge seam).
      const rules = reconstructPackedFrameRules(frame);
      const cars: PhysicsCarState[] = [];
      for (let carIndex = 0; carIndex < frame.agentCount; carIndex++) {
        cars.push(reconstructPackedFrameCar(frame, carIndex));
      }
      // Step 3: Return the typed snapshot carrying header scalars verbatim.
      return {
        fromTick: frame.fromTick,
        toTick: frame.toTick,
        agentCount: frame.agentCount,
        featureFlags: frame.featureFlags,
        rules,
        cars,
      };
    },
  };
}
