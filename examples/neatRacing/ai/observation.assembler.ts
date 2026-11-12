/**
 * NeatRacing observation assembler (slice `P2-S2-B`, AC-027).
 *
 * THE VECTOR: the deterministic per-car observation vector of the
 * NeatRacing voxel racer. AC-027 pins THREE prefixes of the SAME channel
 * layout, one per ladder level, so the NGE genotype/phenotype controller
 * of Step 03 can grow without any channel ever moving:
 *
 * - Tier A (level 1) = 82 channels: base 70 + wheel slip 4 + rival gaps 8
 * - Tier B (level 2) = 89 channels: Tier A + self radio 7
 * - Tier C (level 3) = 110 channels: Tier B + teammate radio 3 x 7 = 21
 *
 * THE PREFIX (the whole point): the base channels stay `[0..69]`, the
 * wheel-slip block stays `[70..73]`, the rival-gap block stays
 * `[74..81]`, the self-radio block stays `[82..88]` and the
 * teammate-radio block stays `[89..109]` on EVERY level. A Tier A
 * network's input space is a strict slice of Tier B's and Tier C's.
 *
 * THE MIRROR: the base-70 block is the owner-local mirror of the
 * `racing_curriculum` observation vocabulary — the same 20 scalar,
 * 40 look-ahead and 10 memory-trace channels with the SAME recycled
 * §11.1 scales, re-expressed for `PhysicsCarState` without a single
 * cross-example runtime import (only the shared frozen `TrackSpec` /
 * `SplineSample` TYPES flow in, exactly like `physics/ai.autopilot.ts`).
 *
 * THE ONE SOURCE OF TRUTH (AC-086): the optimal-line lateral offset and
 * the curvature-derived target-speed channels are derived from the
 * PUBLISHED racing-line table (`resolveAutopilotRacingLine`), the same
 * memoized table the autopilot drives on. There is NO second geometry
 * anywhere in the repo.
 *
 * THE WHEEL SLIP (LD-15): the physics publishes ONE car-level
 * `yawRate`/`slipAngle` pair, so the four per-wheel channels are the
 * DETERMINISTIC SIDE PROJECTION of that pair (see
 * `collectNeatRacingWheelSlipChannels` for the exact projection the
 * fixture pins).
 *
 * THE REALM: this module is a PURE headless seam — no `self`, no
 * `postMessage`, no DOM, no RNG, no wall-clock reads. The same (state,
 * track, level) triple yields a byte-identical vector for the batch-83
 * replay contract.
 *
 * @module
 */

import type {
  SplineSample,
  TrackSpec,
} from '../../racing_curriculum/track/track.generator.types';
import { resolveAutopilotRacingLine } from '../physics/ai.autopilot';

/* ------------------------------------------------------------------------- *
 * AC-027 channel structure (§9.3)
 * ------------------------------------------------------------------------- */

/** Base channel count shared by every tier (the recycled owner-local block). */
export const NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT = 70;

/** Scalar channel count of the base block (car state, tier and lane info). */
export const NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT = 20;

/** Look-ahead segment slice count of the base block. */
export const NEATRACING_OBSERVATION_LOOKAHEAD_SEGMENT_COUNT = 5;

/** Per-segment look-ahead channel count (position, tail, tangent, width, gap). */
export const NEATRACING_OBSERVATION_CHANNELS_PER_LOOKAHEAD_SEGMENT = 8;

/** Look-ahead channel count of the base block (5 segments x 8 channels). */
export const NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT = 40;

/** Recurrent memory-trace channel count of the base block. */
export const NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT = 10;

/**
 * Look-ahead SAMPLE STRIDE between consecutive look-ahead slices.
 *
 * The recycled vocabulary strides 18 samples per look-ahead segment — the
 * shared spline sample density — so slice `i` reads the sample
 * `closest + i x 18` (wrapped around the closed ring). The constant mirrors
 * that stride as a NeatRacing-owned number instead of importing the shared
 * runtime constant.
 */
export const NEATRACING_OBSERVATION_LOOKAHEAD_SAMPLE_STRIDE = 18;

/** Per-wheel slip channel count appended after the base block (AC-027). */
export const NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT = 4;

/**
 * Slip-angle saturation scale of the wheel-slip derivation (LD-15).
 *
 * `PhysicsCarState.slipAngle` is a signed angle in radians bounded to
 * `(-pi/2, pi/2)`, so this recycled saturation angle maps the published
 * slip onto the unit interval. Reused UNCHANGED from the recycled scalar
 * block (`slipAngle / (pi / 2)`), per the §11.1 no-renormalization clause.
 */
export const NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE = Math.PI / 2;

/**
 * Yaw-rate scale of the wheel-slip derivation (LD-15).
 *
 * The wheel-slip contributor consumes `yawRateRadiansPerSecond` with the
 * recycled yaw-rate scale (1 rad/s -> 1.0), so the derivation reuses the
 * SAME normalization the recycled scalar channel 7 uses.
 */
export const NEATRACING_OBSERVATION_WHEEL_SLIP_YAW_RATE_SCALE = 1;

/** Rival-gap channel count appended after the wheel-slip block (AC-027). */
export const NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT = 8;

/**
 * Rival slot capacity covered by the 8 rival-gap channels.
 *
 * AC-027: the 8 rival-gap channels cover the 4-rival case of the level 3
 * ladder (`[0, 1, 1, 1, 1]`), so the block holds FOUR slots of TWO
 * channels (at least 4 rival slots at level 3).
 */
export const NEATRACING_OBSERVATION_RIVAL_GAP_SLOT_COUNT = 4;

/** Per-rival gap channel count: [longitudinal body gap, lateral body gap]. */
export const NEATRACING_OBSERVATION_RIVAL_GAP_CHANNELS_PER_SLOT = 2;

/** Self-radio channel count of the Tier B tail (raw recycled radio field). */
export const NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT = 7;

/** Teammate radio channel count of the Tier C tail (3 slots x 7 channels). */
export const NEATRACING_OBSERVATION_TEAMMATE_RADIO_CHANNEL_COUNT = 21;

/** Teammate radio slot capacity of the Tier C tail. */
export const NEATRACING_OBSERVATION_TEAMMATE_RADIO_SLOT_COUNT = 3;

/** Per-teammate slot channel count (the recycled 7-channel radio slot). */
export const NEATRACING_OBSERVATION_CHANNELS_PER_TEAMMATE_SLOT = 7;

/** Tier A (level 1) vector length: 70 + 4 + 8. */
export const NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT = 82;

/** Tier B (level 2) vector length: 82 + 7. */
export const NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT = 89;

/** Tier C (level 3) vector length: 89 + 21. */
export const NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT = 110;

/* ------------------------------------------------------------------------- *
 * Recycled §11.1 normalization scales (reused VERBATIM — no renormalization)
 * ------------------------------------------------------------------------- */

/** World-position normalization scale (96 world units -> 1.0). */
export const NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE = 96;

/** Distance normalization scale (64 world units -> 1.0). */
export const NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE = 64;

/** Speed normalization scale (108 world units/s -> 1.0). */
export const NEATRACING_OBSERVATION_SPEED_WORLD_SCALE = 108;

/** Lateral-speed normalization scale (54 world units/s -> 1.0). */
export const NEATRACING_OBSERVATION_LATERAL_SPEED_WORLD_SCALE = 54;

/** Boundary distance and lane-width normalization scale (24 units -> 1.0). */
export const NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE = 24;

/** Optimal-line lateral-offset normalization scale (18 units -> 1.0). */
export const NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE = 18;

/** Yaw-rate normalization scale (1 rad/s -> 1.0). */
export const NEATRACING_OBSERVATION_YAW_RATE_RADIANS_PER_SECOND_SCALE = 1;

/** Fallback curvature target speed when the racing-line table is absent. */
export const NEATRACING_OBSERVATION_DEFAULT_TARGET_SPEED_WORLD = 72;

/* ------------------------------------------------------------------------- *
 * Tier mapping and team layouts (E2)
 * ------------------------------------------------------------------------- */

/**
 * NeatRacing observation level (the ladder rung that selects the tail).
 *
 * The spec E2 mapping is level -> tier: level 1 -> Tier A (82 channels),
 * level 2 -> Tier B (89 channels) and level 3 -> Tier C (110 channels).
 */
export type NeatRacingObservationLevel = 1 | 2 | 3;

/**
 * The E2 team layouts of the 1/2/5-car ladder, pinned exactly as specified.
 *
 * - level 1 (`[0]`): the solo rung — one car, no rivals, no teammates.
 * - level 2 (`[0, 1]`): two entries (team A vs team B).
 * - level 3 (`[0, 1, 1, 1, 1]`): the human/team A car 0 and FOUR team B
 *   rivals — the case the 8 rival-gap channels were sized for.
 */
export const NEATRACING_TEAM_LAYOUTS: Readonly<
  Record<NeatRacingObservationLevel, readonly number[]>
> = Object.freeze({
  1: Object.freeze([0]),
  2: Object.freeze([0, 1]),
  3: Object.freeze([0, 1, 1, 1, 1]),
});

/**
 * Resolves the observation channel count of one ladder level.
 *
 * @param level - Ladder level (1, 2 or 3) to measure.
 * @returns The pinned channel count: 82 for level 1, 89 for level 2 and 110 for level 3.
 * @throws RangeError for a level outside the pinned ladder.
 *
 * @example
 * ```ts
 * resolveNeatRacingObservationChannelCount(1); // 82
 * resolveNeatRacingObservationChannelCount(3); // 110
 * ```
 */
export function resolveNeatRacingObservationChannelCount(
  level: NeatRacingObservationLevel,
): number {
  if (level === 1) {
    return NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT;
  }

  if (level === 2) {
    return NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT;
  }

  if (level === 3) {
    return NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT;
  }

  throw new RangeError(
    `NeatRacing observation level must be 1, 2 or 3 (received ${level}).`,
  );
}

/* ------------------------------------------------------------------------- *
 * Seam types (owner-local mirror of the recycled observation vocabulary)
 * ------------------------------------------------------------------------- */

/**
 * One roster car as the seam consumes it for the rival and teammate tails.
 *
 * The vocabulary mirrors the recycled per-car state: world pose, optional
 * speed and the team index. The roster ARRAY INDEX is the car index, so
 * the focal car is excluded by its array position exactly like the
 * recycled roster derivation.
 */
export type NeatRacingObservationCarState = {
  /** World x position in meters. */
  readonly carX: number;
  /** World y position in meters. */
  readonly carY: number;
  /** Heading angle in radians, wrapped to `[-pi, pi)` (PhysicsCarState). */
  readonly carHeading: number;
  /** Longitudinal (forward) speed in m/s (`PhysicsCarState.speed`). */
  readonly speedWorld?: number;
  /** Team index (`0 = team A`, `1 = team B` per the E2 ladder). */
  readonly teamIndex: 0 | 1;
};

/**
 * The observation seam state of ONE focal car.
 *
 * Required fields cover the `PhysicsCarState` kinematics the vector needs;
 * every optional field mirrors a recycled observation extension and falls
 * back to its deterministic derived default when absent (see
 * `resolveNeatRacingObservationState`). The seam keeps the recycled field
 * names (`carX`, `carHeading`, `radioField`, `memoryTrace`, ...) so the
 * Step 03 controller wiring needs no translation layer, while remaining a
 * NeatRacing-owned type with NO cross-example import.
 */
export type NeatRacingObservationState = {
  /** Focal car world x position in meters. */
  readonly carX: number;
  /** Focal car world y position in meters. */
  readonly carY: number;
  /** Focal car heading in radians, wrapped to `[-pi, pi)` (PhysicsCarState). */
  readonly carHeading: number;
  /** Focal car forward speed world units/s (`PhysicsCarState.speed`). */
  readonly forwardSpeedWorld?: number;
  /** Focal car lateral speed world units/s (positive LEFT of the heading). */
  readonly lateralSpeedWorld?: number;
  /** Focal car total speed override; derived from the parts when absent. */
  readonly speedWorld?: number;
  /**
   * Focal car yaw rate in RADIANS PER SECOND (LD-15).
   *
   * `PhysicsCarState.yawRate` is radians per 60 Hz tick (house precedent),
   * so callers convert with `yawRate * 60` before feeding this seam.
   */
  readonly yawRateRadiansPerSecond?: number;
  /** Focal car slip angle in radians (`PhysicsCarState.slipAngle`, LD-15). */
  readonly slipAngleRadians?: number;
  /** Focal team index override; derived from the roster entry when absent. */
  readonly teamIndex?: 0 | 1;
  /** Lap progress in `[0, 1]`; derived from the nearest sample when absent. */
  readonly progress01?: number;
  /** Full-lap progress in `[0, 1]`; falls back to `progress01` when absent. */
  readonly lapProgress01?: number;
  /** Left boundary distance; derived from the lane half-width when absent. */
  readonly boundaryDistanceLeftWorld?: number;
  /** Right boundary distance; derived from the lane half-width when absent. */
  readonly boundaryDistanceRightWorld?: number;
  /** Hazard distance; defaults to the distance scale (a full unit of 1.0). */
  readonly hazardDistanceWorld?: number;
  /** Waypoint distance; derived from the nearest spline sample when absent. */
  readonly waypointDistanceWorld?: number;
  /**
   * Optimal-line lateral offset override; derived from the AC-086 racing
   * line when absent (signed lateral offset minus the published line offset).
   */
  readonly optimalLineLateralOffsetWorld?: number;
  /** Optimal-line heading error override; derived from the tangent when absent. */
  readonly optimalLineHeadingErrorRadians?: number;
  /** Target speed override; derived from the racing-line table when absent. */
  readonly targetSpeedWorld?: number;
  /** Recurrent memory trace (clamped into `[-1, 1]`, zero-padded to 10). */
  readonly memoryTrace?: readonly number[];
  /** Self-radio field (7 raw channels; the Tier B tail passes it through). */
  readonly radioField?: Float32Array | readonly number[];
  /** Ordered car roster (array index = car index) driving the rival/teammate tails. */
  readonly cars?: readonly NeatRacingObservationCarState[];
  /** Roster index of the focal car (default 0). */
  readonly carIndex?: number;
};

/** Options selecting the ladder level whose tail the vector must carry. */
export type NeatRacingObservationAssemblerOptions = {
  /** Ladder level to assemble: 1 -> Tier A (82), 2 -> Tier B (89), 3 -> Tier C (110). */
  readonly level: NeatRacingObservationLevel;
};

/* ------------------------------------------------------------------------- *
 * Internal shapes
 * ------------------------------------------------------------------------- */

/** Spline sample frame (tangent + left normal) mirrored owner-locally. */
type NeatRacingSplineSampleFrame = {
  /** Tangent heading in radians (chord toward the next ring sample). */
  readonly tangentHeadingRadians: number;
  /** Left-normal x component (positive to the left of travel). */
  readonly normalX: number;
  /** Left-normal y component (positive to the left of travel). */
  readonly normalY: number;
};

/** Bounding box + center used by the position channels (recycled contract). */
type NeatRacingTrackBounds = {
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
  readonly centerX: number;
  readonly centerY: number;
};

/** Fully resolved observation state (no optional field left unresolved). */
type ResolvedNeatRacingObservationState = {
  readonly forwardSpeedWorld: number;
  readonly lateralSpeedWorld: number;
  readonly speedWorld: number;
  readonly yawRateRadiansPerSecond: number;
  readonly slipAngleRadians: number;
  readonly progress01: number;
  readonly lapProgress01: number;
  readonly boundaryDistanceLeftWorld: number;
  readonly boundaryDistanceRightWorld: number;
  readonly hazardDistanceWorld: number;
  readonly waypointDistanceWorld: number;
  readonly optimalLineLateralOffsetWorld: number;
  readonly optimalLineHeadingErrorRadians: number;
  readonly targetSpeedWorld: number;
  readonly memoryTrace: readonly number[];
  readonly radioField: readonly number[];
};

/** Assembly context threaded through the collector helpers. */
type NeatRacingObservationContext = {
  readonly envState: NeatRacingObservationState;
  readonly trackSpec: TrackSpec;
  readonly trackBounds: NeatRacingTrackBounds;
  readonly closestSplineSampleIndex: number;
  readonly resolvedObservationState: ResolvedNeatRacingObservationState;
};

/* ------------------------------------------------------------------------- *
 * Wheel-slip projection (LD-15)
 * ------------------------------------------------------------------------- */

/**
 * Per-wheel SIGN of the shared slide, ordered front-left, front-right,
 * rear-left, rear-right.
 *
 * THE DERIVATION (the fixture-owning detail of this slice): the physics
 * publishes ONE car-level slip pair, so each wheel channel reads the SAME
 * slide intensity with SIDE polarity — the left-side pair (+1) and the
 * right-side pair (-1) see opposite signs of the car's slide, and the
 * intensity blends the two LD-15 contributors at their recycled scales
 * (slip angle / (pi/2), yaw rate rad/s / 1). A straight drive keeps every
 * channel exactly 0; a slide makes the wheel sides diverge.
 */
const NEATRACING_WHEEL_SLIP_SIDE_POLARITY: readonly [1, -1, 1, -1] = [
  1, -1, 1, -1,
];

/* ------------------------------------------------------------------------- *
 * Internal defaults
 * ------------------------------------------------------------------------- */

/** Empty radio field used when the environment has not produced one yet. */
const EMPTY_RADIO_FIELD: readonly number[] = [];

/** Zeroed memory trace used when the environment has not produced one yet. */
const EMPTY_MEMORY_TRACE: readonly number[] = new Array<number>(
  NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT,
).fill(0);

/* ------------------------------------------------------------------------- *
 * Public entry point
 * ------------------------------------------------------------------------- */

/**
 * Assembles the deterministic NeatRacing observation vector of one focal car.
 *
 * THE CONTRACT (AC-027): the returned vector is a prefix-preserving
 * Float32Array whose length is pinned by the ladder level — 82 (Tier A),
 * 89 (Tier B) or 110 (Tier C) channels:
 *
 * - `[0..69]` the recycled base block (20 scalars + 5 x 8 look-ahead + 10
 *   memory-trace) with the §11.1 scales reused verbatim;
 * - `[70..73]` the per-wheel slip channels derived from the LD-15 pair;
 * - `[74..81]` the four 2-channel rival-gap slots (body-frame longitudinal
 *   and lateral gap, roster order, zero-padded beyond the roster);
 * - `[82..88]` the raw self-radio tail (level 2 and 3);
 * - `[89..109]` the three 7-channel teammate radio slots (level 3 only),
 *   zero-padded beyond the same-team roster.
 *
 * @param envState - Focal car observation seam state (see `NeatRacingObservationState`).
 * @param trackSpec - Frozen track geometry (the shared `TrackSpec` contract).
 * @param options - The ladder level whose tail the vector must carry.
 * @returns The normalized observation vector.
 * @throws RangeError when the track spline-sample ring is empty.
 *
 * @example
 * ```ts
 * const vector = assembleNeatRacingObservationVector(
 *   { carX: 2, carY: 0, carHeading: 0, forwardSpeedWorld: 10 },
 *   trackSpec,
 *   { level: 3 },
 * );
 * vector.length; // 110
 * Array.from(vector.slice(0, 82)); // the Tier A prefix
 * ```
 */
export function assembleNeatRacingObservationVector(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
  options: NeatRacingObservationAssemblerOptions,
): Float32Array {
  const observationContext = createNeatRacingObservationContext(
    envState,
    trackSpec,
  );

  // Step 1: Assemble the 70-channel base block (scalars + look-ahead + memory).
  const baseChannels = collectNeatRacingBaseChannels(observationContext);

  // Step 2: Derive the four LD-15 wheel-slip channels.
  const wheelSlipChannels = collectNeatRacingWheelSlipChannels(
    observationContext.resolvedObservationState,
  );

  // Step 3: Derive the four 2-channel rival-gap slots (roster order).
  const rivalGapChannels = collectNeatRacingRivalGapChannels(
    observationContext.envState,
  );

  // Step 4: Resolve the ladder level and its prefix-preserving length.
  const level = options.level;
  const channelCount = resolveNeatRacingObservationChannelCount(level);
  const observationVector = new Float32Array(channelCount);

  observationVector.set(baseChannels, 0);
  observationVector.set(
    wheelSlipChannels,
    NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT,
  );
  observationVector.set(
    rivalGapChannels,
    NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
      NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
  );

  // Step 5: Append the raw self-radio tail for Tier B and Tier C.
  if (level >= 2) {
    observationVector.set(
      collectNeatRacingSelfRadioChannels(
        observationContext.resolvedObservationState.radioField,
      ),
      NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
        NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT +
        NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT,
    );
  }

  // Step 6: Append the teammate radio tail for Tier C.
  if (level >= 3) {
    observationVector.set(
      collectNeatRacingTeammateRadioChannels(envState),
      NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
        NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT +
        NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT +
        NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT,
    );
  }

  return observationVector;
}

/* ------------------------------------------------------------------------- *
 * Context resolution
 * ------------------------------------------------------------------------- */

/**
 * Builds the assembly context (nearest sample, track bounds, resolved state).
 *
 * @param envState - Focal car observation seam state.
 * @param trackSpec - Frozen track geometry.
 * @returns The resolved assembly context.
 * @throws RangeError when the spline-sample ring is empty (the frozen
 * `TrackSpec` contract guarantees a non-empty closed loop).
 */
function createNeatRacingObservationContext(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
): NeatRacingObservationContext {
  if (trackSpec.splineSamples.length === 0) {
    throw new RangeError(
      'NeatRacing observation assembly requires a non-empty spline-sample loop (frozen TrackSpec contract).',
    );
  }

  const closestSplineSampleIndex = findClosestNeatRacingSplineSampleIndex(
    envState,
    trackSpec,
  );

  return {
    envState,
    trackSpec,
    trackBounds: resolveNeatRacingTrackBounds(trackSpec),
    closestSplineSampleIndex,
    resolvedObservationState: resolveNeatRacingObservationState(
      envState,
      trackSpec,
      closestSplineSampleIndex,
    ),
  };
}

/**
 * Resolves all optional observation fields, deriving safe defaults when the
 * live environment has not produced them yet.
 *
 * THE OPTIMAL LINE (AC-086 single source of truth): the lateral-offset and
 * target-speed defaults are resolved from the PUBLISHED racing-line table
 * (`resolveAutopilotRacingLine`) at the nearest spline sample — the same
 * memoized table the autopilot drives on. The signed lateral offset of the
 * car relative to that line is the car's own centerline offset minus the
 * line's published offset (positive when the car sits LEFT of the line).
 *
 * @param envState - Focal car observation seam state.
 * @param trackSpec - Frozen track geometry.
 * @param closestSplineSampleIndex - Global index of the nearest spline sample.
 * @returns The complete observation state ready for normalization.
 */
function resolveNeatRacingObservationState(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
): ResolvedNeatRacingObservationState {
  const closestSplineSample =
    trackSpec.splineSamples[closestSplineSampleIndex]!;
  const closestSplineSampleFrame = resolveNeatRacingSplineSampleFrame(
    trackSpec.splineSamples,
    closestSplineSample.globalIndex,
  );
  const signedLateralOffsetWorld = resolveNeatRacingSignedLateralOffsetWorld(
    envState,
    closestSplineSample,
    closestSplineSampleFrame,
  );
  const racingLine = resolveAutopilotRacingLine(trackSpec.splineSamples);
  const racingLinePoint = racingLine[closestSplineSampleIndex];
  // THE SINGLE SOURCE OF TRUTH: the racing-line table publishes a signed
  // lateral offset (positive LEFT of travel) — subtracting it from the
  // car's own signed centerline offset yields the car's offset relative
  // to the OPTIMAL line, with NO second geometry behind it.
  const optimalLineLateralOffsetWorld =
    signedLateralOffsetWorld - (racingLinePoint?.offsetM ?? 0);
  const forwardSpeedWorld = envState.forwardSpeedWorld ?? 0;
  const lateralSpeedWorld = envState.lateralSpeedWorld ?? 0;
  const speedWorld =
    envState.speedWorld ?? Math.hypot(forwardSpeedWorld, lateralSpeedWorld);
  const halfTrackWidthWorld = closestSplineSample.width / 2;
  const derivedWaypointDistanceWorld = Math.hypot(
    closestSplineSample.x - envState.carX,
    closestSplineSample.y - envState.carY,
  );
  const derivedProgress01 =
    closestSplineSampleIndex / trackSpec.splineSamples.length;

  return {
    forwardSpeedWorld,
    lateralSpeedWorld,
    speedWorld,
    yawRateRadiansPerSecond: envState.yawRateRadiansPerSecond ?? 0,
    slipAngleRadians: envState.slipAngleRadians ?? 0,
    progress01: clamp01NeatRacing(envState.progress01 ?? derivedProgress01),
    lapProgress01: clamp01NeatRacing(
      envState.lapProgress01 ?? envState.progress01 ?? derivedProgress01,
    ),
    boundaryDistanceLeftWorld: Math.max(
      0,
      envState.boundaryDistanceLeftWorld ??
        halfTrackWidthWorld - signedLateralOffsetWorld,
    ),
    boundaryDistanceRightWorld: Math.max(
      0,
      envState.boundaryDistanceRightWorld ??
        halfTrackWidthWorld + signedLateralOffsetWorld,
    ),
    hazardDistanceWorld:
      envState.hazardDistanceWorld ??
      NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
    waypointDistanceWorld:
      envState.waypointDistanceWorld ?? derivedWaypointDistanceWorld,
    optimalLineLateralOffsetWorld:
      envState.optimalLineLateralOffsetWorld ?? optimalLineLateralOffsetWorld,
    optimalLineHeadingErrorRadians:
      envState.optimalLineHeadingErrorRadians ??
      wrapNeatRacingAngleToMinusPiPi(
        closestSplineSampleFrame.tangentHeadingRadians - envState.carHeading,
      ),
    targetSpeedWorld:
      envState.targetSpeedWorld ??
      racingLinePoint?.targetSpeedUnitsPerS ??
      NEATRACING_OBSERVATION_DEFAULT_TARGET_SPEED_WORLD,
    memoryTrace: resolveNeatRacingMemoryTrace(envState.memoryTrace),
    radioField: resolveNeatRacingRadioTail(envState.radioField),
  };
}

/**
 * Finds the spline sample whose lane-center point is closest to the car.
 *
 * @param envState - Focal car observation seam state.
 * @param trackSpec - Frozen track geometry.
 * @returns Global index of the nearest spline sample.
 */
function findClosestNeatRacingSplineSampleIndex(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
): number {
  let bestSampleIndex = 0;
  let bestDistanceWorld = Number.POSITIVE_INFINITY;

  for (const splineSample of trackSpec.splineSamples) {
    const distanceToSampleWorld = Math.hypot(
      splineSample.x - envState.carX,
      splineSample.y - envState.carY,
    );

    if (distanceToSampleWorld < bestDistanceWorld) {
      bestDistanceWorld = distanceToSampleWorld;
      bestSampleIndex = splineSample.globalIndex;
    }
  }

  return bestSampleIndex;
}

/**
 * Resolves the track bounds used to normalize the world-position channels.
 *
 * @param trackSpec - Frozen track geometry.
 * @returns Bounding box plus center point of the spline-sample ring.
 */
function resolveNeatRacingTrackBounds(
  trackSpec: TrackSpec,
): NeatRacingTrackBounds {
  const initialSplineSample = trackSpec.splineSamples[0]!;
  let minX = initialSplineSample.x;
  let maxX = initialSplineSample.x;
  let minY = initialSplineSample.y;
  let maxY = initialSplineSample.y;

  for (const splineSample of trackSpec.splineSamples) {
    minX = Math.min(minX, splineSample.x);
    maxX = Math.max(maxX, splineSample.x);
    minY = Math.min(minY, splineSample.y);
    maxY = Math.max(maxY, splineSample.y);
  }

  return {
    minX,
    maxX,
    minY,
    maxY,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
  };
}

/* ------------------------------------------------------------------------- *
 * Base block collectors (recycled base-70 mirror)
 * ------------------------------------------------------------------------- */

/**
 * Collects the 70 base channels (scalars + look-ahead + memory trace).
 *
 * @param observationContext - Resolved assembly context.
 * @returns The normalized base block.
 */
function collectNeatRacingBaseChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array {
  const scalarChannels = collectNeatRacingScalarChannels(observationContext);
  const lookAheadChannels =
    collectNeatRacingLookAheadChannels(observationContext);
  const memoryTraceChannels =
    collectNeatRacingMemoryTraceChannels(observationContext);
  const baseChannels = new Float32Array(
    NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT,
  );

  baseChannels.set(scalarChannels, 0);
  baseChannels.set(
    lookAheadChannels,
    NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT,
  );
  baseChannels.set(
    memoryTraceChannels,
    NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT +
      NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT,
  );

  return baseChannels;
}

/**
 * Collects the 20 scalar channels that describe the current car state.
 *
 * The channel ORDER and the §11.1 scales mirror the recycled vocabulary
 * verbatim: position (96), heading (sin/cos), body-frame speeds (108/54),
 * speed (108), yaw rate (1), slip angle (pi/2), progress probabilities,
 * boundary distances (24) and balance, hazard/waypoint distances (64),
 * optimal-line offset (18) and heading error (pi), target speed (108) and
 * the target-speed margin (108).
 *
 * @param observationContext - Resolved assembly context.
 * @returns Twenty normalized scalar channels.
 */
function collectNeatRacingScalarChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array {
  const { envState, trackBounds, resolvedObservationState } =
    observationContext;

  return Float32Array.from([
    normalizeNeatRacingSignedValue(
      envState.carX - trackBounds.centerX,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      envState.carY - trackBounds.centerY,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    Math.sin(envState.carHeading),
    Math.cos(envState.carHeading),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.forwardSpeedWorld,
      NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.lateralSpeedWorld,
      NEATRACING_OBSERVATION_LATERAL_SPEED_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.speedWorld,
      NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.yawRateRadiansPerSecond,
      NEATRACING_OBSERVATION_YAW_RATE_RADIANS_PER_SECOND_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.slipAngleRadians,
      NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE,
    ),
    normalizeNeatRacingProbability(resolvedObservationState.progress01),
    normalizeNeatRacingProbability(resolvedObservationState.lapProgress01),
    normalizeNeatRacingUnsignedValue(
      resolvedObservationState.boundaryDistanceLeftWorld,
      NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
    ),
    normalizeNeatRacingUnsignedValue(
      resolvedObservationState.boundaryDistanceRightWorld,
      NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolveNeatRacingBoundaryBalance(
        resolvedObservationState.boundaryDistanceLeftWorld,
        resolvedObservationState.boundaryDistanceRightWorld,
      ),
      1,
    ),
    normalizeNeatRacingUnsignedValue(
      resolvedObservationState.hazardDistanceWorld,
      NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
    ),
    normalizeNeatRacingUnsignedValue(
      resolvedObservationState.waypointDistanceWorld,
      NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.optimalLineLateralOffsetWorld,
      NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.optimalLineHeadingErrorRadians,
      Math.PI,
    ),
    normalizeNeatRacingUnsignedValue(
      resolvedObservationState.targetSpeedWorld,
      NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      resolvedObservationState.targetSpeedWorld -
        resolvedObservationState.speedWorld,
      NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
  ]);
}

/**
 * Collects the 40 look-ahead channels used for track anticipation.
 *
 * Slice `i` (0-based) reads the spline sample `closest + i x 18` (wrapped
 * around the closed ring) plus its neighbor, mirroring the recycled
 * per-segment layout: the sample and next-sample offsets to the car (96),
 * the tangent sin/cos, the lane width (24) and the car-to-sample gap (64).
 *
 * @param observationContext - Resolved assembly context.
 * @returns Forty normalized look-ahead channels.
 */
function collectNeatRacingLookAheadChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array {
  const lookAheadChannels = new Float32Array(
    NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT,
  );

  for (
    let lookAheadOffset = 0;
    lookAheadOffset < NEATRACING_OBSERVATION_LOOKAHEAD_SEGMENT_COUNT;
    lookAheadOffset++
  ) {
    const channelOffset =
      lookAheadOffset * NEATRACING_OBSERVATION_CHANNELS_PER_LOOKAHEAD_SEGMENT;
    const splineSample = resolveNeatRacingLookAheadSample(
      observationContext,
      lookAheadOffset,
    );
    const nextSplineSample = resolveNeatRacingRelativeSplineSample(
      observationContext.trackSpec.splineSamples,
      splineSample.globalIndex,
      1,
    );
    const splineSampleFrame = resolveNeatRacingSplineSampleFrame(
      observationContext.trackSpec.splineSamples,
      splineSample.globalIndex,
    );

    lookAheadChannels.set(
      [
        normalizeNeatRacingSignedValue(
          splineSample.x - observationContext.envState.carX,
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        ),
        normalizeNeatRacingSignedValue(
          splineSample.y - observationContext.envState.carY,
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        ),
        normalizeNeatRacingSignedValue(
          nextSplineSample.x - observationContext.envState.carX,
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        ),
        normalizeNeatRacingSignedValue(
          nextSplineSample.y - observationContext.envState.carY,
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        ),
        Math.sin(splineSampleFrame.tangentHeadingRadians),
        Math.cos(splineSampleFrame.tangentHeadingRadians),
        normalizeNeatRacingUnsignedValue(
          splineSample.width,
          NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
        ),
        normalizeNeatRacingUnsignedValue(
          Math.hypot(
            splineSample.x - observationContext.envState.carX,
            splineSample.y - observationContext.envState.carY,
          ),
          NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
        ),
      ],
      channelOffset,
    );
  }

  return lookAheadChannels;
}

/**
 * Resolves one look-ahead spline sample by wrapping around the closed ring.
 *
 * @param observationContext - Resolved assembly context.
 * @param lookAheadOffset - Zero-based look-ahead slice offset.
 * @returns The spline sample encoding into that slice.
 */
function resolveNeatRacingLookAheadSample(
  observationContext: NeatRacingObservationContext,
  lookAheadOffset: number,
): SplineSample {
  return resolveNeatRacingRelativeSplineSample(
    observationContext.trackSpec.splineSamples,
    observationContext.closestSplineSampleIndex,
    lookAheadOffset * NEATRACING_OBSERVATION_LOOKAHEAD_SAMPLE_STRIDE,
  );
}

/**
 * Resolves one spline sample relative to a global index on the closed ring.
 *
 * @param splineSamples - Frozen closed-loop spline samples.
 * @param sampleIndex - Base global sample index.
 * @param sampleOffset - Positive ring offset in samples.
 * @returns The wrapped spline sample.
 */
function resolveNeatRacingRelativeSplineSample(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
  sampleOffset: number,
): SplineSample {
  const wrappedSampleIndex = wrapNeatRacingSplineSampleIndex(
    splineSamples.length,
    sampleIndex + sampleOffset,
  );

  return splineSamples[wrappedSampleIndex]!;
}

/**
 * Collects the 10-channel recurrent memory trace.
 *
 * @param observationContext - Resolved assembly context.
 * @returns Ten normalized memory-trace channels (zeros when absent).
 */
function collectNeatRacingMemoryTraceChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array {
  const memoryTraceChannels = new Float32Array(
    NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT,
  );

  for (
    let memoryIndex = 0;
    memoryIndex < NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT;
    memoryIndex++
  ) {
    memoryTraceChannels[memoryIndex] = clampNeatRacingNormalizedValue(
      observationContext.resolvedObservationState.memoryTrace[memoryIndex] ?? 0,
    );
  }

  return memoryTraceChannels;
}

/* ------------------------------------------------------------------------- *
 * AC-027 tail collectors
 * ------------------------------------------------------------------------- */

/**
 * Collects the four per-wheel slip channels (AC-027, LD-15).
 *
 * THE DERIVATION: NeatRacing publishes ONE car-level `slipAngle` and ONE
 * car-level `yawRate` (LD-15), so the per-wheel indicator is the DETERMINED
 * side projection of that shared pair rather than an invented fourth-scale
 * physical model:
 *
 * 1. the slip contributor normalizes the published slip angle by its
 *    recycled saturation scale (`pi/2` -> 1.0);
 * 2. the yaw contributor normalizes the yaw rate by its recycled scale
 *    (1 rad/s -> 1.0);
 * 3. the mean of the two becomes the slide intensity;
 * 4. the wheel sides read it with opposing polarity (left +1, right -1)
 *    so opposite wheels see opposite signs of the same slide.
 *
 * A straight drive keeps every channel exactly 0.
 *
 * @param resolvedObservationState - Fully resolved observation state.
 * @returns Four normalized wheel-slip channels (FL, FR, RL, RR).
 */
function collectNeatRacingWheelSlipChannels(
  resolvedObservationState: ResolvedNeatRacingObservationState,
): Float32Array {
  const wheelSlipChannels = new Float32Array(
    NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
  );
  const slipContributor = normalizeNeatRacingSignedValue(
    resolvedObservationState.slipAngleRadians,
    NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE,
  );
  const yawContributor = normalizeNeatRacingSignedValue(
    resolvedObservationState.yawRateRadiansPerSecond,
    NEATRACING_OBSERVATION_WHEEL_SLIP_YAW_RATE_SCALE,
  );
  const slideIntensity = clampNeatRacingNormalizedValue(
    (slipContributor + yawContributor) / 2,
  );

  for (
    let wheelIndex = 0;
    wheelIndex < NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT;
    wheelIndex++
  ) {
    const projectedSlide =
      NEATRACING_WHEEL_SLIP_SIDE_POLARITY[wheelIndex] * slideIntensity;

    // THE ZERO STATE: the right-side polarity flips +0 into -0, so a
    // straight drive re-normalizes to a clean zero on EVERY wheel channel
    // (the channel contract reads exactly 0 when the car is not sliding).
    wheelSlipChannels[wheelIndex] = projectedSlide === 0 ? 0 : projectedSlide;
  }

  return wheelSlipChannels;
}

/**
 * Collects the four 2-channel rival-gap slots (AC-027).
 *
 * THE ENCODING: slot `s` channels are the body-frame gap of the `s`-th
 * tracked rival — `[longitudinal / 96, lateral / 96]`, where longitudinal
 * is positive AHEAD and lateral is positive LEFT of the focal heading.
 * Rivals are the opposing-team roster cars in deterministic roster order
 * (the recycled roster contract); missing slots stay zero-padded. The
 * position delta scale (96) mirrors the recycled body-frame offsets.
 *
 * @param envState - Focal car observation seam state (carries the roster).
 * @returns Eight normalized rival-gap channels.
 */
function collectNeatRacingRivalGapChannels(
  envState: NeatRacingObservationState,
): Float32Array {
  const rivalGapChannels = new Float32Array(
    NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT,
  );
  const rivals = resolveNeatRacingTrackedRivals(envState);

  for (let slotIndex = 0; slotIndex < rivals.length; slotIndex++) {
    const rival = rivals[slotIndex]!;
    const gapChannels = buildNeatRacingRivalGapSlot(rival, envState);

    rivalGapChannels.set(
      gapChannels,
      slotIndex * NEATRACING_OBSERVATION_RIVAL_GAP_CHANNELS_PER_SLOT,
    );
  }

  return rivalGapChannels;
}

/**
 * Resolves the tracked rivals for the focal car in deterministic roster order.
 *
 * @param envState - Focal car observation seam state (carries the roster).
 * @returns Up to four opposing-team cars, roster order, focal excluded.
 */
function resolveNeatRacingTrackedRivals(
  envState: NeatRacingObservationState,
): readonly NeatRacingObservationCarState[] {
  const cars = envState.cars ?? [];
  const focalCarIndex = envState.carIndex ?? 0;
  const focalTeamIndex = resolveNeatRacingFocalTeamIndex(envState);

  return cars
    .filter(
      (car, index) =>
        index !== focalCarIndex && car.teamIndex !== focalTeamIndex,
    )
    .slice(0, NEATRACING_OBSERVATION_RIVAL_GAP_SLOT_COUNT);
}

/**
 * Builds the two-channel body-frame gap slot of one rival.
 *
 * @param rival - The rival car state to encode.
 * @param envState - Focal car observation seam state (the body-frame origin).
 * @returns The two-channel slot `[longitudinal / 96, lateral / 96]`.
 */
function buildNeatRacingRivalGapSlot(
  rival: NeatRacingObservationCarState,
  envState: NeatRacingObservationState,
): Float32Array {
  const deltaX = rival.carX - envState.carX;
  const deltaY = rival.carY - envState.carY;
  const longitudinalGapWorld =
    deltaX * Math.cos(envState.carHeading) +
    deltaY * Math.sin(envState.carHeading);
  const lateralGapWorld =
    -deltaX * Math.sin(envState.carHeading) +
    deltaY * Math.cos(envState.carHeading);

  return Float32Array.from([
    normalizeNeatRacingSignedValue(
      longitudinalGapWorld,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      lateralGapWorld,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
  ]);
}

/**
 * Collects the raw seven-channel self-radio tail (AC-027 Tier B).
 *
 * THE TAIL: the recycled Tier-2 seam passes the provider-owned radio field
 * through WITHOUT renormalization (radio payloads are already bounded);
 * missing channels stay zero.
 *
 * @param radioField - Current self-radio field from the seam state.
 * @returns The seven-channel raw radio tail.
 */
function collectNeatRacingSelfRadioChannels(
  radioField: readonly number[],
): Float32Array {
  const selfRadioChannels = new Float32Array(
    NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT,
  );

  for (
    let channelIndex = 0;
    channelIndex < NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT;
    channelIndex++
  ) {
    selfRadioChannels[channelIndex] = radioField[channelIndex] ?? 0;
  }

  return selfRadioChannels;
}

/**
 * Collects the 21-channel teammate radio tail (AC-027 Tier C).
 *
 * THE SLOTS: up to three same-team roster cars (roster order, focal car
 * excluded) encode into the recycled 7-channel teammate slot; the remaining
 * slots stay zero-padded. The NeatRacing ladder teams are size 1 (team A)
 * or size 4 (team B), so the recycled self-broadcast rule for exactly
 * 3-car teams is unreachable here and is not mirrored.
 *
 * @param envState - Focal car observation seam state (carries the roster).
 * @returns The 21 normalized teammate radio channels.
 */
function collectNeatRacingTeammateRadioChannels(
  envState: NeatRacingObservationState,
): Float32Array {
  const teammateRadioChannels = new Float32Array(
    NEATRACING_OBSERVATION_TEAMMATE_RADIO_CHANNEL_COUNT,
  );
  const teammates = resolveNeatRacingTeammates(envState);

  for (let slotIndex = 0; slotIndex < teammates.length; slotIndex++) {
    const teammate = teammates[slotIndex]!;
    const slotChannels = buildNeatRacingTeammateSlot(teammate, envState);

    teammateRadioChannels.set(
      slotChannels,
      slotIndex * NEATRACING_OBSERVATION_CHANNELS_PER_TEAMMATE_SLOT,
    );
  }

  return teammateRadioChannels;
}

/**
 * Resolves the same-team teammates for the focal car in roster order.
 *
 * @param envState - Focal car observation seam state (carries the roster).
 * @returns Up to three same-team cars, roster order, focal excluded.
 */
function resolveNeatRacingTeammates(
  envState: NeatRacingObservationState,
): readonly NeatRacingObservationCarState[] {
  const cars = envState.cars ?? [];
  const focalCarIndex = envState.carIndex ?? 0;
  const focalTeamIndex = resolveNeatRacingFocalTeamIndex(envState);

  return cars
    .filter(
      (car, index) =>
        index !== focalCarIndex && car.teamIndex === focalTeamIndex,
    )
    .slice(0, NEATRACING_OBSERVATION_TEAMMATE_RADIO_SLOT_COUNT);
}

/**
 * Resolves the focal team index from the seam (override, roster entry, then 0).
 *
 * @param envState - Focal car observation seam state.
 * @returns The focal team index (`0` or `1`).
 */
function resolveNeatRacingFocalTeamIndex(
  envState: NeatRacingObservationState,
): 0 | 1 {
  if (envState.teamIndex !== undefined) {
    return envState.teamIndex;
  }

  const cars = envState.cars ?? [];
  const focalCarIndex = envState.carIndex ?? 0;

  return cars[focalCarIndex]?.teamIndex ?? 0;
}

/**
 * Encodes one teammate's state into the recycled 7-channel radio slot.
 *
 * Channel layout (recycled vocabulary verbatim):
 * `[posX / 96, posY / 96, sin(heading), speed / 108,
 *   relOffsetX / 96, relOffsetY / 96, sin(relHeading)]`.
 *
 * @param teammate - The teammate car state to encode.
 * @param envState - Focal car observation seam state (the relative origin).
 * @returns The seven-channel slot with the recycled normalization scales.
 */
function buildNeatRacingTeammateSlot(
  teammate: NeatRacingObservationCarState,
  envState: NeatRacingObservationState,
): Float32Array {
  const relOffsetX = teammate.carX - envState.carX;
  const relOffsetY = teammate.carY - envState.carY;
  const relHeading = teammate.carHeading - envState.carHeading;

  return Float32Array.from([
    normalizeNeatRacingSignedValue(
      teammate.carX,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      teammate.carY,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    Math.sin(teammate.carHeading),
    normalizeNeatRacingSignedValue(
      teammate.speedWorld ?? 0,
      NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      relOffsetX,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    normalizeNeatRacingSignedValue(
      relOffsetY,
      NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    ),
    Math.sin(relHeading),
  ]);
}

/* ------------------------------------------------------------------------- *
 * Spline geometry helpers (owner-local frame mirror)
 * ------------------------------------------------------------------------- */

/**
 * Resolves the tangent + left-normal frame of one spline sample.
 *
 * THE FRAME: the tangent is the chord toward the NEXT ring sample (the
 * deterministic discretization the spline ribbon implies) and the left
 * normal is the tangent's 90-degree CCW rotation `(x, y) -> (-y, x)`,
 * matching the frozen `SplineSample` convention of "positive along the
 * unit left normal".
 *
 * @param splineSamples - Frozen closed-loop spline samples.
 * @param sampleIndex - Global index of the sample to frame.
 * @returns The unit-length spline sample frame.
 */
function resolveNeatRacingSplineSampleFrame(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): NeatRacingSplineSampleFrame {
  const nextSplineSample = resolveNeatRacingRelativeSplineSample(
    splineSamples,
    sampleIndex,
    1,
  );
  const tangentHeadingRadians = wrapNeatRacingAngleToMinusPiPi(
    Math.atan2(
      nextSplineSample.y - splineSamples[sampleIndex]!.y,
      nextSplineSample.x - splineSamples[sampleIndex]!.x,
    ),
  );

  return {
    tangentHeadingRadians,
    normalX: -Math.sin(tangentHeadingRadians),
    normalY: Math.cos(tangentHeadingRadians),
  };
}

/**
 * Resolves the car's signed lateral offset from a spline sample.
 *
 * @param envState - Focal car observation seam state.
 * @param splineSample - The reference lane-center sample.
 * @param splineSampleFrame - The sample's tangent/left-normal frame.
 * @returns Signed lateral offset in world units (positive LEFT of travel).
 */
function resolveNeatRacingSignedLateralOffsetWorld(
  envState: NeatRacingObservationState,
  splineSample: SplineSample,
  splineSampleFrame: NeatRacingSplineSampleFrame,
): number {
  const deltaX = envState.carX - splineSample.x;
  const deltaY = envState.carY - splineSample.y;

  return (
    deltaX * splineSampleFrame.normalX + deltaY * splineSampleFrame.normalY
  );
}

/** Resolves the recurrent memory trace while preserving a fixed width. */
function resolveNeatRacingMemoryTrace(
  memoryTrace?: readonly number[],
): readonly number[] {
  if (!memoryTrace || memoryTrace.length === 0) {
    return EMPTY_MEMORY_TRACE;
  }

  return Array.from(
    { length: NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT },
    (_, memoryIndex) => {
      return memoryTrace[memoryIndex] ?? 0;
    },
  );
}

/** Resolves the raw self-radio tail without applying any renormalization. */
function resolveNeatRacingRadioTail(
  radioField?: Float32Array | readonly number[],
): readonly number[] {
  if (!radioField) {
    return EMPTY_RADIO_FIELD;
  }

  return Array.from(
    { length: NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT },
    (_, channelIndex) => {
      return radioField[channelIndex] ?? 0;
    },
  );
}

/* ------------------------------------------------------------------------- *
 * Numeric helpers (recycled normalization mirrors)
 * ------------------------------------------------------------------------- */

/**
 * Computes the signed left-versus-right lane balance.
 *
 * @param boundaryDistanceLeftWorld - Distance to the left boundary.
 * @param boundaryDistanceRightWorld - Distance to the right boundary.
 * @returns Symmetric balance in `[-1, 1]` (positive means right of center).
 */
function resolveNeatRacingBoundaryBalance(
  boundaryDistanceLeftWorld: number,
  boundaryDistanceRightWorld: number,
): number {
  const totalBoundaryDistanceWorld =
    boundaryDistanceLeftWorld + boundaryDistanceRightWorld;

  if (totalBoundaryDistanceWorld <= Number.EPSILON) {
    return 0;
  }

  return (
    (boundaryDistanceRightWorld - boundaryDistanceLeftWorld) /
    totalBoundaryDistanceWorld
  );
}

/**
 * Converts a probability-style scalar in `[0, 1]` to the controller's
 * symmetric `[-1, 1]` range.
 *
 * @param probabilityValue - Probability-like scalar.
 * @returns Symmetric normalized value.
 */
function normalizeNeatRacingProbability(probabilityValue: number): number {
  return clampNeatRacingNormalizedValue(
    clamp01NeatRacing(probabilityValue) * 2 - 1,
  );
}

/**
 * Converts a signed scalar into the normalized `[-1, 1]` range.
 *
 * @param value - Signed source scalar.
 * @param scale - Absolute scale corresponding to magnitude 1.
 * @returns Clamped normalized value.
 */
function normalizeNeatRacingSignedValue(value: number, scale: number): number {
  return clampNeatRacingNormalizedValue(value / scale);
}

/**
 * Converts an unsigned scalar into the normalized `[0, 1]` range.
 *
 * @param value - Unsigned source scalar.
 * @param scale - Maximum reference scale for value 1.
 * @returns Normalized value in `[0, 1]`.
 */
function normalizeNeatRacingUnsignedValue(
  value: number,
  scale: number,
): number {
  return clampNeatRacingNormalizedValue(Math.max(0, value) / scale);
}

/**
 * Clamps any scalar into the controller's accepted normalized range.
 *
 * @param value - Source scalar.
 * @returns Value clamped to `[-1, 1]` (non-finite input reads as 0).
 */
function clampNeatRacingNormalizedValue(value: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
}

/**
 * Clamps a probability-like scalar to `[0, 1]`.
 *
 * @param value - Source scalar.
 * @returns Value clamped to `[0, 1]` (non-finite input reads as 0).
 */
function clamp01NeatRacing(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

/**
 * Wraps a spline sample index around the closed ring.
 *
 * @param sampleCount - Number of spline samples on the ring.
 * @param sampleIndex - Raw (possibly out-of-range) sample index.
 * @returns The wrapped index in `[0, sampleCount)`.
 */
function wrapNeatRacingSplineSampleIndex(
  sampleCount: number,
  sampleIndex: number,
): number {
  return ((sampleIndex % sampleCount) + sampleCount) % sampleCount;
}

/**
 * Wraps an angle to the closed interval `[-pi, pi]`.
 *
 * @param angleRadians - Raw angle in radians.
 * @returns Wrapped angle in `[-pi, pi]`.
 */
function wrapNeatRacingAngleToMinusPiPi(angleRadians: number): number {
  let wrappedAngleRadians = angleRadians;

  while (wrappedAngleRadians > Math.PI) {
    wrappedAngleRadians -= Math.PI * 2;
  }

  while (wrappedAngleRadians < -Math.PI) {
    wrappedAngleRadians += Math.PI * 2;
  }

  return wrappedAngleRadians;
}
