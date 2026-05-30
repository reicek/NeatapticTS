import type {
  EnvironmentState,
  TireStateTuple,
} from '../environment/environment.types';
import type { SplineSample, TrackSpec } from '../track/track.generator.types';
import {
  TRACK_SPLINE_SAMPLES_PER_SEGMENT,
  resolveSplineSampleFrame,
  type SplineSampleFrame,
} from '../track/track.spline.utils';

/** Tier 1 observation width before any radio channels are appended. */
const TIER_ONE_CHANNEL_COUNT = 70;
/** Tier 2 adds a seven-float self-radio tail to the Tier 1 base vector. */
const TIER_TWO_RADIO_CHANNEL_COUNT = 7;
/** Tier 3 adds 21 teammate-radio channels (3 slots × 7 channels each). */
const TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT = 21;
/** Tier 4 and 5 append four own-car tire-health channels to the Tier 3 vector. */
const TIRE_CHANNEL_COUNT = 4;
/** Number of teammate slots in the Tier 3 radio layout. */
const TIER_THREE_TEAMMATE_SLOT_COUNT = 3;
/** Number of channels per teammate radio slot. */
const TIER_THREE_CHANNELS_PER_TEAMMATE_SLOT = 7;
/** Number of look-ahead segments encoded into the observation vector. */
const TRACK_LOOKAHEAD_SEGMENT_COUNT = 5;
/** Number of channels emitted per look-ahead segment. */
const TRACK_CHANNELS_PER_SEGMENT = 8;
/** Number of recurrent memory-trace channels preserved in the observation tail. */
const MEMORY_TRACE_CHANNEL_COUNT = 10;
/** Normalization scale for position deltas and segment endpoints in world units. */
const TRACK_POSITION_WORLD_SCALE = 96;
/** Normalization scale for boundary and waypoint distances in world units. */
const DISTANCE_WORLD_SCALE = 64;
/** Normalization scale for forward and target speeds in world units per second. */
const SPEED_WORLD_SCALE = 36;
/** Normalization scale for lateral speed in world units per second. */
const LATERAL_SPEED_WORLD_SCALE = 18;
/** Normalization scale for boundary distances from the car centerline. */
const BOUNDARY_DISTANCE_WORLD_SCALE = 24;
/** Normalization scale for track-relative lateral offset from the optimal line. */
const OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE = 18;
/** Normalization scale for yaw-rate channels. */
const YAW_RATE_RADIANS_PER_SECOND_SCALE = 1;
/** Fallback target speed used when the environment has not produced one yet. */
const DEFAULT_TARGET_SPEED_WORLD = 24;
/** Shared empty radio field for Tier 1 or pre-radio Tier 2 states. */
const EMPTY_RADIO_FIELD = new Float32Array(0);
/** Shared zero-filled memory trace used when no recurrent trace is available yet. */
const EMPTY_MEMORY_TRACE = new Array<number>(MEMORY_TRACE_CHANNEL_COUNT).fill(
  0,
);
/** Shared empty spline sample used for empty-track safety fallbacks. */
const EMPTY_SPLINE_SAMPLE: SplineSample = {
  x: 0,
  y: 0,
  width: BOUNDARY_DISTANCE_WORLD_SCALE,
  segmentIndex: 0,
  sampleIndexWithinSegment: 0,
  globalIndex: 0,
};
/** Shared fully-healthy tire tuple used when Tier 4 state has not filled it yet. */
const DEFAULT_TIER_FOUR_TIRE_STATE: TireStateTuple = [1, 1, 1, 1];

/** Additional Tier 1–3 sensory fields layered on top of the base environment state. */
type ObservationExtensions = {
  forwardSpeedWorld?: number;
  lateralSpeedWorld?: number;
  speedWorld?: number;
  yawRateRadiansPerSecond?: number;
  slipAngleRadians?: number;
  progress01?: number;
  lapProgress01?: number;
  boundaryDistanceLeftWorld?: number;
  boundaryDistanceRightWorld?: number;
  hazardDistanceWorld?: number;
  waypointDistanceWorld?: number;
  optimalLineLateralOffsetWorld?: number;
  optimalLineHeadingErrorRadians?: number;
  targetSpeedWorld?: number;
  memoryTrace?: readonly number[];
  radioField?: Float32Array;
  teammateRadioSlots?: readonly (Float32Array | readonly number[])[];
  tireState?: TireStateTuple;
};

/** Concrete environment shape consumed by the Tier 1–3 controller seam. */
export type RacingObservationState = EnvironmentState & ObservationExtensions;

/**
 * Tier selector for the owner-local observation seam.
 *
 * - `1` — 70-channel base vector (20 scalar + 40 look-ahead + 10 memory-trace).
 * - `2` — Tier 1 plus the seven-channel self-radio tail for a 77-channel vector.
 * - `3` — Tier 1 plus three teammate-radio slots for a 91-channel vector; smaller
 *   packs keep any missing slot zero-padded.
 * - `4` — Tier 3 plus own-car tire health at `[91..94]` ordered as
 *   `[frontLeft, frontRight, rearLeft, rearRight]`, producing 95 channels.
 * - `5` — The same 95-channel byte layout as Tier 4, but the 3v3 six-car seam can
 *   fully populate all three teammate-radio rows before the tire tail is appended.
 */
export type ObservationTier = 1 | 2 | 3 | 4 | 5;

/**
 * Options that select which suffix is appended to the 70-channel driving base.
 *
 * `tier` decides whether callers receive only the base observation, the seven-channel
 * self-radio tail, the 21-channel teammate-radio tail, or the four-channel own-tire
 * suffix that keeps Tier 4 and Tier 5 byte-stable at 95 channels.
 */
export interface ObservationAssemblerOptions {
  /** Active curriculum tier for the controller. */
  readonly tier: ObservationTier;
}

type TrackBounds = {
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
  readonly centerX: number;
  readonly centerY: number;
};

type ResolvedObservationState = {
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
  readonly radioField: Float32Array;
};

type ObservationContext = {
  readonly envState: RacingObservationState;
  readonly trackSpec: TrackSpec;
  readonly closestSplineSampleIndex: number;
  readonly trackBounds: TrackBounds;
  readonly resolvedObservationState: ResolvedObservationState;
};

/**
 * Builds the flat normalized observation vector used by the racing NGE controller.
 *
 * The Tier 1 base vector contains 70 channels:
 * 1. Twenty scalar channels describing the car and immediate driving context.
 * 2. Forty channels describing five look-ahead track segments.
 * 3. Ten recurrent memory-trace channels.
 *
 * Tier 2 reuses the Tier 1 base and appends the seven self-radio channels at the
 * tail without re-normalizing them. Tier 3 reuses the same base and appends three
 * seven-channel teammate-radio slots. Tier 4 and 5 both keep the 95-channel tail
 * shape that appends the querying car's four tire channels.
 *
 * @param envState - Current environment snapshot plus optional Tier 1–5 fields.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @param options - Tier selector that decides which radio or tire tail is appended.
 * @returns Normalized Tier 1 vector (70 channels), Tier 2 vector (77 channels), Tier 3 vector (91 channels), or Tier 4/5 vector (95 channels).
 */
export function assembleNormalizedObservationVector(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
  options: ObservationAssemblerOptions,
): Float32Array {
  const observationContext = createObservationContext(envState, trackSpec);

  // Step 1: Assemble the 70-channel Tier 1 base vector.
  const scalarChannels = collectScalarChannels(observationContext);
  const lookAheadChannels = collectLookAheadChannels(observationContext);
  const memoryTraceChannels = collectMemoryTraceChannels(observationContext);
  const tierOneVector = new Float32Array(TIER_ONE_CHANNEL_COUNT);
  tierOneVector.set(scalarChannels, 0);
  tierOneVector.set(lookAheadChannels, scalarChannels.length);
  tierOneVector.set(
    memoryTraceChannels,
    scalarChannels.length + lookAheadChannels.length,
  );

  if (options.tier === 1) {
    return tierOneVector;
  }

  if (options.tier === 3) {
    return assembleTier3Observation(envState, trackSpec);
  }

  if (options.tier === 4) {
    return assembleTier4Observation(envState, trackSpec);
  }

  if (options.tier === 5) {
    return assembleTier5Observation(envState, trackSpec);
  }

  // Step 2: Append the raw seven-channel self-radio tail for Tier 2.
  const tierTwoVector = new Float32Array(
    TIER_ONE_CHANNEL_COUNT + TIER_TWO_RADIO_CHANNEL_COUNT,
  );
  tierTwoVector.set(tierOneVector, 0);
  tierTwoVector.set(
    resolveRadioTail(observationContext.resolvedObservationState.radioField),
    TIER_ONE_CHANNEL_COUNT,
  );
  return tierTwoVector;
}

/**
 * Builds the Tier 3 observation vector with a fixed teammate-radio extension.
 *
 * The layout stays stable so controller weights can treat teammate radio rows
 * as a predictable suffix: `[0..69]` is the Tier 1 driving baseline,
 * `[70..76]` is teammate slot 0, `[77..83]` is teammate slot 1, and
 * `[84..90]` is teammate slot 2. `envState.teammateRadioSlots` feeds those
 * slots directly.
 *
 * Missing slots are zero-padded. That is the honest Phase 3 2v2 behavior: a
 * race pack can supply at most two teammate rows, so the unused tail remains
 * silent rather than fabricating extra agents.
 *
 * @param envState - Current environment snapshot plus optional teammate radio rows.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @returns Ninety-one-channel Tier 3 observation vector.
 * @example
 * ```ts
 * const observation = assembleTier3Observation(
 *   {
 *     ...envState,
 *     teammateRadioSlots: [
 *       Float32Array.from([1, 0, 0, 0, 0, 0, 0]),
 *       Float32Array.from([0, 1, 0, 0, 0, 0, 0]),
 *     ],
 *   },
 *   trackSpec,
 * );
 *
 * observation.length; // 91
 * observation.slice(84, 91); // zero-padded in 2v2
 * ```
 */
export function assembleTier3Observation(
  envState: RacingObservationState & {
    teammateRadioSlots?: readonly (Float32Array | readonly number[])[];
  },
  trackSpec: TrackSpec,
): Float32Array {
  // Step 1: Build the 70-channel Tier 1 base vector.
  const baseVector = assembleNormalizedObservationVector(envState, trackSpec, {
    tier: 1,
  });

  // Step 2: Build the 91-channel Tier 3 vector with the teammate-radio tail.
  const tierThreeVector = new Float32Array(
    TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
  );
  tierThreeVector.set(baseVector, 0);

  // Step 3: Fill the teammate slots while zero-padding any unused slots.
  const teammateSlots = envState.teammateRadioSlots ?? [];
  for (
    let teammateSlotIndex = 0;
    teammateSlotIndex < TIER_THREE_TEAMMATE_SLOT_COUNT;
    teammateSlotIndex++
  ) {
    const slotOffset =
      TIER_ONE_CHANNEL_COUNT +
      teammateSlotIndex * TIER_THREE_CHANNELS_PER_TEAMMATE_SLOT;
    const slotData = teammateSlots[teammateSlotIndex];

    if (slotData !== undefined) {
      tierThreeVector.set(
        slotData.slice(0, TIER_THREE_CHANNELS_PER_TEAMMATE_SLOT),
        slotOffset,
      );
    }
  }

  return tierThreeVector;
}

/**
 * Creates the reusable `{ tier: 3 }` selector for the 91-channel observation layout.
 *
 * Use this helper when a caller wants the teammate-aware Tier 3 vector without
 * re-allocating the options object by hand.
 *
 * @returns Immutable Tier 3 observation options object.
 */
export function createTier3ObservationOptions(): { readonly tier: 3 } {
  return { tier: 3 };
}

/**
 * Builds the Tier 4 observation vector by appending own-car tire health.
 *
 * The first 91 channels are byte-for-byte identical to Tier 3. The new suffix
 * occupies `[91..94]` and stores `[frontLeft, frontRight, rearLeft, rearRight]`.
 * That keeps every pre-existing Tier 3 feature aligned while exposing only the
 * querying car's four tire channels as the new Tier 4 sensory delta.
 *
 * @param envState - Current environment snapshot plus optional Tier 4 tire state.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @returns Ninety-five-channel Tier 4 observation vector.
 * @example
 * ```ts
 * const observation = assembleTier4Observation(
 *   { ...envState, tireState: [1, 0.9, 0.8, 0.7] },
 *   trackSpec,
 * );
 *
 * observation.length; // 95
 * observation.slice(91, 95); // Float32Array [1, 0.9, 0.8, 0.7]
 * ```
 */
export function assembleTier4Observation(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): Float32Array {
  return appendOwnTireState(
    assembleTier3Observation(envState, trackSpec),
    envState.tireState ?? DEFAULT_TIER_FOUR_TIRE_STATE,
  );
}

/**
 * Creates the reusable `{ tier: 4 }` selector for the 95-channel observation layout.
 *
 * Use this helper when callers need the canonical Tier 4 pack shape without
 * re-allocating the options object by hand.
 *
 * @returns Immutable Tier 4 observation options object.
 */
export function createTier4ObservationOptions(): { readonly tier: 4 } {
  return { tier: 4 };
}

/**
 * Builds the Tier 5 observation vector with the canonical 95-channel 3v3 layout.
 *
 * Channel layout:
 * - `[0..69]` — 70-channel base observation containing pose, speed, track geometry,
 *   and recurrent memory trace.
 * - `[70..90]` — team radio (`3 × 7 = 21` channels). In 3v3 all three rows can be
 *   populated; smaller packs keep any missing row zero-padded.
 * - `[91..94]` — own-car tire state `[frontLeft, frontRight, rearLeft, rearRight]`.
 *
 * Tier 5 is byte-stable with Tier 4: both tiers emit the same 95 floats in the
 * same order. The difference is radio population, not vector shape.
 *
 * @param envState - Current environment snapshot plus optional Tier 5 teammate radio rows and tire state.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @returns Ninety-five-channel Tier 5 observation vector with the stable Tier 4 byte layout.
 * @example
 * ```ts
 * const observation = assembleTier5Observation(
 *   {
 *     ...envState,
 *     teammateRadioSlots: [
 *       Float32Array.from([1, 0, 0, 0, 0, 0, 0]),
 *       Float32Array.from([0, 1, 0, 0, 0, 0, 0]),
 *       Float32Array.from([0, 0, 1, 0, 0, 0, 0]),
 *     ],
 *     tireState: [1, 0.9, 0.8, 0.7],
 *   },
 *   trackSpec,
 * );
 *
 * observation.length; // TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT + TIRE_CHANNEL_COUNT
 * observation.slice(
 *   TIER_ONE_CHANNEL_COUNT,
 *   TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
 * ); // three teammate radio rows
 * observation.slice(
 *   TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
 * ); // own-car tire channels
 * ```
 */
export function assembleTier5Observation(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): Float32Array {
  return appendOwnTireState(
    assembleTier3Observation(envState, trackSpec),
    envState.tireState ?? DEFAULT_TIER_FOUR_TIRE_STATE,
  );
}

/**
 * Creates the reusable `{ tier: 5 }` selector for the 95-channel Tier 5 layout.
 *
 * Pass this helper to `assembleNormalizedObservationVector` when the caller wants
 * the byte-stable Tier 4/5 observation shape while allowing all three teammate-radio
 * rows to be populated in a six-car 3v3 pack.
 *
 * @returns Immutable Tier 5 observation options object.
 * @example
 * ```ts
 * const options = createTier5ObservationOptions();
 * const observation = assembleNormalizedObservationVector(envState, trackSpec, options);
 *
 * options.tier; // 5
 * observation.length; // TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT + TIRE_CHANNEL_COUNT
 * ```
 */
export function createTier5ObservationOptions(): { readonly tier: 5 } {
  return { tier: 5 };
}

/**
 * Appends the querying car's tire-health tuple to a base observation vector.
 *
 * @param baseVector - Base observation vector.
 * @param tireState - Four-channel tire-health tuple ordered by wheel corner.
 * @returns Observation vector with the tire-health tail appended.
 */
function appendOwnTireState(
  baseVector: Float32Array,
  tireState: TireStateTuple,
): Float32Array {
  const tireAwareVector = new Float32Array(baseVector.length + TIRE_CHANNEL_COUNT);

  tireAwareVector.set(baseVector, 0);
  tireAwareVector.set(tireState, baseVector.length);
  return tireAwareVector;
}

/**
 * Creates the resolved controller context used by the assembler helpers.
 *
 * @param envState - Current environment snapshot.
 * @param trackSpec - Frozen track geometry.
 * @returns Stable context containing derived state and the closest segment index.
 */
function createObservationContext(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): ObservationContext {
  const trackBounds = resolveTrackBounds(trackSpec);
  const closestSplineSampleIndex = findClosestSplineSampleIndex(
    envState,
    trackSpec,
  );

  return {
    envState,
    trackSpec,
    closestSplineSampleIndex,
    trackBounds,
    resolvedObservationState: resolveObservationState(
      envState,
      trackSpec,
      closestSplineSampleIndex,
    ),
  };
}

/**
 * Collects the 20 scalar channels that describe the current car state.
 *
 * @param observationContext - Resolved controller context.
 * @returns Twenty normalized scalar channels.
 */
function collectScalarChannels(
  observationContext: ObservationContext,
): Float32Array {
  const { envState, trackBounds, resolvedObservationState } =
    observationContext;

  return Float32Array.from([
    normalizeSignedValue(
      envState.carX - trackBounds.centerX,
      TRACK_POSITION_WORLD_SCALE,
    ),
    normalizeSignedValue(
      envState.carY - trackBounds.centerY,
      TRACK_POSITION_WORLD_SCALE,
    ),
    Math.sin(envState.carHeading),
    Math.cos(envState.carHeading),
    normalizeSignedValue(
      resolvedObservationState.forwardSpeedWorld,
      SPEED_WORLD_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.lateralSpeedWorld,
      LATERAL_SPEED_WORLD_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.speedWorld,
      SPEED_WORLD_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.yawRateRadiansPerSecond,
      YAW_RATE_RADIANS_PER_SECOND_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.slipAngleRadians,
      Math.PI / 2,
    ),
    normalizeProbability(resolvedObservationState.progress01),
    normalizeProbability(resolvedObservationState.lapProgress01),
    normalizeUnsignedValue(
      resolvedObservationState.boundaryDistanceLeftWorld,
      BOUNDARY_DISTANCE_WORLD_SCALE,
    ),
    normalizeUnsignedValue(
      resolvedObservationState.boundaryDistanceRightWorld,
      BOUNDARY_DISTANCE_WORLD_SCALE,
    ),
    clampNormalizedValue(
      resolveBoundaryBalance(
        resolvedObservationState.boundaryDistanceLeftWorld,
        resolvedObservationState.boundaryDistanceRightWorld,
      ),
    ),
    normalizeUnsignedValue(
      resolvedObservationState.hazardDistanceWorld,
      DISTANCE_WORLD_SCALE,
    ),
    normalizeUnsignedValue(
      resolvedObservationState.waypointDistanceWorld,
      DISTANCE_WORLD_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.optimalLineLateralOffsetWorld,
      OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.optimalLineHeadingErrorRadians,
      Math.PI,
    ),
    normalizeUnsignedValue(
      resolvedObservationState.targetSpeedWorld,
      SPEED_WORLD_SCALE,
    ),
    normalizeSignedValue(
      resolvedObservationState.targetSpeedWorld -
        resolvedObservationState.speedWorld,
      SPEED_WORLD_SCALE,
    ),
  ]);
}

/**
 * Collects the 40 look-ahead segment channels used for track anticipation.
 *
 * @param observationContext - Resolved controller context.
 * @returns Forty normalized channels describing five upcoming segments.
 */
function collectLookAheadChannels(
  observationContext: ObservationContext,
): Float32Array {
  const lookAheadChannels = new Float32Array(
    TRACK_LOOKAHEAD_SEGMENT_COUNT * TRACK_CHANNELS_PER_SEGMENT,
  );

  for (
    let lookAheadOffset = 0;
    lookAheadOffset < TRACK_LOOKAHEAD_SEGMENT_COUNT;
    lookAheadOffset++
  ) {
    const channelOffset = lookAheadOffset * TRACK_CHANNELS_PER_SEGMENT;
    const splineSample = resolveLookAheadSplineSample(
      observationContext.trackSpec,
      observationContext.closestSplineSampleIndex,
      lookAheadOffset,
    );
    const nextSplineSample = resolveRelativeSplineSample(
      observationContext.trackSpec.splineSamples,
      splineSample.globalIndex,
      1,
    );
    const splineSampleFrame = resolveSplineSampleFrame(
      observationContext.trackSpec.splineSamples,
      splineSample.globalIndex,
    );

    lookAheadChannels.set(
      [
        normalizeSignedValue(
          splineSample.x - observationContext.envState.carX,
          TRACK_POSITION_WORLD_SCALE,
        ),
        normalizeSignedValue(
          splineSample.y - observationContext.envState.carY,
          TRACK_POSITION_WORLD_SCALE,
        ),
        normalizeSignedValue(
          nextSplineSample.x - observationContext.envState.carX,
          TRACK_POSITION_WORLD_SCALE,
        ),
        normalizeSignedValue(
          nextSplineSample.y - observationContext.envState.carY,
          TRACK_POSITION_WORLD_SCALE,
        ),
        Math.sin(splineSampleFrame.tangentHeadingRadians),
        Math.cos(splineSampleFrame.tangentHeadingRadians),
        normalizeUnsignedValue(
          splineSample.width,
          BOUNDARY_DISTANCE_WORLD_SCALE,
        ),
        normalizeUnsignedValue(
          Math.hypot(
            splineSample.x - observationContext.envState.carX,
            splineSample.y - observationContext.envState.carY,
          ),
          DISTANCE_WORLD_SCALE,
        ),
      ],
      channelOffset,
    );
  }

  return lookAheadChannels;
}

/**
 * Collects the 10-channel recurrent memory trace.
 *
 * @param observationContext - Resolved controller context.
 * @returns Ten normalized recurrent channels.
 */
function collectMemoryTraceChannels(
  observationContext: ObservationContext,
): Float32Array {
  const memoryTraceChannels = new Float32Array(MEMORY_TRACE_CHANNEL_COUNT);

  for (
    let memoryIndex = 0;
    memoryIndex < MEMORY_TRACE_CHANNEL_COUNT;
    memoryIndex++
  ) {
    memoryTraceChannels[memoryIndex] = clampNormalizedValue(
      observationContext.resolvedObservationState.memoryTrace[memoryIndex] ?? 0,
    );
  }

  return memoryTraceChannels;
}

/**
 * Resolves all optional Tier 1–2 observation fields, deriving safe defaults when
 * the live environment has not produced them yet.
 *
 * @param envState - Current environment snapshot.
 * @param trackSpec - Frozen track geometry.
 * @param closestSegmentIndex - Segment nearest to the car position.
 * @returns Complete observation state ready for normalization.
 */
function resolveObservationState(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
): ResolvedObservationState {
  const closestSplineSample =
    trackSpec.splineSamples[closestSplineSampleIndex] ?? EMPTY_SPLINE_SAMPLE;
  const closestSplineSampleFrame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    closestSplineSample.globalIndex,
  );
  const signedLateralOffsetWorld = resolveSignedLateralOffsetWorld(
    envState,
    closestSplineSample,
    closestSplineSampleFrame,
  );
  const forwardSpeedWorld = envState.forwardSpeedWorld ?? 0;
  const lateralSpeedWorld = envState.lateralSpeedWorld ?? 0;
  const speedWorld =
    envState.speedWorld ?? Math.hypot(forwardSpeedWorld, lateralSpeedWorld);
  const halfTrackWidthWorld =
    closestSplineSample.width / 2;
  const derivedWaypointDistanceWorld = Math.hypot(
    closestSplineSample.x - envState.carX,
    closestSplineSample.y - envState.carY,
  );
  const derivedProgress01 =
    trackSpec.splineSamples.length === 0
      ? 0
      : closestSplineSampleIndex / trackSpec.splineSamples.length;

  return {
    forwardSpeedWorld,
    lateralSpeedWorld,
    speedWorld,
    yawRateRadiansPerSecond: envState.yawRateRadiansPerSecond ?? 0,
    slipAngleRadians: envState.slipAngleRadians ?? 0,
    progress01: clamp01(envState.progress01 ?? derivedProgress01),
    lapProgress01: clamp01(
      envState.lapProgress01 ?? envState.progress01 ?? derivedProgress01,
    ),
    boundaryDistanceLeftWorld:
      envState.boundaryDistanceLeftWorld ??
      resolveBoundaryDistanceWorld(
        halfTrackWidthWorld - signedLateralOffsetWorld,
      ),
    boundaryDistanceRightWorld:
      envState.boundaryDistanceRightWorld ??
      resolveBoundaryDistanceWorld(
        halfTrackWidthWorld + signedLateralOffsetWorld,
      ),
    hazardDistanceWorld: envState.hazardDistanceWorld ?? DISTANCE_WORLD_SCALE,
    waypointDistanceWorld:
      envState.waypointDistanceWorld ?? derivedWaypointDistanceWorld,
    optimalLineLateralOffsetWorld:
      envState.optimalLineLateralOffsetWorld ?? signedLateralOffsetWorld,
    optimalLineHeadingErrorRadians:
      envState.optimalLineHeadingErrorRadians ??
      wrapAngleToMinusPiPi(
        closestSplineSampleFrame.tangentHeadingRadians - envState.carHeading,
      ),
    targetSpeedWorld: envState.targetSpeedWorld ?? DEFAULT_TARGET_SPEED_WORLD,
    memoryTrace: resolveMemoryTrace(envState.memoryTrace),
    radioField: envState.radioField ?? EMPTY_RADIO_FIELD,
  };
}

/**
 * Finds the shared spline sample whose lane-center point is closest to the car.
 *
 * @param envState - Current environment snapshot.
 * @param trackSpec - Frozen track geometry.
 * @returns Index of the nearest spline sample.
 */
function findClosestSplineSampleIndex(
  envState: RacingObservationState,
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
 * Resolves the track bounds used to normalize car position channels.
 *
 * @param trackSpec - Frozen track geometry.
 * @returns Bounding box plus center point.
 */
function resolveTrackBounds(trackSpec: TrackSpec): TrackBounds {
  const initialSplineSample =
    trackSpec.splineSamples[0] ?? EMPTY_SPLINE_SAMPLE;
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

/**
 * Resolves one look-ahead spline sample by wrapping around the closed loop.
 *
 * @param trackSpec - Frozen track geometry.
 * @param closestSplineSampleIndex - Nearest spline sample index for the car.
 * @param lookAheadOffset - Positive offset from the nearest segment.
 * @returns Spline sample to encode into the observation vector.
 */
function resolveLookAheadSplineSample(
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
  lookAheadOffset: number,
): SplineSample {
  return resolveRelativeSplineSample(
    trackSpec.splineSamples,
    closestSplineSampleIndex,
    lookAheadOffset * TRACK_SPLINE_SAMPLES_PER_SEGMENT,
  );
}

function resolveRelativeSplineSample(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
  sampleOffset: number,
): SplineSample {
  if (splineSamples.length === 0) {
    return EMPTY_SPLINE_SAMPLE;
  }

  const wrappedSampleIndex = wrapSplineSampleIndex(
    splineSamples.length,
    sampleIndex + sampleOffset,
  );

  return splineSamples[wrappedSampleIndex] ?? EMPTY_SPLINE_SAMPLE;
}

/**
 * Resolves the recurrent memory trace while preserving a fixed width.
 *
 * @param memoryTrace - Incoming recurrent trace values, if any.
 * @returns Ten-value trace ready for normalization.
 */
function resolveMemoryTrace(
  memoryTrace?: readonly number[],
): readonly number[] {
  if (!memoryTrace || memoryTrace.length === 0) {
    return EMPTY_MEMORY_TRACE;
  }

  return Array.from(
    { length: MEMORY_TRACE_CHANNEL_COUNT },
    (_, memoryIndex) => {
      return memoryTrace[memoryIndex] ?? 0;
    },
  );
}

/**
 * Resolves the raw Tier 2 radio tail without applying any additional normalization.
 *
 * @param radioField - Current self-radio buffer.
 * @returns Seven-channel radio tail in the original order.
 */
function resolveRadioTail(radioField: Float32Array): Float32Array {
  const radioTail = new Float32Array(TIER_TWO_RADIO_CHANNEL_COUNT);

  for (
    let channelIndex = 0;
    channelIndex < TIER_TWO_RADIO_CHANNEL_COUNT;
    channelIndex++
  ) {
    radioTail[channelIndex] = radioField[channelIndex] ?? 0;
  }

  return radioTail;
}

/**
 * Converts a probability-style scalar in [0, 1] to the controller's symmetric
 * [-1, 1] range.
 *
 * @param probabilityValue - Probability-like scalar.
 * @returns Symmetric normalized value.
 */
function normalizeProbability(probabilityValue: number): number {
  return clampNormalizedValue(clamp01(probabilityValue) * 2 - 1);
}

/**
 * Converts a signed scalar into the normalized [-1, 1] range.
 *
 * @param value - Signed source scalar.
 * @param scale - Absolute scale corresponding to magnitude 1.
 * @returns Clamped normalized value.
 */
function normalizeSignedValue(value: number, scale: number): number {
  return clampNormalizedValue(value / Math.max(scale, Number.EPSILON));
}

/**
 * Converts an unsigned scalar into the normalized [0, 1] range while keeping the
 * result inside the controller's accepted bounds.
 *
 * @param value - Unsigned source scalar.
 * @param scale - Maximum reference scale for value 1.
 * @returns Normalized value in [0, 1].
 */
function normalizeUnsignedValue(value: number, scale: number): number {
  return clampNormalizedValue(
    Math.max(0, value) / Math.max(scale, Number.EPSILON),
  );
}

/**
 * Clamps any scalar into the controller's accepted normalized range.
 *
 * @param value - Source scalar.
 * @returns Value clamped to [-1, 1].
 */
function clampNormalizedValue(value: number): number {
  return Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
}

/**
 * Clamps a probability-like scalar to [0, 1].
 *
 * @param value - Source scalar.
 * @returns Value clamped to [0, 1].
 */
function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

/**
 * Computes the signed left-versus-right balance used by the controller to judge
 * how centered the car is within the lane.
 *
 * @param boundaryDistanceLeftWorld - Distance from the car to the left boundary.
 * @param boundaryDistanceRightWorld - Distance from the car to the right boundary.
 * @returns Symmetric balance in [-1, 1].
 */
function resolveBoundaryBalance(
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

function resolveSignedLateralOffsetWorld(
  envState: RacingObservationState,
  splineSample: SplineSample,
  splineSampleFrame: SplineSampleFrame,
): number {
  const deltaX = envState.carX - splineSample.x;
  const deltaY = envState.carY - splineSample.y;

  return (
    deltaX * splineSampleFrame.normalX + deltaY * splineSampleFrame.normalY
  );
}

function resolveBoundaryDistanceWorld(boundaryDistanceWorld: number): number {
  return Math.max(0, Number.isFinite(boundaryDistanceWorld) ? boundaryDistanceWorld : 0);
}

function wrapSplineSampleIndex(sampleCount: number, sampleIndex: number): number {
  return ((sampleIndex % sampleCount) + sampleCount) % sampleCount;
}

/**
 * Wraps an angle to the closed interval [-π, π].
 *
 * @param angleRadians - Raw angle in radians.
 * @returns Wrapped angle in [-π, π].
 */
function wrapAngleToMinusPiPi(angleRadians: number): number {
  let wrappedAngleRadians = angleRadians;

  while (wrappedAngleRadians > Math.PI) {
    wrappedAngleRadians -= Math.PI * 2;
  }

  while (wrappedAngleRadians < -Math.PI) {
    wrappedAngleRadians += Math.PI * 2;
  }

  return wrappedAngleRadians;
}
