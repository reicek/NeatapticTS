import type {
  EnvironmentState,
  PitStrategyState,
  RacingCarState,
  TireStateTuple,
} from '../environment/environment.types';
import type { SplineSample, TrackSpec } from '../track/track.generator.types';
import {
  TRACK_SPLINE_SAMPLES_PER_SEGMENT,
  resolveInnerLaneCenterlineOffsetWorld,
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
/** Number of pit/strategy channels appended after the tire-health tail. */
const PIT_STRATEGY_CHANNEL_COUNT = 8;
/**
 * Total Tier 4/5 controller input size after the tire and pit/strategy tails.
 *
 * This is `70 + 21 + 4 + 8 = 103` channels and must stay byte-stable with the
 * coevolution service's controller input dimension.
 */
export const TOTAL_TIER4_INPUT_SIZE =
  TIER_ONE_CHANNEL_COUNT +
  TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT +
  TIRE_CHANNEL_COUNT +
  PIT_STRATEGY_CHANNEL_COUNT;
/** Number of opponent slots encoded in the Tier 6 opponent-perception tail. */
const TIER6_OPPONENT_SLOT_COUNT = 3;
/** Number of ego-relative channels encoded per Tier 6 opponent slot. */
const TIER6_OPPONENT_SLOT_CHANNELS = 7;
/**
 * Total Tier 6 controller input size after appending the opponent-perception tail.
 *
 * This is `103 + 3 × 7 = 124` channels and must stay byte-stable with the
 * coevolution service's Tier 6 controller input dimension.
 */
export const TIER6_TOTAL_INPUT_SIZE =
  TOTAL_TIER4_INPUT_SIZE +
  TIER6_OPPONENT_SLOT_COUNT * TIER6_OPPONENT_SLOT_CHANNELS;
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
const SPEED_WORLD_SCALE = 108;
/** Normalization scale for lateral speed in world units per second. */
const LATERAL_SPEED_WORLD_SCALE = 54;
/** Normalization scale for boundary distances from the car centerline. */
const BOUNDARY_DISTANCE_WORLD_SCALE = 24;
/** Normalization scale for track-relative lateral offset from the optimal line. */
const OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE = 18;
/** Normalization scale for yaw-rate channels. */
const YAW_RATE_RADIANS_PER_SECOND_SCALE = 1;
/** Fallback target speed used when the environment has not produced one yet. */
const DEFAULT_TARGET_SPEED_WORLD = 72;
/** Shared empty radio field for Tier 1 or pre-radio Tier 2 states. */
const EMPTY_RADIO_FIELD = new Float32Array(0);
/** Shared zero-filled memory trace used when no recurrent trace is available yet. */
const EMPTY_MEMORY_TRACE = new Array<number>(MEMORY_TRACE_CHANNEL_COUNT).fill(
  0,
);
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
  /** Optional pre-computed opponent slots for Tier 6; derived from `envState.cars` when absent. */
  opponentPerceptionSlots?: readonly (Float32Array | readonly number[])[];
  tireState?: TireStateTuple;
};

/** Concrete environment shape consumed by the Tier 1–3 controller seam. */
export type RacingObservationState = EnvironmentState &
  ObservationExtensions &
  PitStrategyState;

/**
 * Tier selector for the owner-local observation seam.
 *
 * - `1` — 70-channel base vector (20 scalar + 40 look-ahead + 10 memory-trace).
 * - `2` — Tier 1 plus the seven-channel self-radio tail for a 77-channel vector.
 * - `3` — Tier 1 plus three teammate-radio slots for a 91-channel vector; smaller
 *   packs keep any missing slot zero-padded.
 * - `4` — Tier 3 plus own-car tire health at `[91..94]` ordered as
 *   `[frontLeft, frontRight, rearLeft, rearRight]`, then 8 pit/strategy channels
 *   at `[95..102]`, producing 103 channels.
 * - `5` — The same 103-channel byte layout as Tier 4, but the 3v3 six-car seam can
 *   fully populate all three teammate-radio rows before the tire/pit tail is appended.
 * - `6` — 103-channel Tier 4/5 base plus 21 opponent-perception channels
 *   (3 opponent slots × 7 ego-relative channels) for a 124-channel vector.
 */
export type ObservationTier = 1 | 2 | 3 | 4 | 5 | 6;

/**
 * Options that select which suffix is appended to the 70-channel driving base.
 *
 * `tier` decides whether callers receive only the base observation, the seven-channel
 * self-radio tail, the 21-channel teammate-radio tail, the four-channel own-tire
 * suffix, the 8-channel pit/strategy suffix that keeps Tier 4 and Tier 5
 * byte-stable at 103 channels, or the 21-channel opponent-perception suffix that
 * produces the Tier 6 124-channel vector.
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
 * seven-channel teammate-radio slots. Tier 4 and 5 both keep the 103-channel tail
 * shape that appends the querying car's four tire channels plus eight pit/strategy
 * channels.
 *
 * @param envState - Current environment snapshot plus optional Tier 1–5 fields.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @param options - Tier selector that decides which radio, tire, or pit tail is appended.
 * @returns Normalized Tier 1 vector (70 channels), Tier 2 vector (77 channels), Tier 3 vector (91 channels), or Tier 4/5 vector (103 channels).
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

  if (options.tier === 6) {
    return assembleTier6Observation(envState, trackSpec);
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
 * Derives a per-car observation state from a multi-car environment snapshot.
 *
 * This helper copies the selected car's pose, team index, and tire state into a
 * new observation state object while preserving all other environment fields
 * (track sample, boundary distances, memory trace, etc.). The existing
 * `assembleNormalizedObservationVector` can then be called unchanged; the
 * team-aware optimal-line offset will automatically follow the selected car's
 * team because `teamIndex` is taken from the car rather than the top-level state.
 *
 * @param envState - Current multi-car environment snapshot plus observation extensions.
 * @param carIndex - Index of the car to observe within `envState.cars`.
 * @returns Observation state scoped to the requested car.
 * @throws RangeError when `envState.cars` is missing or `carIndex` is out of bounds.
 *
 * @example
 * ```ts
 * const blueState = derivePerCarObservationState(envState, 0);
 * const blueVector = assembleNormalizedObservationVector(
 *   blueState,
 *   trackSpec,
 *   { tier: 1 },
 * );
 * ```
 */
export function derivePerCarObservationState(
  envState: RacingObservationState,
  carIndex: number,
): RacingObservationState {
  const selectedCar = envState.cars?.[carIndex];
  if (selectedCar === undefined) {
    throw new RangeError(
      `envState.cars[${carIndex}] is undefined; cannot derive observation state.`,
    );
  }

  // Step 1: Populate teammate radio slots from same-team cars.
  const teammateRadioSlots = buildTeammateRadioSlots(
    envState.cars!,
    carIndex,
    selectedCar.teamIndex,
    selectedCar.carX,
    selectedCar.carY,
    selectedCar.carHeading,
  );

  // Step 2: Populate opponent-perception slots for Tier 6.
  const opponentPerceptionSlots = buildOpponentPerceptionSlots(
    envState.cars!,
    carIndex,
    selectedCar,
  );

  return {
    ...envState,
    carX: selectedCar.carX,
    carY: selectedCar.carY,
    carHeading: selectedCar.carHeading,
    teamIndex: selectedCar.teamIndex,
    tireState: selectedCar.tireState,
    teammateRadioSlots,
    opponentPerceptionSlots,
  };
}

/**
 * Builds the teammate radio slots for a focal car from the multi-car roster.
 *
 * Each slot encodes seven channels: teammate position x/y (normalized),
 * teammate heading (sin), teammate speed (normalized, 0 when unavailable),
 * relative offset x/y (normalized), and relative heading difference (sin).
 *
 * In a 2v2 layout only one teammate exists, so slot 0 is populated and
 * slots 1–2 are zero-padded.
 *
 * @param cars - Ordered car roster from the environment state.
 * @param focalCarIndex - Index of the focal car.
 * @param focalTeamIndex - Team index of the focal car.
 * @param focalCarX - Focal car X position in world units.
 * @param focalCarY - Focal car Y position in world units.
 * @param focalCarHeading - Focal car heading in radians.
 * @returns Array of three 7-channel slots (populated or zero-padded).
 */
function buildTeammateRadioSlots(
  cars: readonly RacingCarState[],
  focalCarIndex: number,
  focalTeamIndex: 0 | 1,
  focalCarX: number,
  focalCarY: number,
  focalCarHeading: number,
): readonly (Float32Array | readonly number[])[] {
  // Step 1: Collect teammates (same team, different car index).
  const teammates = cars.filter(
    (car, index) => index !== focalCarIndex && car.teamIndex === focalTeamIndex,
  );

  // Step 2: Count same-team cars including the focal car.
  // For 3-car teams (Tier 5), the focal car's own state is included as a
  // self-broadcast slot so all 3 radio rows carry non-zero data.
  // For 2-car teams (Tier 3–4), the focal car is excluded and the third
  // slot remains zero-padded.
  const sameTeamCount = teammates.length + 1;
  const includeSelfBroadcast = sameTeamCount >= 3;
  const focalCar = cars[focalCarIndex];

  // Step 3: Build up to 3 slots — populate available teammates, then
  // self-broadcast for 3-car teams, zero-pad the rest.
  const slots: (Float32Array | readonly number[])[] = [];
  for (
    let slotIndex = 0;
    slotIndex < TIER_THREE_TEAMMATE_SLOT_COUNT;
    slotIndex++
  ) {
    const teammate = teammates[slotIndex];
    if (teammate !== undefined) {
      slots.push(
        buildTeammateSlot(teammate, focalCarX, focalCarY, focalCarHeading),
      );
    } else if (
      includeSelfBroadcast &&
      focalCar !== undefined &&
      slotIndex === teammates.length
    ) {
      slots.push(
        buildTeammateSlot(focalCar, focalCarX, focalCarY, focalCarHeading),
      );
    } else {
      slots.push(new Float32Array(TIER_THREE_CHANNELS_PER_TEAMMATE_SLOT));
    }
  }

  return slots;
}

/**
 * Encodes one teammate's state into a 7-channel radio slot.
 *
 * Channel layout: [posX, posY, headingSin, speed, relOffsetX, relOffsetY, relHeadingSin].
 * Position channels are normalized by {@link TRACK_POSITION_WORLD_SCALE}, speed by
 * {@link SPEED_WORLD_SCALE}, and all channels stay within [-1, 1].
 *
 * @param teammate - The teammate car state to encode.
 * @param focalCarX - Focal car X position in world units.
 * @param focalCarY - Focal car Y position in world units.
 * @param focalCarHeading - Focal car heading in radians.
 * @returns Seven-channel Float32Array with normalized teammate state.
 */
function buildTeammateSlot(
  teammate: RacingCarState,
  focalCarX: number,
  focalCarY: number,
  focalCarHeading: number,
): Float32Array {
  const relOffsetX = teammate.carX - focalCarX;
  const relOffsetY = teammate.carY - focalCarY;
  const relHeading = teammate.carHeading - focalCarHeading;

  return Float32Array.from([
    teammate.carX / TRACK_POSITION_WORLD_SCALE,
    teammate.carY / TRACK_POSITION_WORLD_SCALE,
    Math.sin(teammate.carHeading),
    (teammate.speedWorld ?? 0) / SPEED_WORLD_SCALE,
    relOffsetX / TRACK_POSITION_WORLD_SCALE,
    relOffsetY / TRACK_POSITION_WORLD_SCALE,
    Math.sin(relHeading),
  ]);
}

/**
 * Builds the opponent-perception slots for a focal car from the multi-car roster.
 *
 * Opponents are taken from all cars that are not the focal car and that are not
 * on the same team as the focal car, preserving deterministic roster order.
 * Missing slots are zero-padded by leaving them empty; the caller is expected
 * to fill unused slots with zeros.
 *
 * @param cars - Ordered car roster from the environment state.
 * @param focalCarIndex - Index of the focal car.
 * @param focalCar - Focal car state used as the body-frame origin.
 * @returns Array of up to three 7-channel opponent slots.
 */
function buildOpponentPerceptionSlots(
  cars: readonly RacingCarState[],
  focalCarIndex: number,
  focalCar: RacingCarState,
): readonly (Float32Array | readonly number[])[] {
  const opponents = cars.filter(
    (car, index) =>
      index !== focalCarIndex && car.teamIndex !== focalCar.teamIndex,
  );
  const slots: (Float32Array | readonly number[])[] = [];

  for (let slotIndex = 0; slotIndex < TIER6_OPPONENT_SLOT_COUNT; slotIndex++) {
    const opponent = opponents[slotIndex];
    slots.push(
      opponent !== undefined
        ? buildOpponentSlot(focalCar, opponent)
        : new Float32Array(TIER6_OPPONENT_SLOT_CHANNELS),
    );
  }

  return slots;
}

/**
 * Encodes one opponent's state into a 7-channel ego-relative slot.
 *
 * Channel layout:
 *   0. `relForwardEgo` — forward distance in the focal car's body frame,
 *      normalized by {@link TRACK_POSITION_WORLD_SCALE}.
 *   1. `relLeftEgo` — left distance in the focal car's body frame,
 *      normalized by {@link TRACK_POSITION_WORLD_SCALE}.
 *   2. `sinHeadingDeltaEgo` — sine of the opponent heading minus the focal heading.
 *   3. `cosHeadingDeltaEgo` — cosine of the opponent heading minus the focal heading.
 *   4. `relSpeedForwardEgo` — relative forward speed along the focal car's forward
 *      axis, normalized by {@link SPEED_WORLD_SCALE}.
 *   5. `relSpeedLateralEgo` — relative lateral speed along the focal car's left
 *      axis, normalized by {@link LATERAL_SPEED_WORLD_SCALE}.
 *   6. `directDistance` — Euclidean distance between the two cars, normalized by
 *      {@link DISTANCE_WORLD_SCALE}.
 *
 * @param focalCar - Focal car state that defines the body-frame origin.
 * @param opponentCar - Opponent car state to encode.
 * @returns Seven-channel Float32Array with normalized ego-relative state.
 */
function buildOpponentSlot(
  focalCar: RacingCarState,
  opponentCar: RacingCarState,
): Float32Array {
  const dx = opponentCar.carX - focalCar.carX;
  const dy = opponentCar.carY - focalCar.carY;
  const focalHeading = focalCar.carHeading;
  const headingDelta = opponentCar.carHeading - focalHeading;
  const cosHeading = Math.cos(focalHeading);
  const sinHeading = Math.sin(focalHeading);

  const focalWorldVx =
    (focalCar.forwardSpeedWorld ?? 0) * cosHeading -
    (focalCar.lateralSpeedWorld ?? 0) * sinHeading;
  const focalWorldVy =
    (focalCar.forwardSpeedWorld ?? 0) * sinHeading +
    (focalCar.lateralSpeedWorld ?? 0) * cosHeading;
  const opponentCosHeading = Math.cos(opponentCar.carHeading);
  const opponentSinHeading = Math.sin(opponentCar.carHeading);
  const opponentWorldVx =
    (opponentCar.forwardSpeedWorld ?? 0) * opponentCosHeading -
    (opponentCar.lateralSpeedWorld ?? 0) * opponentSinHeading;
  const opponentWorldVy =
    (opponentCar.forwardSpeedWorld ?? 0) * opponentSinHeading +
    (opponentCar.lateralSpeedWorld ?? 0) * opponentCosHeading;

  const relWorldVx = opponentWorldVx - focalWorldVx;
  const relWorldVy = opponentWorldVy - focalWorldVy;

  return Float32Array.from([
    (dx * cosHeading + dy * sinHeading) / TRACK_POSITION_WORLD_SCALE,
    (-dx * sinHeading + dy * cosHeading) / TRACK_POSITION_WORLD_SCALE,
    Math.sin(headingDelta),
    Math.cos(headingDelta),
    (relWorldVx * cosHeading + relWorldVy * sinHeading) / SPEED_WORLD_SCALE,
    (-relWorldVx * sinHeading + relWorldVy * cosHeading) /
      LATERAL_SPEED_WORLD_SCALE,
    Math.hypot(dx, dy) / DISTANCE_WORLD_SCALE,
  ]);
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
 * Missing slots are zero-padded. In a 2v2 race pack at most two teammate
 * rows are populated, so the unused tail remains silent rather than
 * fabricating extra agents.
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
 * Builds the Tier 4 observation vector by appending own-car tire health and
 * pit/strategy state.
 *
 * The first 91 channels are byte-for-byte identical to Tier 3. The new suffix
 * occupies `[91..94]` and stores `[frontLeft, frontRight, rearLeft, rearRight]`,
 * followed by 8 pit/strategy channels at `[95..102]`. That keeps every pre-existing
 * Tier 3 feature aligned while exposing only the querying car's tire and strategy
 * state as the new Tier 4 sensory delta.
 *
 * @param envState - Current environment snapshot plus optional Tier 4 tire and pit/strategy state.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @returns 103-channel Tier 4 observation vector.
 * @example
 * ```ts
 * const observation = assembleTier4Observation(
 *   { ...envState, tireState: [1, 0.9, 0.8, 0.7] },
 *   trackSpec,
 * );
 *
 * observation.length; // 103
 * observation.slice(91, 95); // Float32Array [1, 0.9, 0.8, 0.7]
 * observation.slice(95, 103); // 8 pit/strategy channels
 * ```
 */
export function assembleTier4Observation(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): Float32Array {
  const tireAwareVector = appendOwnTireState(
    assembleTier3Observation(envState, trackSpec),
    envState.tireState ?? DEFAULT_TIER_FOUR_TIRE_STATE,
  );
  return appendPitStrategyState(tireAwareVector, envState);
}

/**
 * Creates the reusable `{ tier: 4 }` selector for the 103-channel observation layout.
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
 * Builds the Tier 5 observation vector with the canonical 103-channel 3v3 layout.
 *
 * Channel layout:
 * - `[0..69]` — 70-channel base observation containing pose, speed, track geometry,
 *   and recurrent memory trace.
 * - `[70..90]` — team radio (`3 × 7 = 21` channels). In 3v3 all three rows can be
 *   populated; smaller packs keep any missing row zero-padded.
 * - `[91..94]` — own-car tire state `[frontLeft, frontRight, rearLeft, rearRight]`.
 * - `[95..102]` — 8 pit/strategy channels.
 *
 * Tier 5 is byte-stable with Tier 4: both tiers emit the same 103 floats in the
 * same order. The difference is radio population, not vector shape.
 *
 * @param envState - Current environment snapshot plus optional Tier 5 teammate radio rows, tire state, and pit/strategy state.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @returns 103-channel Tier 5 observation vector with the stable Tier 4 byte layout.
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
 * observation.length; // TOTAL_TIER4_INPUT_SIZE
 * observation.slice(
 *   TIER_ONE_CHANNEL_COUNT,
 *   TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
 * ); // three teammate radio rows
 * observation.slice(
 *   TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
 *   TIER_ONE_CHANNEL_COUNT +
 *     TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT +
 *     TIRE_CHANNEL_COUNT,
 * ); // own-car tire channels
 * observation.slice(
 *   TIER_ONE_CHANNEL_COUNT +
 *     TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT +
 *     TIRE_CHANNEL_COUNT,
 * ); // 8 pit/strategy channels
 * ```
 */
export function assembleTier5Observation(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): Float32Array {
  const tireAwareVector = appendOwnTireState(
    assembleTier3Observation(envState, trackSpec),
    envState.tireState ?? DEFAULT_TIER_FOUR_TIRE_STATE,
  );
  return appendPitStrategyState(tireAwareVector, envState);
}

/**
 * Creates the reusable `{ tier: 5 }` selector for the 103-channel Tier 5 layout.
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
 * observation.length; // TOTAL_TIER4_INPUT_SIZE
 * ```
 */
export function createTier5ObservationOptions(): { readonly tier: 5 } {
  return { tier: 5 };
}

/**
 * Builds the Tier 6 observation vector with the 124-channel opponent-perception layout.
 *
 * Channel layout:
 * - `[0..69]` — 70-channel base observation containing pose, speed, track geometry,
 *   and recurrent memory trace.
 * - `[70..90]` — team radio (`3 × 7 = 21` channels).
 * - `[91..94]` — own-car tire state `[frontLeft, frontRight, rearLeft, rearRight]`.
 * - `[95..102]` — 8 pit/strategy channels.
 * - `[103..123]` — 21 opponent-perception channels (`3 × 7` ego-relative slots).
 *
 * Tier 6 is byte-stable with Tier 4/5 for the first 103 channels. The 21-channel
 * opponent tail is appended after the pit/strategy suffix.
 *
 * @param envState - Current environment snapshot plus Tier 6 opponent perception slots.
 * @param trackSpec - Frozen track geometry used to derive look-ahead features.
 * @returns 124-channel Tier 6 observation vector.
 * @example
 * ```ts
 * const observation = assembleTier6Observation(
 *   {
 *     ...envState,
 *     teammateRadioSlots: [
 *       Float32Array.from([1, 0, 0, 0, 0, 0, 0]),
 *       Float32Array.from([0, 1, 0, 0, 0, 0, 0]),
 *       Float32Array.from([0, 0, 1, 0, 0, 0, 0]),
 *     ],
 *     opponentPerceptionSlots: [
 *       Float32Array.from([0.5, 0, 0, 1, 0, 0, 0.25]),
 *     ],
 *   },
 *   trackSpec,
 * );
 *
 * observation.length; // TIER6_TOTAL_INPUT_SIZE
 * observation.slice(103, 110); // opponent slot 0
 * ```
 */
export function assembleTier6Observation(
  envState: RacingObservationState & {
    opponentPerceptionSlots?: readonly (Float32Array | readonly number[])[];
  },
  trackSpec: TrackSpec,
): Float32Array {
  const baseVector = assembleTier5Observation(envState, trackSpec);
  const tierSixVector = new Float32Array(TIER6_TOTAL_INPUT_SIZE);

  tierSixVector.set(baseVector, 0);

  // Append up to three 7-channel opponent-perception slots after the 103-channel base.
  const opponentSlots = envState.opponentPerceptionSlots ?? [];
  const opponentSlotStartOffset = TOTAL_TIER4_INPUT_SIZE;
  for (
    let opponentSlotIndex = 0;
    opponentSlotIndex < TIER6_OPPONENT_SLOT_COUNT;
    opponentSlotIndex++
  ) {
    const slotData = opponentSlots[opponentSlotIndex];
    if (slotData !== undefined) {
      tierSixVector.set(
        slotData.slice(0, TIER6_OPPONENT_SLOT_CHANNELS),
        opponentSlotStartOffset +
          opponentSlotIndex * TIER6_OPPONENT_SLOT_CHANNELS,
      );
    }
  }

  return tierSixVector;
}

/**
 * Creates the reusable `{ tier: 6 }` selector for the 124-channel Tier 6 layout.
 *
 * @returns Immutable Tier 6 observation options object.
 */
export function createTier6ObservationOptions(): { readonly tier: 6 } {
  return { tier: 6 };
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
  const tireAwareVector = new Float32Array(
    baseVector.length + TIRE_CHANNEL_COUNT,
  );

  tireAwareVector.set(baseVector, 0);
  tireAwareVector.set(tireState, baseVector.length);
  return tireAwareVector;
}

/**
 * Appends the querying car's pit/strategy state as an 8-channel tail.
 *
 * The channels are ordered and map to offsets `[95..102]` of the Tier 4/5
 * observation vector:
 *   0. `pitDistanceToEntrance01` — distance to pit entrance, normalized to `[0, 1]`
 *   1. `pitOccupancyStatus` — pit-box occupancy for the car's team (`0` empty, `1` occupied)
 *   2. `lapsSincePit` — laps since the car's last pit stop, normalized to `[0, 1]`
 *   3. `teammatePitStatus` — team pit box occupied by any team member, including the
 *      querying car itself when it is pitting (`0` free, `1` occupied)
 *   4. `tireDegradationRate` — tire degradation rate, normalized to `[0, 1]`
 *   5. `estimatedLapsBeforeFailure` — estimated laps before tire failure, normalized to `[0, 1]`
 *   6. `reservedPitContext1` — reserved expansion channel
 *   7. `reservedPitContext2` — reserved expansion channel
 *
 * Any missing field is treated as zero so the vector length stays stable even
 * when the race-pack service has not populated pit/strategy data yet.
 *
 * @param baseVector - Base observation vector (usually already tire-aware).
 * @param pitStrategy - Pit/strategy state computed by the race-pack service.
 * @returns Observation vector with the 8-channel pit/strategy tail appended.
 */
function appendPitStrategyState(
  baseVector: Float32Array,
  pitStrategy: PitStrategyState,
): Float32Array {
  const tail: number[] = [
    pitStrategy.pitDistanceToEntrance01 ?? 0,
    pitStrategy.pitOccupancyStatus ?? 0,
    pitStrategy.lapsSincePit ?? 0,
    pitStrategy.teammatePitStatus ?? 0,
    pitStrategy.tireDegradationRate ?? 0,
    pitStrategy.estimatedLapsBeforeFailure ?? 0,
    pitStrategy.reservedPitContext1 ?? 0,
    pitStrategy.reservedPitContext2 ?? 0,
  ];
  const pitAwareVector = new Float32Array(
    baseVector.length + PIT_STRATEGY_CHANNEL_COUNT,
  );

  pitAwareVector.set(baseVector, 0);
  pitAwareVector.set(tail, baseVector.length);
  return pitAwareVector;
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
      observationContext.resolvedObservationState.memoryTrace[memoryIndex]!,
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
    trackSpec.splineSamples[closestSplineSampleIndex]!;
  const closestSplineSampleFrame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    closestSplineSample.globalIndex,
  );
  const signedLateralOffsetWorld = resolveSignedLateralOffsetWorld(
    envState,
    closestSplineSample,
    closestSplineSampleFrame,
  );
  const innerLaneCenterlineOffsetWorld =
    resolveInnerLaneCenterlineOffsetWorld(closestSplineSample);
  // Team 0 (blue) targets the inner-lane centerline, Team 1 (red) the outer.
  const targetLaneCenterlineOffsetWorld =
    envState.teamIndex === 1
      ? -innerLaneCenterlineOffsetWorld
      : innerLaneCenterlineOffsetWorld;
  const optimalLineLateralOffsetWorld =
    signedLateralOffsetWorld - targetLaneCenterlineOffsetWorld;
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
      envState.optimalLineLateralOffsetWorld ?? optimalLineLateralOffsetWorld,
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
  const wrappedSampleIndex = wrapSplineSampleIndex(
    splineSamples.length,
    sampleIndex + sampleOffset,
  );

  return splineSamples[wrappedSampleIndex]!;
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
  return clampNormalizedValue(value / scale);
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
  return clampNormalizedValue(Math.max(0, value) / scale);
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
  return Math.max(0, boundaryDistanceWorld);
}

function wrapSplineSampleIndex(
  sampleCount: number,
  sampleIndex: number,
): number {
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
