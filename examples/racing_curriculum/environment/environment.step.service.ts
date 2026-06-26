import { generateTrack } from '../track/track.generator';
import type { TrackAabb, TrackSpec } from '../track/track.generator.types';
import { resolveSplineSampleFrame } from '../track/track.spline.utils';
import type {
  CarControlOutput,
  EnvironmentState,
  PitOccupancyRecord,
  PitOccupancyState,
  RacingCarState,
  TireStateTuple,
} from './environment.types';

const FIXED_TIMESTEP_SECONDS = 1 / 60;
const MAX_FORWARD_SPEED_UNITS_PER_SECOND = 108;
const MAX_TURN_RADIANS_PER_STEP = 0.05;
const DEFAULT_TRACK_SEED = 42;
const DEFAULT_TRACK_LAYOUT_VERSION = 1;
const DEFAULT_TRACK_SIZE_BUCKET = 'medium';
const NO_CAR_INDEX = 255;
const PIT_STOP_TICKS = 4;
const TEAM_COUNT = 2;
const PIT_SLOTS_PER_TEAM = 3;
const PIT_SLOT_COUNT = TEAM_COUNT * PIT_SLOTS_PER_TEAM;
/** Car bounding-box half-dimensions in world units, mirroring renderer constants. */
const CAR_HALF_WIDTH = 2.2;
const CAR_HALF_LENGTH = 3.8;
/** Minimum Euclidean center-to-center distance enforced between any car pair. */
const CAR_MIN_CENTER_SEPARATION = CAR_HALF_WIDTH * 2 + 1;
/** Small margin added to AABB separation to avoid floating-point edge-touching. */
const SEPARATION_MARGIN = 0.01;
/** Maximum iterations for the pairwise overlap resolution solver. */
const SEPARATION_MAX_ITERATIONS = 200;
/** Penalty assigned when a car is clamped back onto the track ribbon. */
const OFF_TRACK_CLAMP_REWARD = -1;
/** Penalty assigned when a car moves opposite the track tangent. */
const WRONG_DIRECTION_REWARD = -1;
/** Base lateral wear contribution per step for the tire-health model. */
const TIRE_DECAY_LATERAL_FACTOR = 0.00012;
/** Base longitudinal wear contribution per step for the tire-health model. */
const TIRE_DECAY_LONGITUDINAL_FACTOR = 0.00006;
/** Base speed wear contribution per step for the tire-health model. */
const TIRE_DECAY_SPEED_FACTOR = 0.000006;
const TIRE_DECAY_ACCELERATION_FACTOR = 0.5;
const TEAM_LAYOUT: readonly [0, 0, 0, 1, 1, 1] = [0, 0, 0, 1, 1, 1];
const DEFAULT_TIRE_STATE: TireStateTuple = [1, 1, 1, 1];
const DEFAULT_TRACK_SPEC = generateTrack({
  seed: DEFAULT_TRACK_SEED,
  layoutVersion: DEFAULT_TRACK_LAYOUT_VERSION,
  sizeBucket: DEFAULT_TRACK_SIZE_BUCKET,
});

/** Accepted control input for one deterministic environment step. */
type EnvironmentControlInput = CarControlOutput | readonly CarControlOutput[];

type MutablePitOccupancyState = PitOccupancyRecord[];

/**
 * Creates the canonical start-of-episode environment state.
 *
 * Tier 5 seeds the owner-local 3v3 layout up front so tire decay, pit status,
 * observation assembly, and renderer fallbacks all share the same initial state.
 * Both `pitOccupancy` and the compatibility alias `pitStatus` point at the same
 * empty six-slot shelf, and every car starts with fully healthy tires.
 *
 * @returns Fresh environment state with six cars, healthy tires, and empty pits.
 */
export function createInitialState(): EnvironmentState {
  const cars = createInitialCars();
  const pitOccupancy = createEmptyPitOccupancy();
  const primaryCar = cars[0];

  return {
    tick: 0,
    carX: primaryCar.carX,
    carY: primaryCar.carY,
    carHeading: primaryCar.carHeading,
    teamIndex: primaryCar.teamIndex,
    tireState: primaryCar.tireState,
    cars,
    trackSpec: DEFAULT_TRACK_SPEC,
    pitOccupancy,
    pitStatus: pitOccupancy,
  };
}

/**
 * Decays all four tire channels using the pinned Tier 4 degradation formula.
 *
 * The base wear for one step is
 * `|lateralForce| * lateralFactor + |longitudinalForce| * longitudinalFactor + |speed| * speedFactor`.
 * Each tire then multiplies that base wear by `1 + (1 - tireHealth) * accelerationFactor`,
 * so already-degraded tires lose grip faster than fresh tires under the same load.
 * The final value is always clamped to the closed `[0.0, 1.0]` interval.
 *
 * @param current - Current tire-health tuple `[FL, FR, RL, RR]`.
 * @param lateralForce - Absolute lateral load for the current step.
 * @param longitudinalForce - Absolute longitudinal load for the current step.
 * @param speed - Scalar speed used by the degradation model.
 * @returns New clamped tire-health tuple after one decay step.
 * @example
 * ```ts
 * const freshTires = decayTireState([1, 1, 1, 1], 0.2, 0.4, 10);
 * const wornTires = decayTireState([0.3, 0.3, 0.3, 0.3], 0.2, 0.4, 10);
 *
 * wornTires[0] < freshTires[0]; // already-damaged tires decay faster
 * ```
 */
export function decayTireState(
  current: readonly [number, number, number, number],
  lateralForce: number,
  longitudinalForce: number,
  speed: number,
): [number, number, number, number] {
  const baseDecay =
    Math.abs(lateralForce) * TIRE_DECAY_LATERAL_FACTOR +
    Math.abs(longitudinalForce) * TIRE_DECAY_LONGITUDINAL_FACTOR +
    Math.abs(speed) * TIRE_DECAY_SPEED_FACTOR;

  if (baseDecay === 0) {
    return [...current];
  }

  return current.map((tireHealth) => {
    const degradationAcceleration =
      1 + (1 - tireHealth) * TIRE_DECAY_ACCELERATION_FACTOR;
    const nextTireHealth = tireHealth - baseDecay * degradationAcceleration;

    return clampUnitInterval(nextTireHealth);
  }) as [number, number, number, number];
}

/**
 * Advances the environment by exactly one fixed timestep.
 *
 * The returned state is a new object; the input state is never mutated.
 * `tick` is incremented by exactly one. The stepping path is generic over the
 * resolved car roster length, so solo callers may still pass a single control
 * object while multi-car packs supply one control per car. In the 3v3 slice,
 * `state.cars.length = 6` and the same loop structure advances all six cars.
 *
 * Pit lifecycle semantics are intentionally narrow and deterministic: each team
 * owns three pit slots (`255` means a slot is empty), a car that touches one of
 * its team's entrance-corridor AABBs claims an available owned slot for a fixed
 * four-tick stop, and the environment restores all four tire channels to `1`
 * when that stop expires. Entry stays deterministic: cars claim in roster order
 * and each car can claim at most one slot per tick.
 *
 * @param state - Current environment state.
 * @param control - Single-car control or ordered per-car controls for this tick.
 * @returns New environment state after one fixed timestep.
 */
export function stepEnvironment(
  state: EnvironmentState,
  control: EnvironmentControlInput,
): EnvironmentState {
  // Step 1: Clone the resolved roster so solo, 2v2, and 3v3 packs share one stepping path.
  const currentCars = resolveCars(state).map((car) => ({
    ...car,
    tireState: [...car.tireState] as TireStateTuple,
  }));
  // Step 2: Expand the caller input to the exact car count for this pack.
  const controls = resolveControls(control, currentCars.length);
  // Step 3: Resolve the active track geometry (used by clamp, direction, and pit checks).
  const trackSpec = state.trackSpec ?? DEFAULT_TRACK_SPEC;
  // Step 4: Tick pit timers before motion so expired slots release deterministically.
  const releasedCars = new Set<number>();
  const pitOccupancy = tickPitOccupancy(
    resolvePitOccupancy(state),
    currentCars,
    releasedCars,
  );
  // Step 5: Advance every car that is not currently stopped in its team's pit.
  const steppedCars = currentCars.map((car, carIndex) => {
    if (isCarStoppedInPit(pitOccupancy, carIndex)) {
      return car;
    }

    return stepCarKinematics(car, controls[carIndex]);
  });
  // Step 6: Detect wrong-direction motion before clamping so the original velocity is read.
  const wrongDirectionFlags = detectWrongDirection(
    currentCars,
    steppedCars,
    trackSpec,
  );
  // Step 7: Enforce track boundary walls and assign off-track/wrong-direction rewards.
  const boundedCars = steppedCars.map((car, carIndex) => {
    if (isCarStoppedInPit(pitOccupancy, carIndex)) {
      return car;
    }

    const clamped = clampCarToTrackBounds(car, trackSpec);
    const wasClamped = clamped.carX !== car.carX || clamped.carY !== car.carY;
    let reward = 0;
    if (wasClamped) {
      reward += OFF_TRACK_CLAMP_REWARD;
    }
    if (wrongDirectionFlags[carIndex]) {
      reward += WRONG_DIRECTION_REWARD;
    }

    return reward === 0 ? clamped : { ...clamped, reward };
  });
  // Step 8: Push overlapping cars apart so two cars cannot share the same space.
  const separatedCars = separateCars(boundedCars);
  // Step 9: Let each car claim one available own-team slot in deterministic roster order.
  const nextPitOccupancy = resolvePitEntries(
    separatedCars,
    pitOccupancy,
    trackSpec,
    releasedCars,
  );
  const primaryCar = separatedCars[0] ?? createFallbackPrimaryCar();

  return {
    ...state,
    tick: state.tick + 1,
    carX: primaryCar.carX,
    carY: primaryCar.carY,
    carHeading: primaryCar.carHeading,
    teamIndex: primaryCar.teamIndex,
    tireState: primaryCar.tireState,
    cars: separatedCars,
    trackSpec,
    pitOccupancy: nextPitOccupancy,
    pitStatus: nextPitOccupancy,
  };
}

/**
 * Advances the environment by a fixed number of deterministic timesteps.
 *
 * @param state - Current environment state.
 * @param control - Controller output applied for each batched step.
 * @param stepCount - Number of fixed timesteps to apply.
 * @returns New environment state after the batch completes.
 */
export function stepEnvironmentBatch(
  state: EnvironmentState,
  control: CarControlOutput,
  stepCount: number,
): EnvironmentState {
  let nextState = state;

  for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
    nextState = stepEnvironment(nextState, control);
  }

  return nextState;
}

/**
 * Creates the canonical six-car owner-local roster.
 *
 * @returns Ordered 3v3 car list `[A0, A1, A2, B0, B1, B2]` with full tire health.
 */
function createInitialCars(): readonly RacingCarState[] {
  return TEAM_LAYOUT.map((teamIndex) => ({
    carX: 0,
    carY: 0,
    carHeading: 0,
    teamIndex,
    tireState: [...DEFAULT_TIRE_STATE] as TireStateTuple,
  }));
}

/**
 * Creates an empty six-slot pit occupancy shelf.
 *
 * @returns Six-slot pit occupancy shelf with no cars assigned.
 */
function createEmptyPitOccupancy(): PitOccupancyState {
  return Array.from({ length: PIT_SLOT_COUNT }, () => createEmptyPitRecord());
}

/**
 * Creates one empty pit occupancy record.
 *
 * @returns Empty record using the `255` no-car sentinel.
 */
function createEmptyPitRecord(): PitOccupancyRecord {
  return { occupyingCarIndex: NO_CAR_INDEX, remainingStopTicks: 0 };
}

/**
 * Resolves the active car roster from legacy or Tier 4 state shapes.
 *
 * @param state - Current environment state.
 * @returns Ordered car list to step for the current tick.
 */
function resolveCars(state: EnvironmentState): readonly RacingCarState[] {
  if (state.cars?.length) {
    return state.cars;
  }

  return [
    {
      carX: state.carX,
      carY: state.carY,
      carHeading: state.carHeading,
      teamIndex: state.teamIndex ?? 0,
      tireState: [...(state.tireState ?? DEFAULT_TIRE_STATE)] as TireStateTuple,
    },
  ];
}

/**
 * Resolves the ordered control list for the active car roster.
 *
 * @param control - Single-car or per-car control input.
 * @param carCount - Number of cars that will be stepped this tick.
 * @returns Per-car control list aligned to the roster order.
 */
function resolveControls(
  control: EnvironmentControlInput,
  carCount: number,
): readonly CarControlOutput[] {
  if (Array.isArray(control)) {
    const perCarControl = control as readonly CarControlOutput[];

    return Array.from(
      { length: carCount },
      (_, carIndex) => perCarControl[carIndex] ?? { throttle: 0, steer: 0 },
    );
  }

  const primaryControl = control as CarControlOutput;

  return Array.from({ length: carCount }, (_, carIndex) =>
    carIndex === 0 ? primaryControl : { throttle: 0, steer: 0 },
  );
}

/**
 * Finds the spline sample nearest to a world-space point.
 *
 * Used by both boundary clamping and wrong-direction detection so both
 * features agree on the local track frame.
 *
 * @param x - Point X coordinate.
 * @param y - Point Y coordinate.
 * @param trackSpec - Active track geometry.
 * @returns Index of the nearest spline sample.
 */
function resolveNearestSampleIndex(
  x: number,
  y: number,
  trackSpec: TrackSpec,
): number {
  const { splineSamples } = trackSpec;
  let nearestSampleIndex = 0;
  let nearestDistanceSquared = Infinity;

  for (let sampleIndex = 0; sampleIndex < splineSamples.length; sampleIndex++) {
    const sample = splineSamples[sampleIndex]!;
    const deltaX = x - sample.x;
    const deltaY = y - sample.y;
    const distanceSquared = deltaX * deltaX + deltaY * deltaY;

    if (distanceSquared < nearestDistanceSquared) {
      nearestDistanceSquared = distanceSquared;
      nearestSampleIndex = sampleIndex;
    }
  }

  return nearestSampleIndex;
}

/**
 * Detects cars that moved opposite to the track tangent during this step.
 *
 * A car is flagged when its displacement vector has a negative dot product
 * with the forward tangent at its pre-step nearest sample. Stationary cars are
 * never flagged, so a parked car cannot accumulate wrong-direction penalties.
 *
 * @param beforeCars - Car roster before the kinematic update.
 * @param afterCars - Car roster after the kinematic update.
 * @param trackSpec - Active track geometry.
 * @returns Per-car boolean flags; `true` means wrong-direction motion.
 */
function detectWrongDirection(
  beforeCars: readonly RacingCarState[],
  afterCars: readonly RacingCarState[],
  trackSpec: TrackSpec,
): readonly boolean[] {
  const { splineSamples } = trackSpec;

  if (splineSamples.length === 0) {
    return Array.from({ length: beforeCars.length }, () => false);
  }

  return beforeCars.map((beforeCar, carIndex) => {
    const afterCar = afterCars[carIndex];

    if (afterCar === undefined) {
      return false;
    }

    const deltaX = afterCar.carX - beforeCar.carX;
    const deltaY = afterCar.carY - beforeCar.carY;
    const displacement = Math.hypot(deltaX, deltaY);

    if (displacement < 1e-9) {
      return false;
    }

    const nearestSampleIndex = resolveNearestSampleIndex(
      beforeCar.carX,
      beforeCar.carY,
      trackSpec,
    );
    const sampleFrame = resolveSplineSampleFrame(
      splineSamples,
      nearestSampleIndex,
    );
    // The unit tangent is perpendicular to the left normal.
    const tangentX = sampleFrame.normalY;
    const tangentY = -sampleFrame.normalX;
    const dotProduct = deltaX * tangentX + deltaY * tangentY;

    return dotProduct < -1e-9;
  });
}

/**
 * Pushes overlapping car centers apart so bounding boxes never overlap.
 *
 * Uses axis-aligned bounding boxes with half-extents {@link CAR_HALF_WIDTH}
 * along X and {@link CAR_HALF_LENGTH} along Y. For each overlapping pair the
 * required center-to-center distance is computed along the connecting line so
 * that either the X gap exceeds the combined half-widths or the Y gap exceeds
 * the combined half-lengths, whichever is smaller. A minimum Euclidean
 * separation of {@link CAR_MIN_CENTER_SEPARATION} is also enforced. The solver
 * iterates up to {@link SEPARATION_MAX_ITERATIONS} times to resolve cascading
 * overlaps in multi-car stacks.
 *
 * @param cars - Car roster after track-boundary clamping.
 * @returns New roster with overlapping cars separated in place.
 */
function separateCars(
  cars: readonly RacingCarState[],
): readonly RacingCarState[] {
  if (cars.length < 2) {
    return cars;
  }

  const mutableCars = cars.map((car) => ({ ...car }));

  for (let iteration = 0; iteration < SEPARATION_MAX_ITERATIONS; iteration++) {
    let resolved = true;

    for (let firstIndex = 0; firstIndex < mutableCars.length; firstIndex++) {
      for (
        let secondIndex = firstIndex + 1;
        secondIndex < mutableCars.length;
        secondIndex++
      ) {
        const firstCar = mutableCars[firstIndex]!;
        const secondCar = mutableCars[secondIndex]!;
        const deltaX = secondCar.carX - firstCar.carX;
        const deltaY = secondCar.carY - firstCar.carY;
        const distance = Math.hypot(deltaX, deltaY);

        const overlapX = CAR_HALF_WIDTH * 2 - Math.abs(deltaX);
        const overlapY = CAR_HALF_LENGTH * 2 - Math.abs(deltaY);
        const aabbOverlapping = overlapX > 0 && overlapY > 0;
        const tooClose = distance < CAR_MIN_CENTER_SEPARATION;

        if (!aabbOverlapping && !tooClose) {
          continue;
        }

        let unitX: number;
        let unitY: number;
        if (distance < 1e-9) {
          unitX = 1;
          unitY = 0;
        } else {
          unitX = deltaX / distance;
          unitY = deltaY / distance;
        }

        const aabbDistX =
          Math.abs(unitX) > 1e-9
            ? (CAR_HALF_WIDTH * 2 + SEPARATION_MARGIN) / Math.abs(unitX)
            : Infinity;
        const aabbDistY =
          Math.abs(unitY) > 1e-9
            ? (CAR_HALF_LENGTH * 2 + SEPARATION_MARGIN) / Math.abs(unitY)
            : Infinity;
        const aabbDist = Math.min(aabbDistX, aabbDistY);
        const required = Math.max(CAR_MIN_CENTER_SEPARATION, aabbDist) + 1e-6;

        if (distance >= required) {
          continue;
        }

        resolved = false;

        const push = (required - distance) / 2;
        firstCar.carX -= unitX * push;
        firstCar.carY -= unitY * push;
        secondCar.carX += unitX * push;
        secondCar.carY += unitY * push;
      }
    }

    if (resolved) {
      break;
    }
  }

  return mutableCars;
}

/**
 * Resolves the current pit occupancy shelf from either Tier 4 field name.
 *
 * @param state - Current environment state.
 * @returns Six-slot pit occupancy shelf.
 */
function resolvePitOccupancy(state: EnvironmentState): PitOccupancyState {
  const activePitOccupancy = state.pitOccupancy ?? state.pitStatus;

  if (activePitOccupancy !== undefined) {
    return normalizePitOccupancy(activePitOccupancy);
  }

  return createEmptyPitOccupancy();
}

/**
 * Normalizes incoming pit occupancy state to the fixed six-slot layout.
 *
 * Legacy two-slot inputs are mapped from `[teamA, teamB]` to
 * `[A0, A1, A2, B0, B1, B2]` by placing Team A at slot `0` and Team B at
 * slot `3`.
 *
 * @param pitOccupancy - Incoming pit occupancy state.
 * @returns Six-slot normalized pit occupancy shelf.
 */
function normalizePitOccupancy(
  pitOccupancy: PitOccupancyState,
): PitOccupancyState {
  if (pitOccupancy.length === PIT_SLOT_COUNT) {
    return clonePitOccupancy(pitOccupancy);
  }

  const normalizedPitOccupancy = clonePitOccupancy(createEmptyPitOccupancy());

  if (pitOccupancy.length === TEAM_COUNT) {
    normalizedPitOccupancy[0] = { ...pitOccupancy[0] };
    normalizedPitOccupancy[PIT_SLOTS_PER_TEAM] = { ...pitOccupancy[1] };
    return normalizedPitOccupancy;
  }

  for (
    let pitSlotIndex = 0;
    pitSlotIndex < Math.min(pitOccupancy.length, PIT_SLOT_COUNT);
    pitSlotIndex++
  ) {
    normalizedPitOccupancy[pitSlotIndex] = { ...pitOccupancy[pitSlotIndex] };
  }

  return normalizedPitOccupancy;
}

/**
 * Ticks active pit stops forward and restores tires on release.
 *
 * A record remains active while `remainingStopTicks > 0`. When the counter
 * reaches zero, the occupying car's tire tuple is reset to `[1, 1, 1, 1]`, the
 * released car index is marked for same-tick re-entry blocking, and the record
 * returns to the `255 = no car` sentinel state.
 *
 * @param pitOccupancy - Current pit occupancy shelf.
 * @param cars - Ordered car roster for the current tick.
 * @param releasedCars - Mutable set filled with car indices released this tick.
 * @returns Updated pit occupancy shelf after decrementing stop timers.
 */
function tickPitOccupancy(
  pitOccupancy: PitOccupancyState,
  cars: RacingCarState[],
  releasedCars: Set<number>,
): PitOccupancyState {
  const nextPitOccupancy = clonePitOccupancy(pitOccupancy);

  for (const [pitSlotIndex, record] of nextPitOccupancy.entries()) {
    if (
      record.remainingStopTicks <= 0 ||
      record.occupyingCarIndex === NO_CAR_INDEX
    ) {
      continue;
    }

    const nextRemainingStopTicks = record.remainingStopTicks - 1;

    if (nextRemainingStopTicks > 0) {
      nextPitOccupancy[pitSlotIndex] = {
        occupyingCarIndex: record.occupyingCarIndex,
        remainingStopTicks: nextRemainingStopTicks,
      };
      continue;
    }

    const occupyingCar = cars[record.occupyingCarIndex];
    if (occupyingCar !== undefined) {
      cars[record.occupyingCarIndex] = {
        ...occupyingCar,
        tireState: [...DEFAULT_TIRE_STATE] as TireStateTuple,
      };
    }

    releasedCars.add(record.occupyingCarIndex);
    nextPitOccupancy[pitSlotIndex] = createEmptyPitRecord();
  }

  return nextPitOccupancy;
}

/**
 * Returns whether the given car is currently waiting out an active pit stop.
 *
 * @param pitOccupancy - Current pit occupancy shelf.
 * @param carIndex - Car index being checked.
 * @returns True when the car is locked in a pit box this tick.
 */
function isCarStoppedInPit(
  pitOccupancy: PitOccupancyState,
  carIndex: number,
): boolean {
  return pitOccupancy.some(
    (record) =>
      record.occupyingCarIndex === carIndex && record.remainingStopTicks > 0,
  );
}

/**
 * Pulls a car's world position back onto the drivable ribbon if it has crossed
 * either the inner or outer track edge.
 *
 * The ribbon is approximated by the nearest spline sample: the car is clamped
 * so its signed lateral offset stays within `[-halfWidth, halfWidth]`. Cars that
 * are stopped in a pit box are intentionally skipped, because pit stalls live
 * outside the drivable surface.
 *
 * @param car - Car whose position should be clamped.
 * @param trackSpec - Frozen track geometry used for the boundary lookup.
 * @returns Car state with its position bounded to the track ribbon.
 */
function clampCarToTrackBounds(
  car: RacingCarState,
  trackSpec: TrackSpec,
): RacingCarState {
  const { splineSamples } = trackSpec;
  if (splineSamples.length === 0) {
    return car;
  }

  const nearestSampleIndex = resolveNearestSampleIndex(
    car.carX,
    car.carY,
    trackSpec,
  );
  const nearestSample = splineSamples[nearestSampleIndex]!;
  const sampleFrame = resolveSplineSampleFrame(
    splineSamples,
    nearestSampleIndex,
  );
  const halfWidth = nearestSample.width / 2;
  const offsetX = car.carX - nearestSample.x;
  const offsetY = car.carY - nearestSample.y;
  const signedOffset =
    offsetX * sampleFrame.normalX + offsetY * sampleFrame.normalY;

  if (Math.abs(signedOffset) <= halfWidth + 1e-9) {
    return car;
  }

  const clampedOffset = Math.max(-halfWidth, Math.min(halfWidth, signedOffset));
  const correction = clampedOffset - signedOffset;

  return {
    ...car,
    carX: car.carX + sampleFrame.normalX * correction,
    carY: car.carY + sampleFrame.normalY * correction,
  };
}

/**
 * Steps one car forward using the pinned Tier 4 grip and decay rules.
 *
 * @param car - Current car state.
 * @param control - Controller output for the car.
 * @returns New car state after one fixed timestep.
 */
function stepCarKinematics(
  car: RacingCarState,
  control: CarControlOutput,
): RacingCarState {
  const clampedThrottle = clampControlValue(control.throttle);
  const clampedSteer = clampControlValue(control.steer);
  const gripMultiplier = Math.sqrt(resolveMeanTireHealth(car.tireState));
  const effectiveSteer = clampedSteer * gripMultiplier;
  const effectiveThrottle =
    clampedThrottle < 0 ? clampedThrottle * gripMultiplier : clampedThrottle;
  const nextHeading =
    car.carHeading + effectiveSteer * MAX_TURN_RADIANS_PER_STEP;
  const forwardDistance =
    effectiveThrottle *
    MAX_FORWARD_SPEED_UNITS_PER_SECOND *
    FIXED_TIMESTEP_SECONDS;
  const speed =
    Math.abs(effectiveThrottle) * MAX_FORWARD_SPEED_UNITS_PER_SECOND;

  return {
    ...car,
    carHeading: nextHeading,
    carX: car.carX + Math.cos(nextHeading) * forwardDistance,
    carY: car.carY + Math.sin(nextHeading) * forwardDistance,
    tireState: decayTireState(
      car.tireState,
      Math.abs(clampedSteer),
      Math.abs(clampedThrottle),
      speed,
    ),
  };
}

/**
 * Detects new pit entries after the current tick's car updates complete.
 *
 * Entry is based on any of the team's `entranceCorridor` axis-aligned boxes
 * inside the frozen `TrackSpec`. Cars may claim up to one own-team slot each,
 * but cars released earlier in the same tick cannot re-enter immediately.
 *
 * @param cars - Updated car roster.
 * @param pitOccupancy - Pit occupancy shelf after ticking active stops.
 * @param trackSpec - Active track metadata.
 * @param releasedCars - Cars released this tick and therefore blocked from re-entry.
 * @returns Final pit occupancy shelf for the next state.
 */
function resolvePitEntries(
  cars: readonly RacingCarState[],
  pitOccupancy: PitOccupancyState,
  trackSpec: TrackSpec,
  releasedCars: ReadonlySet<number>,
): PitOccupancyState {
  const nextPitOccupancy = clonePitOccupancy(pitOccupancy);
  const pitBoxes = trackSpec.pitBoxes ?? [];

  for (let carIndex = 0; carIndex < cars.length; carIndex++) {
    const car = cars[carIndex];

    if (
      releasedCars.has(carIndex) ||
      isCarStoppedInPit(nextPitOccupancy, carIndex)
    ) {
      continue;
    }

    const availablePitSlotIndex = pitBoxes.findIndex(
      (candidatePitBox, candidatePitSlotIndex) =>
        candidatePitSlotIndex < nextPitOccupancy.length &&
        candidatePitBox.teamIndex === car.teamIndex &&
        nextPitOccupancy[candidatePitSlotIndex].occupyingCarIndex ===
          NO_CAR_INDEX &&
        isPointInsideAabb(car.carX, car.carY, candidatePitBox.entranceCorridor),
    );

    if (availablePitSlotIndex >= 0) {
      nextPitOccupancy[availablePitSlotIndex] = {
        occupyingCarIndex: carIndex,
        remainingStopTicks: PIT_STOP_TICKS,
      };
    }
  }

  return nextPitOccupancy;
}

/**
 * Computes the mean tire-health value used by the grip model.
 *
 * @param tireState - Ordered tire-health tuple.
 * @returns Mean health across all four corners.
 */
function resolveMeanTireHealth(tireState: TireStateTuple): number {
  return (
    tireState.reduce((sum, tireHealth) => sum + tireHealth, 0) /
    tireState.length
  );
}

/**
 * Clones the fixed six-slot pit shelf into a mutable array.
 *
 * @param pitOccupancy - Source pit occupancy state.
 * @returns Mutable clone suitable for in-step updates.
 */
function clonePitOccupancy(
  pitOccupancy: PitOccupancyState,
): MutablePitOccupancyState {
  return pitOccupancy.map((record) => ({ ...record }));
}

/**
 * Returns whether the point lies inside the provided axis-aligned rectangle.
 *
 * @param x - Point X coordinate.
 * @param y - Point Y coordinate.
 * @param axisAlignedBox - Rectangle to test.
 * @returns True when the point is inside or on the rectangle boundary.
 */
function isPointInsideAabb(
  x: number,
  y: number,
  axisAlignedBox: TrackAabb,
): boolean {
  return (
    x >= axisAlignedBox.x &&
    x <= axisAlignedBox.x + axisAlignedBox.width &&
    y >= axisAlignedBox.y &&
    y <= axisAlignedBox.y + axisAlignedBox.height
  );
}

/**
 * Creates a primary-car fallback used only when the roster is unexpectedly empty.
 *
 * @returns Neutral Team A fallback car.
 */
function createFallbackPrimaryCar(): RacingCarState {
  return {
    carX: 0,
    carY: 0,
    carHeading: 0,
    teamIndex: 0,
    tireState: [...DEFAULT_TIRE_STATE] as TireStateTuple,
  };
}

/**
 * Clamps control values to the accepted controller range.
 *
 * @param value - Incoming control signal.
 * @returns Value clamped to [-1, 1].
 */
function clampControlValue(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

/**
 * Clamps a value to the closed `[0, 1]` interval.
 *
 * @param value - Incoming floating-point value.
 * @returns Value clamped to the unit interval.
 */
function clampUnitInterval(value: number): number {
  return Math.max(0, Math.min(1, value));
}
