import { generateTrack } from '../track/track.generator';
import type { TrackAabb, TrackSpec } from '../track/track.generator.types';
import type {
  CarControlOutput,
  EnvironmentState,
  PitOccupancyRecord,
  PitOccupancyState,
  RacingCarState,
  TireStateTuple,
} from './environment.types';

const FIXED_TIMESTEP_SECONDS = 1 / 60;
const MAX_FORWARD_SPEED_UNITS_PER_SECOND = 36;
const MAX_TURN_RADIANS_PER_STEP = 0.05;
const DEFAULT_TRACK_SEED = 42;
const DEFAULT_TRACK_LAYOUT_VERSION = 1;
const DEFAULT_TRACK_SIZE_BUCKET = 'medium';
const NO_CAR_INDEX = 255;
const PIT_STOP_TICKS = 4;
const TIRE_DECAY_LATERAL_FACTOR = 0.002;
const TIRE_DECAY_LONGITUDINAL_FACTOR = 0.001;
const TIRE_DECAY_SPEED_FACTOR = 0.0001;
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

type MutablePitOccupancyState = [PitOccupancyRecord, PitOccupancyRecord];

/**
 * Creates the canonical start-of-episode environment state.
 *
 * Tier 5 seeds the owner-local 3v3 layout up front so tire decay, pit status,
 * observation assembly, and renderer fallbacks all share the same initial state.
 * Both `pitOccupancy` and the compatibility alias `pitStatus` point at the same
 * empty two-team shelf, and every car starts with fully healthy tires.
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
 * owns one pit slot, `255` means the slot is empty, a car that touches its own
 * entrance-corridor AABB serves a fixed four-tick stop, and the environment
 * restores all four tire channels to `1` when that stop expires. If multiple
 * teammates reach the same team pit on one tick, the first eligible car in
 * roster order claims the slot and the others stay on track until it clears.
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
  // Step 3: Tick pit timers before motion so released cars can rejoin this frame.
  const releasedTeams = new Set<number>();
  const pitOccupancy = tickPitOccupancy(
    resolvePitOccupancy(state),
    currentCars,
    releasedTeams,
  );
  // Step 4: Advance every car that is not currently stopped in its team's pit.
  const steppedCars = currentCars.map((car, carIndex) => {
    if (isCarStoppedInPit(pitOccupancy, carIndex)) {
      return car;
    }

    return stepCarKinematics(car, controls[carIndex]);
  });
  // Step 5: Let the first eligible teammate claim the single pit slot for its team.
  const nextPitOccupancy = resolvePitEntries(
    steppedCars,
    pitOccupancy,
    state.trackSpec ?? DEFAULT_TRACK_SPEC,
    releasedTeams,
  );
  const primaryCar = steppedCars[0] ?? createFallbackPrimaryCar();

  return {
    ...state,
    tick: state.tick + 1,
    carX: primaryCar.carX,
    carY: primaryCar.carY,
    carHeading: primaryCar.carHeading,
    teamIndex: primaryCar.teamIndex,
    tireState: primaryCar.tireState,
    cars: steppedCars,
    trackSpec: state.trackSpec ?? DEFAULT_TRACK_SPEC,
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
 * Creates an empty per-team pit occupancy shelf.
 *
 * @returns Two-slot pit occupancy tuple with no cars assigned.
 */
function createEmptyPitOccupancy(): PitOccupancyState {
  return [createEmptyPitRecord(), createEmptyPitRecord()];
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
 * Resolves the current pit occupancy shelf from either Tier 4 field name.
 *
 * @param state - Current environment state.
 * @returns Two-slot pit occupancy tuple.
 */
function resolvePitOccupancy(state: EnvironmentState): PitOccupancyState {
  const activePitOccupancy = state.pitOccupancy ?? state.pitStatus;

  if (activePitOccupancy !== undefined) {
    return clonePitOccupancy(activePitOccupancy);
  }

  return createEmptyPitOccupancy();
}

/**
 * Ticks active pit stops forward and restores tires on release.
 *
 * A record remains active while `remainingStopTicks > 0`. When the counter
 * reaches zero, the occupying car's tire tuple is reset to `[1, 1, 1, 1]`, the
 * team is marked as released for the current tick, and the record returns to the
 * `255 = no car` sentinel state.
 *
 * @param pitOccupancy - Current pit occupancy shelf.
 * @param cars - Ordered car roster for the current tick.
 * @param releasedTeams - Mutable set filled with team indices released this tick.
 * @returns Updated pit occupancy shelf after decrementing stop timers.
 */
function tickPitOccupancy(
  pitOccupancy: PitOccupancyState,
  cars: RacingCarState[],
  releasedTeams: Set<number>,
): PitOccupancyState {
  const nextPitOccupancy = clonePitOccupancy(pitOccupancy);

  for (const [teamIndex, record] of nextPitOccupancy.entries()) {
    if (
      record.remainingStopTicks <= 0 ||
      record.occupyingCarIndex === NO_CAR_INDEX
    ) {
      continue;
    }

    const nextRemainingStopTicks = record.remainingStopTicks - 1;

    if (nextRemainingStopTicks > 0) {
      nextPitOccupancy[teamIndex] = {
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

    releasedTeams.add(teamIndex);
    nextPitOccupancy[teamIndex] = createEmptyPitRecord();
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
 * Entry is based on the team's `entranceCorridor` axis-aligned box inside the
 * frozen `TrackSpec`. A team cannot claim a second pit slot while one is active,
 * and a car released earlier in the same tick cannot re-enter immediately.
 *
 * @param cars - Updated car roster.
 * @param pitOccupancy - Pit occupancy shelf after ticking active stops.
 * @param trackSpec - Active track metadata.
 * @param releasedTeams - Teams released this tick and therefore blocked from re-entry.
 * @returns Final pit occupancy shelf for the next state.
 */
function resolvePitEntries(
  cars: readonly RacingCarState[],
  pitOccupancy: PitOccupancyState,
  trackSpec: TrackSpec,
  releasedTeams: ReadonlySet<number>,
): PitOccupancyState {
  const nextPitOccupancy = clonePitOccupancy(pitOccupancy);
  const pitBoxes = trackSpec.pitBoxes ?? [];

  for (let carIndex = 0; carIndex < cars.length; carIndex++) {
    const car = cars[carIndex];
    const teamIndex = car.teamIndex;

    if (
      nextPitOccupancy[teamIndex].occupyingCarIndex !== NO_CAR_INDEX ||
      releasedTeams.has(teamIndex)
    ) {
      continue;
    }

    const pitBox = pitBoxes.find(
      (candidatePitBox) => candidatePitBox.teamIndex === teamIndex,
    );

    if (
      pitBox !== undefined &&
      isPointInsideAabb(car.carX, car.carY, pitBox.entranceCorridor)
    ) {
      nextPitOccupancy[teamIndex] = {
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
 * Clones the fixed two-slot pit shelf into a mutable tuple.
 *
 * @param pitOccupancy - Source pit occupancy state.
 * @returns Mutable clone suitable for in-step updates.
 */
function clonePitOccupancy(
  pitOccupancy: PitOccupancyState,
): MutablePitOccupancyState {
  return [{ ...pitOccupancy[0] }, { ...pitOccupancy[1] }];
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
