import type { TrackSpec } from '../track/track.generator.types';

/**
 * Ordered tire-health tuple for one car.
 *
 * The layout is always `[frontLeft, frontRight, rearLeft, rearRight]`, with
 * each channel clamped to the closed `[0, 1]` interval.
 */
export type TireStateTuple = readonly [number, number, number, number];

/**
 * Fixed pit occupancy record for one team's single pit slot.
 *
 * The race pack always exposes one record per team. `occupyingCarIndex = 255`
 * means the pit is empty; otherwise `remainingStopTicks` counts down the fixed
 * stop duration before the environment releases the car and restores all four
 * tire channels back to full health.
 */
export type PitOccupancyRecord = {
  /** Car index currently in the pit, or `255` when no car is occupying it. */
  occupyingCarIndex: number;
  /** Remaining fixed stop ticks before the car is released and its tires are reset. */
  remainingStopTicks: number;
};

/**
 * Fixed two-record pit shelf shared by Team A and Team B.
 *
 * Slot `0` is Team A's pit and slot `1` is Team B's pit. The shelf width does
 * not change with roster size, so single-car, 2v2, and 3v3 packs all reuse the
 * same `255 = no car` sentinel and stop-tick contract.
 */
export type PitOccupancyState = readonly [
  PitOccupancyRecord,
  PitOccupancyRecord,
];

/**
 * Per-car racing state tracked by the environment.
 *
 * Tire health stays owner-local on each car so grip, pit restore, worker
 * projection, and observation assembly all read the same state regardless of
 * whether the active pack is solo, 2v2, or 3v3.
 */
export type CarState = {
  /** Car X position in logical world units. */
  carX: number;
  /** Car Y position in logical world units. */
  carY: number;
  /** Car heading in radians (0 = facing positive X axis). */
  carHeading: number;
  /** Team index for the packed multi-car layout (`0 = Team A`, `1 = Team B`). */
  teamIndex: 0 | 1;
  /** Ordered tire-health tuple `[FL, FR, RL, RR]`. */
  tireState: TireStateTuple;
};

/** Concrete racing-car alias kept for the browser and worker seams. */
export type RacingCarState = CarState;

/**
 * Runtime state for one simulation episode.
 *
 * The legacy top-level `carX`, `carY`, and `carHeading` fields stay in place so
 * the solo browser path remains stable. Multi-car packs layer the ordered `cars`
 * roster on top; the six-car 3v3 slice simply sets `cars.length = 6` while
 * reusing the same per-team `pitOccupancy` and compatibility `pitStatus` shelf.
 */
export type EnvironmentState = {
  /** Monotonic fixed-timestep counter. Starts at 0 and increments by 1 per step. */
  tick: number;
  /** Primary car X position in logical world units. */
  carX: number;
  /** Primary car Y position in logical world units. */
  carY: number;
  /** Primary car heading in radians (0 = facing positive X axis). */
  carHeading: number;
  /** Optional primary-car team index for single-car compatibility seams. */
  teamIndex?: 0 | 1;
  /** Optional primary-car tire tuple for observation and renderer fallbacks. */
  tireState?: TireStateTuple;
  /** Optional ordered car roster for multi-car tiers; the 3v3 slice uses six cars in canonical team order. */
  cars?: readonly RacingCarState[];
  /** Optional frozen track metadata owned by the current episode. */
  trackSpec?: TrackSpec;
  /** Fixed per-team pit shelf; `255` marks an empty pit and stop ticks count down to tire reset. */
  pitOccupancy?: PitOccupancyState;
  /** Backward-compatible alias exposing the same two-record pit shelf to UI/tests. */
  pitStatus?: PitOccupancyState;
};

/**
 * Control signal from the car's controller for one simulation tick.
 *
 * Both values are normalised to [-1, 1] and clamped inside `stepEnvironment`.
 */
export type CarControlOutput = {
  /** Forward/backward throttle. Positive = accelerate forward. Range [-1, 1]. */
  throttle: number;
  /** Steering. Positive = turn right. Range [-1, 1]. */
  steer: number;
};
