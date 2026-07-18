import type { TrackSpec } from '../track/track.generator.types';

/**
 * Ordered tire-health tuple for one car.
 *
 * The layout is always `[frontLeft, frontRight, rearLeft, rearRight]`, with
 * each channel clamped to the closed `[0, 1]` interval.
 */
export type TireStateTuple = readonly [number, number, number, number];

/**
 * Fixed pit occupancy record for one team-owned pit slot.
 *
 * `occupyingCarIndex = 255` means the slot is empty; otherwise
 * `remainingStopTicks` counts down the fixed stop duration before the
 * environment releases the car and restores all four tire channels back to
 * full health.
 */
export type PitOccupancyRecord = {
  /** Car index currently in the pit, or `255` when no car is occupying it. */
  occupyingCarIndex: number;
  /** Remaining fixed stop ticks before the car is released and its tires are reset. */
  remainingStopTicks: number;
};

/**
 * Fixed six-record pit shelf shared by Team A and Team B.
 *
 * The canonical slot order is `[A0, A1, A2, B0, B1, B2]`. The shelf width does
 * not change with roster size, so single-car, 2v2, and 3v3 packs all reuse this
 * same six-slot state with the `255 = no car` sentinel and fixed stop-tick
 * contract.
 */
export type PitOccupancyState = readonly PitOccupancyRecord[];

/**
 * Pit/strategy sensory state appended to the Tier 4/5 observation tail.
 *
 * All channels are normalized to `[0, 1]` (or zero when unavailable). The
 * assembler treats every field as optional so that Tier 1–3 code paths stay
 * unchanged until the race-pack service explicitly provides pit/strategy data.
 *
 * The canonical channel order matches offsets `[95..102]` of the Tier 4/5
 * observation vector:
 *   1. `pitDistanceToEntrance01`
 *   2. `pitOccupancyStatus`
 *   3. `lapsSincePit`
 *   4. `teammatePitStatus`
 *   5. `tireDegradationRate`
 *   6. `estimatedLapsBeforeFailure`
 *   7. `reservedPitContext1`
 *   8. `reservedPitContext2`
 *
 * `teammatePitStatus` is high whenever the team pit box is occupied by any
 * team member, including the querying car itself. Because each team has its
 * own box, the channel functions as a "team box busy" signal rather than a
 * strict teammate-other-than-self flag.
 */
export type PitStrategyState = {
  /** Normalized distance from the car to its team's pit entrance. */
  pitDistanceToEntrance01?: number;
  /** Normalized team pit-box occupancy (`0` empty, `1` occupied). */
  pitOccupancyStatus?: number;
  /** Normalized laps elapsed since the car's last pit stop. */
  lapsSincePit?: number;
  /**
   * Normalized flag indicating the team pit box is occupied by any team member,
   * including the querying car itself when it is pitting (`0` free, `1` occupied).
   */
  teammatePitStatus?: number;
  /** Normalized tire-degradation rate (`0` fresh, `1` fully degraded). */
  tireDegradationRate?: number;
  /** Normalized estimate of remaining laps before tire failure. */
  estimatedLapsBeforeFailure?: number;
  /** Reserved expansion channel for future pit context. */
  reservedPitContext1?: number;
  /** Reserved expansion channel for future pit context. */
  reservedPitContext2?: number;
};

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
  /** Signed forward speed in world units per second (negative when reversing). */
  forwardSpeedWorld?: number;
  /** Signed lateral speed in world units per second; `0` until a lateral-velocity model is added. */
  lateralSpeedWorld?: number;
  /** Unsigned world speed in world units per second; always `Math.abs(forwardSpeedWorld)`. */
  speedWorld?: number;
  /**
   * Per-step reward/penalty produced by the local physics step.
   *
   * This is intentionally transient: it is written by `stepEnvironment` when a
   * car leaves the track or drives the wrong direction, and is read by callers
   * that need a per-car training signal for the current tick.
   */
  reward?: number;
};

/** Concrete racing-car alias kept for the browser and worker seams. */
export type RacingCarState = CarState;

/**
 * Runtime state for one simulation episode.
 *
 * The legacy top-level `carX`, `carY`, and `carHeading` fields stay in place so
 * the solo browser path remains stable. Multi-car packs layer the ordered `cars`
 * roster on top; the six-car 3v3 slice simply sets `cars.length = 6` while
 * reusing the same six-slot `pitOccupancy` and compatibility `pitStatus` shelf.
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
  /** Fixed six-slot pit shelf; `255` marks an empty slot and stop ticks count down to tire reset. */
  pitOccupancy?: PitOccupancyState;
  /** Backward-compatible alias exposing the same six-record pit shelf to UI/tests. */
  pitStatus?: PitOccupancyState;
  /** Per-car consecutive border-contact tick counts for escalating penalties. */
  consecutiveBorderContactTicks?: readonly number[];
  /** Per-car consecutive wrong-direction tick counts for escalating penalties. */
  consecutiveWrongDirectionTicks?: readonly number[];
  /** Guidance overlay alpha in [0, 1]; 0 means the guide line is unavailable. */
  guidanceAlpha?: number;
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
