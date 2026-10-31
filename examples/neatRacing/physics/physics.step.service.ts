/**
 * Deterministic fixed-step physics step service for the neatRacing example
 * (plan slice P1-S7-B; AC-023; spec Â§5.1â5.3, LD-15, TP-01/TP-02/TP-04).
 *
 * The simulation truth is a top-down 2D world: the car state carries world
 * `x`/`y` (meters; 1 world unit = 1 meter), `heading` (radians), a
 * longitudinal `speed` (meters/second; TP-01: ~89 u/s â‰ˆ 320 km/h top-speed
 * feel), and the LD-15 drift channel â `yawRate` (heading change in radians
 * PER 60 Hz tick, `rad/s = yawRate * 60`) and `slipAngle` (bounded
 * `atan2`-style angle between the velocity direction and the heading) â
 * exactly the fields Phase-2 sensors/observations consume unchanged.
 *
 * The DRIFT CHANNEL is nested on the car state (AC-087):
 * `drift.committed` (the latched physical slide) and `drift.pose` (the
 * visible yaw accumulator) - the drift model in `./drift.model` owns every
 * drift law and this service only sequences the tick.
 *
 * The TRACTION SEVERITY (`tractionSeverity`, AC-088) is the ONE shared
 * scalar published on the car state: it is derived from the POST-TRACTION
 * speed and the PUBLISHED slip, so the camera follow, the car pass, the
 * HUD and the debug seam all read the SAME number and no consumer
 * re-derives it.
 *
 * The DEMANDED TURN RATE (Step 14S, AC-091) is resolved ONCE per tick and
 * is the ONE turn quantity the tick produces: the per-tick heading delta
 * is DERIVED from it (the rate times the fixed `dt`) and the SAME number
 * is fed to the drift commitment, so the yaw and the momentum law can
 * never disagree about the turn the driver demanded.
 *
 * THE TURN-RATE LAW (Step 14T, AC-092/AC-093) resolves that one number on
 * TWO separate channels that never collide: the USER channel (the steer
 * axis times the steering rate times the speed percentile, softened by a
 * MILD speed-proportional reduction that never reaches zero) and the AI
 * channel (the OPTIONAL turn-rate percent, which is AUTHORITATIVE - it is
 * resolved BEFORE the user softening and never sees it). A PRESENT
 * percent wins; an ABSENT percent keeps the steer-axis law.
 *
 * The tuning baseline is the locked javascript-racer adaptation table
 * (spec Â§5.2), transcribed as named constants with their js-racer
 * derivations preserved: `ACCEL = maxSpeed / 5`,
 * `BRAKING = -maxSpeed`, `NATURAL_DECEL = -maxSpeed / 5`,
 * `OFF_ROAD_DECEL = -maxSpeed / 2` with the passive off-road floor at
 * `maxSpeed / 4` (so a coasting off-road car never decelerates to zero),
 * the `STEERING_RATE = 2` steering rate scaled by the speed percentile
 * (TP-04: `dx = dt Â· 2 Â· speedPercent`, a standing car does not rotate).
 *
 * Determinism (spec Â§5.3): the service is PURE â it calls no `Math.random`,
 * reads the surface at tick start through the resolved resolver, and
 * integrates in one fixed documented order, so the same
 * `(seed, control stream, surface stream)` produces byte-identical
 * JSON-serialized states on the same runtime (reproducibility rung
 * Level 2, ordered deterministic; cross-environment byte equality is NOT
 * claimed).
 *
 * Scope notes: off-track/ribbon detection is wired by the surface-resolver
 * consumer (P1-S7-C); per-wheel slip channels and the
 * worker-level reproducibility tuple are Phase-2 concerns and deliberately
 * out of scope here. `curb` shares the road longitudinal model in this
 * slice; curb-specific rumble shaping arrives with the collision/
 * off-track surface utils (P1-S7-C).
 *
 * The CAR-TO-CAR COLLISION is wired HERE (AC-090): the named seam
 * `resolveLiveCarCollisions` is the ONE owner of the resolver call
 * (`resolveCarCollisions` in `./collision.utils`), and the pack tick
 * `stepPhysicsFleet` calls it AFTER the drift integration and BEFORE the
 * states are published. The single-car tick (`stepPhysics`) stays the hot
 * path of the live session and never resolves a fleet.
 *
 * The ROAD CONTAINMENT is wired HERE too (AC-081-WIRE): the named seam
 * `applyTrackContainment` is the ONE owner of the drivable-surface bound
 * (`resolveTrackBoundaryCar` in `./collision.utils`), and the single-car
 * tick applies it AFTER the translation and BEFORE the publish, through
 * the OPTIONAL fourth parameter `trackSegments`. Without that geometry
 * (an absent parameter or an empty segment list) the tick is a byte-
 * identical no-op, so every existing caller and every pinned tick stays
 * untouched.
 */

import { resolveCarCollisions, resolveTrackBoundaryCar } from './collision.utils';
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';
import {
  computeLongitudinalSpeedScrub,
  computeNextLateralVelocity,
  computeSlipAngle,
  computeTractionSeverity,
  createZeroDriftChannelState,
  stepDriftCommitment,
  stepDriftPoseSpring,
} from './drift.model';
import type { DriftChannelState } from './drift.model';

/** Surface label for one tick, matching the track slab palette (`track.slabs`). */
export type PhysicsSurfaceType = 'road' | 'curb' | 'grass';

/**
 * Three-axis keyboard control sample applied for one fixed tick (D5: the
 * input is sampled once per frame by the host and replayed per tick, so
 * the same `PhysicsStepInput` object is passed to every catch-up step).
 *
 * Throttle and brake are clamped into `[0, 1]`; brake takes precedence
 * over throttle. Steer is clamped into `[-1, 1]` (positive steers
 * counterclockwise). Axes are analog-capable so later input transports
 * can pass analog values without a contract change.
 *
 * The OPTIONAL turn-rate percent (Step 14T, AC-093) is the AI channel: it
 * rides ALONGSIDE the steer axis (it never replaces it) and it is NOT
 * clamped, because the AI owns the full rate range (a percent may exceed
 * 100 and it may be negative - the direction travels with the percent).
 */
export interface PhysicsStepInput {
  /** Forward throttle axis in `[0, 1]`. */
  throttle: number;
  /** Brake axis in `[0, 1]`; a positive brake wins over throttle that tick. */
  brake: number;
  /** Steering axis in `[-1, 1]`; positive steers counterclockwise (left). */
  steer: number;
  /**
   * The OPTIONAL turn-rate percent (Step 14T, AC-093): the AI channel.
   *
   * A PRESENT percent is AUTHORITATIVE: the tick resolves the demanded
   * turn rate from it (a percent of the nominal turn rate, signed - the
   * direction travels with the percent and never with the steer axis) and
   * NEVER applies the user speed-proportional softening to it. An ABSENT
   * percent keeps the steer-axis law (the USER channel, the
   * backward-compatible default).
   */
  turnRatePercent?: number;
}

/**
 * Full deterministic car state for one 60 Hz fixed step. JSON-serializable
 * on purpose: the batch-83 replay contract compares byte-identical
 * `JSON.stringify` snapshots of this shape across repeated runs.
 */
export interface PhysicsCarState {
  /** World x position in meters. */
  x: number;
  /** World y position in meters. */
  y: number;
  /** Heading angle in radians, wrapped to `[-Ï€, Ï€)`. */
  heading: number;
  /** Longitudinal (forward) speed in meters/second along the heading. */
  speed: number;
  /**
   * Yaw response in radians PER 60 Hz tick (house precedent: radians-per-
   * step units, cf. `racing_curriculum`'s `MAX_TURN_RADIANS_PER_STEP`);
   * convert to `rad/s` with `yawRate * 60`. Nonzero only while the steer
   * axis is engaged and the car has speed percent.
   */
  yawRate: number;
  /**
   * Drift (slip) angle in radians between the velocity direction and the
   * heading, bounded to `(-Ï€/2, Ï€/2)` by the `atan2` form. Zero while the
   * car is not moving; grows only under lateral push, decays via grip.
   */
  slipAngle: number;
  /**
   * The TRAVEL direction (rad, wrapped to `[-Ï€, Ï€)`): the heading the car
   * is actually MOVING along, i.e. the NOSE heading plus the PUBLISHED slip
   * angle.
   *
   * Step 14P (the G-M1 break line): the service always integrated the
   * position along `heading + slipAngle` and then DISCARDED that velocity
   * heading, so no consumer could know which way the car was travelling and
   * the camera had to re-derive a slide it could not see. The published
   * field is the EXACT heading the integration used (it is wrapped ONCE and
   * the x/y advance reads that very number), so the camera, the body and the
   * physics can never disagree about the travel direction.
   */
  travelHeadingRad: number;
  /**
   * The ONE shared TRACTION SEVERITY (AC-088): how much traction this car
   * is spending, in the closed unit interval `[0, 1]`.
   *
   * The severity is the shared "how much traction is left" measure: the
   * DEEPER the slide AND the FASTER the car, the higher the severity, and
   * it is EXACTLY 0 on a straight line (the published slip is exactly 0
   * there). It is derived from the POST-TRACTION speed (the very number
   * the service publishes as `speed`, after the drift scrub and the
   * off-road floor) and from the PUBLISHED slip (the very number the
   * service publishes as `slipAngle`), so the tires, the HUD, the camera
   * and the debug seam can never disagree about how much traction is left
   * - and NO consumer re-derives it.
   *
   * The law itself lives in `./drift.model` (`computeTractionSeverity`),
   * so the published value and the law value are the SAME number.
   */
  tractionSeverity: number;
  /**
   * Lateral velocity in meters/second, positive = left of the heading.
   * Driven by the TP-02 outward push, drained by the grip rate; shapes
   * the velocity direction (and therefore `slipAngle`) without moving the
   * car sideways on its own while `speed` is zero.
   */
  lateralVelocity: number;
  /**
   * The NESTED drift channel (AC-087): the committed physical slide and
   * the visual pose accumulator in ONE carrier. The channel is the single
   * home of the visible drift yaw (the retired flat `driftYawOffset` field
   * is gone), so a bounce can zero the WHOLE channel atomically and the
   * render bridge reads `drift.pose.pose`.
   */
  drift: DriftChannelState;
  /** Completed fixed-tick count since spawn (0 = the spawn state itself). */
  tick: number;
}

/**
 * Fixed simulation timestep in milliseconds (spec Â§5.1: deterministic
 * 60 Hz fixed step inside the simulation worker; host pacing uses the
 * accumulator loop, not this service).
 */
export const NEATRACING_PHYSICS_FIXED_TIMESTEP_MS = 1000 / 60;

/**
 * Fixed simulation timestep in seconds, the `dt` of every integration
 * step in this service (js-racer parity: all rates are per-second values
 * scaled by `dt`).
 */
export const NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS = 1 / 60;

/**
 * Top speed in meters/second â the locked tuning anchor (TP-01): ~320
 * km/h arcade feel with 1 world unit = 1 meter. Every derived tuning
 * constant below is expressed from this anchor, exactly as javascript-
 * racer derives `ACCEL`/`BRAKING`/`NATURAL_DECEL`/`OFF_ROAD_DECEL` from
 * its `maxSpeed`.
 */
export const NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S = 89;

/**
 * Forward acceleration rate: `maxSpeed / 5` (spec Â§5.2 locked table,
 * javascript-racer `ACCEL`), i.e. 0-to-top in ~5 s of full throttle.
 */
export const NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2 =
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 5;

/**
 * Braking deceleration rate: `-maxSpeed` (spec Â§5.2 locked table,
 * javascript-racer `BRAKING`), i.e. a full brake pin stops the car in ~1 s.
 */
export const NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2 =
  -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;

/**
 * Natural deceleration (coast drag): `-maxSpeed / 5` (spec Â§5.2 locked
 * table, javascript-racer `DECELERATION`), applied on ticks with neither
 * throttle nor brake.
 */
export const NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2 =
  -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 5;

/**
 * Off-road deceleration drag: `-maxSpeed / 2` (spec Â§5.2 locked table,
 * javascript-racer `OFF_ROAD_DECELERATION`), stacked on top of the passive
 * rate on off-road ticks. `javascript-racer`'s value is transcribed with
 * the same `maxSpeed` derivation (G10 forbids only the raw segment-units
 * `5`, which this constant does not transcribe).
 */
export const NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2 =
  -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 2;

/**
 * Off-road speed floor: `maxSpeed / 4` (spec Â§5.2 locked table). While the
 * car enters an off-road tick at/above this speed and is NOT braking, the
 * passive decelerations are floored here so an off-road car coasts down to
 * this cruise floor instead of decelerating to zero.
 */
export const NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S =
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 4;

/**
 * Steering rate: `dx = dt Â· 2 Â· speedPercent` (spec Â§5.2 locked table,
 * TP-04 "cross the road in ~1 s at top speed"). Top-down adaptation: the
 * heading change per tick is `input.steer Â· STEERING_RATE Â·
 * speedPercent Â· dt` (radians per tick), so steering authority grows with
 * speed and a standing car does not rotate.
 */
export const NEATRACING_PHYSICS_STEERING_RATE = 2;

/**
 * The USER turn-rate reduction at the TOP speed (Step 14T, AC-092).
 *
 * MEASURED (the RED decision, Step 14T): the mild reduction takes a
 * FIFTH of the nominal turn rate away at the top speed, so the top-speed
 * USER turn rate reads `0.8 * 2 = 1.6` rad/s - above the RED-measured
 * mild floor (`0.75 * 2 = 1.5` rad/s) and far away from zero, so
 * turning AND drifting stay possible at every speed.
 */
export const NEATRACING_PHYSICS_TURN_RATE_SPEED_REDUCTION_FRACTION = 0.2;

/**
 * The turn-rate speed-factor FLOOR (Step 14T, AC-092).
 *
 * The floor is the RED-measured MILD bound (a quarter of the nominal rate
 * may never be taken away) and it is the hard guarantee that the speed
 * factor NEVER reaches zero: the user turn rate stays a REAL rate at
 * every speed, so the reduction can never silently switch the steering
 * off. The floor binds the FEEL and never the physics (at the floor the
 * top-speed full-steer lane still commits a 0.443 rad slide, 98.5 % of
 * the 0.45 rad depth budget).
 */
export const NEATRACING_PHYSICS_TURN_RATE_SPEED_FACTOR_FLOOR = 0.75;

/**
 * The low-speed handling band (m/s) BELOW which the USER turn rate is
 * NOT reduced (Step 14T, AC-092).
 *
 * MEASURED (the RED decision, Step 14T): the slowest REAL drifting lane
 * of the build is the 10 m/s crawl lane (the momentum crossover sits at
 * about 9.43 m/s), so the reduction starts ABOVE it: the low-speed
 * handling keeps the nominal law BIT-IDENTICALLY and the reduction bites
 * on the high-speed lanes, where the handling actually needs it. The band
 * is the crawl band of the RED-measured window, so the crawl turn rate
 * stays at the existing nominal law.
 */
export const NEATRACING_PHYSICS_TURN_RATE_REDUCTION_BAND_UNITS_PER_S = 10;

/**
 * Lateral spawn spread in meters: the seeded spawn lane offset is drawn
 * deterministically from the spawn seed within `Â±spread / 2` so the same
 * seed always spawns on the same lane line.
 */
export const NEATRACING_PHYSICS_SPAWN_LATERAL_SPREAD_M = 2;

/** Heading at spawn in radians: straight along +x (top-down race start). */
export const NEATRACING_PHYSICS_SPAWN_HEADING_RADIANS = 0;

/** Spawn seed used when a caller does not provide one (deterministic). */
export const NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED = 0;

/** Surface treated as the baseline when no resolver is supplied. */
const DEFAULT_SURFACE: PhysicsSurfaceType = 'road';

/** Off-road surface label that arms the off-road drag and passive floor. */
const OFF_ROAD_SURFACE: PhysicsSurfaceType = 'grass';

/** Standing-start longitudinal speed (m/s). */
const INITIAL_SPEED_UNITS_PER_S = 0;

/** Spawn lateral offset centerline (m) â the offset is centered here. */
const SPAWN_Y_M = 0;

/** Control axis ratio bounds shared by throttle and brake axes. */
const CONTROL_RATIO_MIN = 0;
const CONTROL_RATIO_MAX = 1;
/** Negative steering bound for the steer axis. */
const STEER_RATIO_MIN = -1;

/** Full-turn wrap period for heading normalization. */
const HEADING_WRAP_TAU_RADIANS = Math.PI * 2;

/**
 * Multiplier of the integer seed-mixing recipe (`imul`-based avalanche;
 * deterministic, no `Math.random`).
 */
const SEED_MIX_MULTIPLIER = 0x45d9f3b;

/** Unsigned-32-bit denominator used to normalize the mixed seed to `[0, 1]`. */
const MIXED_SEED_UINT32_MAX = 0xffffffff;

/**
 * Creates the deterministic spawn car state for a given seed (spec Â§5.3
 * determinism tuple: the seed selects the spawn lane via a pure integer
 * mix â no `Math.random` anywhere, so the same seed always yields the
 * byte-identical spawn state). The car starts standing (zero speed, zero
 * yaw, zero slip) heading straight along +x at the world origin line.
 *
 * @param seed - Replay seed that fixes the spawn lane offset; defaults to
 *   `NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED` when omitted. Non-integer or
 *   out-of-range values are truncated deterministically (`| 0` treats
 *   `NaN` as `0`) â never a silent nondeterminism source.
 * @returns Fresh `PhysicsCarState` at `tick = 0`.
 * @remarks The lateral spread keeps the spawn inside ~Â±1 m of the center
 *   line; a future track-side spawn wrapper (P1-S7-D) may re-base this
 *   pose on the generated track while still relying on the same seed rule.
 *
 * @example
 * ```ts
 * const car = createInitialPhysicsCarState(42);
 * console.log(car.speed === 0); // standing start
 * ```
 */
/**
 * Optional spawn-pose override for callers that already know the track-side
 * position and heading. When supplied, the pose fields replace the defaults;
 * the seed is ignored because the explicit position already fixes the lane.
 */
export interface PhysicsCarSpawnPose {
  /** World x position in meters. */
  readonly x: number;
  /** World y position in meters. */
  readonly y: number;
  /** Heading angle in radians. */
  readonly heading: number;
}

/**
 * Creates the deterministic spawn car state for a given seed (spec Â§5.3
 * determinism tuple: the seed selects the spawn lane via a pure integer
 * mix Ã¢ no `Math.random` anywhere, so the same seed always yields the
 * byte-identical spawn state). The car starts standing (zero speed, zero
 * yaw, zero slip, a zeroed drift channel) heading straight along +x at the
 * world origin line.
 *
 * @param seed - Replay seed that fixes the spawn lane offset; defaults to
 *   `NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED` when omitted. Non-integer or
 *   out-of-range values are truncated deterministically (`| 0` treats
 *   `NaN` as `0`) Ã¢ never a silent nondeterminism source.
 * @param pose - Optional explicit spawn pose; when supplied, the seed is
 *   ignored because the explicit position already fixes the lane.
 * @returns Fresh `PhysicsCarState` at `tick = 0`.
 * @remarks The lateral spread keeps the spawn inside ~ÃÂ±1 m of the center
 *   line; a future track-side spawn wrapper (P1-S7-D) may re-base this
 *   pose on the generated track while still relying on the same seed rule.
 *
 * @example
 * ```ts
 * const car = createInitialPhysicsCarState(42);
 * console.log(car.speed === 0); // standing start
 * ```
 */
export function createInitialPhysicsCarState(
  seed?: number,
  pose?: PhysicsCarSpawnPose,
): PhysicsCarState {
  const resolvedSeed = seed ?? NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED;
  const spawnHeadingRad =
    pose?.heading ?? NEATRACING_PHYSICS_SPAWN_HEADING_RADIANS;
  return {
    x: pose?.x ?? resolveSpawnLateralOffset(resolvedSeed),
    y: pose?.y ?? SPAWN_Y_M,
    heading: spawnHeadingRad,
    speed: INITIAL_SPEED_UNITS_PER_S,
    yawRate: 0,
    slipAngle: 0,
    // The spawn state carries NO slide, so the travel direction IS the nose
    // heading - the exact heading the first integration would advance along.
    travelHeadingRad: spawnHeadingRad,
    // The spawn state carries NO slide and NO speed, so the traction
    // severity is EXACTLY 0 - the very value the law returns for a zero
    // slip.
    tractionSeverity: 0,
    lateralVelocity: 0,
    drift: createZeroDriftChannelState(),
    tick: 0,
  };
}

/**
 * Advances one car state by exactly one deterministic 60 Hz fixed tick
 * under the given control sample and surface.
 *
 * The tick pipeline is fixed and side-effect free (spec Â§5.1âÂ§5.3):
 *
 * 1. resolve the surface at tick start through the supplied resolver
 *    (omitted means road; the resolver sees the tick-start state, which
 *    is the integration read for an explicit integrator);
 * 2. resolve the DEMANDED turn rate ONCE from the steer axis and the
 *    tick-start speed percentile (TP-04, standing cars do not rotate),
 *    DERIVE the per-tick yaw delta from that one number and integrate
 *    the heading - the SAME number is handed to the drift commitment in
 *    step 4a, so the yaw and the momentum law read ONE value;
 * 3. integrate the longitudinal speed through the locked Â§5.2 rates â
 *    braking wins over throttle, passive coast drag otherwise, off-road
 *    drag stacked on grass with the passive off-road floor (deliberate
 *    braking is never floored and can stop the car), clamped to
 *    `[0, maxSpeed]`;
 *
 *    the SAME off-road floor is applied again to the POST-SCRUB speed
 *    in step 4c (RC-1), so a drifting off-road car never decays below
 *    the floor and never traps at 0 m/s;
 * 4. advance the DRIFT CHANNEL (AC-087) in a LOAD-BEARING order: 4a the
 *    commitment (the latched physical slide, with the MIN_SPEED gate and
 *    the RE-ARM token inside the drift model, fed the DEMANDED turn rate
 *    resolved in step 2), 4b the committed chase (the
 *    lateral velocity chases the committed slide), 4c the speed scrub
 *    (AC-079: a sliding tire cannot hold the forward speed, so a
 *    full-speed turn is scrubbed down instead of being taken at the whole
 *    top speed), 4d the published slip angle (the bounded LD-15 `atan2`),
 *    and 4e the pose spring (the visible yaw accumulator, stepped INSIDE
 *    the 60 Hz tick and never in the render path; its target is the
 *    PUBLISHED slip, the ONE shared scalar);
 * 5. translate along the velocity direction (heading + slipAngle) so the
 *    drift arc bends the path, and PUBLISH that velocity heading as the
 *    state's travel direction (`travelHeadingRad`);
 * 6. derive and PUBLISH the ONE shared traction severity
 *    (`tractionSeverity`) from the POST-TRACTION speed and the PUBLISHED
 *    slip, so the tires, the HUD, the camera and the debug seam all read
 *    the SAME number and NO consumer re-derives it;
 * 7. apply the TRACK CONTAINMENT (AC-081-WIRE) when the caller supplied
 *    the ribbon geometry: the drivable-surface bound reseats a car that
 *    left the road, and the published scalars are RE-DERIVED from the
 *    POST-containment pose, so the tick never publishes pre-bounce
 *    numbers;
 * 8. count the tick.
 *
 * @param state - Current car state (treated read-only; never mutated).
 * @param input - Control sample applied for this single tick.
 * @param surfaceResolver - Optional surface probe; receives the tick-start
 *   state and must return a `PhysicsSurfaceType`. When omitted the tick
 *   behaves as on-road. Off-track detection (which road/curb/grass the
 *   ribbon assigns) is the consumer's job (P1-S7-C).
 * @param trackSegments - Optional TRACK GEOMETRY (the ribbon segments).
 *   When present, the drivable-surface containment (AC-081) is applied
 *   AFTER the translation and BEFORE the publish. An ABSENT parameter and
 *   an EMPTY list are both no-ops: the tick stays byte-identical and
 *   nothing throws (AC-081-WIRE-NOOP).
 * @returns A new `PhysicsCarState` (input argument is never mutated).
 *
 * @example
 * ```ts
 * const car = stepPhysics(createInitialPhysicsCarState(42), {
 *   throttle: 1, brake: 0, steer: 0,
 * });
 * console.log(car.speed > 0); // one tick of full throttle
 * ```
 */
export function stepPhysics(
  state: PhysicsCarState,
  input: PhysicsStepInput,
  surfaceResolver?: (carState: PhysicsCarState) => PhysicsSurfaceType,
  trackSegments?: readonly TrackSegment[],
): PhysicsCarState {
  // Step 1: Resolve the surface under the car at tick start (explicit read).
  const surface = resolveSurfaceAtTickStart(surfaceResolver, state);
  const clampedInput = clampStepInput(input);
  const speedPercent = resolveSpeedPercent(state.speed);

  // Step 2: Resolve the DEMANDED turn rate of this tick ONCE, derive the
  // yaw response from it and rotate the heading.
  //
  // THE ONE RESOLVED NUMBER (Step 14S, AC-091): the demanded turn rate
  // (rad/s) is the ONLY turn quantity this tick resolves. The per-tick
  // heading delta is DERIVED from it (the rate times the fixed `dt`, in
  // the SAME product order as before, so the published heading stays
  // BIT-IDENTICAL) and the drift commitment in step 4a consumes the very
  // same number, so the yaw and the momentum law can never disagree about
  // the turn the driver demanded.
  //
  // THE TWO CHANNELS (Step 14T, AC-092/AC-093): the AI channel (a
  // PRESENT turn-rate percent) is authoritative and is never softened by
  // the user adjustment; the USER channel (an ABSENT percent) keeps the
  // steer-axis law, softened by the mild speed-proportional factor.
  const demandedTurnRateRadPerS = resolveDemandedTurnRateRadPerS(
    clampedInput.steer,
    speedPercent,
    clampedInput.turnRatePercent,
  );
  const yawRatePerTick = resolveYawRatePerTick(demandedTurnRateRadPerS);
  const nextHeading = wrapHeading(state.heading + yawRatePerTick);

  // Step 3: Integrate the longitudinal speed (locked Â§5.2 rates).
  const nextSpeed = integrateLongitudinalSpeed(
    state.speed,
    clampedInput,
    surface,
  );

  // Step 4: the DRIFT CHANNEL (AC-087). The order is LOAD-BEARING:
  // 4a the commitment, 4b the committed chase, 4c the speed scrub,
  // 4d the slip, 4e the pose spring.
  // Step 4a: the COMMITMENT - the latched physical slide. The MIN_SPEED
  // gate and the RE-ARM token live inside the drift model; the service
  // only feeds the tick-start sample and the DEMANDED turn rate it
  // resolved in step 2 (the SAME number the yaw delta was derived from).
  const nextCommitted = stepDriftCommitment(state.drift.committed, {
    steer: clampedInput.steer,
    speed: state.speed,
    speedPercent,
    dt: NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
    lateralVelocity: state.lateralVelocity,
    slipAngle: state.slipAngle,
    demandedTurnRateRadPerS,
  });
  // Step 4b: the COMMITTED CHASE - the lateral velocity chases the
  // committed slide (the grip budget, the countersteer authority and the
  // hard zero are inside the model).
  const nextLateralVelocity = computeNextLateralVelocity(nextCommitted, {
    steer: clampedInput.steer,
    speed: state.speed,
    speedPercent,
    dt: NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
    lateralVelocity: state.lateralVelocity,
    slipAngle: state.slipAngle,
  });
  // Step 4c: the SPEED SCRUB (AC-079): the sliding tires lose the forward
  // speed they cannot hold, so the scrub is subtracted from the tick-end
  // speed. The scrub reads the tick-start slip, like every other
  // tick-start read in this integrator.
  const scrubbedSpeed = Math.max(
    0,
    nextSpeed -
      computeLongitudinalSpeedScrub(
        state.slipAngle,
        nextSpeed,
        speedPercent,
      ) * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  );
  // Step 4c-bis: the OFF-ROAD FLOOR on the POST-SCRUB speed (RC-1).
  // The floor used to be applied to the PRE-scrub speed ONLY, so the
  // drift scrub pushed an off-road car BELOW the floor; on the next tick
  // the tick-start gate no longer held the floor, and the car decayed to
  // a permanent 0 m/s trap (the "gets stuck" user symptom). The SAME
  // gate now guards the post-scrub speed, so the floor holds with OR
  // without a drift. The floor is never a clip: it only lifts a speed
  // that the SCRUB pushed below the floor, never the scrub itself.
  const nextSpeedAfterTraction = isOffRoadSpeedFloorEngaged(
    surface,
    clampedInput,
    state.speed,
  )
    ? Math.max(
        scrubbedSpeed,
        NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
      )
    : scrubbedSpeed;
  // Step 4d: the published slip angle (LD-15), from the POST-scrub speed.
  const nextSlipAngle = computeSlipAngle(
    nextLateralVelocity,
    nextSpeedAfterTraction,
  );
  // Step 4e: the POSE SPRING - the visible yaw accumulator. Its target is
  // the PUBLISHED slip (the ONE shared scalar: the same number the
  // traction law, the debug seam and the chase camera read), NOT the
  // committed slip - the committed slip is the COMMAND, and driving the
  // visible crab from it was the G-V3 defect (the body showed a slide the
  // car was not taking). The spring is stepped INSIDE the 60 Hz tick and
  // NEVER in the render path.
  const nextPose = stepDriftPoseSpring(state.drift.pose, {
    steer: clampedInput.steer,
    speedPercent,
    slipAngle: nextSlipAngle,
    dt: NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  });

  // Step 5: Translate along the velocity direction (drift arc path). The
  // velocity heading is the TRAVEL direction: the nose heading plus the
  // PUBLISHED slip. It is wrapped ONCE and the SAME number advances x and
  // y and is PUBLISHED on the state, so the travel direction the camera
  // reads can never diverge from the heading the car actually moved along.
  const velocityHeading = wrapHeading(nextHeading + nextSlipAngle);

  // Step 6: the ONE shared TRACTION SEVERITY (AC-088). It is derived from
  // the POST-TRACTION speed (the very number published as `speed`) and
  // from the PUBLISHED slip (the very number published as `slipAngle`),
  // so the tires, the HUD, the camera and the debug seam read the SAME
  // number and NO consumer re-derives it. The law itself lives in
  // `./drift.model` (`computeTractionSeverity`).
  const nextTractionSeverity = computeTractionSeverity(
    nextSlipAngle,
    nextSpeedAfterTraction,
  );

  const nextTick = state.tick + 1;

  const advancedCar: PhysicsCarState = {
    x:
      state.x +
      Math.cos(velocityHeading) *
        nextSpeedAfterTraction *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
    y:
      state.y +
      Math.sin(velocityHeading) *
        nextSpeedAfterTraction *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
    heading: nextHeading,
    speed: nextSpeedAfterTraction,
    yawRate: yawRatePerTick,
    slipAngle: nextSlipAngle,
    travelHeadingRad: velocityHeading,
    tractionSeverity: nextTractionSeverity,
    lateralVelocity: nextLateralVelocity,
    drift: { committed: nextCommitted, pose: nextPose },
    tick: nextTick,
  };

  // Step 7: the TRACK CONTAINMENT (AC-081-WIRE). The containment runs
  // AFTER the translation and BEFORE the publish, so the published pose is
  // the CONTAINED one and the rendered frame can never show a car that
  // the simulation already bounced back onto the road.
  return applyTrackContainment(advancedCar, trackSegments);
}

/**
 * The LIVE car-to-car collision seam (AC-090): the ONE named owner of the
 * car-to-car resolution inside the live physics tick.
 *
 * The seam is the ONLY live call site of the resolver (`resolveCarCollisions`
 * in `./collision.utils`), so the resolution has a single owner and a single
 * call site: the pack tick calls it AFTER every member finished the drift
 * integration (so the whole fleet sits at ONE time base) and BEFORE the
 * states are published (so the published states and the rendered frame can
 * never disagree).
 *
 * The resolver separates an overlapping pair and exchanges the world
 * velocities along the collision normal (LD-05, the arcade elastic
 * exchange). The seam adds the PUBLISH rule the resolver cannot know about:
 * a separated member is republished with its published scalars re-derived
 * from its BOUNCED pose, because `travelHeadingRad` and `tractionSeverity`
 * are documented as the exact values the integration used - a bounced car
 * must never publish the pre-bounce numbers.
 *
 * @param fleet - The advanced fleet states (treated read-only; never
 *   mutated).
 * @returns Fresh resolved states in the same order; a member that did NOT
 *   overlap keeps every published value.
 */
export function resolveLiveCarCollisions(
  fleet: readonly PhysicsCarState[],
): PhysicsCarState[] {
  const resolved = resolveCarCollisions(fleet);
  for (let index = 0; index < resolved.length; index += 1) {
    const member = resolved[index];
    const source = fleet[index];
    // A member that did NOT overlap comes back with the SAME position, so
    // its published values are the untouched tick output and are kept
    // byte-identical.
    if (member.x === source.x && member.y === source.y) {
      continue;
    }
    resolved[index] = republishSeparatedCar(member);
  }
  return resolved;
}

/**
 * Republishes a resolved car with the published scalars re-derived from
 * its POST-RESOLUTION pose.
 *
 * The car-to-car collision (AC-090) and the road containment (AC-081-WIRE)
 * both recompose the car around a new world velocity (a new heading, a new
 * speed, a zeroed slip and a zeroed drift channel), so the published
 * travel heading and the published traction severity of the
 * pre-resolution tick are STALE. They are re-derived here - from the
 * resolved values ONLY, never from the pre-resolution ones - so the
 * published state and the rendered frame stay one frame of reference.
 *
 * @param car - The resolved car state (treated read-only).
 * @returns A fresh state whose published scalars match the resolved pose.
 */
function republishSeparatedCar(car: PhysicsCarState): PhysicsCarState {
  return {
    ...car,
    travelHeadingRad: wrapHeading(car.heading + car.slipAngle),
    tractionSeverity: computeTractionSeverity(car.slipAngle, car.speed),
  };
}

/**
 * The TRACK-CONTAINMENT seam (AC-081-WIRE): the ONE named owner of the
 * road-containment bound inside the live physics tick.
 *
 * The bound (`resolveTrackBoundaryCar` in `./collision.utils`) is the
 * LANDED AC-081 drivable-surface bound: a car that left the drivable
 * surface is reseated on its edge, and only an APPROACHING car takes the
 * elastic rebound. The seam is the ONLY live call site of the bound, so
 * the bound has a single owner and a single call site - the Step 14I
 * contract "there is ONE bound in the codebase" stays true.
 *
 * The NO-OP contract (AC-081-WIRE-NOOP): an ABSENT geometry and an EMPTY
 * segment list both leave the tick BYTE-IDENTICAL - the bound is never
 * applied and NOTHING throws (`resolveNearestChordPoint` throws on an
 * empty list, so the empty case is guarded BEFORE the bound is called).
 *
 * The PUBLISH rule (AC-081-WIRE-SCALARS): the bound recomposes the
 * velocity channels of a bounced car, so the published travel heading and
 * the published traction severity of the pre-containment tick are STALE.
 * They are re-derived from the POST-containment pose - exactly as the
 * AC-090 seam does - so the tick never publishes pre-bounce numbers.
 *
 * @param car - The advanced car state (treated read-only; never
 *   mutated).
 * @param trackSegments - The ribbon geometry; an absent or an empty list
 *   is a no-op.
 * @returns A fresh state inside the drivable-surface bound.
 */
function applyTrackContainment(
  car: PhysicsCarState,
  trackSegments: readonly TrackSegment[] | undefined,
): PhysicsCarState {
  // The EMPTY guard comes FIRST: the bound throws on an empty segment
  // list, so the no-op must never reach it.
  if (trackSegments === undefined || trackSegments.length === 0) {
    return car;
  }
  const contained = resolveTrackBoundaryCar(car, trackSegments);
  // A car INSIDE the bound keeps every published value: the bound hands it
  // back unchanged, so there is nothing stale to re-derive.
  if (contained.x === car.x && contained.y === car.y) {
    return contained;
  }
  return republishSeparatedCar(contained);
}

/**
 * Advances a whole fleet by exactly ONE deterministic 60 Hz fixed tick and
 * resolves the car-to-car overlaps (AC-090).
 *
 * The ORDER is LOAD-BEARING: every member runs the SAME tick pipeline
 * FIRST (so the whole fleet sits at ONE time base) and the car-to-car
 * overlaps are resolved AFTER the drift integration and BEFORE the states
 * are published. The single-car tick (`stepPhysics`) stays the hot path of
 * the live session and never resolves a fleet; this pack tick is the entry
 * point a multi-car session drives, and it resolves the fleet EXACTLY ONCE
 * per tick (a per-car resolution would resolve the whole fleet N times per
 * tick).
 *
 * @param fleet - The fleet states at TICK START (treated read-only; never
 *   mutated), index-aligned with `inputs`.
 * @param inputs - The control sample of every member, index-aligned with
 *   `fleet` (a missing sample is a caller error).
 * @param surfaceResolver - Optional surface probe, shared by the fleet; it
 *   receives the tick-start state of the probed member.
 * @returns Fresh published states in the same order.
 *
 * @example
 * ```ts
 * const fleet = stepPhysicsFleet([carA, carB], [inputA, inputB]);
 * console.log(fleet.length); // 2
 * ```
 */
export function stepPhysicsFleet(
  fleet: readonly PhysicsCarState[],
  inputs: readonly PhysicsStepInput[],
  surfaceResolver?: (carState: PhysicsCarState) => PhysicsSurfaceType,
): PhysicsCarState[] {
  const advanced = fleet.map((car, index) =>
    stepPhysics(car, inputs[index], surfaceResolver),
  );
  return resolveLiveCarCollisions(advanced);
}

/** Resolves the tick-start surface, defaulting to road without a resolver. */
function resolveSurfaceAtTickStart(
  surfaceResolver:
    ((carState: PhysicsCarState) => PhysicsSurfaceType) | undefined,
  state: PhysicsCarState,
): PhysicsSurfaceType {
  return surfaceResolver?.(state) ?? DEFAULT_SURFACE;
}

/** Clamps the three control axes into their contract ranges. */
function clampStepInput(input: PhysicsStepInput): PhysicsStepInput {
  return {
    throttle: clampControlRatio(input.throttle),
    brake: clampControlRatio(input.brake),
    steer: clampSteeringRatio(input.steer),
    // The turn-rate percent is NOT clamped: the AI channel is signed and
    // may exceed 100 (the AI keeps the full rate range), so it rides
    // through UNCHANGED and stays authoritative.
    turnRatePercent: input.turnRatePercent,
  };
}

/** Normalizes the longitudinal speed into the `[0, 1]` speed percentile. */
function resolveSpeedPercent(speed: number): number {
  return clampControlRatio(speed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);
}

/**
 * Resolves the USER turn-rate speed factor of a speed (a unitless factor).
 *
 * THE NAMED SPEED-PROPORTIONAL FACTOR (Step 14T, AC-092): the factor is
 * `1` at a standstill and in the whole low-speed handling band (the
 * reduction starts ABOVE the band), and it falls LINEARLY with the speed
 * excess above the band until it reaches the measured mild value at the
 * top speed (`1 - 0.2 = 0.8` of the nominal rate). It is MONOTONIC in
 * the speed (the factor never grows with the speed) and it is floored
 * (the factor NEVER reaches zero, so the user turn rate stays a real rate
 * at every speed - the reduction can never switch the steering off).
 *
 * THE BASIS (the RED-measured window, Step 14T): the factor is resolved
 * from the SAME speed percent the existing steer-axis law uses (the
 * normalized speed in `[0, 1]`), so the two laws can never disagree about
 * the speed of the tick. The MEASURED ladder reads `1` at the 10 m/s
 * crawl lane, `0.9620253164556962` at 25 m/s, `0.9113924050632911` at
 * 45 m/s, `0.8556962025316456` at 67 m/s and `0.8` at the 89 m/s top
 * speed (the top-speed user turn rate is `1.6` rad/s, above the
 * RED-measured mild floor of `1.5` rad/s).
 *
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @returns The speed factor in `(0, 1]`; `1` below the reduction band.
 */
export function resolveTurnRateSpeedFactor(speedPercent: number): number {
  const bandPercent =
    NEATRACING_PHYSICS_TURN_RATE_REDUCTION_BAND_UNITS_PER_S /
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  const excessPercent =
    clampControlRatio(speedPercent - bandPercent) / (1 - bandPercent);
  const factor =
    1 -
    NEATRACING_PHYSICS_TURN_RATE_SPEED_REDUCTION_FRACTION * excessPercent;
  return Math.max(NEATRACING_PHYSICS_TURN_RATE_SPEED_FACTOR_FLOOR, factor);
}

/**
 * Resolves the DEMANDED turn rate of a tick (radians per SECOND).
 *
 * THE TWO CHANNELS (Step 14T, AC-092/AC-093): the rate is resolved on
 * TWO separate laws that never collide.
 *
 * THE AI CHANNEL (a PRESENT turn-rate percent, AC-093): the rate is
 * resolved from the percent of the NOMINAL turn rate - INDEPENDENT of
 * the steer axis and NEVER softened by the user adjustment. The percent
 * is SIGNED, so the direction travels with the percent and never with the
 * steer axis. A percent of `100` commands the nominal rate, a percent of
 * `0` commands no turn at all and a percent above `100` commands a rate
 * above the nominal one (the AI keeps the full range).
 *
 * THE USER CHANNEL (an ABSENT turn-rate percent, AC-092): the steer-axis
 * law (the steer axis times the steering rate times the speed percentile,
 * the TP-04 form) softened by the MILD speed-proportional factor. The
 * factor is applied LAST, in the SAME product order as the pre-14T law,
 * so every lane below the reduction band stays BIT-IDENTICAL.
 *
 * THE RATE is the one turn quantity a tick resolves; the per-tick heading
 * delta is DERIVED from it and the drift commitment CONSUMES it, so the
 * yaw and the momentum law read the SAME number (Step 14S, AC-091).
 *
 * @param steer - The clamped steering axis in `[-1, 1]`.
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @param turnRatePercent - The optional AI turn-rate percent, in percent.
 * @returns The demanded turn rate (rad/s).
 */
function resolveDemandedTurnRateRadPerS(
  steer: number,
  speedPercent: number,
  turnRatePercent?: number,
): number {
  // THE AI CHANNEL WINS: a present, finite percent is authoritative, so
  // the user speed reduction never touches it. A non-finite percent is
  // treated as ABSENT (the backward-compatible default), so a malformed
  // AI sample can never poison the heading with NaN.
  if (turnRatePercent !== undefined && Number.isFinite(turnRatePercent)) {
    return (turnRatePercent / 100) * NEATRACING_PHYSICS_STEERING_RATE;
  }
  // THE USER CHANNEL: the steer-axis law, softened by the mild
  // speed-proportional factor (the factor is applied LAST, so the
  // product order of the pre-14T law is preserved).
  return (
    steer *
    NEATRACING_PHYSICS_STEERING_RATE *
    speedPercent *
    resolveTurnRateSpeedFactor(speedPercent)
  );
}

/**
 * Resolves the per-tick heading delta (radians per tick) FROM the resolved
 * demanded turn rate: the rate times the fixed `dt`.
 *
 * THE DERIVATION (Step 14S, AC-091): the per-tick delta is NEVER
 * re-derived from the steer axis - it is the single resolved demanded turn
 * rate scaled by the fixed timestep, in the SAME product order as the
 * pre-14S law, so the published heading values stay BIT-IDENTICAL while
 * the drift commitment reads the very same number.
 *
 * @param demandedTurnRateRadPerS - The demanded turn rate of this tick (rad/s).
 * @returns The per-tick heading delta (rad per tick).
 */
function resolveYawRatePerTick(demandedTurnRateRadPerS: number): number {
  return demandedTurnRateRadPerS * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;
}

/**
 * Resolves whether the OFF-ROAD SPEED FLOOR holds for the current tick.
 *
 * THE GATE (js-racer limit-gate parity): the floor holds only while the
 * car ENTERS the tick at/above the floor and is NOT braking - a
 * deliberate brake is never floored and may stop the car.
 *
 * ONE GATE, TWO READS (RC-1): the SAME gate guards the pre-scrub floor
 * (inside the longitudinal integrator) and the post-scrub floor (the
 * drift-scrub read in the tick pipeline), so the two can never disagree
 * about whether the floor applies to the tick. Before RC-1 the pre-scrub
 * floor was the only read, so the drift scrub pushed an off-road car
 * below the floor and the tick-start gate stopped holding it.
 *
 * @param surface - The tick-start surface under the car.
 * @param input - The clamped control sample applied to this tick.
 * @param tickStartSpeed - The tick-start longitudinal speed in m/s.
 * @returns True when the off-road speed floor applies to this tick.
 */
function isOffRoadSpeedFloorEngaged(
  surface: PhysicsSurfaceType,
  input: PhysicsStepInput,
  tickStartSpeed: number,
): boolean {
  return (
    surface === OFF_ROAD_SURFACE &&
    input.brake <= 0 &&
    tickStartSpeed >= NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S
  );
}

/**
 * Integrates the longitudinal speed for one tick per the Â§5.2 locked
 * table: braking wins and is never floor-protected (deliberate braking
 * may stop the car); passive ticks stack the off-road drag on grass and
 * floor the result at the off-road floor only while the car enters the
 * tick at/above it (js-racer limit-gate parity, so the floor actually
 * holds). Speeds are clamped to `[0, maxSpeed]`.
 *
 * The floor gate is SHARED with the post-scrub floor of the tick
 * pipeline (RC-1) through `isOffRoadSpeedFloorEngaged`, so the
 * integrator and the drift scrub can never disagree about the floor.
 */
function integrateLongitudinalSpeed(
  speed: number,
  input: PhysicsStepInput,
  surface: PhysicsSurfaceType,
): number {
  if (input.brake > 0) {
    return Math.max(
      speed +
        input.brake *
          NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2 *
          NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
      CONTROL_RATIO_MIN,
    );
  }

  let nextSpeed =
    speed +
    resolvePassiveRate(input.throttle) *
      NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;
  if (surface === OFF_ROAD_SURFACE) {
    nextSpeed +=
      NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2 *
      NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;
    if (isOffRoadSpeedFloorEngaged(surface, input, speed)) {
      nextSpeed = Math.max(
        nextSpeed,
        NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
      );
    }
  }
  return Math.min(
    Math.max(nextSpeed, CONTROL_RATIO_MIN),
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  );
}

/** Resolves the passive (non-braking) longitudinal rate for one tick. */
function resolvePassiveRate(throttle: number): number {
  if (throttle > 0) {
    return throttle * NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2;
  }
  return NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2;
}

/** Wraps a heading into `[-Ï€, Ï€)` with one O(1) pass (no spin-up loops). */
function wrapHeading(radians: number): number {
  return (
    radians -
    HEADING_WRAP_TAU_RADIANS *
      Math.floor((radians + Math.PI) / HEADING_WRAP_TAU_RADIANS)
  );
}

/**
 * Mixes a spawn seed through a deterministic `imul`-based avalanche
 * (pure integer arithmetic; identical inputs always produce identical
 * mixes on any runtime honoring the ES specification for `Math.imul`).
 */
function mixSeed(seed: number): number {
  let mixed = seed | 0;
  mixed = Math.imul(mixed ^ (mixed >>> 16), SEED_MIX_MULTIPLIER);
  mixed = Math.imul(mixed ^ (mixed >>> 16), SEED_MIX_MULTIPLIER);
  mixed ^= mixed >>> 16;
  return mixed >>> 0;
}

/** Resolves the seeded spawn lane offset within `Â±spread / 2`. */
function resolveSpawnLateralOffset(seed: number): number {
  const normalized = mixSeed(seed) / MIXED_SEED_UINT32_MAX;
  return (normalized - 0.5) * NEATRACING_PHYSICS_SPAWN_LATERAL_SPREAD_M;
}

/** Clamps a `[0, 1]` control axis ratio (throttle, brake, speed percentile). */
function clampControlRatio(value: number): number {
  return Math.min(CONTROL_RATIO_MAX, Math.max(CONTROL_RATIO_MIN, value));
}

/** Clamps the steering ratio into `[-1, 1]`. */
function clampSteeringRatio(value: number): number {
  return Math.min(CONTROL_RATIO_MAX, Math.max(STEER_RATIO_MIN, value));
}
