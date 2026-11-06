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
 * THE TURN SUFFICIENCY (Step 14Y, AC-098(a)) lifts the USER channel
 * inside a speed-localized BAND (the crawl band up to the band's upper
 * edge), so the channel can HOLD the tightest measured corner instead of
 * falling short of its kinematic requirement: the sufficiency peaks at
 * the corner's own target speed and it is exactly `1` outside the band,
 * so the crawl rate stays BIT-IDENTICAL and the ratified mild reduction
 * keeps its own full meaning above the band. The sufficiency belongs to
 * the USER channel alone - the AI channel is untouched by it.
 *
 * The tuning baseline is the locked javascript-racer adaptation table
 * (spec Â§5.2), transcribed as named constants with their js-racer
 * derivations preserved: `ACCEL = maxSpeed / 5`,
 * `BRAKING = -maxSpeed`, `NATURAL_DECEL = -maxSpeed / 5`,
 * `OFF_ROAD_DECEL = -maxSpeed / 2` with the USER-named off-track speed
 * cap at `20 km/h` (AC-090 VERDICT 8 - the off-track limit the USER
 * named, `20 / 3.6` m/s, so a coasting off-road car never decelerates
 * to zero and never exceeds the USER-named limit),
 * the `STEERING_RATE = 2` steering rate scaled by the speed percentile
 * (TP-04: `dx = dt Â· 2 Â· speedPercent`, a standing car does not rotate).
 *
 * Determinism (spec Â§5.3): the service is PURE â it calls no `Math.random`,
 * reads the surface ONCE from the POST-MOVE pose through the resolved
 * resolver, and
 * integrates in one fixed documented order, so the same
 * `(seed, control stream, surface stream)` produces byte-identical
 * JSON-serialized states on the same runtime (reproducibility rung
 * Level 2, ordered deterministic; cross-environment byte equality is NOT
 * claimed).
 *
 * THE ONE SURFACE ORDER (Step 14W, AC-094 (c)): the surface the physics
 * integrates against is the surface the car is ON, never the surface it
 * was on at the tick start. The tick resolves the surface ONCE, from the
 * POST-MOVE pose (the pose the car reached in this tick), so a car that
 * has left the road really gets the band surface in the SAME tick and the
 * EXISTING named off-road law (the off-road deceleration constant and the
 * off-road speed-floor constant) actually applies. The reach pose itself
 * depends on the surface, so the reach is computed against the NEUTRAL
 * (road) surface FIRST and the tick is RE-INTEGRATED against the resolved
 * surface when that surface is not the neutral one - so the published tick
 * is always the tick that integrated against the surface the car is on.
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
 * The TRACK BOUND IS GONE (AC-090 VERDICT 8, superseding the AC-081
 * containment): there is NO invisible wall near the track edge. A car
 * that leaves the ribbon really goes off track and keeps rolling - the
 * ONLY limit out there is the USER-named off-track SPEED cap (the named
 * `20 km/h` constant below), never a geometric bound, so the single-car
 * tick needs NO geometry seam at all.
 */

import { resolveCarCollisions } from './collision.utils';
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
 * The USER-named OFF-TRACK SPEED LIMIT, in km/h (AC-090 VERDICT 8).
 *
 * THE USER'S OWN NUMBER: "the car should be able to go off track, but
 * with the speed limited to 20km/h". The off-track limit is therefore
 * 20 km/h - a NAMED constant, never a magic number, and the ONE place
 * the off-track limit is authored.
 */
export const NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH = 20;

/**
 * Off-track speed cap in m/s: the USER-named 20 km/h (`20 / 3.6`).
 *
 * While the car enters an off-road tick at/above this speed and is NOT
 * braking, the passive decelerations are floored here, so an off-road
 * car coasts down to the USER-named limit and HOLDS it instead of
 * decelerating to zero - and never runs faster than the USER named.
 *
 * THE DECEL LAW STAYS THE SAND MECHANISM: the off-road deceleration
 * constant above is what actually slows the car down; this cap is the
 * CEILING the USER named, and it SUPERSEDES the retired `maxSpeed / 4`
 * floor (22.25 m/s = 80.1 km/h, ~4x the USER-named limit).
 */
export const NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S =
  NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH / 3.6;

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
 * The tightest holdable corner radius of the USER channel (m) (Step 14Y,
 * AC-098(a)).
 *
 * THE RED-MEASURED VALUE: the tightest MEASURED layout of the repo (the
 * R = 8 m stadium) is the corner the USER channel has to be able to
 * HOLD. A corner of curvature `kappa` demands the turn rate `v x kappa`
 * (the kinematic requirement) at the speed the racing line drives it at,
 * so the radius is the model-side half of the sufficiency: it turns the
 * corner into a turn RATE the commanded rate can be compared with.
 */
export const NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_RADIUS_M = 8;

/**
 * The target speed of that tightest corner (m/s) (Step 14Y, AC-098(a)).
 *
 * THE RED-MEASURED VALUE: the published target speed of the tightest
 * corner of the tightest MEASURED layout of the repo (measured through
 * the racing line's own curvature-derived target speed). The
 * sufficiency PEAKS at this speed - it is the speed the corner is
 * driven at, so it is where the corner's kinematic requirement is
 * largest and where the USER channel is measured.
 */
export const NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_SPEED_UNITS_PER_S =
  17.8;

/**
 * The command-sufficiency margin of the USER channel (1) (Step 14Y,
 * AC-098(a)).
 *
 * THE MARGIN (the same rationale as the AI's own command-sufficiency
 * margin): a commanded rate that MERELY meets the corner's kinematic
 * requirement leaves the corner on the boundary, where any float noise
 * loses it. The margin is the small slack that keeps the commanded rate
 * STRICTLY above the requirement, so the corner is HOLDABLE rather than
 * borderline.
 */
export const NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_MARGIN = 1.1;

/**
 * The upper edge of the USER turn-rate sufficiency band (m/s) (Step 14Y,
 * AC-098(a)).
 *
 * WHY THE SUFFICIENCY IS A BAND AND NOT A GLOBAL RAISE: the ratified
 * AC-092 reduction factor is measured against the NOMINAL law of the
 * SAME speed, so the rate may never grow above the crawl band's rate
 * (the factor may only FALL with the speed) - a global raise of the
 * nominal rate would invert that factor at the crawl band. The
 * sufficiency is therefore a BAND: it is raised inside the band and it
 * is back at zero at this edge, so the ratified factor stays monotonic
 * across the ratified speed ladder and the top-speed rate stays
 * strictly below the nominal law (the AC-092 rails).
 *
 * THE VALUE: 18.0 m/s - the first speed above the RED-measured corner
 * speed where the ratified law's own rate is read again.
 */
export const NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_BAND_END_UNITS_PER_S = 18.0;

/**
 * The transition half-width of the USER turn-rate sufficiency window
 * (m/s) (Step 14Y, AC-098(a)).
 *
 * THE SHAPE: the sufficiency is a raised-cosine window that peaks at
 * the RED-measured corner speed. This constant caps how far from the
 * peak the window is allowed to open on EITHER side. It keeps the R=8
 * hold fully boosted at 17.8 m/s while making the boost fade long before
 * the demo drive reaches the ~18.4 m/s point where the wide original
 * window pushed the car off the served track.
 */
export const NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_TRANSITION_HALF_WIDTH_UNITS_PER_S = 0.2;

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

/**
 * The NEUTRAL surface of the REACH pass (Step 14W, AC-094 (c)).
 *
 * The pose a car reaches in a tick DEPENDS on the surface (the off-road
 * law slows the car down, so the car covers a shorter distance), so the
 * reach pose is computed against this NEUTRAL surface FIRST and the
 * resolver is asked about that pose. It is the ROAD surface because the
 * road arms NO extra law: the reach must never depend on the surface it
 * is resolving.
 */
const REACH_SURFACE: PhysicsSurfaceType = 'road';

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
 * 1. resolve the surface ONCE from the POST-MOVE pose (Step 14W,
 *    AC-094 (c)): the tick integrates against the surface the car is ON,
 *    never the surface it was on at the tick start. The pose the car
 *    reaches is computed against the NEUTRAL (road) surface, the resolver
 *    is asked about THAT pose (omitted means road; the resolver sees the
 *    pose the car reached, never the tick-start state), and the tick is
 *    RE-INTEGRATED against the resolved surface when that surface is not
 *    the neutral one;
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
 * 7. (THE RETIRED SEAM - AC-090 VERDICT 8) the published pose is the
 *    REACHED pose: the AC-081 TRACK CONTAINMENT is retired with the
 *    invisible wall it made, so the car that left the ribbon keeps
 *    rolling under the USER-named off-track speed cap (the ONE named
 *    `NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH`) - the geometry
 *    argument rides the retired NO-OP seam and the SHIPPED 4-arg
 *    live-tick form is preserved;
 * 8. count the tick.
 *
 * @param state - Current car state (treated read-only; never mutated).
 * @param input - Control sample applied for this single tick.
 * @param surfaceResolver - Optional surface probe; receives the POST-MOVE
 *   pose (the pose the car reached in this tick - never the tick-start
 *   state) and must return a `PhysicsSurfaceType`. When omitted the tick
 *   behaves as on-road. Off-track detection (which road/curb/grass the
 *   ribbon assigns) is the consumer's job (P1-S7-C).
 * @param trackSegments - The RETIRED TRACK GEOMETRY parameter (the
 *   ribbon segments). The AC-081 containment bound was RETIRED with the
 *   invisible wall it made (AC-090 VERDICT 8: "it's now an invisible
 *   wall ... the car should be able to go off track, but with the speed
 *   limited to 20km/h"), so this argument is NEVER READ any more: an
 *   ABSENT parameter, an EMPTY list and a FULL ribbon all produce THE
 *   SAME tick (the retired seam is a NO-OP - the AC-081-WIRE-NOOP
 *   contract re-contracted by the retirement). The parameter STAYS the
 *   SHIPPED 4-arg live-tick form, so the served closure
 *   (`browser-entry`), the served lane and the contract lanes keep ONE
 *   tick form; the off-track limit is only ever the USER-named speed
 *   cap.
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
  const clampedInput = clampStepInput(input);

  // Step 1: THE ONE SURFACE ORDER (Step 14W, AC-094 (c)).
  //
  // THE ORDER: the surface the physics integrates against is the surface
  // the car is ON - never the surface it was on at the tick start. The
  // resolver is therefore asked ONCE per tick, about the POST-MOVE pose
  // (the pose the car reached in this tick), so a car that has left the
  // road really gets the band surface in the SAME tick and the EXISTING
  // named off-road law (the off-road deceleration constant and the
  // off-road speed-floor constant) actually applies.
  //
  // THE REACH FIRST: the pose a car reaches depends on the surface (the
  // off-road law slows the car down, so the car covers a shorter
  // distance), so the reach is computed against the NEUTRAL (road)
  // surface FIRST and the resolver is asked about THAT pose. When the
  // resolved surface is not the neutral one, the tick is RE-INTEGRATED
  // against the resolved surface, so the published tick is always the
  // tick that integrated against the surface the car is on.
  const reached = advanceCarState(state, clampedInput, REACH_SURFACE);
  const surface = resolveSurfaceAtPostMove(surfaceResolver, reached);
  const advancedCar =
    surface === REACH_SURFACE
      ? reached
      : advanceCarState(state, clampedInput, surface);

  // Step 7: THE RETIRED SEAM (AC-090 VERDICT 8). The published pose is
  // the REACHED pose - a car that crossed the red edge keeps rolling
  // under the USER-named off-track speed cap and is NEVER clamped,
  // stopped or reseated by any bound (the NO-WALL contract the USER
  // named). The geometry argument rides the retired NO-OP seam, so the
  // shipped 4-arg live-tick form is preserved byte-identically.
  return applyTrackContainment(advancedCar, trackSegments);
}

/**
 * Advances ONE car by ONE fixed tick against ONE surface.
 *
 * THE PIPELINE (the documented order of the tick): the demanded turn rate
 * is resolved ONCE and drives both the yaw and the drift commitment, the
 * longitudinal speed is integrated through the locked speed-table rates
 * (with the surface's off-road law), the drift channel advances in its
 * load-bearing order, the published slip angle and the ONE shared
 * traction severity are derived from the POST-scrub speed, and the car is
 * translated along the velocity direction.
 *
 * THE SURFACE IS THE ONLY INPUT THAT CHANGES THE NUMBERS: the off-road
 * deceleration and the off-road speed floor are the two surface-dependent
 * reads, so the same call with the same surface always produces the same
 * numbers. That is what lets the tick resolve the surface from the
 * POST-MOVE pose and re-integrate against it without changing a single
 * pinned tick.
 *
 * @param state - The tick-start car state (treated read-only; never
 *   mutated).
 * @param clampedInput - The clamped control sample applied to this tick.
 * @param surface - The surface this tick integrates against.
 * @returns The advanced car state (the published pose - the AC-081
 *   containment seam is retired, so the reached pose IS the published
 *   pose; AC-090 VERDICT 8).
 */
function advanceCarState(
  state: PhysicsCarState,
  clampedInput: PhysicsStepInput,
  surface: PhysicsSurfaceType,
): PhysicsCarState {
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

  return advancedCar;
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
 * from its RESOLVED pose, because `travelHeadingRad` and `tractionSeverity`
 * are documented as the exact values the integration used - a resolved car
 * must never publish the pre-resolution numbers.
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
 * The car-to-car collision (AC-090) recomposes the car around a new
 * world velocity (a new heading, a new speed, a zeroed slip and a zeroed
 * drift channel), so the published travel heading and the published
 * traction severity of the pre-resolution tick are STALE. They are
 * re-derived here - from the resolved values ONLY, never from the
 * pre-resolution ones - so the published state and the rendered frame
 * stay one frame of reference. (The road containment that shared this
 * republish path was RETIRED with the AC-081 bound - AC-090 VERDICT 8 -
 * so this republish now serves the car-to-car resolution ONLY.)
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
 * The RETIRED TRACK-CONTAINMENT seam (AC-090 VERDICT 8, superseding the
 * landed AC-081-WIRE seam of the Step 14I/14W iterations): the ONE named
 * owner of the ONE track bound - which is now NO BOUND AT ALL.
 *
 * THE RETIREMENT (the USER observable, the 2026-11-05 verdict-8 record):
 * the landed iteration-2 non-reflecting clamp HELD the pose at the band
 * outer edge (288/288 post-touch ticks pinned EXACTLY 3.0 m past the
 * drivable edge while full throttle was applied - the plan's measured W1
 * record), so the boundary read as an INVISIBLE WALL. The USER named the
 * next-order defect verbatim: "the car should be able to go off track,
 * but with the speed limited to 20km/h". The measured classification
 * recorded the hold as a true pose-hold (NOT a slowdown), the corridor
 * beyond the bound as geometrically CLEAN (7.0 m to the AC-096 10 m
 * prop floor), and the published off-track floor as 4.005x the
 * USER-named limit - so the bound is RETIRED ENTIRELY and the ONLY
 * limit out there is the USER-named off-track speed cap (the named
 * `NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH` constant in this
 * module; the off-road decel law STAYS the slowdown mechanism).
 *
 * THE NO-OP CONTRACT (AC-081-WIRE-NOOP, re-contracted by the
 * retirement): EVERY input is a no-op - an ABSENT geometry, an EMPTY
 * segment list and a FULL ribbon all leave the tick byte-identical,
 * and the returned car IS the reached car. The ONE-BOUND and
 * SHIPPED-TICK clauses carry over the retirement: this seam stays the
 * ONE named owner of the (now retired) bound contract, the second
 * parameter list stays the SHIPPED 4-ARG live-tick form (the served
 * closure and the contract lanes keep ONE tick form), and the AC-081
 * pose-independence clause is re-authored to hold TRUE BY
 * CONSTRUCTION: with no bound at all, no published value depends on
 * any pose.
 *
 * @param car - The advanced car state (treated read-only; never
 *   mutated).
 * @param trackSegments - The RETIRED ribbon geometry; accepted for the
 *   shipped 4-arg live-tick form and never read (`void` marks the
 *   intentional non-read of the retirement).
 * @returns The SAME car state, unchanged.
 */
function applyTrackContainment(
  car: PhysicsCarState,
  trackSegments: readonly TrackSegment[] | undefined,
): PhysicsCarState {
  // THE RETIRED BOUND: there is NO containment left to apply - no
  // clamp, no reseat, no hold. The argument is received and retired
  // (never read); the `void` marks the intentional non-read so the
  // unused-vars lane stays honest about the retirement instead of
  // papering over it.
  void trackSegments;
  return car;
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
 *   receives the POST-MOVE pose the probed member reached in this tick
 *   (never the tick-start state).
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

/**
 * Resolves the surface of the POST-MOVE pose (Step 14W, AC-094 (c)).
 *
 * THE ONE READ: the resolver is asked ONCE per tick, about the pose the
 * car REACHED in this tick (never the tick-start state), so the surface
 * the physics integrates against is the surface the car is ON. An absent
 * resolver keeps the tick on the road.
 *
 * @param surfaceResolver - The optional surface probe.
 * @param reached - The pose the car reached in this tick.
 * @returns The surface under the reached pose; road without a resolver.
 */
function resolveSurfaceAtPostMove(
  surfaceResolver:
    ((carState: PhysicsCarState) => PhysicsSurfaceType) | undefined,
  reached: PhysicsCarState,
): PhysicsSurfaceType {
  return surfaceResolver?.(reached) ?? DEFAULT_SURFACE;
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
 * Resolves the turn-sufficiency WINDOW of a speed (a unitless factor in
 * `[0, 1]`).
 *
 * THE WINDOW (Step 14Y, AC-098(a)): the window is a raised cosine that is
 * EXACTLY zero at both edges of the sufficiency band (the crawl band
 * below, the band's upper edge above) and peaks at the RED-MEASURED
 * corner speed. The transition half-width caps how far from the peak the
 * cosine is allowed to open on each side, so the boost is concentrated on
 * the tight corner and fades before the higher-speed straightaway. The
 * window is continuous and smooth at the peak and at both edges, so the
 * commanded rate never steps.
 *
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @returns The window in `[0, 1]`; `0` outside the sufficiency band.
 */
function resolveTurnRateSufficiencyWindow(speedPercent: number): number {
  const startPercent =
    NEATRACING_PHYSICS_TURN_RATE_REDUCTION_BAND_UNITS_PER_S /
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  const endPercent =
    NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_BAND_END_UNITS_PER_S /
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  const peakPercent =
    NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_SPEED_UNITS_PER_S /
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  if (speedPercent <= startPercent || speedPercent >= endPercent) {
    return 0;
  }
  const transitionHalfWidthPercent =
    NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_TRANSITION_HALF_WIDTH_UNITS_PER_S /
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  const halfWidth =
    speedPercent < peakPercent
      ? Math.min(peakPercent - startPercent, transitionHalfWidthPercent)
      : Math.min(endPercent - peakPercent, transitionHalfWidthPercent);
  const phase = Math.min(
    1,
    Math.max(-1, (speedPercent - peakPercent) / halfWidth),
  );
  return 0.5 * (1 + Math.cos(Math.PI * phase));
}

/**
 * Resolves the turn-sufficiency FACTOR of a speed (a unitless factor
 * `>= 1`).
 *
 * THE FACTOR (Step 14Y, AC-098(a)): the factor is `1` outside the
 * sufficiency band, so the ratified rate is left BIT-IDENTICAL there, and
 * inside the band it scales the ratified rate by the window. At the
 * RED-MEASURED corner speed the factor lifts the commanded rate to the
 * corner's kinematic requirement (`speed / radius`) plus the sufficiency
 * margin, so the USER channel can HOLD that corner instead of falling
 * short of it.
 *
 * THE BASIS: the peak is resolved from the ratified law of the corner
 * speed itself (the steer-axis law times the ratified mild factor), so
 * the sufficiency is measured against the very law it lifts and it never
 * needs its own copy of the reduction factor.
 *
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @returns The sufficiency factor in `[1, ...]`; exactly `1` outside the band.
 */
function resolveTurnRateSufficiencyFactor(speedPercent: number): number {
  const window = resolveTurnRateSufficiencyWindow(speedPercent);
  if (window <= 0) {
    return 1;
  }
  const peakPercent =
    NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_SPEED_UNITS_PER_S /
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  const peakRequirementRadPerS =
    NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_MARGIN *
    (NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_SPEED_UNITS_PER_S /
      NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_RADIUS_M);
  const peakBaseRateRadPerS =
    NEATRACING_PHYSICS_STEERING_RATE *
    peakPercent *
    resolveTurnRateSpeedFactor(peakPercent);
  return 1 + (peakRequirementRadPerS / peakBaseRateRadPerS - 1) * window;
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
 * THE TURN SUFFICIENCY (Step 14Y, AC-098(a)): inside the sufficiency band
 * the user rate is raised by a speed-localized sufficiency factor, so the
 * channel can HOLD the tightest measured corner instead of falling short
 * of its kinematic requirement. The factor is exactly `1` outside the
 * band - below the crawl band (the rate stays BIT-IDENTICAL) and above
 * the band's upper edge (the ratified mild reduction keeps its own full
 * meaning) - so the sufficiency never touches a ratified lane's reading.
 * The sufficiency belongs to the USER channel alone: the AI channel
 * above is untouched by it.
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
  // product order of the pre-14T law is preserved) and raised INSIDE the
  // turn-sufficiency band (Step 14Y, AC-098(a)).
  //
  // THE SUFFICIENCY FACTOR is applied after the mild factor and it is
  // exactly `1` outside its band, so every lane below the reduction band
  // (and every lane above the sufficiency band) stays BIT-IDENTICAL.
  return (
    steer *
    NEATRACING_PHYSICS_STEERING_RATE *
    speedPercent *
    resolveTurnRateSpeedFactor(speedPercent) *
    resolveTurnRateSufficiencyFactor(speedPercent)
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
 * @param surface - The resolved surface the tick integrates against.
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
