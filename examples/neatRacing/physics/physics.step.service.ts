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
 * The tuning baseline is the locked javascript-racer adaptation table
 * (spec Â§5.2), transcribed as named constants with their js-racer
 * derivations preserved: `ACCEL = maxSpeed / 5`,
 * `BRAKING = -maxSpeed`, `NATURAL_DECEL = -maxSpeed / 5`,
 * `OFF_ROAD_DECEL = -maxSpeed / 2` with the passive off-road floor at
 * `maxSpeed / 4` (so a coasting off-road car never decelerates to zero),
 * the `CENTRIFUGAL_MULTIPLIER = 0.3` outward push (TP-02), and the
 * `STEERING_RATE = 2` steering rate scaled by the speed percentile
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
 * Scope notes: off-track/ribbon detection and collision are wired by the
 * surface-resolver consumer (P1-S7-C); per-wheel slip channels and the
 * worker-level reproducibility tuple are Phase-2 concerns and deliberately
 * out of scope here. `curb` shares the road longitudinal model in this
 * slice; curb-specific rumble shaping arrives with the collision/
 * off-track surface utils (P1-S7-C).
 */

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
 */
export interface PhysicsStepInput {
  /** Forward throttle axis in `[0, 1]`. */
  throttle: number;
  /** Brake axis in `[0, 1]`; a positive brake wins over throttle that tick. */
  brake: number;
  /** Steering axis in `[-1, 1]`; positive steers counterclockwise (left). */
  steer: number;
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
   * Lateral velocity in meters/second, positive = left of the heading.
   * Driven by the TP-02 outward push, drained by the grip rate; shapes
   * the velocity direction (and therefore `slipAngle`) without moving the
   * car sideways on its own while `speed` is zero.
   */
  lateralVelocity: number;
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
 * Centrifugal outward-push multiplier: `0.3` (spec Â§5.2 locked table,
 * TP-02). Top-down adaptation: the outward lateral push per tick is
 * `CENTRIFUGAL_MULTIPLIER Â· yawRate Â· speed` (algebraically the
 * `0.3 Â· curvature Â· speedPercentÂ²` form with the top-down curvature
 * `curvature = yawRate / speed`, guarded to zero at standstill).
 */
export const NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER = 0.3;

/**
 * Steering rate: `dx = dt Â· 2 Â· speedPercent` (spec Â§5.2 locked table,
 * TP-04 "cross the road in ~1 s at top speed"). Top-down adaptation: the
 * heading change per tick is `input.steer Â· STEERING_RATE Â·
 * speedPercent Â· dt` (radians per tick), so steering authority grows with
 * speed and a standing car does not rotate.
 */
export const NEATRACING_PHYSICS_STEERING_RATE = 2;

/**
 * Lateral grip drain rate in 1/s: how fast the lateral drift velocity
 * bleeds back to zero when the outward push stops. An internal arcade
 * shaping constant for the LD-15 slip channel (not part of the locked
 * js-racer table; `6` keeps the drift visible per tick yet settles in
 * ~250 ms so releasing the steer axis ends a slide quickly).
 */
export const NEATRACING_PHYSICS_LATERAL_GRIP_RATE_PER_S = 6;

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

export function createInitialPhysicsCarState(
  seed?: number,
  pose?: PhysicsCarSpawnPose,
): PhysicsCarState {
  const resolvedSeed = seed ?? NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED;
  return {
    x: pose?.x ?? resolveSpawnLateralOffset(resolvedSeed),
    y: pose?.y ?? SPAWN_Y_M,
    heading: pose?.heading ?? NEATRACING_PHYSICS_SPAWN_HEADING_RADIANS,
    speed: INITIAL_SPEED_UNITS_PER_S,
    yawRate: 0,
    slipAngle: 0,
    lateralVelocity: 0,
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
 * 2. resolve the yaw response from the steer axis and the tick-start
 *    speed percentile (TP-04, standing cars do not rotate) and integrate
 *    the heading;
 * 3. integrate the longitudinal speed through the locked Â§5.2 rates â
 *    braking wins over throttle, passive coast drag otherwise, off-road
 *    drag stacked on grass with the passive off-road floor (deliberate
 *    braking is never floored and can stop the car), clamped to
 *    `[0, maxSpeed]`;
 * 4. integrate the lateral drift velocity (TP-02 outward push, grip
 *    drain) and reduce it to the bounded LD-15 slip angle via `atan2`;
 * 5. translate along the velocity direction (heading + slipAngle) so the
 *    drift arc bends the path;
 * 6. count the tick.
 *
 * @param state - Current car state (treated read-only; never mutated).
 * @param input - Control sample applied for this single tick.
 * @param surfaceResolver - Optional surface probe; receives the tick-start
 *   state and must return a `PhysicsSurfaceType`. When omitted the tick
 *   behaves as on-road. Off-track detection (which road/curb/grass the
 *   ribbon assigns) is the consumer's job (P1-S7-C).
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
): PhysicsCarState {
  // Step 1: Resolve the surface under the car at tick start (explicit read).
  const surface = resolveSurfaceAtTickStart(surfaceResolver, state);
  const clampedInput = clampStepInput(input);
  const speedPercent = resolveSpeedPercent(state.speed);

  // Step 2: Resolve the yaw response for this tick and rotate the heading.
  const yawRatePerTick = resolveYawRatePerTick(
    clampedInput.steer,
    speedPercent,
  );
  const nextHeading = wrapHeading(state.heading + yawRatePerTick);

  // Step 3: Integrate the longitudinal speed (locked Â§5.2 rates).
  const nextSpeed = integrateLongitudinalSpeed(
    state.speed,
    clampedInput,
    surface,
  );

  // Step 4: Integrate the lateral drift dynamics and the slip angle (LD-15).
  const nextLateralVelocity = integrateLateralVelocity(
    state.lateralVelocity,
    yawRatePerTick,
    state.speed,
  );
  const nextSlipAngle = resolveSlipAngle(nextLateralVelocity, nextSpeed);

  // Step 5: Translate along the velocity direction (drift arc path).
  const velocityHeading = nextHeading + nextSlipAngle;

  const nextTick = state.tick + 1;

  return {
    x:
      state.x +
      Math.cos(velocityHeading) *
        nextSpeed *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
    y:
      state.y +
      Math.sin(velocityHeading) *
        nextSpeed *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
    heading: nextHeading,
    speed: nextSpeed,
    yawRate: yawRatePerTick,
    slipAngle: nextSlipAngle,
    lateralVelocity: nextLateralVelocity,
    tick: nextTick,
  };
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
  };
}

/** Normalizes the longitudinal speed into the `[0, 1]` speed percentile. */
function resolveSpeedPercent(speed: number): number {
  return clampControlRatio(speed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);
}

/**
 * Resolves the per-tick heading delta (radians per tick): the steer axis
 * scaled by `STEERING_RATE`, the speed percentile, and the fixed `dt`
 * (TP-04 form `dt Â· 2 Â· speedPercent`, mirrored for the top-down world).
 */
function resolveYawRatePerTick(steer: number, speedPercent: number): number {
  return (
    steer *
    NEATRACING_PHYSICS_STEERING_RATE *
    speedPercent *
    NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS
  );
}

/**
 * Integrates the longitudinal speed for one tick per the Â§5.2 locked
 * table: braking wins and is never floor-protected (deliberate braking
 * may stop the car); passive ticks stack the off-road drag on grass and
 * floor the result at the off-road floor only while the car enters the
 * tick at/above it (js-racer limit-gate parity, so the floor actually
 * holds). Speeds are clamped to `[0, maxSpeed]`.
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
    if (speed >= NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S) {
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

/**
 * Integrates the lateral drift velocity for one tick: the TP-02 outward
 * push (`âCENTRIFUGAL_MULTIPLIER Â· yawRate Â· speed`, zero at standstill)
 * charges the lateral velocity, then the grip rate drains it toward zero.
 */
function integrateLateralVelocity(
  lateralVelocity: number,
  yawRatePerTick: number,
  speed: number,
): number {
  const outwardPushPerTick =
    speed > 0
      ? -NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER * yawRatePerTick * speed
      : 0;
  const pushed = lateralVelocity + outwardPushPerTick;
  return (
    pushed *
    (1 -
      NEATRACING_PHYSICS_LATERAL_GRIP_RATE_PER_S *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS)
  );
}

/**
 * Resolves the LD-15 slip angle via `atan2(lateralVelocity, speed)`, which
 * keeps the angle bounded to `(-Ï€/2, Ï€/2)` by construction. A stopped
 * car reports zero slip (a bare `atan2` with zero speed would return
 * `Â±Ï€/2` and breach the AC-023 envelope for a phantom drift state).
 */
function resolveSlipAngle(lateralVelocity: number, speed: number): number {
  if (speed <= CONTROL_RATIO_MIN) {
    return 0;
  }
  return Math.atan2(lateralVelocity, speed);
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
