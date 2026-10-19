/**
 * Arcade drift model for the neatRacing example (slice P1-arcade-drift;
 * AC-047; spec §5.2 drift amendment).
 *
 * The model is intentionally small and pure: it takes a previous drift
 * state and a single-tick input and returns the next drift state. All
 * tuning is expressed through named constants so the spec amendment can
 * reproduce the numbers verbatim.
 *
 * The drift channel is made of two observable values:
 * - `lateralVelocity` (m/s, positive = left of the heading)
 * - `slipAngle` (rad, bounded `(-π/2, π/2)` via `atan2`)
 *
 * One integration step applies, in order:
 * 1. outward push from the centrifugal term (`-multiplier · yawRate · speed`);
 * 2. countersteer recovery when the steer axis opposes the current slip;
 * 3. grip falloff as the slip angle grows past a threshold;
 * 4. a grip drain toward zero lateral velocity.
 *
 * The result is an arcade drift feel: controlled oversteer that builds
 * under sustained steering, persists at large slip angles because the
 * grip factor falls off, and settles quickly when the driver releases
 * the steer axis or countersteers.
 *
 * Determinism (spec §5.3): the model contains no `Math.random` and no
 * side effects; the same `(state, input)` always yields the same
 * byte-identical output on the same runtime.
 */

/** Absolute slip-angle ceiling used by the spec and the UI envelope. */
export const NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = Math.PI / 2;

/**
 * Slip-angle threshold where the grip falloff curve begins. Below this
 * angle the tire retains full lateral grip (`falloff factor = 1`).
 */
export const NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD = 0.25;

/**
 * Rate at which grip falls off for each radian past the threshold.
 * The effective grip factor is clamped to `MIN_GRIP_FACTOR` so the
 * car never becomes completely frictionless.
 */
export const NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD = 2;

/**
 * Floor for the grip falloff factor. A small residual grip remains even
 * at the maximum slip angle, keeping the drift controllable and bounded.
 */
export const NEATRACING_DRIFT_MIN_GRIP_FACTOR = 0.25;

/**
 * How fast opposing steer reduces the lateral drift velocity. Scaled by
 * the steering axis magnitude and the speed percentile so countersteer
 * authority grows with speed.
 */
export const NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S = 4;

/**
 * Base lateral grip drain rate in 1/s. At zero slip this drains the
 * lateral velocity toward zero; at larger slip the drain is softened by
 * the grip falloff factor.
 */
export const NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S = 6;

/** Read-only drift state carried by the physics car state. */
export interface DriftModelState {
  /** Lateral velocity in m/s, positive = left of the heading. */
  lateralVelocity: number;
  /** Slip angle in rad between velocity direction and heading. */
  slipAngle: number;
}

/** Input describing one fixed-tick arcade-drift integration. */
export interface DriftStepInput {
  /** Steering axis in `[-1, 1]`; positive steers counterclockwise. */
  steer: number;
  /** Heading change for this tick in radians (radians per 60 Hz step). */
  yawRatePerTick: number;
  /** Longitudinal speed in m/s at the start of the drift integration. */
  speed: number;
  /** Normalized speed in `[0, 1]`. */
  speedPercent: number;
  /** Fixed timestep in seconds. */
  dt: number;
  /** Centrifugal multiplier from the locked §5.2 table (default 0.3). */
  centrifugalMultiplier: number;
}

/** Creates the deterministic zero-drift state used on spawn. */
export function createZeroDriftState(): DriftModelState {
  return { lateralVelocity: 0, slipAngle: 0 };
}

/**
 * Advances the arcade drift state by one fixed tick.
 *
 * The integration uses the previous slip angle to look up the grip
 * falloff and the current speed for the outward push. This keeps the
 * model side-effect free and easy to replay deterministically.
 *
 * @param prev - Drift state at tick start (read-only).
 * @param input - Tick input sample.
 * @returns Next drift state.
 */
export function stepDriftModel(
  prev: DriftModelState,
  input: DriftStepInput,
): DriftModelState {
  const outwardPush = computeOutwardPush(
    input.yawRatePerTick,
    input.speed,
    input.centrifugalMultiplier,
  );
  const slipAngle = computeSlipAngle(prev.lateralVelocity, input.speed);
  const gripFalloff = computeGripFalloff(slipAngle);
  const counterDelta = computeCountersteerDelta(
    input.steer,
    slipAngle,
    input.speedPercent,
    input.dt,
  );

  const effectiveGrip = NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S * gripFalloff;
  const nextLateralVelocity =
    (prev.lateralVelocity + outwardPush + counterDelta) *
    (1 - effectiveGrip * input.dt);

  return {
    lateralVelocity: nextLateralVelocity,
    slipAngle: computeSlipAngle(nextLateralVelocity, input.speed),
  };
}

/**
 * Computes the next lateral velocity for one tick without recomputing
 * the slip angle. This seam lets `physics.step.service.ts` match its
 * existing integration order (lateral velocity at tick-start speed,
 * slip angle at tick-end speed).
 *
 * @param prevLateralVelocity - Lateral velocity at tick start.
 * @param input - Tick input sample.
 * @returns Lateral velocity at tick end.
 */
export function computeNextLateralVelocity(
  prevLateralVelocity: number,
  input: DriftStepInput,
): number {
  const outwardPush = computeOutwardPush(
    input.yawRatePerTick,
    input.speed,
    input.centrifugalMultiplier,
  );
  const slipAngle = computeSlipAngle(prevLateralVelocity, input.speed);
  const gripFalloff = computeGripFalloff(slipAngle);
  const counterDelta = computeCountersteerDelta(
    input.steer,
    slipAngle,
    input.speedPercent,
    input.dt,
  );

  const effectiveGrip = NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S * gripFalloff;
  return (
    (prevLateralVelocity + outwardPush + counterDelta) *
    (1 - effectiveGrip * input.dt)
  );
}

/**
 * Computes the lateral outward push from a heading change and speed.
 *
 * Equation: `push = -centrifugalMultiplier · yawRatePerTick · speed`
 * (zero at standstill).
 *
 * @param yawRatePerTick - Heading change for this tick in radians.
 * @param speed - Longitudinal speed in m/s.
 * @param centrifugalMultiplier - Outward-push multiplier.
 * @returns Signed lateral push in m/s for this tick.
 */
export function computeOutwardPush(
  yawRatePerTick: number,
  speed: number,
  centrifugalMultiplier: number,
): number {
  if (speed <= 0) {
    return 0;
  }
  return -centrifugalMultiplier * yawRatePerTick * speed;
}

/**
 * Computes the grip falloff factor for a given slip angle.
 *
 * Full grip (`1`) is kept below the threshold. Past the threshold the
 * factor falls linearly with `GRIP_FALLOFF_RATE_PER_RAD` and is clamped
 * to `MIN_GRIP_FACTOR`.
 *
 * @param slipAngle - Current slip angle in rad.
 * @returns Grip factor in `[MIN_GRIP_FACTOR, 1]`.
 */
export function computeGripFalloff(slipAngle: number): number {
  const overThreshold = Math.max(
    0,
    Math.abs(slipAngle) - NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD,
  );
  return Math.max(
    NEATRACING_DRIFT_MIN_GRIP_FACTOR,
    1 - overThreshold * NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD,
  );
}

/**
 * Computes the countersteer-induced lateral-velocity delta.
 *
 * The delta is nonzero only when the steering axis opposes the current
 * slip sign. It reduces the lateral velocity magnitude, helping the
 * driver end a slide. Same-direction steer or zero slip returns zero.
 *
 * @param steer - Steering axis in `[-1, 1]`.
 * @param slipAngle - Current slip angle in rad.
 * @param speedPercent - Normalized speed in `[0, 1]`.
 * @param dt - Fixed timestep in seconds.
 * @returns Signed lateral-velocity delta in m/s for this tick.
 */
export function computeCountersteerDelta(
  steer: number,
  slipAngle: number,
  speedPercent: number,
  dt: number,
): number {
  if (slipAngle === 0 || steer === 0) {
    return 0;
  }
  const slipSign = Math.sign(slipAngle);
  if (steer * slipSign >= 0) {
    return 0;
  }
  return (
    -slipSign *
    Math.abs(steer) *
    speedPercent *
    NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S *
    dt
  );
}

/**
 * Resolves the bounded slip angle via `atan2(lateralVelocity, speed)`.
 *
 * A stopped car reports zero slip so a bare `atan2` with zero speed
 * cannot create a phantom drift state.
 *
 * @param lateralVelocity - Lateral velocity in m/s.
 * @param speed - Longitudinal speed in m/s.
 * @returns Slip angle in `(-π/2, π/2)` rad, or `0` when stopped.
 */
export function computeSlipAngle(
  lateralVelocity: number,
  speed: number,
): number {
  if (speed <= 0) {
    return 0;
  }
  return Math.atan2(lateralVelocity, speed);
}
