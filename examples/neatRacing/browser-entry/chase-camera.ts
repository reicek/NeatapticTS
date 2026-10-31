/**
 * The Neat Racing chase camera (Step 14P, slice
 * `P1-drift-rewire-travel-camera`).
 *
 * THE DEFECT THIS MODULE CLOSES: the chase camera was a pure function of the
 * car NOSE heading. The camera eye, the look target and the starfield yaw
 * were all re-derived from `car.heading` every frame, and because the eye
 * was re-placed along the nose FROM the car's own position, the frame
 * cancelled the off-axis travel BY CONSTRUCTION: a 25.8 degree slide
 * produced EXACTLY ZERO screen-space translation (the frozen-camera trap,
 * RED-4). The 3D view could not move, so the drift could not be seen.
 *
 * THE LAW: the camera aims along the TRAVEL direction - the travel heading
 * the physics publishes on the car state - and the follow that gets it
 * there is PROPORTIONAL to the speed times the shared drift amount (the
 * user law S-1/S-4), never a fixed time constant (a fixed tau is the S-1
 * violation the retired `NEATRACING_CHASE_CAMERA_FOLLOW_TAU_S` and
 * `NEATRACING_CHASE_CAMERA_TRAVEL_AIM_GAIN` names stood for; those names
 * are FORBIDDEN and do not exist).
 *
 * THE OWNERSHIP: this module owns the chase GEOMETRY (the eye, the look
 * target, the forward vector) and the follow LAW. The browser entry owns
 * the renderer and reads BOTH through the ONE aim law, so the camera
 * placement, the look target and the starfield yaw can never disagree about
 * which way the car is travelling.
 *
 * @packageDocumentation
 */

import { NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS } from '../physics/physics.step.service';

/**
 * The chase-camera follow timestep (s): the NAMED 60 Hz physics tick dt.
 *
 * The follow is stepped in TICKS, never in frames, so the aim is frame-rate
 * independent: sixty ticks delivered as one frame or as sixty frames
 * produce the IDENTICAL aim (RED-6). The dt is IMPORTED from the physics
 * service instead of being inlined, so the camera can never tick at a
 * different rate than the simulation it follows.
 */
export const NEATRACING_CHASE_CAMERA_FOLLOW_DT_S =
  NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;

/** Chase-camera distance behind the car (world units). */
export const NEATRACING_CHASE_CAMERA_BACK = 5.5;
/** Chase-camera height above the ground plane (world units). */
export const NEATRACING_CHASE_CAMERA_HEIGHT = 2.4;
/** Chase-camera look-ahead distance along the camera aim (world units). */
export const NEATRACING_CHASE_CAMERA_LOOKAHEAD = 8;

/**
 * The chase-camera follow gain: how fast the camera aim closes on the
 * PUBLISHED travel direction, in rad/s per (m/s of car speed) per rad of
 * shared drift amount.
 *
 * WHY 1: the follow rate scales with the SPEED times the shared drift
 * amount (the user law S-1/S-4), so the camera eases onto the slide at low
 * speed and locks onto it at speed - the drift reads as a MOTION, not as a
 * cut. The bound is never reached inside the physics envelope: the fastest
 * possible step is `89 m/s * 0.452 rad / 60 Hz = 0.67` of the remaining
 * divergence per tick, so the follow NEVER snaps onto the target.
 */
const NEATRACING_CHASE_CAMERA_FOLLOW_GAIN = 1;

/**
 * The chase-camera follow state: the camera's slide offset from the nose
 * heading and the deterministic tick the follow was last advanced to.
 *
 * EXACTLY TWO fields, by contract: the offset is the ONLY carried state of
 * the follow, and the tick is the frame-rate-independent clock that
 * advances it.
 */
export interface ChaseCameraFollowState {
  /**
   * The camera aim offset from the nose heading (rad), signed: negative
   * means the camera aims BELOW the nose, i.e. along the travel direction of
   * a left-hand slide.
   */
  offsetRad: number;
  /** The deterministic simulation tick the follow was last advanced to. */
  lastTick: number;
}

/**
 * The step descriptor the follow consumes: the simulation ticks that
 * elapsed, the nose heading, the PUBLISHED travel heading, the car speed and
 * the tick index the step belongs to.
 */
export interface ChaseCameraFollowStep {
  /** The number of physics ticks this step advances the follow by. */
  ticks: number;
  /** The car nose heading (rad). */
  headingRad: number;
  /** The published travel heading (rad) - the direction the car moves. */
  travelHeadingRad: number;
  /** The car speed (m/s). */
  speedMps: number;
  /** The physics tick index this step belongs to. */
  tick: number;
}

/**
 * The chase-camera basis in scene space: the camera eye, the look target
 * and the normalized camera forward.
 */
export interface ChaseCameraBasis {
  /** The camera eye in scene space (x, y, z). */
  cameraWorld: [number, number, number];
  /** The look target in scene space (x, y, z). */
  lookTarget: [number, number, number];
  /** The normalized camera forward in scene space (x, y, z). */
  forwardVector: [number, number, number];
}

/**
 * Wraps an angle into the `[-pi, pi)` band.
 *
 * @param angleRad - The angle to wrap (rad).
 * @returns The wrapped angle (rad).
 */
function wrapRad(angleRad: number): number {
  const twoPi = 2 * Math.PI;
  return angleRad - twoPi * Math.floor((angleRad + Math.PI) / twoPi);
}

/**
 * The SIGNED SHORTEST-PATH delta from one angle to another, so a heading
 * crossing plus or minus pi never turns the camera by 2 pi (RED-7).
 *
 * @param fromRad - The angle travelled from (rad).
 * @param toRad - The angle travelled to (rad).
 * @returns The shortest signed delta (rad).
 */
function signedShortestDeltaRad(fromRad: number, toRad: number): number {
  return wrapRad(toRad - fromRad);
}

/**
 * Creates the neutral chase-camera follow state: the camera aims along the
 * nose heading and the follow has not advanced yet.
 *
 * @returns A fresh follow state.
 */
export function createChaseCameraFollowState(): ChaseCameraFollowState {
  return { offsetRad: 0, lastTick: 0 };
}

/**
 * The ONE aim law: the camera aim is the nose heading plus the follow
 * offset.
 *
 * Every live consumer (the camera placement, the look target and the
 * starfield yaw) reads THIS function, so the frame can never show a camera
 * that looks one way and a sky that slides another (RED-3).
 *
 * @param headingRad - The car nose heading (rad).
 * @param followOffsetRad - The chase-camera follow offset (rad).
 * @returns The camera aim (rad).
 */
export function resolveChaseCameraAimRad(
  headingRad: number,
  followOffsetRad: number,
): number {
  return headingRad + followOffsetRad;
}

/**
 * Advances the chase-camera follow over the elapsed simulation ticks.
 *
 * THE LAW (RED-5, the user law S-1/S-4): the per-tick aim change is
 * PROPORTIONAL to the car speed times the shared drift amount, so the
 * follow is a slight ANIMATION with no snap: it eases onto the published
 * slide at low speed, locks onto it at speed, and it holds with ZERO
 * steady-state error against the travel divergence. A fixed time constant
 * (the retired, FORBIDDEN `NEATRACING_CHASE_CAMERA_FOLLOW_TAU_S`) would
 * leave the rate unchanged when the speed doubles - the S-1 violation.
 *
 * THE MECHANICS:
 *
 * - `ticks: 0` HOLDS the aim (a frame that released no simulation tick
 *   must not move the camera);
 * - a tick REGRESSION (`tick < follow.lastTick` - a new car or a mode
 *   switch rewinds the deterministic tick counter) resets the offset to
 *   EXACTLY 0, so no stale slide survives a respawn (RED-7). This is the
 *   ONE reset owner;
 * - the aim is always approached along the SHORTEST PATH, so a heading
 *   crossing plus or minus pi never turns the camera by 2 pi (RED-7).
 *
 * @param follow - The current follow state (read-only; never mutated).
 * @param step - The step descriptor (the elapsed ticks, the nose heading,
 *   the published travel heading, the speed and the tick index).
 * @param out - Optional destination bag. When supplied, the next state is
 *   written INTO it and returned, so a hot render path can refresh a
 *   STABLE bag with no per-frame allocation. When omitted, a fresh state
 *   is returned.
 * @returns The NEXT follow state.
 */
export function stepChaseCameraFollow(
  follow: ChaseCameraFollowState,
  step: ChaseCameraFollowStep,
  out?: ChaseCameraFollowState,
): ChaseCameraFollowState {
  // The tick REGRESSION reset: a new car or a mode switch rewinds the
  // deterministic tick counter. The stale slide offset is dropped to
  // EXACTLY 0 and the follow re-anchors on the new tick.
  if (step.tick < follow.lastTick) {
    return writeChaseCameraFollowState(out, 0, step.tick);
  }
  // A zero-tick frame HOLDS the aim: the follow advances only when the
  // simulation actually ticked.
  if (step.ticks <= 0) {
    return writeChaseCameraFollowState(
      out,
      follow.offsetRad,
      follow.lastTick,
    );
  }
  let offsetRad = follow.offsetRad;
  for (let tick = 0; tick < step.ticks; tick += 1) {
    offsetRad += resolveChaseCameraFollowStepRad(offsetRad, step);
  }
  return writeChaseCameraFollowState(out, offsetRad, step.tick);
}

/**
 * Resolves the ONE per-tick follow step of the camera aim.
 *
 * The step is the PROPORTIONAL law: the rate scales with the car SPEED
 * times the shared drift amount, the per-tick change is bounded (no snap)
 * and the aim is approached along the shortest path (no overshoot, no
 * discontinuity).
 *
 * @param offsetRad - The current camera aim offset from the nose (rad).
 * @param step - The step descriptor.
 * @returns The signed offset change for this tick (rad).
 */
function resolveChaseCameraFollowStepRad(
  offsetRad: number,
  step: ChaseCameraFollowStep,
): number {
  // The PUBLISHED travel divergence: the travel heading IS the nose
  // heading plus the published slip, so this is the ONE shared scalar the
  // physics publishes and the body reads.
  const divergenceRad = signedShortestDeltaRad(
    step.headingRad,
    step.travelHeadingRad,
  );
  // What is LEFT to close: the remaining divergence between the camera aim
  // and the published travel direction.
  const aimErrorRad = signedShortestDeltaRad(
    step.headingRad + offsetRad,
    step.travelHeadingRad,
  );
  // The shared drift amount: the published slide, or the camera's own
  // remaining divergence, whichever is larger. The maximum keeps the follow
  // closing the slide after the physics has already RELEASED it, so a
  // release decays back to the nose-aligned aim instead of sticking.
  const driftAmountRad = Math.max(
    Math.abs(divergenceRad),
    Math.abs(aimErrorRad),
  );
  // The PROPORTIONAL rate: the speed times the shared drift amount.
  const followRateRadPerSecond =
    NEATRACING_CHASE_CAMERA_FOLLOW_GAIN *
    step.speedMps *
    driftAmountRad;
  // The bounded per-tick step (no snap): the aim never jumps the whole
  // remaining divergence in a single tick.
  const alpha = Math.min(
    1,
    followRateRadPerSecond * NEATRACING_CHASE_CAMERA_FOLLOW_DT_S,
  );
  return aimErrorRad * alpha;
}

/**
 * Writes a chase-camera follow state into the destination bag.
 *
 * @param out - Optional destination bag; a fresh state is created when it
 *   is omitted.
 * @param offsetRad - The next camera aim offset (rad).
 * @param lastTick - The tick the follow is now anchored on.
 * @returns The written follow state.
 */
function writeChaseCameraFollowState(
  out: ChaseCameraFollowState | undefined,
  offsetRad: number,
  lastTick: number,
): ChaseCameraFollowState {
  const next = out ?? { offsetRad, lastTick };
  next.offsetRad = offsetRad;
  next.lastTick = lastTick;
  return next;
}

/**
 * Resolves the chase-camera basis from the camera aim ALONE: the camera
 * eye, the look target and the normalized camera forward.
 *
 * The basis is a pure function of the aim, so the three live call sites
 * (the camera placement, the look target and the starfield yaw) can only
 * agree if they all read the ONE aim law (RED-3), and a straight lane (a
 * published slip of EXACTLY 0) reproduces the pre-wave basis BIT-FOR-BIT
 * (RED-4).
 *
 * The scene mirrors the top-down y into z (`worldZ = -y`) - the SAME
 * handedness seam as `carSceneWorldZ` in the browser entry. The mirror is
 * kept LOCAL here because the browser entry imports this module: importing
 * the helper back would close an import cycle.
 *
 * @param carX - The car x coordinate (m).
 * @param carY - The car top-down y coordinate (m).
 * @param aimRad - The camera aim (rad).
 * @param out - Optional destination bag. When supplied, the basis is
 *   written INTO it and returned, so a hot render path can refresh a
 *   STABLE bag with no per-frame allocation. When omitted, a fresh basis
 *   is returned.
 * @returns The chase-camera basis.
 */
export function resolveChaseCameraBasis(
  carX: number,
  carY: number,
  aimRad: number,
  out?: ChaseCameraBasis,
): ChaseCameraBasis {
  const basis = out ?? {
    cameraWorld: [0, 0, 0] as [number, number, number],
    lookTarget: [0, 0, 0] as [number, number, number],
    forwardVector: [0, 0, 0] as [number, number, number],
  };
  const forwardX = Math.cos(aimRad);
  const sceneForwardZ = -Math.sin(aimRad);
  // The ground-plane handedness seam: the scene mirrors the top-down y
  // into z. The signed zero is normalized so the world origin maps to +0.
  const carSceneZ = carY === 0 ? 0 : -carY;
  basis.cameraWorld[0] =
    carX - forwardX * NEATRACING_CHASE_CAMERA_BACK;
  basis.cameraWorld[1] = NEATRACING_CHASE_CAMERA_HEIGHT;
  basis.cameraWorld[2] =
    carSceneZ - sceneForwardZ * NEATRACING_CHASE_CAMERA_BACK;
  basis.lookTarget[0] =
    carX + forwardX * NEATRACING_CHASE_CAMERA_LOOKAHEAD;
  basis.lookTarget[1] = NEATRACING_CHASE_CAMERA_HEIGHT / 2;
  basis.lookTarget[2] =
    carSceneZ + sceneForwardZ * NEATRACING_CHASE_CAMERA_LOOKAHEAD;
  const toTargetX = basis.lookTarget[0] - basis.cameraWorld[0];
  const toTargetY = basis.lookTarget[1] - basis.cameraWorld[1];
  const toTargetZ = basis.lookTarget[2] - basis.cameraWorld[2];
  const forwardDistance = Math.hypot(toTargetX, toTargetY, toTargetZ) || 1;
  basis.forwardVector[0] = toTargetX / forwardDistance;
  basis.forwardVector[1] = toTargetY / forwardDistance;
  basis.forwardVector[2] = toTargetZ / forwardDistance;
  return basis;
}
