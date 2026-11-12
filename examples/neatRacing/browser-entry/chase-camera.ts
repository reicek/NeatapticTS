/**
 * The Neat Racing chase camera (Step 14P, slice
 * `P1-drift-rewire-travel-camera`; the DRIFT-CAMERA law of Step 14X, slice
 * `P1-driftcam-law`).
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
 * THE DRIFT-CAMERA LAW (Step 14X, AC-097): the yaw aim offset is CAPPED
 * at the NAMED `NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD` - EXACTLY HALF
 * of the MEASURED pre-wave peak offset - and the cap is enforced AT THE
 * SOURCE (inside this law), so EVERY reader can only ever see the CAPPED
 * offset. The follow is DESENSITISED by the NAMED
 * `NEATRACING_CHASE_CAMERA_FOLLOW_GAIN` (HALF of the MEASURED pre-wave
 * gain), so the camera reacts LESS to the same published slide and drifts
 * into it instead of snapping onto it. The RELEASE is a CONSTANT fraction
 * of the aim (`NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION`), so the
 * RELEASE term never closes more than that fraction of the aim's value in
 * ONE tick: the MULTIPLICATIVE snap-back enabler (the pre-wave release
 * rate was MULTIPLIED by the camera's OWN offset, so the camera snapped
 * back harder the further it already sat from the nose) is DELETED at its
 * source.
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
 * The chase-camera MAXIMUM aim offset (rad): the NAMED CAP on the yaw aim
 * offset the camera may carry away from the nose heading.
 *
 * UNIT: rad (an angle). SIDE: the cap is SYMMETRIC - it bounds the
 * magnitude of the signed offset, so a left-hand slide (negative offset,
 * the camera aiming below the nose) and a right-hand slide (positive
 * offset) are bounded by the SAME value.
 *
 * MEASURED: the RED slice (P1-driftcam-red) MEASURED the pre-wave peak
 * aim offset of 0.34897858999999976 rad over a 1200-tick hold and
 * RECORDED this cap as EXACTLY HALF of that measured peak (the half is
 * the only USER-NAMED number of the wave). The cap is enforced AT THE
 * SOURCE (inside this law), so every reader - the HUD reader and the
 * browser reader - can only ever see the CAPPED offset.
 */
export const NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD = 0.17448929499999988;

/**
 * The chase-camera follow gain: how fast the camera aim closes on the
 * PUBLISHED travel direction, in rad/s per (m/s of car speed) per rad of
 * shared drift amount.
 *
 * UNIT: rad/s per (m/s * rad). SIDE: SIGNLESS - the gain scales the
 * magnitude of the published drift amount and never flips the side of the
 * slide.
 *
 * MEASURED: the RED slice (P1-driftcam-red) MEASURED the pre-wave gain
 * as 1 and RECORDED the pre-wave peak aim offset of 0.34897858999999976
 * rad; this gain is HALF of that MEASURED pre-wave gain, so the camera
 * responds DESENSITISED (a weaker pull on the published slide). The
 * follow stays PROPORTIONAL to the speed times the shared drift amount
 * (the user law S-1/S-4), never a fixed time constant.
 */
export const NEATRACING_CHASE_CAMERA_FOLLOW_GAIN = 0.5;

/**
 * The chase-camera release smooth fraction: the maximum fraction of its
 * own value the aim may close in ONE release tick.
 *
 * UNIT: a dimensionless fraction (0..1) of the aim offset. SIDE:
 * SIGNLESS - it bounds the MAGNITUDE of the release step, and the release
 * never flips the side of the aim.
 *
 * MEASURED: the RED slice (P1-driftcam-red) MEASURED the pre-wave
 * one-tick release close fraction as 0.10498288025082125 at the peak
 * offset (the pre-wave rate was MULTIPLIED by the camera's own offset -
 * the multiplicative snap-back enabler this law deletes) and DECIDED
 * this RECORDED smooth fraction below it, so the one-tick snap-back is
 * impossible.
 */
export const NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION = 0.05;

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
 * Clamps a chase-camera aim offset to the NAMED symmetric cap.
 *
 * WHY: the cap is a PUBLISHED invariant (the cap is enforced AT THE
 * SOURCE, so EVERY reader can only ever see the CAPPED offset), so every
 * path that publishes an offset - the supported aim, the next aim and the
 * HOLD path - clamps through this ONE helper and no reader can ever see an
 * over-cap offset.
 *
 * @param offsetRad - The aim offset to clamp (rad).
 * @returns The offset clamped to the NAMED cap (rad).
 */
function clampChaseCameraFollowOffsetRad(offsetRad: number): number {
  return Math.min(
    NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD,
    Math.max(-NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD, offsetRad),
  );
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
 * slide at low speed and locks onto the CAPPED slide at speed. The
 * steady-state error against the travel divergence is ZERO only while the
 * published divergence stays INSIDE the cap; when the divergence is WIDER
 * the aim HOLDS AT THE CAP and the residual steady-state error is the
 * divergence MAGNITUDE minus the cap. A fixed time constant (the retired,
 * FORBIDDEN `NEATRACING_CHASE_CAMERA_FOLLOW_TAU_S`) would leave the rate
 * unchanged when the speed doubles - the S-1 violation.
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
      clampChaseCameraFollowOffsetRad(follow.offsetRad),
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
 * THE LAW (Step 14X, AC-097): the aim is pulled toward the SUPPORTED aim
 * - the PUBLISHED travel divergence CAPPED at the NAMED
 * `NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD` - by a PROPORTIONAL rate
 * (the car SPEED times the shared drift amount, scaled by the DESENSITISED
 * named gain, never a fixed time constant), and the part of the aim the
 * published slide no longer supports is RELEASED at a CONSTANT fraction of
 * the aim.
 *
 * THE MULTIPLICATIVE SNAP-BACK ENABLER IS DELETED AT ITS SOURCE: the
 * pre-wave law fed the camera's OWN remaining divergence into the follow
 * rate (`max(|published slip|, |aim error|)`), so the release closed
 * FASTER the further the camera already sat from the nose. The rate here
 * reads the PUBLISHED slide ONLY, so the release never accelerates and the
 * one-tick snap-back is impossible.
 *
 * The per-tick change is bounded (no snap) and the aim is approached along
 * the shortest path (no overshoot, no discontinuity).
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
  // The SUPPORTED aim: the published divergence, CAPPED at the NAMED cap.
  // The cap is enforced HERE, at the source, so no reader can ever see an
  // uncapped offset.
  const targetOffsetRad = clampChaseCameraFollowOffsetRad(divergenceRad);
  // The PROPORTIONAL follow rate: the speed times the shared drift amount
  // (the published slide ONLY - the camera's own divergence is NOT part
  // of the rate, so the release is never multiplied by it).
  const followRateRadPerSecond =
    NEATRACING_CHASE_CAMERA_FOLLOW_GAIN *
    step.speedMps *
    Math.abs(divergenceRad);
  // The bounded per-tick follow fraction (no snap): the aim never jumps
  // the whole remaining gap in a single tick.
  const followFraction = Math.min(
    1,
    followRateRadPerSecond * NEATRACING_CHASE_CAMERA_FOLLOW_DT_S,
  );
  // The FOLLOW step: the aim eases toward the supported aim.
  const followStepRad = (targetOffsetRad - offsetRad) * followFraction;
  // The RESIDUAL: the part of the aim the published slide no longer
  // supports (the release). While the slide holds, the residual is zero.
  const residualRad =
    Math.sign(offsetRad) *
    Math.max(0, Math.abs(offsetRad) - Math.abs(targetOffsetRad));
  // The RELEASE step: the residual eases back at the RECORDED smooth
  // fraction - a CONSTANT fraction of the aim, so the release never
  // accelerates the further the aim already sits from the nose.
  const releaseStepRad =
    -residualRad * NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION;
  // The next aim: the follow plus the release, CLAMPED at the NAMED cap.
  const nextOffsetRad = clampChaseCameraFollowOffsetRad(
    offsetRad + followStepRad + releaseStepRad,
  );
  // The step the caller applies: the CHANGE of the aim. The caller
  // ADVANCES the aim by the returned delta, so the aim it ends up with is
  // the CLAMPED next aim above - the cap is enforced AT THE SOURCE and no
  // reader can ever see an uncapped offset.
  return nextOffsetRad - offsetRad;
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
  basis.cameraWorld[0] = carX - forwardX * NEATRACING_CHASE_CAMERA_BACK;
  basis.cameraWorld[1] = NEATRACING_CHASE_CAMERA_HEIGHT;
  basis.cameraWorld[2] =
    carSceneZ - sceneForwardZ * NEATRACING_CHASE_CAMERA_BACK;
  basis.lookTarget[0] = carX + forwardX * NEATRACING_CHASE_CAMERA_LOOKAHEAD;
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
