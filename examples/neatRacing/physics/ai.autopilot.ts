/**
 * NeatRacing AI autopilot (slice P1-physics-drive, user-gap patch batch 108;
 * Step 14M, slice `P1-ai-centerline`: AC-083..AC-086).
 *
 * The step-14B user gap: the served `/` page must open as a running race,
 * not a parked scene. Phase 1 ships one deterministic arcade autopilot that
 * drives the human car around the generated closed loop while the race-mode
 * selector is on the `ai` option; switching to `manual` hands control back
 * to the keyboard sample seam with no other contract change.
 *
 * Pure seam: the function reads only the car's kinematic snapshot and the
 * frozen track spline samples, and returns one analog step input. No
 * mutation, no RNG, no wall-clock reads, so the same (car, samples) pair
 * yields byte-identical inputs for the batch-83 replay contract.
 *
 * Steering law: chase a lookahead sample on the lane centerline ahead of
 * the nearest spline sample, steer proportional to the wrapped heading
 * error (positive error steers counterclockwise per the step-input
 * contract), and brake when the error exceeds a named arcade threshold.
 *
 * THE RACING LINE (Step 14M, AC-083..AC-086): the autopilot now CONSUMES
 * the deterministic racing-line table of
 * `examples/neatRacing/track/track.racing-line.ts` - the SAME table the
 * Phase 2 observation assembler and the scripted controller read, so there
 * is NO second geometry in the repo:
 *
 * - the STEER TARGET is displaced by the table's SIGNED LATERAL OFFSET
 *   (positive to the LEFT of travel, exactly zero on a straight), so the
 *   AI drives the racing line instead of the raw spline point;
 * - a CENTERLINE-CORRECTING steer term is added, proportional to the
 *   car's OWN signed lateral offset to the road centerline and therefore
 *   EXACTLY zero on the centerline (the DECIDED R1 reading of AC-083);
 * - a CURVATURE FEED-FORWARD term is added, proportional to the table's
 *   published signed curvature of the road ahead and therefore exactly
 *   zero on a straight, so the AI commands the steer the corner ahead
 *   actually needs (the arcade model turns at a speed-proportional rate,
 *   so the pure-pursuit chase alone under-steers every corner);
 * - the THROTTLE/BRAKE follow the table's CURVATURE-DERIVED TARGET SPEED,
 *   so the AI slows for the corners ahead instead of entering them at the
 *   top speed (that published target is a BRAKING INTENT - the 12 m
 *   window is shorter than the braking distance, so the brake is issued
 *   before the corner and the car enters it above the target);
 *
 * THE RETAINED LAW: the 12 m lookahead and the 2.2 rad^-1 heading-error
 * gain are UNCHANGED, the named `NEATRACING_AUTOPILOT_*` constants keep
 * their names, and the emergent speed behaviour on a straight is the same
 * as before (full throttle, no brake, at the top speed).
 *
 * Turn-rate channel (Step 14T, AC-093): the autopilot drives the physics
 * through the DEDICATED turn-rate-percent channel, never through the user
 * steer axis. The percent it sets is the SIGNED, BEHAVIOUR-PRESERVING
 * re-expression of the rate the AI already asked for
 * (`100 x steerAI x speedPercent`), so the physics resolves
 * `rate = STEERING_RATE x (percent / 100)` - the AI keeps the FULL range
 * (a percent above 100 commands a rate above the nominal rate) and is
 * EXEMPT from the user steer-axis softening factor.
 *
 * @module
 */

import {
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_STEERING_RATE,
  type PhysicsCarState,
  type PhysicsStepInput,
} from './physics.step.service';
import {
  buildTrackRacingLine,
  type TrackRacingLinePoint,
} from '../track/track.racing-line';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';

/** Lookahead distance along the loop from the nearest spline sample (m). */
export const NEATRACING_AUTOPILOT_LOOKAHEAD_M = 12;

/** Proportional steering gain from wrapped heading error (rad^-1). */
export const NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD = 2.2;

/** Full-throttle heading-error threshold below which the car holds throttle. */
export const NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD = 0.35;

/** Full-brake heading-error threshold above which the car brakes. */
export const NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD = 0.7;

/**
 * The centerline-correcting steer gain from the signed lateral offset
 * (1/m).
 *
 * THE CENTERLINE-CORRECTING TERM (Step 14M, AC-083): the term is
 * `-gain x the signed lateral offset to the road centerline` (the offset
 * is POSITIVE to the LEFT of travel), so it is EXACTLY zero on the
 * centerline and always steers the car BACK toward the centerline.
 *
 * THE GAIN IS DERIVED, NOT TUNED: it is the RETAINED 2.2 rad^-1 heading
 * gain divided by the RETAINED 12 m lookahead, so the correction is the
 * lateral offset expressed as the heading error the retained lookahead
 * would produce (`offset / lookahead`) and scaled by the retained gain.
 * The classic Stanley cross-track term therefore reuses the two
 * constants the slice RETAINS, and the term stays exactly zero on the
 * centerline because the offset is.
 */
export const NEATRACING_AUTOPILOT_CENTERLINE_STEER_GAIN_PER_M =
  NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD / NEATRACING_AUTOPILOT_LOOKAHEAD_M;

/**
 * The curvature feed-forward gain of the racing line (m).
 *
 * THE FEED-FORWARD (Step 14M, AC-083..AC-085): the racing line publishes
 * the SIGNED curvature of the road ahead, and the physics realizes a turn
 * radius of `max speed / (steering rate x steer)`, so the steer command
 * that HOLDS a corner of radius `R` is `steer = (max speed / steering
 * rate) x curvature`. The gain is therefore the model's own minimum turn
 * radius, DERIVED from the physics constants - never a tuned number.
 *
 * WHY THE CHASE ALONE IS NOT ENOUGH (MEASURED, not a formula): the
 * arcade model turns at a rate proportional to the speed, so the chase's
 * CONTROL SCALE is `max speed / (steering rate x the 2.2 rad^-1 gain)` =
 * `89 / (2 x 2.2)` = 20.227 m - wider than the retained 12 m lookahead.
 * That scale is a CONTROL magnitude, NOT a curvature law: the
 * pure-pursuit `L^2 / (2R)` offset formula does NOT describe this
 * proportional-heading law, so NO offset is derived from it. The offset
 * was MEASURED instead: on the R = 60 m lap fixture the retained chase
 * law alone settles at a STEADY state of 2.6368 m (inside the 3 m
 * AC-085 bound) but peaks at 3.3230 m on the corner-entry TRANSIENT
 * (outside it), and the PRE-14M model on the same fixture read 6.000 m -
 * the containment's own drivable edge. The feed-forward supplies the
 * corner's own steer command, so the AI holds the corner instead of
 * under-steering into the containment edge.
 *
 * THE ZERO: the published curvature is EXACTLY zero on a straight, so the
 * term adds nothing on a straight and the emergent straight-line
 * behaviour is unchanged.
 */
export const NEATRACING_AUTOPILOT_CURVATURE_FEEDFORWARD_GAIN_M =
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / NEATRACING_PHYSICS_STEERING_RATE;

/**
 * The brake gain from the relative speed excess over the curvature-derived
 * target speed (1).
 *
 * THE PROPORTIONAL BRAKE (Step 14M, AC-084): the brake is the speed
 * excess over the target speed as a RATIO of that target, so the AI
 * brakes in proportion to how far it is over the target and the brake
 * saturates at the full axis when the car is twice the target speed.
 */
export const NEATRACING_AUTOPILOT_BRAKE_GAIN_PER_SPEED_RATIO = 1;

/** Autopilot throttle axis under steering control (`[0, 1]`). */
export const NEATRACING_AUTOPILOT_THROTTLE = 1;

/** Autopilot brake axis under steering control (`[0, 1]`). */
export const NEATRACING_AUTOPILOT_BRAKE = 1;

/**
 * The percent scale of the AI turn-rate channel (percent per unit of the
 * signed AI turn command).
 *
 * A percent of `100` commands the NOMINAL turn rate (the steering rate),
 * a percent of `0` commands no turn at all and a percent above `100`
 * commands a rate above the nominal one - the AI keeps the full rate
 * range, because the AI channel is never softened.
 */
export const NEATRACING_AUTOPILOT_TURN_RATE_PERCENT_SCALE = 100;

/**
 * Neutral step input for degenerate (empty or zero-width) sample lists.
 *
 * The neutral input OWNS the turn-rate channel too: the AI percent is
 * `0`, so the degenerate lane commands NO turn through the AI channel
 * (and never falls back to the user channel by accident).
 */
const NEUTRAL_STEP_INPUT: PhysicsStepInput = {
  throttle: 0,
  brake: 0,
  steer: 0,
  turnRatePercent: 0,
};

/** Minimal kinematic snapshot consumed by the autopilot step (no RNG fields). */
export type AutopilotCarSnapshot = Pick<
  PhysicsCarState,
  'x' | 'y' | 'heading' | 'speed'
>;

/**
 * The per-layout racing-line MEMO (the zero per-frame allocation seam).
 *
 * THE MEMO: the racing-line table is a PURE function of the frozen sample
 * list, so it is built ONCE per track layout and reused for every frame -
 * the per-frame AI path reads the published table and allocates NO TABLE
 * (the per-frame step still builds its own small step-input object and
 * rescans the sample list).
 * The memo is a WeakMap, so a layout is released together with its sample
 * list and the module never keeps a track alive.
 */
const RACING_LINE_MEMO = new WeakMap<
  readonly SplineSample[],
  readonly TrackRacingLinePoint[]
>();

/**
 * Resolves the racing-line table of a track layout (AC-086).
 *
 * THE TRACK-RELATIVE SURFACE: the table is the ONE published surface of
 * the signed lateral offset, the signed curvature profile and the
 * curvature-derived target speed, so the Phase 2 observation assembler
 * (AC-027) and the scripted controller (AC-028) read the SAME table the
 * autopilot drives on - there is NO second geometry and NO second source
 * of truth anywhere in the repo.
 *
 * THE MEMO: the table is built once per layout and cached against the
 * frozen sample list, so the hot path never rebuilds it and never
 * allocates a TABLE (the per-frame step still builds its own small
 * step-input object and rescans the sample list).
 *
 * @param splineSamples - Frozen track spline samples (ordered closed loop).
 * @returns The racing-line points, one per sample, in the sample order.
 * @example
 * const line = resolveAutopilotRacingLine(trackSpec.splineSamples);
 * // line[0].targetSpeedUnitsPerS -> the curvature target speed (m/s)
 */
export function resolveAutopilotRacingLine(
  splineSamples: readonly SplineSample[],
): readonly TrackRacingLinePoint[] {
  const memoized = RACING_LINE_MEMO.get(splineSamples);
  if (memoized !== undefined) {
    return memoized;
  }
  const built = buildTrackRacingLine(splineSamples);
  RACING_LINE_MEMO.set(splineSamples, built);
  return built;
}

/**
 * Wraps an angle into `[-π, π)` via the atan2 half-plane projection so
 * steering errors stay continuous across the ±π discontinuity.
 *
 * @param radians - Raw angle difference in radians.
 * @returns Angle wrapped into `[-π, π)`.
 */
function wrapSignedAngle(radians: number): number {
  return Math.atan2(Math.sin(radians), Math.cos(radians));
}

/**
 * Clamps a steering command into the analog axis range.
 *
 * @param value - Raw steering command.
 * @returns Value clamped into `[-1, 1]`.
 */
function clampSteer(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

/**
 * Clamps a control ratio into the closed unit interval.
 *
 * @param value - Raw ratio.
 * @returns Value clamped into `[0, 1]`.
 */
function clampRatio(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * Resolves the normalized speed percentile of a snapshot speed.
 *
 * THE SAME PERCENTILE (Step 14T, AC-093): the AI reads the speed
 * percentile the physics service resolves for its own laws - the same
 * ratio, clamped the same way - so the AI's behaviour-preserving
 * re-expression resolves to exactly the turn rate the AI commanded
 * before the channel was adopted.
 *
 * @param speed - The car speed in units per second.
 * @returns The normalized speed in `[0, 1]`.
 */
function resolveSpeedPercent(speed: number): number {
  return clampRatio(speed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);
}

/**
 * Resolves the SIGNED turn-rate percent the autopilot commands.
 *
 * THE SIGNED RE-EXPRESSION (Step 14T, AC-093): the percent is the signed
 * re-expression of the turn the AI's own heading-error law already asks
 * for - `100 x steerAI x speedPercent` - so the physics service resolves
 * `rate = STEERING_RATE x (percent / 100)`.
 *
 * THE EXEMPTION: because the percent is a DEFINED channel, the service
 * resolves the AI rate from the percent ALONE, so the AI is EXEMPT from
 * the user steer-axis softening factor and keeps the full rate range (a
 * percent above 100 commands a rate above the nominal rate).
 *
 * @param steer - The clamped AI steering command in `[-1, 1]`.
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @returns The signed turn-rate percent.
 */
function resolveTurnRatePercent(steer: number, speedPercent: number): number {
  return NEATRACING_AUTOPILOT_TURN_RATE_PERCENT_SCALE * steer * speedPercent;
}

/**
 * Resolves the index of the spline sample nearest to the car (squared
 * distance scan over the frozen list; the sample count is small enough
 * that a linear scan beats a maintained spatial index for this arcade
 * autopilot).
 *
 * @param samples - Frozen track spline samples.
 * @param x - Car world x.
 * @param y - Car world y.
 * @returns The nearest sample index (0 when the list is empty).
 */
function nearestSampleIndex(
  samples: readonly SplineSample[],
  x: number,
  y: number,
): number {
  let best = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    const dx = sample.x - x;
    const dy = sample.y - y;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}

/**
 * Resolves the lookahead spline sample index a named distance ahead of the
 * nearest sample along the closed loop (wrapped walk accumulating the
 * centerline chord lengths).
 *
 * @param samples - Frozen track spline samples.
 * @param nearestIndex - Index of the nearest sample (the walk origin).
 * @returns The index of the lookahead sample.
 */
function lookaheadSampleIndex(
  samples: readonly SplineSample[],
  nearestIndex: number,
): number {
  const total = samples.length;
  let remaining = NEATRACING_AUTOPILOT_LOOKAHEAD_M;
  let cursor = nearestIndex;
  for (let steps = 0; steps < total; steps += 1) {
    const from = samples[cursor];
    const to = samples[(cursor + 1) % total];
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    remaining -= Math.hypot(dx, dy);
    cursor = (cursor + 1) % total;
    if (remaining <= 0) {
      break;
    }
  }
  return cursor;
}

/**
 * Measures the SIGNED lateral offset of a point from the road centerline.
 *
 * THE MEASUREMENT: the offset is the signed perpendicular distance to the
 * centerline chord at the nearest spline sample, POSITIVE to the LEFT of
 * the travel direction (the repo's left-normal convention). It is
 * measured DIRECTLY from the samples - never inferred from the road
 * containment - so a contained-but-off-center car still reads off-center.
 *
 * THE ZERO: a car that sits exactly on the centerline chord reads EXACTLY
 * zero, which is what makes the centerline-correcting steer term zero on
 * the centerline.
 *
 * @param samples - Frozen track spline samples.
 * @param nearestIndex - Index of the nearest sample (the chord origin).
 * @param x - Car world x.
 * @param y - Car world y.
 * @returns The signed lateral offset (m); positive to the LEFT.
 */
function signedLateralOffsetM(
  samples: readonly SplineSample[],
  nearestIndex: number,
  x: number,
  y: number,
): number {
  const from = samples[nearestIndex];
  const to = samples[(nearestIndex + 1) % samples.length];
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) {
    return 0;
  }
  const length = Math.sqrt(lengthSquared);
  const projection = Math.max(
    0,
    Math.min(1, ((x - from.x) * dx + (y - from.y) * dy) / lengthSquared),
  );
  const ex = x - (from.x + projection * dx);
  const ey = y - (from.y + projection * dy);
  return (ex * -dy + ey * dx) / length;
}

/**
 * Resolves the curvature-derived TARGET SPEED of the lookahead window.
 *
 * THE WINDOW: the samples from the nearest one up to (and including) the
 * lookahead one, so the TIGHTEST curvature in the window ahead of the car
 * sets the target speed - the AI slows for a corner while it is still
 * straight, not once it is already in the corner.
 *
 * THE TARGET IS A BRAKING INTENT, NOT AN ACHIEVED SPEED: the window is
 * 12 m long, while the arcade brake rate
 * (`NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2` = -89 m/s^2) needs
 * `v^2 / (2 x 89)` of distance to shed the excess - 31.1 m to drop from
 * 89 m/s to the R = 60 m fixture's 48.75 m/s target, and about 18.5 m to
 * drop to the demo track's ~68 m/s target. Both are LONGER than the
 * 12 m window, so the brake is issued about 12 m (0.135 s) BEFORE the
 * corner and the car ENTERS it above the published target. The corner
 * stays holdable - the model's minimum turn radius is 44.5 m - but the
 * published target speed is the INTENT the brake aims at, not a speed the
 * car reaches on entry. Widen the horizon by the speed-dependent braking
 * distance if an ACHIEVED speed is ever needed.
 *
 * @param racingLine - The published racing-line table of the layout.
 * @param nearestIndex - Index of the nearest sample (the window origin).
 * @param lookaheadIndex - Index of the lookahead sample (the window end).
 * @returns The target speed (m/s); the top speed on a straight.
 */
function resolveTargetSpeedUnitsPerS(
  racingLine: readonly TrackRacingLinePoint[],
  nearestIndex: number,
  lookaheadIndex: number,
): number {
  const count = racingLine.length;
  if (count === 0) {
    return NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  }
  let targetSpeedUnitsPerS = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  let index = nearestIndex;
  for (let step = 0; step <= count; step += 1) {
    const point = racingLine[index];
    if (
      point !== undefined &&
      point.targetSpeedUnitsPerS < targetSpeedUnitsPerS
    ) {
      targetSpeedUnitsPerS = point.targetSpeedUnitsPerS;
    }
    if (index === lookaheadIndex) {
      break;
    }
    index = (index + 1) % count;
  }
  return targetSpeedUnitsPerS;
}

/**
 * Computes one analog physics step input that chases the RACING LINE:
 * steer proportional to the wrapped heading error toward the racing-line
 * steer target (displaced by the signed lateral offset) MINUS the
 * centerline-correcting term, and throttle or brake from the
 * curvature-derived target speed and the named error thresholds.
 *
 * THE TURN-RATE CHANNEL (Step 14T, AC-093): the input ALWAYS carries a
 * DEFINED, signed turn-rate percent - the behaviour-preserving
 * re-expression of the AI's own turn (`100 x steerAI x speedPercent`) -
 * so the physics drives the AI through the AI channel and the AI is
 * exempt from the user steer-axis softening factor.
 *
 * THE RACING LINE (Step 14M, AC-083..AC-086): the seam reads the ONE
 * published racing-line table of the layout (memoized, so the hot path
 * allocates NO TABLE - it still builds its own small step-input object
 * and rescans the sample list) - the steer target is displaced by the
 * table's SIGNED LATERAL OFFSET, the steer command gains the CURVATURE
 * FEED-FORWARD of the road ahead, the throttle and the brake follow its
 * CURVATURE-DERIVED TARGET SPEED, and the car's own signed lateral
 * offset adds the centerline-correcting steer term (EXACTLY zero on the
 * centerline).
 *
 * Deterministic pure seam — no RNG, no clock reads, no mutation.
 *
 * @param car - Kinematic car snapshot (x, y, heading, speed).
 * @param splineSamples - Frozen track spline samples (ordered closed loop).
 * @returns One step input for the 60 Hz fixed step.
 * @example
 * const input = stepAutopilotInput(car, trackSpec.splineSamples);
 * // { throttle: 1, brake: 0, steer: 0.21, turnRatePercent: 12.3 }
 * // chasing the racing line of the loop
 */
export function stepAutopilotInput(
  car: AutopilotCarSnapshot,
  splineSamples: readonly SplineSample[],
): PhysicsStepInput {
  if (splineSamples.length === 0) {
    return NEUTRAL_STEP_INPUT;
  }
  const racingLine = resolveAutopilotRacingLine(splineSamples);
  const nearest = nearestSampleIndex(splineSamples, car.x, car.y);
  const lookaheadIndex = lookaheadSampleIndex(splineSamples, nearest);
  const target = splineSamples[lookaheadIndex];
  const offsetM = racingLine[lookaheadIndex]?.offsetM ?? 0;
  const chordTo = splineSamples[(lookaheadIndex + 1) % splineSamples.length];
  const chordX = chordTo.x - target.x;
  const chordY = chordTo.y - target.y;
  const chordLength = Math.sqrt(chordX * chordX + chordY * chordY);
  let aimX = target.x;
  let aimY = target.y;
  if (chordLength > 0) {
    // THE LEFT NORMAL of the travel direction: the racing-line offset is
    // positive to the LEFT, so the steer target moves with it.
    aimX += (-chordY / chordLength) * offsetM;
    aimY += (chordX / chordLength) * offsetM;
  }
  const desiredHeading = Math.atan2(aimY - car.y, aimX - car.x);
  const headingError = wrapSignedAngle(desiredHeading - car.heading);
  const absoluteError = Math.abs(headingError);
  const lateralOffsetM = signedLateralOffsetM(
    splineSamples,
    nearest,
    car.x,
    car.y,
  );
  const curvatureAheadPerM = racingLine[lookaheadIndex]?.curvaturePerM ?? 0;
  const steer = clampSteer(
    headingError * NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD -
      lateralOffsetM * NEATRACING_AUTOPILOT_CENTERLINE_STEER_GAIN_PER_M +
      curvatureAheadPerM * NEATRACING_AUTOPILOT_CURVATURE_FEEDFORWARD_GAIN_M,
  );
  const targetSpeedUnitsPerS = resolveTargetSpeedUnitsPerS(
    racingLine,
    nearest,
    lookaheadIndex,
  );
  const speedExcessUnitsPerS = car.speed - targetSpeedUnitsPerS;
  const speedBrake =
    speedExcessUnitsPerS > 0
      ? clampRatio(
          (speedExcessUnitsPerS / targetSpeedUnitsPerS) *
            NEATRACING_AUTOPILOT_BRAKE_GAIN_PER_SPEED_RATIO,
        )
      : 0;
  const errorBrake =
    absoluteError > NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD
      ? NEATRACING_AUTOPILOT_BRAKE
      : 0;
  const brake = Math.max(speedBrake, errorBrake);
  const throttle =
    speedExcessUnitsPerS > 0 ||
    absoluteError > NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD
      ? 0
      : NEATRACING_AUTOPILOT_THROTTLE;
  const speedPercent = resolveSpeedPercent(car.speed);
  return {
    throttle,
    brake,
    steer,
    turnRatePercent: resolveTurnRatePercent(steer, speedPercent),
  };
}
