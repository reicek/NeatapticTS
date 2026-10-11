/**
 * NeatRacing AI autopilot (slice P1-physics-drive, user-gap patch batch 108).
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
 * @module
 */

import type { PhysicsCarState, PhysicsStepInput } from './physics.step.service';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';

/** Lookahead distance along the loop from the nearest spline sample (m). */
export const NEATRACING_AUTOPILOT_LOOKAHEAD_M = 12;

/** Proportional steering gain from wrapped heading error (rad^-1). */
export const NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD = 2.2;

/** Full-throttle heading-error threshold below which the car holds throttle. */
export const NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD = 0.35;

/** Full-brake heading-error threshold above which the car brakes. */
export const NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD = 0.7;

/** Autopilot throttle axis under steering control (`[0, 1]`). */
export const NEATRACING_AUTOPILOT_THROTTLE = 1;

/** Autopilot brake axis under steering control (`[0, 1]`). */
export const NEATRACING_AUTOPILOT_BRAKE = 1;

/** Neutral step input for degenerate (empty or zero-width) sample lists. */
const NEUTRAL_STEP_INPUT: PhysicsStepInput = {
  throttle: 0,
  brake: 0,
  steer: 0,
};

/** Minimal kinematic snapshot consumed by the autopilot step (no RNG fields). */
export type AutopilotCarSnapshot = Pick<
  PhysicsCarState,
  'x' | 'y' | 'heading' | 'speed'
>;

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
 * Computes one analog physics step input that chases the loop: steer
 * proportional to the wrapped heading error toward the lookahead sample,
 * and throttle or brake by named error thresholds.
 *
 * Deterministic pure seam — no RNG, no clock reads, no mutation.
 *
 * @param car - Kinematic car snapshot (x, y, heading, speed).
 * @param splineSamples - Frozen track spline samples (ordered closed loop).
 * @returns One step input for the 60 Hz fixed step.
 * @example
 * const input = stepAutopilotInput(car, trackSpec.splineSamples);
 * // { throttle: 1, brake: 0, steer: 0.21 } chasing the loop centerline
 */
export function stepAutopilotInput(
  car: AutopilotCarSnapshot,
  splineSamples: readonly SplineSample[],
): PhysicsStepInput {
  if (splineSamples.length === 0) {
    return NEUTRAL_STEP_INPUT;
  }
  const nearest = nearestSampleIndex(splineSamples, car.x, car.y);
  const target = splineSamples[lookaheadSampleIndex(splineSamples, nearest)];
  const desiredHeading = Math.atan2(target.y - car.y, target.x - car.x);
  const headingError = wrapSignedAngle(desiredHeading - car.heading);
  const absoluteError = Math.abs(headingError);
  return {
    throttle:
      absoluteError <= NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD
        ? NEATRACING_AUTOPILOT_THROTTLE
        : 0,
    brake:
      absoluteError > NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD
        ? NEATRACING_AUTOPILOT_BRAKE
        : 0,
    steer: clampSteer(headingError * NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD),
  };
}
