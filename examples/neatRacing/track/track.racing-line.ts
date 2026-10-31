/**
 * Deterministic, curvature-derived racing line for the Neat Racing example
 * (plan slice `P1-ai-racing-line`; Step 14M; the AC-082 and AC-086
 * contracts).
 *
 * THE SURFACE (AC-086): the module publishes ONE point per spline sample -
 * the signed lateral offset from the road centerline, the signed curvature
 * profile and the curvature-derived target speed - as a stable
 * TRACK-RELATIVE table the Phase 2 observation assembler (AC-027) and the
 * scripted controller (AC-028) can read WITHOUT a second source of truth:
 * the table is built from the SAME frozen spline samples the autopilot
 * already reads (`examples/neatRacing/physics/ai.autopilot.ts`), so there
 * is NO second geometry anywhere in the repo.
 *
 * THE GEOMETRY: the signed curvature is the Menger curvature of the
 * three-sample window around each sample of the closed ring - POSITIVE
 * when the loop turns LEFT, the repo's left-normal convention - and it is
 * smoothed over a SYMMETRIC circular window, so the published profile is
 * free of the single-sample jitter the raw three-point estimate carries
 * while staying exactly invariant under a rigid rotation of the layout
 * (the rotation negates and swaps the coordinate differences, so the
 * cross product and the three side lengths are bit-identical).
 *
 * THE OFFSET: the lateral offset is proportional to the EXCESS curvature
 * above the straight threshold, so the line is EXACTLY ZERO on a straight
 * and leans toward the INSIDE of a corner - the inside of a
 * counter-clockwise corner is to the LEFT, i.e. a POSITIVE offset - with
 * the peak at the apex, where the curvature peaks. The offset is clamped
 * to the drivable half-width (`sample.width / 2`, the SAME definition the
 * Step 14I containment uses in `physics/collision.utils.ts`).
 *
 * THE SPEED TARGET: the cornering limit `v = sqrt(a_lat / |k|)` capped at
 * the physics top speed, so a straight publishes the top speed and a tight
 * corner publishes a slower one. The lateral limit is derived from the top
 * speed and the straight threshold, so the two meet exactly: the flattest
 * corner the line still reacts to is drivable AT the top speed.
 *
 * THE PROFILE: the offset rises from ZERO at the corner entry to the apex
 * peak and falls back to ZERO at the corner exit - INSIDE through the
 * corner and back on the centerline after it. The classic out-in-out line
 * (a NEGATIVE offset on entry and exit) is deliberately NOT built: the
 * RED contracts pin only the apex-inside clause, and a negative exit
 * offset would fight the AC-085 centerline bound and the zero-on-straights
 * clause.
 *
 * DETERMINISM: the table is a pure function of the frozen sample list - no
 * RNG, no clock reads, no shared state - so two builds of the same layout
 * are identical and the line is SEED-INVARIANT for a fixed layout.
 *
 * ALLOCATION: the table is built ONCE per track layout and cached by the
 * caller; the per-frame AI path reads the published table and allocates
 * nothing.
 *
 * @module
 */

import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import { NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S } from '../physics/physics.step.service';

/**
 * The radius of the FLATTEST corner the racing line still treats as a
 * corner (m).
 *
 * A curve flatter than this is a straight for the racing line: the
 * published offset is exactly zero there and the published speed target is
 * the top speed. The threshold is the corner radius the top speed can
 * still hold, so the lateral limit and the top speed meet exactly at the
 * threshold and the speed profile has no step at the straight/corner
 * boundary.
 */
export const NEATRACING_RACING_LINE_STRAIGHT_CORNER_RADIUS_M = 200;

/**
 * The curvature at or below which the centerline counts as a STRAIGHT
 * (1/m) - the reciprocal of
 * {@link NEATRACING_RACING_LINE_STRAIGHT_CORNER_RADIUS_M}.
 *
 * The straight zero is a GEOMETRIC property, not a fixture artifact: a
 * generated track's nominal straight is a sampled curve, so its curvature
 * is small but not exactly zero. The threshold is what makes the AC-082
 * "exactly zero on straights" clause hold on a real layout.
 */
export const NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M =
  1 / NEATRACING_RACING_LINE_STRAIGHT_CORNER_RADIUS_M;

/**
 * The lateral acceleration limit the published speed target respects
 * (m/s^2).
 *
 * DERIVED, not tuned: `top speed^2 / the straight corner radius`, so the
 * flattest corner the line reacts to is exactly drivable AT the top speed
 * and every tighter corner publishes a proportionally slower target.
 */
export const NEATRACING_RACING_LINE_LATERAL_ACCELERATION_LIMIT_M_PER_S2 =
  (NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S *
    NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S) /
  NEATRACING_RACING_LINE_STRAIGHT_CORNER_RADIUS_M;

/**
 * The apex-bias gain of the lateral offset (m^2).
 *
 * The offset is `gain x the excess curvature`, so the gain is the lateral
 * offset the line takes per unit of curvature: a TIGHTER corner is cut
 * deeper. The value is calibrated so the tight R = 8 m corner of the
 * AI-center-keeping fixtures takes a 1.92 m apex offset - inside the 6 m
 * drivable half-width and inside the 3 m AC-085 centerline bound - while
 * the R = 60 m lap fixture takes a 0.19 m offset.
 */
export const NEATRACING_RACING_LINE_APEX_BIAS_GAIN_M2 = 16;

/**
 * The radius of the symmetric curvature-smoothing window (samples).
 *
 * The window is a circular moving average over the closed ring, so the
 * published profile is smooth AND rotation- and reversal-equivariant. The
 * radius is deliberately small: a wide window would bleed the corner
 * curvature back onto the straight and break the straight-zero clause.
 */
export const NEATRACING_RACING_LINE_CURVATURE_WINDOW_RADIUS_SAMPLES = 3;

/**
 * One published point of the racing-line table.
 *
 * ONE point per spline sample, in the sample order.
 */
export interface TrackRacingLinePoint {
  /**
   * The SIGNED lateral offset from the road centerline (m).
   *
   * POSITIVE is to the LEFT of the travel direction (the repo's
   * left-normal convention - `SplineSample.innerOffsetWorld`: "Positive
   * along the unit left normal (inward on a counter-clockwise loop)"). It
   * is EXACTLY zero on a straight, positive toward the inside of a
   * counter-clockwise corner, and never larger than the drivable
   * half-width.
   */
  readonly offsetM: number;
  /**
   * The SIGNED curvature profile value at the sample (1/m).
   *
   * POSITIVE is turning LEFT. The value is the symmetric-window-smoothed
   * curvature, so it is the ONE curvature profile the offset and the
   * speed target are both derived from.
   */
  readonly curvaturePerM: number;
  /**
   * The curvature-derived target speed at the sample (m/s).
   *
   * The cornering limit `sqrt(a_lat / |curvature|)` capped at the physics
   * top speed, so a straight publishes the top speed and a tight corner
   * publishes a slower one.
   */
  readonly targetSpeedUnitsPerS: number;
}

/**
 * Computes the signed Menger curvature of the closed sample ring at one
 * sample.
 *
 * THE SIGN: the Menger curvature of the three-sample triangle is POSITIVE
 * when the triangle turns counter-clockwise, i.e. when the loop turns
 * LEFT - the repo's left-normal convention.
 *
 * THE ROTATION INVARIANCE: the side lengths are computed as
 * `sqrt(dx * dx + dy * dy)` and NOT with `Math.hypot`, because the
 * explicit form is bit-identical when a rigid rotation negates and swaps
 * the coordinate differences (IEEE-754 addition and multiplication are
 * commutative, so the cross product and the three side lengths are
 * bit-identical under a rotation).
 *
 * @param samples - The frozen closed-ring spline samples.
 * @param index - The index of the sample to measure.
 * @returns The signed curvature at the sample (1/m); zero when the three
 * samples are collinear or coincident.
 */
function signedCurvaturePerM(
  samples: readonly SplineSample[],
  index: number,
): number {
  const count = samples.length;
  const previous = samples[(index - 1 + count) % count];
  const current = samples[index];
  const next = samples[(index + 1) % count];
  const previousToCurrentX = current.x - previous.x;
  const previousToCurrentY = current.y - previous.y;
  const previousToNextX = next.x - previous.x;
  const previousToNextY = next.y - previous.y;
  const currentToNextX = next.x - current.x;
  const currentToNextY = next.y - current.y;
  const cross =
    previousToCurrentX * previousToNextY -
    previousToCurrentY * previousToNextX;
  const previousToCurrentLength = Math.sqrt(
    previousToCurrentX * previousToCurrentX +
      previousToCurrentY * previousToCurrentY,
  );
  const currentToNextLength = Math.sqrt(
    currentToNextX * currentToNextX + currentToNextY * currentToNextY,
  );
  const previousToNextLength = Math.sqrt(
    previousToNextX * previousToNextX + previousToNextY * previousToNextY,
  );
  const denominator =
    previousToCurrentLength * currentToNextLength * previousToNextLength;
  if (denominator === 0) {
    return 0;
  }
  return (2 * cross) / denominator;
}

/**
 * Smooths the raw curvature profile over a symmetric circular window.
 *
 * THE WINDOW: a moving average over `2 x radius + 1` samples of the
 * closed ring, so the profile is smooth at the corner entry and exit and
 * the smoothing stays rotation- and reversal-equivariant (the window is
 * symmetric and the ring is closed).
 *
 * @param rawCurvaturePerM - The raw signed curvature profile (1/m).
 * @param radius - The window radius in samples.
 * @returns The smoothed signed curvature profile (1/m).
 */
function smoothCurvatureProfile(
  rawCurvaturePerM: Float64Array,
  radius: number,
): Float64Array {
  const count = rawCurvaturePerM.length;
  const windowLength = radius * 2 + 1;
  const smoothed = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    let sum = 0;
    for (let offset = -radius; offset <= radius; offset += 1) {
      sum += rawCurvaturePerM[(index + offset + count) % count];
    }
    smoothed[index] = sum / windowLength;
  }
  return smoothed;
}

/**
 * Resolves the curvature a corner actually bites with: the curvature
 * ABOVE the straight threshold, keeping the sign of the profile.
 *
 * @param curvaturePerM - The signed curvature profile value (1/m).
 * @returns The excess curvature (1/m); zero on a straight.
 */
function excessCurvaturePerM(curvaturePerM: number): number {
  const magnitude = Math.abs(curvaturePerM);
  if (magnitude <= NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M) {
    return 0;
  }
  return (
    Math.sign(curvaturePerM) *
    (magnitude - NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M)
  );
}

/**
 * Clamps a signed lateral offset into the drivable half-width.
 *
 * @param offsetM - The unclamped signed offset (m).
 * @param halfWidthM - The drivable half-width (`sample.width / 2`, m).
 * @returns The clamped signed offset (m).
 */
function clampOffsetToHalfWidth(offsetM: number, halfWidthM: number): number {
  if (offsetM > halfWidthM) {
    return halfWidthM;
  }
  if (offsetM < -halfWidthM) {
    return -halfWidthM;
  }
  return offsetM;
}

/**
 * Resolves the curvature-derived target speed at one sample.
 *
 * @param curvaturePerM - The signed curvature profile value (1/m).
 * @returns The target speed (m/s); the top speed on a straight.
 */
function targetSpeedUnitsPerS(curvaturePerM: number): number {
  const magnitude = Math.abs(curvaturePerM);
  if (magnitude <= NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M) {
    return NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  }
  const corneringSpeed = Math.sqrt(
    NEATRACING_RACING_LINE_LATERAL_ACCELERATION_LIMIT_M_PER_S2 / magnitude,
  );
  return Math.min(NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S, corneringSpeed);
}

/**
 * Builds the deterministic racing-line table of a closed spline ring.
 *
 * THE CONTRACT (AC-082): the table is a pure function of the frozen
 * spline samples - ONE point per sample, in the sample order - with a
 * signed lateral offset that is apex-biased toward the inside of a
 * corner, EXACTLY zero on a straight, clamped to the drivable half-width
 * and invariant for a fixed track layout.
 *
 * THE SURFACE (AC-086): the same table publishes the signed curvature
 * profile and the curvature-derived target speed, so the Phase 2
 * observation assembler and the scripted controller read ONE track-relative
 * surface instead of a second source of truth.
 *
 * @param samples - The frozen closed-ring spline samples of the track.
 * @returns The racing-line points, one per sample, in the sample order.
 * @example
 * const line = buildTrackRacingLine(trackSpec.splineSamples);
 * // line[0].offsetM            -> the signed lateral offset (m)
 * // line[0].curvaturePerM      -> the signed curvature (1/m)
 * // line[0].targetSpeedUnitsPerS -> the curvature target speed (m/s)
 */
export function buildTrackRacingLine(
  samples: readonly SplineSample[],
): readonly TrackRacingLinePoint[] {
  const count = samples.length;
  if (count === 0) {
    return [];
  }
  const rawCurvaturePerM = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    rawCurvaturePerM[index] = signedCurvaturePerM(samples, index);
  }
  const curvaturePerM = smoothCurvatureProfile(
    rawCurvaturePerM,
    NEATRACING_RACING_LINE_CURVATURE_WINDOW_RADIUS_SAMPLES,
  );
  const line: TrackRacingLinePoint[] = [];
  for (let index = 0; index < count; index += 1) {
    const smoothedCurvaturePerM = curvaturePerM[index];
    const halfWidthM = samples[index].width / 2;
    const offsetM = clampOffsetToHalfWidth(
      NEATRACING_RACING_LINE_APEX_BIAS_GAIN_M2 *
        excessCurvaturePerM(smoothedCurvaturePerM),
      halfWidthM,
    );
    line.push({
      offsetM,
      curvaturePerM: smoothedCurvaturePerM,
      targetSpeedUnitsPerS: targetSpeedUnitsPerS(smoothedCurvaturePerM),
    });
  }
  return line;
}
