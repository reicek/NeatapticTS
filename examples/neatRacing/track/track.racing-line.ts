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
 * the peak at the apex, where the curvature peaks.
 *
 * THE CLAMP (Step 14V, AC-095): the offset is clamped against the SEAM's
 * ASYMMETRIC half-width PAIR - the OUTER half-width (the inner
 * half-width plus the corner-measured drift space) on the OUTER side of
 * the loop and the INNER half-width on the INNER side - so the AI may use
 * the drift room on the OUTSIDE of a corner and NEVER on the inside. The
 * RETIRED symmetric `sample.width / 2` clamp is DELETED (no shim, no
 * deferred cleanup): the road never narrows, so the inner bound is the
 * half-width the seam publishes and the outer bound is that half-width
 * plus the drift space the corner measure earns.
 *
 * THE SEAM (Step 14V, AC-095): the clamp reads the SAME edge the road
 * reads - the seam published by `examples/neatRacing/track/track.edge.ts`,
 * the edge the rasterizer, the band paint and the containment bound all
 * read - so there is NO second edge anywhere in the repo. The racing line
 * resolves that seam through the closed-loop chord ring the frozen
 * samples describe: the CONTROL ring when the samples name their control
 * segments (the production layout - the reconstruction is BIT-IDENTICAL to
 * the spec's own chord list, because the first sample of a control
 * segment sits exactly on its control point), and the samples' OWN ring
 * when they name a single segment (a sample-only fixture, whose own ring
 * is the only ring it describes).
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

import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';
import {
  resolveTrackEdgeProfile,
  resolveTrackSplineEdges,
  type TrackSplineEdge,
} from './track.edge';
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
   * counter-clockwise corner, and never larger than the SEAM's
   * half-width that applies on its side - the OUTER half-width on the
   * outer side of the loop, the INNER half-width on the inner side
   * (Step 14V, AC-095).
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
    previousToCurrentX * previousToNextY - previousToCurrentY * previousToNextX;
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
 * The closed-loop chord ring the racing line reads the SEAM through.
 *
 * THE RING IS RECONSTRUCTED FROM THE SAMPLES: the racing line is a pure
 * function of the frozen sample list (AC-086), so it resolves the seam
 * from the chord ring those samples describe - never from a second
 * geometry.
 */
interface RacingLineSeamRing {
  /** The closed-loop chords the seam profile is resolved from. */
  readonly segments: readonly TrackSegment[];
  /**
   * The samples the seam's per-sample publication is resolved for, each
   * one naming its OWNING CHORD index.
   */
  readonly samples: readonly SplineSample[];
}

/**
 * Builds the CONTROL chord ring the samples describe.
 *
 * THE RECONSTRUCTION IS BIT-IDENTICAL to the spec's own chord list: the
 * first sample of a control segment carries an interpolation factor of
 * ZERO, so it sits EXACTLY on that segment's control point, and the
 * control chord of segment `k` runs from control point `k` to control
 * point `k + 1` with the width that segment publishes.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The control chord ring and the samples that name it.
 */
function buildControlChordRing(
  samples: readonly SplineSample[],
): RacingLineSeamRing {
  const controlPointBySegmentIndex = new Map<number, SplineSample>();
  for (const sample of samples) {
    if (!controlPointBySegmentIndex.has(sample.segmentIndex)) {
      controlPointBySegmentIndex.set(sample.segmentIndex, sample);
    }
  }

  const chordIndexBySegmentIndex = new Map<number, number>();
  const controlPoints: SplineSample[] = [];
  const segmentIndices = [...controlPointBySegmentIndex.keys()].sort(
    (left, right) => left - right,
  );
  for (const segmentIndex of segmentIndices) {
    const controlPoint = controlPointBySegmentIndex.get(segmentIndex);
    if (controlPoint === undefined) {
      continue;
    }
    chordIndexBySegmentIndex.set(segmentIndex, controlPoints.length);
    controlPoints.push(controlPoint);
  }

  const count = controlPoints.length;
  const segments: TrackSegment[] = [];
  for (let index = 0; index < count; index += 1) {
    const controlPoint = controlPoints[index];
    const nextControlPoint = controlPoints[(index + 1) % count];
    segments.push({
      startX: controlPoint.x,
      startY: controlPoint.y,
      endX: nextControlPoint.x,
      endY: nextControlPoint.y,
      width: controlPoint.width,
    });
  }

  return {
    segments,
    samples: samples.map((sample) => ({
      ...sample,
      segmentIndex: chordIndexBySegmentIndex.get(sample.segmentIndex) ?? 0,
    })),
  };
}

/**
 * Builds the SAMPLE chord ring: one chord per consecutive sample pair,
 * each sample owning the chord that STARTS at it.
 *
 * A fixture whose samples name a SINGLE control segment describes no
 * control ring of its own, so the samples' OWN ring is the only ring it
 * describes and the only ring the seam can be read through.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The sample chord ring and the samples that name it.
 */
function buildSampleChordRing(
  samples: readonly SplineSample[],
): RacingLineSeamRing {
  const count = samples.length;
  const segments: TrackSegment[] = [];
  for (let index = 0; index < count; index += 1) {
    const sample = samples[index];
    const nextSample = samples[(index + 1) % count];
    segments.push({
      startX: sample.x,
      startY: sample.y,
      endX: nextSample.x,
      endY: nextSample.y,
      width: sample.width,
    });
  }

  return {
    segments,
    samples: samples.map((sample, index) => ({
      ...sample,
      segmentIndex: index,
    })),
  };
}

/**
 * Resolves the chord ring the seam is read through.
 *
 * THE CONTROL RING WHEN THE SAMPLES NAME THEIR CONTROL SEGMENTS: the
 * production layout, whose reconstruction is bit-identical to the spec's
 * own chord list. THE SAMPLES' OWN RING OTHERWISE: a fixture whose
 * samples name a single segment describes no control ring of its own.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The chord ring the seam profile is resolved from.
 */
function resolveRacingLineChordRing(
  samples: readonly SplineSample[],
): RacingLineSeamRing {
  const namedSegmentIndices = new Set<number>();
  for (const sample of samples) {
    namedSegmentIndices.add(sample.segmentIndex);
  }
  if (namedSegmentIndices.size < 2) {
    return buildSampleChordRing(samples);
  }
  return buildControlChordRing(samples);
}

/**
 * The SEAM the racing line is clamped by: the published half-width pair
 * of every sample and the loop-global outer-side sign.
 */
interface RacingLineSeam {
  /** The published half-width pair of every sample, in sample order. */
  readonly edges: readonly TrackSplineEdge[];
  /** The loop-global sign of the OUTER side (`-1` or `+1`). */
  readonly outerSideSign: 1 | -1;
}

/**
 * Resolves the SEAM the racing line is clamped by.
 *
 * THE ONE EDGE: the profile is resolved from the chord ring the samples
 * describe and the per-sample edges from THAT profile, so the racing
 * line is bounded by the SAME edge the rasterizer, the band paint and
 * the containment bound read.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The per-sample half-width pairs and the outer-side sign.
 */
function resolveRacingLineSeam(
  samples: readonly SplineSample[],
): RacingLineSeam {
  const ring = resolveRacingLineChordRing(samples);
  const profile = resolveTrackEdgeProfile(ring.segments);
  return {
    edges: resolveTrackSplineEdges(ring.samples, profile),
    outerSideSign: profile.outerSideSign,
  };
}

/**
 * Clamps a signed lateral offset into the SEAM's half-width PAIR.
 *
 * THE PAIR, NOT THE SYMMETRIC HALF-WIDTH: the OUTER half-width (the
 * inner half-width plus the corner-measured drift space) bounds the
 * offset on the OUTER side of the loop and the INNER half-width bounds
 * it on the INNER side, so the AI may use the drift room on the outside
 * of a corner and NEVER on the inside.
 *
 * THE SIDE IS LOOP-GLOBAL: the side is resolved from the seam's own
 * loop-global outer-side sign (the winding of the WHOLE loop), exactly
 * like the seam's own side resolver, so the bound NEVER flips at a
 * straight.
 *
 * @param offsetM - The unclamped signed offset (m).
 * @param edge - The published half-width pair of the sample.
 * @param outerSideSign - The loop-global sign of the outer side.
 * @returns The clamped signed offset (m).
 */
function clampOffsetToSeamPair(
  offsetM: number,
  edge: TrackSplineEdge,
  outerSideSign: 1 | -1,
): number {
  const boundM =
    Math.sign(offsetM) === outerSideSign
      ? edge.outerHalfWidthM
      : edge.innerHalfWidthM;
  if (offsetM > boundM) {
    return boundM;
  }
  if (offsetM < -boundM) {
    return -boundM;
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
 * corner, EXACTLY zero on a straight, clamped to the SEAM's half-width
 * PAIR (Step 14V, AC-095) and invariant for a fixed track layout.
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
  const seam = resolveRacingLineSeam(samples);
  const line: TrackRacingLinePoint[] = [];
  for (let index = 0; index < count; index += 1) {
    const smoothedCurvaturePerM = curvaturePerM[index];
    const offsetM = clampOffsetToSeamPair(
      NEATRACING_RACING_LINE_APEX_BIAS_GAIN_M2 *
        excessCurvaturePerM(smoothedCurvaturePerM),
      seam.edges[index],
      seam.outerSideSign,
    );
    line.push({
      offsetM,
      curvaturePerM: smoothedCurvaturePerM,
      targetSpeedUnitsPerS: targetSpeedUnitsPerS(smoothedCurvaturePerM),
    });
  }
  return line;
}
