/**
 * RED-phase contract tests for the Neat Racing racing line (slice
 * `P1-ai-centerline-red`; Step 14M; the AC-082 and AC-086 contracts).
 *
 * PHASE CORRECTED 2026-10-29 (FP-14M-07): the suite is GREEN - the
 * racing-line module EXISTS and all EIGHT contracts pass (8/8). The
 * RED-phase wording below is kept as the record of HOW the contracts were
 * authored (the 14N import-fails form), so the pinned contract API stays
 * readable exactly as the RED slice wrote it.
 *
 * The AI-center-keeping wave gives the autopilot a real racing line: a
 * DETERMINISTIC, curvature-derived lateral offset from the road
 * centerline, published as a stable TRACK-RELATIVE surface the Phase 2
 * consumers can read. These contracts are the RED home of that module
 * (the GREEN home is the new module
 * `examples/neatRacing/track/track.racing-line.ts` plus the AI call site
 * in `examples/neatRacing/physics/ai.autopilot.ts`).
 *
 * RED FORM (the 14N import-fails form): the not-yet-existing module is
 * imported BY NAME, so this suite fails to compile with
 * `TS2307: Cannot find module './track.racing-line'`. That IS the red -
 * the contract module does not exist yet, and NO placeholder export is
 * authored here to make the import resolve.
 *
 * THE PINNED CONTRACT API (the green slice must export these names
 * VERBATIM):
 *
 * - `TrackRacingLinePoint` - ONE point per spline sample, in the sample
 *   order, with three named fields:
 *   - `offsetM` - the SIGNED lateral offset from the road centerline (m).
 *     POSITIVE is to the LEFT of the travel direction (the repo's
 *     left-normal convention - `SplineSample.innerOffsetWorld`: "Positive
 *     along the unit left normal (inward on a counter-clockwise loop)");
 *   - `curvaturePerM` - the SIGNED curvature at the sample (1/m); POSITIVE
 *     is turning LEFT;
 *   - `targetSpeedUnitsPerS` - the CURVATURE-DERIVED target speed (m/s)
 *     for the sample;
 * - `buildTrackRacingLine(samples)` - builds the table from the SAME
 *   frozen spline samples the AI already reads (`ai.autopilot.ts`), so
 *   there is NO second source of truth (AC-086).
 *
 * THE DECIDED CONVENTIONS (recorded in the plan; the green slice
 * inherits them):
 *
 * - THE SIGN: `offsetM` is POSITIVE to the LEFT of the travel direction.
 * - THE STRAIGHT ZERO: "EXACTLY ZERO on straights" is asserted as
 *   `|offsetM| <= 1e-9 m` (the house float tolerance - cf. the AC-072
 *   "within 1e-9 m" precedent). A hard `toBe(0)` would be brittle
 *   against the last ULP of a curvature product.
 * - THE CLAMP: the drivable half-width is `sample.width / 2` - the SAME
 *   definition the Step 14I containment uses (`collision.utils.ts`:
 *   `halfWidthM: segment.width / 2`).
 * - THE APEX: the apex of a constant-radius corner is the sample at the
 *   middle of the arc, and the AC-082 ledger row pins the corner offset as
 *   "apex-biased toward the inside of the corner".
 *
 * THE FIXTURES are deterministic: a closed stadium with two straights and
 * two R = 8 m half-circle corners, traversed counter-clockwise, plus the
 * REVERSED traversal of the same layout and a rigidly ROTATED copy of it.
 * No RNG, no timers, no wall-clock reads, no shared state - every lane
 * rebuilds its own table from the frozen samples, so there is NOTHING to
 * clean up between lanes.
 *
 * STEP 14V RED CONTRACTS (slice P1-prop-clearance-red, AC-095 / AC-096):
 * the racing-line lateral clamp must be bounded by the SEAM's ASYMMETRIC
 * half-width PAIR (the OUTER half-width on the outer side of a corner, the
 * INNER half-width on the inner side), so the AI line can use the widened
 * outer corner while the inner side is NOT widened for the AI. The lanes
 * below are RED BY ASSERTION against the CURRENT clamp (the RETIRED
 * symmetric `sample.width / 2`), and the ONE-SEAM closure (RED-4) pins
 * that the clamp is bounded by the SAME edge the containment bound reads.
 *
 * @module
 */
import { describe, expect, it } from '@jest/globals';

import {
  buildTrackRacingLine,
  NEATRACING_RACING_LINE_APEX_BIAS_GAIN_M2,
  NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M,
  type TrackRacingLinePoint,
} from './track.racing-line';
import {
  resolveNearestTrackChord,
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAtWorldZ,
  resolveTrackSplineEdges,
} from './track.edge';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';

/** The fixture road width (m) - the same 12 m road the repo fixture uses. */
const FIXTURE_ROAD_WIDTH_M = 12;

/**
 * The tight-corner radius (m).
 *
 * A TIGHT corner on purpose: the curvature of a wide corner is close to
 * the curvature of a straight, so a loose "is this a corner?" threshold
 * could classify it as a straight and make the corner lanes VACUOUS. At
 * R = 8 m the curvature is 0.125 m^-1 - far from any straight.
 */
const FIXTURE_CORNER_RADIUS_M = 8;

/** The straight length (m) of the stadium fixture. */
const FIXTURE_STRAIGHT_M = 40;

/**
 * The samples per half-circle corner.
 *
 * EVEN on purpose: the arc's midpoint (the APEX) then IS a sample, so
 * the apex lane reads a sample instead of interpolating one.
 */
const FIXTURE_ARC_SAMPLES = 24;

/** The float tolerance of the lateral-offset assertions (m). */
const OFFSET_TOLERANCE_M = 1e-9;

/** The drivable half-width of the fixture road (m) - `width / 2`. */
const DRIVABLE_HALF_WIDTH_M = FIXTURE_ROAD_WIDTH_M / 2;

/**
 * The number of MIDDLE straight samples the straight-zero lane reads.
 *
 * The middle of a long straight is far (in samples) from both corner
 * transitions, so the lane stays valid for ANY curvature window the
 * green slice chooses - it measures the straight, never a transition.
 */
const FIXTURE_STRAIGHT_MIDDLE_SAMPLE_COUNT = 10;

/**
 * The landmarks of the stadium fixture (the sample indices the lanes
 * read).
 */
interface StadiumFixture {
  /** The frozen sample list of the closed loop (in travel order). */
  readonly samples: readonly SplineSample[];
  /** The sample indices of the MIDDLE of the first (bottom) straight. */
  readonly straightSampleIndices: readonly number[];
  /** The sample index of the first corner's apex (the arc's midpoint). */
  readonly apexSampleIndex: number;
}

/**
 * Builds the closed stadium fixture: a bottom straight along +x, a right
 * half-circle, a top straight along -x and a left half-circle, all
 * traversed COUNTER-CLOCKWISE (so the inside of both corners is to the
 * LEFT of the travel direction).
 *
 * The sample spacing is uniform (~1 m on the straights, ~1.05 m on the
 * arcs), so the curvature window of the green slice sees the same
 * geometry everywhere.
 *
 * @returns The frozen fixture samples and their landmarks.
 */
const buildStadiumFixture = (): StadiumFixture => {
  const radius = FIXTURE_CORNER_RADIUS_M;
  const straightSamples = FIXTURE_STRAIGHT_M;
  const samples: SplineSample[] = [];
  const push = (x: number, y: number): void => {
    samples.push({
      x,
      y,
      width: FIXTURE_ROAD_WIDTH_M,
      segmentIndex: 0,
      sampleIndexWithinSegment: samples.length,
      globalIndex: samples.length,
    });
  };

  // The bottom straight, travelled along +x at y = -radius (the last
  // sample sits exactly on the right corner's start).
  for (let index = 0; index <= straightSamples; index += 1) {
    push(index, -radius);
  }
  // The right half-circle, centred at (straightSamples, 0).
  for (let step = 1; step <= FIXTURE_ARC_SAMPLES; step += 1) {
    const angle = -Math.PI / 2 + (step * Math.PI) / FIXTURE_ARC_SAMPLES;
    push(straightSamples + radius * Math.cos(angle), radius * Math.sin(angle));
  }
  // The top straight, travelled along -x at y = +radius (the last sample
  // sits exactly on the left corner's start).
  for (let index = 1; index <= straightSamples; index += 1) {
    push(straightSamples - index, radius);
  }
  // The left half-circle, centred at (0, 0); the loop wraps from the last
  // sample back to the first one.
  for (let step = 1; step < FIXTURE_ARC_SAMPLES; step += 1) {
    const angle = Math.PI / 2 + (step * Math.PI) / FIXTURE_ARC_SAMPLES;
    push(radius * Math.cos(angle), radius * Math.sin(angle));
  }

  const firstStraightEnd = straightSamples;
  const firstCornerStart = firstStraightEnd + 1;
  const straightMiddleStart =
    Math.floor((firstStraightEnd - FIXTURE_STRAIGHT_MIDDLE_SAMPLE_COUNT) / 2) +
    1;
  const straightSampleIndices: number[] = [];
  for (
    let offset = 0;
    offset < FIXTURE_STRAIGHT_MIDDLE_SAMPLE_COUNT;
    offset += 1
  ) {
    straightSampleIndices.push(straightMiddleStart + offset);
  }

  return {
    samples,
    straightSampleIndices,
    // The apex is the arc's midpoint (step FIXTURE_ARC_SAMPLES / 2).
    apexSampleIndex: firstCornerStart + FIXTURE_ARC_SAMPLES / 2 - 1,
  };
};

/** The frozen stadium fixture every lane reads. */
const STADIUM = buildStadiumFixture();

/**
 * Reads the largest absolute lateral offset of a racing-line table.
 *
 * @param line - The racing-line table to read.
 * @returns The largest `|offsetM|` in the table (m).
 */
const maxAbsOffsetMOf = (line: readonly TrackRacingLinePoint[]): number =>
  Math.max(...line.map((point) => Math.abs(point.offsetM)));

/**
 * The racing-line contracts (Step 14M, slice `P1-ai-centerline-red`).
 *
 * AC-082 (the racing line): the lateral offset is a DETERMINISTIC,
 * curvature-derived table - apex-biased toward the inside of a corner,
 * EXACTLY ZERO on a straight, clamped to the drivable half-width and
 * INVARIANT for a fixed track layout.
 *
 * AC-086 (the track-relative surface): the racing line AND the
 * curvature-derived speed profile are published as a stable TRACK-RELATIVE
 * surface for the Phase 2 consumers - no Phase 2 training step is started
 * here.
 */
describe('buildTrackRacingLine (Step 14M, slice P1-ai-centerline-red)', () => {
  describe('GREEN (the racing line is published and consumed by the AI)', () => {
    // [RED-14M-RL-1-ONE-POINT-PER-SAMPLE]
    it('AC-082: the racing line carries ONE point per spline sample, in the sample order (RED-14M-RL-1-ONE-POINT-PER-SAMPLE)', () => {
      // Act
      const line = buildTrackRacingLine(STADIUM.samples);

      // Assert
      expect(line.length).toBe(STADIUM.samples.length);
    });

    // [RED-14M-RL-2-STRAIGHT-ZERO]
    it('AC-082: the racing line offset is EXACTLY ZERO on the straights (RED-14M-RL-2-STRAIGHT-ZERO)', () => {
      // Act
      const line = buildTrackRacingLine(STADIUM.samples);

      // Assert
      const worstStraightOffsetM = Math.max(
        ...STADIUM.straightSampleIndices.map((index) =>
          Math.abs(line[index].offsetM),
        ),
      );
      expect(worstStraightOffsetM).toBeLessThanOrEqual(OFFSET_TOLERANCE_M);
    });

    // [RED-14M-RL-3-APEX-INSIDE]
    it('AC-082: the racing line offset at the corner APEX points to the INSIDE of the corner (RED-14M-RL-3-APEX-INSIDE)', () => {
      // Act
      const line = buildTrackRacingLine(STADIUM.samples);

      // Assert
      // The fixture is traversed counter-clockwise, so the inside of the
      // corner is to the LEFT of the travel direction (a POSITIVE offset).
      expect(line[STADIUM.apexSampleIndex].offsetM).toBeGreaterThan(0);
    });

    // [RED-14M-RL-4-TRAVEL-RELATIVE]
    it('AC-082: the racing line follows the TRAVEL direction (the reversed traversal carries the apex offset on the opposite side) (RED-14M-RL-4-TRAVEL-RELATIVE)', () => {
      // Act
      const reversedSamples = [...STADIUM.samples].reverse();
      const reversedLine = buildTrackRacingLine(reversedSamples);

      // Assert
      // The reversed loop traverses the SAME geometry in the opposite
      // direction, so the inside becomes the RIGHT (a NEGATIVE offset).
      // Only the SIGN is asserted: a forward-looking racing line is not
      // bit-mirror-symmetric, so a magnitude comparison would be
      // unsatisfiable.
      const reversedApexIndex =
        STADIUM.samples.length - 1 - STADIUM.apexSampleIndex;
      expect(reversedLine[reversedApexIndex].offsetM).toBeLessThan(0);
    });

    // [RED-14M-RL-5-CLAMPED]
    it('AC-082: the racing line offset is CLAMPED to the drivable half-width at EVERY sample (RED-14M-RL-5-CLAMPED)', () => {
      // Act
      const line = buildTrackRacingLine(STADIUM.samples);

      // Assert
      expect(maxAbsOffsetMOf(line)).toBeLessThanOrEqual(DRIVABLE_HALF_WIDTH_M);
    });

    // [RED-14M-RL-6-INVARIANT]
    it('AC-082: the racing line is INVARIANT for a fixed track layout (two builds of the same samples are identical) (RED-14M-RL-6-INVARIANT)', () => {
      // Act
      const first = buildTrackRacingLine(STADIUM.samples);
      const second = buildTrackRacingLine(STADIUM.samples);

      // Assert
      expect(second).toEqual(first);
    });

    // [RED-14M-RL-7-TRACK-RELATIVE]
    it('AC-086: the racing line is TRACK-RELATIVE (a rigidly rotated layout carries the identical offsets) (RED-14M-RL-7-TRACK-RELATIVE)', () => {
      // Act
      const line = buildTrackRacingLine(STADIUM.samples);
      const rotatedLine = buildTrackRacingLine(
        STADIUM.samples.map((sample) => ({
          ...sample,
          x: -sample.y,
          y: sample.x,
        })),
      );

      // Assert
      const worstDeltaM = Math.max(
        ...line.map((point: TrackRacingLinePoint, index: number) =>
          Math.abs(point.offsetM - rotatedLine[index].offsetM),
        ),
      );
      expect(worstDeltaM).toBeLessThanOrEqual(OFFSET_TOLERANCE_M);
    });

    // [RED-14M-RL-8-SPEED-PROFILE]
    it('AC-086: the curvature speed profile is PUBLISHED (the corner target speed is below the straight one, and it is positive) (RED-14M-RL-8-SPEED-PROFILE)', () => {
      // Act
      const line = buildTrackRacingLine(STADIUM.samples);

      // Assert
      const straightTargetSpeed =
        line[STADIUM.straightSampleIndices[0]].targetSpeedUnitsPerS;
      const cornerTargetSpeed =
        line[STADIUM.apexSampleIndex].targetSpeedUnitsPerS;
      expect(cornerTargetSpeed).toBeLessThan(straightTargetSpeed);
      expect(cornerTargetSpeed).toBeGreaterThan(0);
    });
  });
});

/**
 * The hook fixture (Step 14V, RED-5 / RED-4).
 *
 * A closed "hook" ring: a long bottom straight, a CONVEX corner, a short
 * spur, a CONCAVE corner, an outer sweep, a left sweep and the closing
 * straights. It is built for the Step 14V lanes because it carries BOTH
 * corner handednesses on ONE counter-clockwise loop, so the seam's OUTER
 * side and its INNER side are probed by the SAME table:
 *
 * - the CONVEX apex is a LEFT turn, so the racing line's offset there
 *   points to the loop's INNER side and the seam's INNER half-width
 *   bounds it (the inner side is NOT widened for the AI);
 * - the CONCAVE apex is a RIGHT turn, so the racing line's offset there
 *   points to the loop's OUTER side and the seam's OUTER half-width - the
 *   WIDENED one - bounds it (the drift room the AI may use).
 *
 * The road is 3 m wide on purpose: at the concave apex the seam publishes
 * an outer half-width of 1.625 m (the inner one is 1.5 m), and the
 * unclamped racing-line offset there is 2.5867 m, so the clamp REALLY
 * binds and the bound clause is NOT vacuous.
 */
interface HookFixture {
  /** The frozen sample list of the closed hook loop (in travel order). */
  readonly samples: readonly SplineSample[];
  /** The raw chord segments of the same closed loop. */
  readonly segments: readonly TrackSegment[];
  /** The sample index of the CONVEX apex (the inner-side corner). */
  readonly convexApexIndex: number;
  /** The sample index of the CONCAVE apex (the outer-side corner). */
  readonly concaveApexIndex: number;
}

/** The hook fixture's road width (m) - narrow, so the seam's outer bound binds. */
const HOOK_ROAD_WIDTH_M = 3;

/** The hook fixture's corner radius (m). */
const HOOK_ARC_RADIUS_M = 6;

/** The hook fixture's arc steps per corner - EVEN, so the apex IS a sample. */
const HOOK_ARC_STEPS = 8;

/** The hook fixture's straight spacing (m). */
const HOOK_STRAIGHT_SPACING_M = 4;

/**
 * Builds the closed hook fixture.
 *
 * Every sample owns ITS OWN chord (`segmentIndex` is the sample's own
 * index), so the seam's per-sample edge and its per-chord edge describe
 * the SAME place of the loop and the two granularities can be compared.
 *
 * @returns The frozen hook samples, its raw segments and its two apexes.
 */
const buildHookFixture = (): HookFixture => {
  const radius = HOOK_ARC_RADIUS_M;
  const points: { x: number; y: number }[] = [];
  const push = (x: number, y: number): void => {
    points.push({ x, y });
  };
  const pushStraight = (
    fromX: number,
    fromY: number,
    toX: number,
    toY: number,
  ): void => {
    const length = Math.hypot(toX - fromX, toY - fromY);
    const steps = Math.max(1, Math.round(length / HOOK_STRAIGHT_SPACING_M));

    for (let step = 1; step <= steps; step += 1) {
      const t = step / steps;
      push(fromX + (toX - fromX) * t, fromY + (toY - fromY) * t);
    }
  };
  const pushArc = (
    centreX: number,
    centreY: number,
    startAngleRad: number,
    sweepRad: number,
    steps: number,
  ): void => {
    for (let step = 1; step <= steps; step += 1) {
      const angle = startAngleRad + (sweepRad * step) / steps;
      push(
        centreX + radius * Math.cos(angle),
        centreY + radius * Math.sin(angle),
      );
    }
  };

  // The bottom straight, travelled along +x.
  push(-56, -20);
  pushStraight(-56, -20, 40, -20);
  // The CONVEX corner (a left turn on this counter-clockwise loop).
  const convexArcStart = points.length - 1;
  pushArc(40, -20 + radius, -Math.PI / 2, Math.PI / 2, HOOK_ARC_STEPS);
  pushStraight(40 + radius, -20 + radius, 40 + radius, 4);
  // The CONCAVE corner (a right turn - the OUTER side of the loop).
  const concaveArcStart = points.length - 1;
  pushArc(40 + 2 * radius, 4, Math.PI, -Math.PI / 2, HOOK_ARC_STEPS);
  pushStraight(40 + 2 * radius, 4 + radius, 96, 4 + radius);
  pushArc(96, 4 + 2 * radius, -Math.PI / 2, Math.PI / 2, HOOK_ARC_STEPS);
  pushStraight(96 + radius, 4 + 2 * radius, 96 + radius, 60);
  pushArc(96, 60, 0, Math.PI / 2, HOOK_ARC_STEPS);
  pushStraight(96, 60 + radius, -56, 60 + radius);
  pushArc(-56, 60, Math.PI / 2, Math.PI / 2, HOOK_ARC_STEPS);
  pushStraight(-56 - radius, 60, -56 - radius, -20 + radius);
  pushArc(-56, -20 + radius, Math.PI, Math.PI / 2, HOOK_ARC_STEPS);
  // The loop closes on its first point, so the last sample is dropped.
  points.pop();

  const samples: SplineSample[] = points.map((point, index) => ({
    x: point.x,
    y: point.y,
    width: HOOK_ROAD_WIDTH_M,
    segmentIndex: index,
    sampleIndexWithinSegment: 0,
    globalIndex: index,
  }));
  const segments: TrackSegment[] = samples.map((sample, index) => {
    const next = samples[(index + 1) % samples.length]!;

    return {
      startX: sample.x,
      startY: sample.y,
      endX: next.x,
      endY: next.y,
      width: HOOK_ROAD_WIDTH_M,
    };
  });

  return {
    samples,
    segments,
    convexApexIndex: convexArcStart + HOOK_ARC_STEPS / 2,
    concaveApexIndex: concaveArcStart + HOOK_ARC_STEPS / 2,
  };
};

/** The frozen hook fixture the Step 14V lanes read. */
const HOOK = buildHookFixture();

/**
 * Recomputes the UNCLAMPED racing-line offset of one published point.
 *
 * The clamp is applied by the module, so the value the offset would have
 * had WITHOUT the clamp is recomputed here from the PUBLISHED curvature
 * through the NAMED gain and threshold. The lanes use it to prove that
 * the probed corner REALLY binds the offset (a corner whose racing line
 * never reaches its bound would satisfy the bound clause vacuously).
 *
 * @param point - The published racing-line point to recompute.
 * @returns The unclamped signed offset (m).
 */
const unclampedOffsetMOf = (point: TrackRacingLinePoint): number => {
  const excessCurvaturePerM = Math.max(
    0,
    Math.abs(point.curvaturePerM) -
      NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M,
  );

  return (
    Math.sign(point.curvaturePerM) *
    NEATRACING_RACING_LINE_APEX_BIAS_GAIN_M2 *
    excessCurvaturePerM
  );
};

/**
 * The Step 14V racing-line contracts (slice `P1-prop-clearance-red`).
 *
 * AC-095 (the asymmetric edge - the directive 4): the racing-line lateral
 * clamp is bounded by the SEAM's ASYMMETRIC half-width PAIR - the OUTER
 * half-width on the outer side of a corner and the INNER half-width on the
 * inner side - so the AI line can use the widened outer corner. The INNER
 * side is NOT widened for the AI.
 *
 * AC-095 (the seam is closed - the ONE edge): the bound the racing line
 * is clamped by is the SAME edge the containment bound reads.
 */
describe('the racing-line clamp reads the SEAM pair (Step 14V, slice P1-prop-clearance-red, AC-095)', () => {
  describe('RED-5 (the clamp is bounded by the ASYMMETRIC pair)', () => {
    // [RED-14V-RL-5-OUTER-BOUND]
    it('AC-095: the outer-side corner is bounded by the seam OUTER half-width (RED-14V-RL-5-OUTER-BOUND)', () => {
      // Arrange
      const profile = resolveTrackEdgeProfile(HOOK.segments);
      const splineEdges = resolveTrackSplineEdges(HOOK.samples, profile);
      const line = buildTrackRacingLine(HOOK.samples);
      const apexIndex = HOOK.concaveApexIndex;
      const apexPoint = line[apexIndex]!;
      const apexEdge = splineEdges[apexIndex]!;

      // NON-VACUITY GUARD (asserted BEFORE the relational clause): the
      // probed corner REALLY binds the offset. The UNCLAMPED offset
      // (recomputed from the PUBLISHED curvature through the NAMED gain
      // and threshold) must exceed the seam's outer half-width, the seam
      // must publish an ASYMMETRIC pair there, and the probe's offset
      // must sit on the loop-global OUTER side - otherwise the bound
      // clause below would be satisfied vacuously.
      expect(apexEdge.outerHalfWidthM).toBeGreaterThan(
        apexEdge.innerHalfWidthM,
      );
      expect(Math.abs(unclampedOffsetMOf(apexPoint))).toBeGreaterThan(
        apexEdge.outerHalfWidthM,
      );
      expect(Math.sign(apexPoint.offsetM)).toBe(profile.outerSideSign);

      // Assert
      // TODAY the published offset is clamped to the RETIRED symmetric
      // `sample.width / 2` (1.500000 m), so the AI line can NOT use the
      // widened outer corner. The seam's published outer half-width at
      // this corner is 1.625000 m, and THAT is the bound the AI gets
      // once the clamp is re-based on the seam pair.
      const boundDeltaM = Math.abs(
        Math.abs(apexPoint.offsetM) - apexEdge.outerHalfWidthM,
      );
      expect(boundDeltaM).toBeLessThanOrEqual(OFFSET_TOLERANCE_M);
    });

    // [RED-14V-RL-5-SIDE-BOUND]
    it('AC-095: the published offset never exceeds the bound that applies on ITS side, and the inner side is NOT widened (RED-14V-RL-5-SIDE-BOUND)', () => {
      // Arrange
      const profile = resolveTrackEdgeProfile(HOOK.segments);
      const splineEdges = resolveTrackSplineEdges(HOOK.samples, profile);
      const line = buildTrackRacingLine(HOOK.samples);

      // Act
      const violations: string[] = [];

      line.forEach((point, index) => {
        if (point.offsetM === 0) {
          return;
        }

        const edge = splineEdges[index]!;
        const boundM =
          Math.sign(point.offsetM) === profile.outerSideSign
            ? edge.outerHalfWidthM
            : edge.innerHalfWidthM;

        if (Math.abs(point.offsetM) > boundM + OFFSET_TOLERANCE_M) {
          violations.push(
            `index=${index} offset=${point.offsetM.toFixed(9)} bound=${boundM.toFixed(9)}`,
          );
        }
      });

      // Assert - the SAFETY half of RED-5: the offset never leaves the
      // half-width that applies on its own side (GREEN today and after
      // the re-base).
      expect(violations).toEqual([]);

      // And the INNER side is NOT widened for the AI: at the convex
      // (inner-side) apex the published offset is exactly the seam's
      // INNER half-width.
      const innerApexPoint = line[HOOK.convexApexIndex]!;
      const innerApexEdge = splineEdges[HOOK.convexApexIndex]!;
      expect(
        Math.abs(
          Math.abs(innerApexPoint.offsetM) - innerApexEdge.innerHalfWidthM,
        ),
      ).toBeLessThanOrEqual(OFFSET_TOLERANCE_M);
    });
  });

  describe('RED-4 (the world reads the SAME edge as the road)', () => {
    // [RED-14V-RL-4-ONE-SEAM]
    it('AC-095: the racing line is bounded by the ONE seam edge the CONTAINMENT BOUND reads (RED-14V-RL-4-ONE-SEAM)', () => {
      // Arrange
      const profile = resolveTrackEdgeProfile(HOOK.segments);
      const line = buildTrackRacingLine(HOOK.samples);
      const apexIndex = HOOK.concaveApexIndex;
      const apexSample = HOOK.samples[apexIndex]!;
      const apexFrame = resolveSplineSampleFrame(HOOK.samples, apexIndex);
      const apexOffsetM = line[apexIndex]!.offsetM;

      // The CONTAINMENT BOUND's own reader of the ONE seam edge: the
      // NEAREST chord of the published profile, measured with that
      // chord's PUBLISHED side half-width. The probe is the racing
      // line's OWN point (the sample offset along the left normal), so
      // the reader is asked about the very place the clamp bounds.
      const probeX = apexSample.x + apexOffsetM * apexFrame.normalX;
      const probeY = apexSample.y + apexOffsetM * apexFrame.normalY;
      const nearest = resolveNearestTrackChord(profile, probeX, probeY);
      const nearestChord = profile.chords[nearest.segmentIndex]!;
      const probeSide = resolveTrackEdgeSideAtWorldZ(
        profile,
        nearest.segmentIndex,
        probeX,
        -probeY,
      );
      const containmentBoundM = resolveTrackEdgeHalfWidthM(
        nearestChord,
        probeSide,
      );

      // NON-VACUITY GUARD (asserted BEFORE the relational clause): the
      // lane really compares TWO DIFFERENT readers - the racing line's
      // OWN clamp and the containment bound's reader of the seam - and
      // the seam really publishes an ASYMMETRIC pair at the probe, so
      // the bound that applies is a SPECIFIC number and NOT the retired
      // symmetric `width / 2`.
      expect(nearestChord.outerHalfWidthM).toBeGreaterThan(
        nearestChord.innerHalfWidthM,
      );
      expect(
        Math.abs(containmentBoundM - nearestChord.innerHalfWidthM),
      ).toBeGreaterThan(OFFSET_TOLERANCE_M);

      // Assert
      // TODAY the racing line's clamp is the RETIRED symmetric
      // `sample.width / 2` (1.500000 m), so the bound the AI really
      // gets is NOT the seam edge the road reads (1.625000 m). After the
      // re-base the two are ONE edge.
      const boundDeltaM = Math.abs(Math.abs(apexOffsetM) - containmentBoundM);
      expect(boundDeltaM).toBeLessThanOrEqual(OFFSET_TOLERANCE_M);
    });
  });
});

/**
 * The PRODUCTION layout the CONTROL-ring lane reads.
 *
 * The determinism tuple the generator's own documentation uses: seed 42,
 * layout version 1 and the medium size bucket. The layout carries 126
 * spline samples over SEVEN control chords.
 */
const PRODUCTION_LAYOUT_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'medium',
};

/**
 * Rebuilds the CONTROL chord ring the racing line reads its seam through.
 *
 * THE MIRROR: `track.racing-line.ts` keeps this path PRIVATE (the racing
 * line is a pure function of the frozen sample list, so the seam is
 * reconstructed from those samples and never from a second geometry), so
 * the lane MIRRORS it here: the FIRST sample of every control segment is
 * that segment's control point (its interpolation factor is ZERO), and
 * the control chord of segment `k` runs from control point `k` to control
 * point `k + 1` with the width that control point publishes.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The rebuilt control chords and the samples that name them.
 */
const rebuildControlChordRing = (
  samples: readonly SplineSample[],
): {
  readonly segments: TrackSegment[];
  readonly samples: readonly SplineSample[];
} => {
  const controlPointBySegmentIndex = new Map<number, SplineSample>();
  for (const sample of samples) {
    if (!controlPointBySegmentIndex.has(sample.segmentIndex)) {
      controlPointBySegmentIndex.set(sample.segmentIndex, sample);
    }
  }

  const chordIndexBySegmentIndex = new Map<number, number>();
  const controlPoints: SplineSample[] = [];
  const segmentIndices = [...controlPointBySegmentIndex.keys()].toSorted(
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
};

/**
 * Counts the control segments the samples name.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The number of DISTINCT control segments the samples name.
 */
const namedControlSegmentCountOf = (samples: readonly SplineSample[]): number =>
  new Set(samples.map((sample) => sample.segmentIndex)).size;

/**
 * Counts the control points that sit at interpolation factor ZERO.
 *
 * @param samples - The frozen closed-ring spline samples.
 * @returns The number of named control segments whose FIRST sample carries
 *   the interpolation factor ZERO.
 */
const zeroFactorControlPointCountOf = (
  samples: readonly SplineSample[],
): number => {
  const firstSampleBySegmentIndex = new Map<number, SplineSample>();
  for (const sample of samples) {
    if (!firstSampleBySegmentIndex.has(sample.segmentIndex)) {
      firstSampleBySegmentIndex.set(sample.segmentIndex, sample);
    }
  }

  let count = 0;
  for (const sample of firstSampleBySegmentIndex.values()) {
    if (sample.sampleIndexWithinSegment === 0) {
      count += 1;
    }
  }

  return count;
};

/**
 * The racing line's CONTROL chord ring (Step 14V fix packet
 * `fix-packet-P1-prop-clearance-impl-iteration-1`, observation N3).
 *
 * THE ONE SEAM, RECONSTRUCTED: `track.racing-line.ts` resolves the seam
 * it clamps by from the CONTROL chord ring the samples describe, and it
 * falls back to the samples' OWN ring only for a fixture that names a
 * SINGLE control segment. The reconstruction is claimed to be
 * BIT-IDENTICAL to the spec's own chord list, so the racing line's clamp
 * and the containment bound read ONE edge - a drift in the
 * reconstruction would silently reopen that seam split.
 *
 * THE LANE pins that claim on the PRODUCTION layout (seed 42, layout
 * version 1, medium), whose samples name MORE THAN ONE control segment,
 * so the CONTROL branch is the branch IN FORCE and the SAMPLE fallback is
 * not.
 */
describe('the racing line reads the CONTROL chord ring (Step 14V fix packet, N3)', () => {
  it('AC-095: the CONTROL chord ring the racing line reads is the OWN chord list of the spec, bit-exact (N3)', () => {
    // Arrange - the PRODUCTION layout.
    const spec = generateTrack(PRODUCTION_LAYOUT_INPUT);
    const samples = spec.splineSamples;

    // Act - rebuild the CONTROL chord ring the way the module does.
    const rebuilt = rebuildControlChordRing(samples);

    // NON-VACUITY GUARD (asserted BEFORE the relational clause): the
    // CONTROL branch is the branch IN FORCE. The module falls back to the
    // samples' own ring ONLY for a fixture that names a SINGLE control
    // segment, so the production layout's named control segments are what
    // select the reconstruction under test.
    expect(namedControlSegmentCountOf(samples)).toBeGreaterThan(1);

    // The BIT-EXACTNESS BASIS: every rebuilt chord is anchored on a
    // control point whose interpolation factor is ZERO, which is why the
    // reconstruction sits EXACTLY on the spec's control points.
    expect(zeroFactorControlPointCountOf(samples)).toBe(
      rebuilt.segments.length,
    );

    // Assert - the reconstruction IS the spec's own chord list, exactly.
    // `toEqual` compares the numbers with `Object.is` semantics, so the
    // match is BIT-EXACT: no tolerance is applied anywhere.
    expect(rebuilt.segments).toEqual(spec.segments);
  });
});
