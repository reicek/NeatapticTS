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
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import {
  buildTrackRacingLine,
  type TrackRacingLinePoint,
} from './track.racing-line';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';

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
    push(
      straightSamples + radius * Math.cos(angle),
      radius * Math.sin(angle),
    );
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
    Math.floor((firstStraightEnd - FIXTURE_STRAIGHT_MIDDLE_SAMPLE_COUNT) / 2) + 1;
  const straightSampleIndices: number[] = [];
  for (let offset = 0; offset < FIXTURE_STRAIGHT_MIDDLE_SAMPLE_COUNT; offset += 1) {
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
const maxAbsOffsetMOf = (
  line: readonly TrackRacingLinePoint[],
): number => Math.max(...line.map((point) => Math.abs(point.offsetM)));

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
        ...STADIUM.straightSampleIndices.map(
          (index) => Math.abs(line[index].offsetM),
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
        ...line.map(
          (point: TrackRacingLinePoint, index: number) =>
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
      const straightTargetSpeed = line[STADIUM.straightSampleIndices[0]]
        .targetSpeedUnitsPerS;
      const cornerTargetSpeed =
        line[STADIUM.apexSampleIndex].targetSpeedUnitsPerS;
      expect(cornerTargetSpeed).toBeLessThan(straightTargetSpeed);
      expect(cornerTargetSpeed).toBeGreaterThan(0);
    });
  });
});
