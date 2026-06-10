import type { EnvironmentState } from '../environment/environment.types';
import type { SplineSample, TrackSpec } from '../track/track.generator.types';
import {
  buildTrackSplineSamples,
  resolveSplineSampleFrame,
  type SplineSampleFrame,
} from '../track/track.spline.utils';

/** Shared distance scale used by observation-vector distance channels. */
export const DISTANCE_WORLD_SCALE = 64;

/** Shared boundary-distance scale used by observation-vector lane channels. */
export const BOUNDARY_DISTANCE_WORLD_SCALE = 24;

/** Constant lane width used by the owner-local curved-track red fixture. */
const TEST_TRACK_WIDTH_WORLD = 24;

type WorldPoint = {
  readonly x: number;
  readonly y: number;
};

/** One sampled spline point enriched with segment ownership metadata. */
export type TestSplineSample = SplineSample;

/** Local tangent-frame information for one spline sample. */
export type TestSampleFrame = SplineSampleFrame;

/** Probe position placed slightly toward the left boundary of a spline sample. */
export interface ObservationProbe {
  readonly carX: number;
  readonly carY: number;
  readonly offsetWorld: number;
}

/** Shared result shape returned by the focal-sample selectors. */
export interface FocalSampleSelection {
  readonly splineSamples: readonly TestSplineSample[];
  readonly focalSample: TestSplineSample;
  readonly focalFrame: TestSampleFrame;
}

const CURVED_TRACK_CONTROL_POINTS: readonly WorldPoint[] = Object.freeze([
  Object.freeze({ x: -120, y: 0 }),
  Object.freeze({ x: -40, y: -110 }),
  Object.freeze({ x: 80, y: -55 }),
  Object.freeze({ x: 120, y: 25 }),
  Object.freeze({ x: 35, y: 120 }),
  Object.freeze({ x: -95, y: 85 }),
]);

/**
 * Builds a deterministic curved track whose rendered spline diverges materially
 * from the raw polygon chord geometry.
 *
 * @returns Frozen owner-local `TrackSpec` used by the red seam contracts.
 */
export function createCurvedTrackSpec(): TrackSpec {
  const frozenSegments = CURVED_TRACK_CONTROL_POINTS.map(
    (startPoint, segmentIndex) => {
      const endPoint =
        CURVED_TRACK_CONTROL_POINTS[
          (segmentIndex + 1) % CURVED_TRACK_CONTROL_POINTS.length
        ]!;

      return Object.freeze({
        startX: startPoint.x,
        startY: startPoint.y,
        endX: endPoint.x,
        endY: endPoint.y,
        width: TEST_TRACK_WIDTH_WORLD,
      });
    },
  );
  const frozenSplineSamples = buildTrackSplineSamples(frozenSegments).map(
    (splineSample) => Object.freeze({ ...splineSample }),
  );

  return Object.freeze({
    seed: 7,
    layoutVersion: 1,
    sizeBucket: 'pathtracking-red',
    segments: Object.freeze(frozenSegments),
    splineSamples: Object.freeze(frozenSplineSamples),
  });
}

/**
 * Creates a minimal base environment state for owner-local controller tests.
 *
 * @param overrides - Environment fields to override for a specific scenario.
 * @returns Deterministic environment state for the red tests.
 */
export function createEnvironmentState(
  overrides: Partial<EnvironmentState> = {},
): EnvironmentState {
  return {
    tick: 12,
    carX: 0,
    carY: 0,
    carHeading: 0,
    ...overrides,
  };
}

/**
 * Builds the sampled Catmull-Rom centerline that the renderer currently draws.
 *
 * @param trackSpec - Frozen track geometry.
 * @returns Ordered spline samples with segment ownership metadata.
 */
export function buildSplineSamples(
  trackSpec: TrackSpec,
): readonly TestSplineSample[] {
  return trackSpec.splineSamples;
}

/**
 * Selects the owner-local spline sample with the strongest chord-midpoint gap.
 *
 * @param trackSpec - Frozen curved-track fixture.
 * @returns The strongest observation seam sample plus its tangent frame.
 */
export function selectObservationFocalSample(
  trackSpec: TrackSpec,
): FocalSampleSelection {
  const splineSamples = buildSplineSamples(trackSpec);
  const focalSample = resolveBestObservationSample(trackSpec, splineSamples);

  return {
    splineSamples,
    focalSample,
    focalFrame: resolveSampleFrame(splineSamples, focalSample.globalIndex),
  };
}

/**
 * Selects the owner-local spline sample with the strongest endpoint-vs-tangent gap.
 *
 * @param trackSpec - Frozen curved-track fixture.
 * @returns The strongest controller seam sample plus its tangent frame.
 */
export function selectControllerFocalSample(
  trackSpec: TrackSpec,
): FocalSampleSelection {
  const splineSamples = buildSplineSamples(trackSpec);
  const focalSample = resolveBestControllerSample(trackSpec, splineSamples);

  return {
    splineSamples,
    focalSample,
    focalFrame: resolveSampleFrame(splineSamples, focalSample.globalIndex),
  };
}

/**
 * Places the observation probe slightly toward the left boundary of one spline sample.
 *
 * @param splineSamples - Ordered sampled centerline points.
 * @param focalSampleIndex - Global index of the selected spline sample.
 * @returns Probe position plus the exact lateral offset from the lane center.
 */
export function resolveOffsetObservationProbe(
  splineSamples: readonly TestSplineSample[],
  focalSampleIndex: number,
): ObservationProbe {
  const focalSample = splineSamples[focalSampleIndex]!;
  const focalFrame = resolveSampleFrame(splineSamples, focalSampleIndex);
  const previousSample = splineSamples.at(
    (focalSampleIndex - 1 + splineSamples.length) % splineSamples.length,
  )!;
  const nextSample =
    splineSamples[(focalSampleIndex + 1) % splineSamples.length]!;
  const nearestNeighborDistanceWorld = Math.min(
    Math.hypot(
      previousSample.x - focalSample.x,
      previousSample.y - focalSample.y,
    ),
    Math.hypot(nextSample.x - focalSample.x, nextSample.y - focalSample.y),
  );
  let offsetWorld = Math.min(
    3,
    focalSample.width / 6,
    nearestNeighborDistanceWorld / 4,
  );

  while (offsetWorld > 0.5) {
    const probeX = focalSample.x + focalFrame.normalX * offsetWorld;
    const probeY = focalSample.y + focalFrame.normalY * offsetWorld;

    if (
      resolveNearestSplineSampleIndex(probeX, probeY, splineSamples) ===
      focalSampleIndex
    ) {
      return { carX: probeX, carY: probeY, offsetWorld };
    }

    offsetWorld /= 2;
  }

  return {
    carX: focalSample.x + focalFrame.normalX * offsetWorld,
    carY: focalSample.y + focalFrame.normalY * offsetWorld,
    offsetWorld,
  };
}

/**
 * Resolves the local tangent frame for one sampled spline point.
 *
 * @param splineSamples - Ordered sampled centerline points.
 * @param focalSampleIndex - Global index of the sample to inspect.
 * @returns Tangent heading plus the unit left normal.
 */
export function resolveSampleFrame(
  splineSamples: readonly TestSplineSample[],
  focalSampleIndex: number,
): TestSampleFrame {
  return resolveSplineSampleFrame(splineSamples, focalSampleIndex);
}

/**
 * Wraps an angle into the closed interval `[-π, π]`.
 *
 * @param angleRadians - Raw angle in radians.
 * @returns Wrapped angle in `[-π, π]`.
 */
export function wrapAngleToMinusPiPi(angleRadians: number): number {
  let wrappedAngleRadians = angleRadians;

  while (wrappedAngleRadians > Math.PI) {
    wrappedAngleRadians -= Math.PI * 2;
  }

  while (wrappedAngleRadians < -Math.PI) {
    wrappedAngleRadians += Math.PI * 2;
  }

  return wrappedAngleRadians;
}

function resolveBestObservationSample(
  trackSpec: TrackSpec,
  splineSamples: readonly TestSplineSample[],
): TestSplineSample {
  let bestSample = splineSamples[0]!;
  let bestDeviationWorld = -Infinity;

  for (const splineSample of splineSamples) {
    if (
      splineSample.sampleIndexWithinSegment < 4 ||
      splineSample.sampleIndexWithinSegment > 13 ||
      resolveNearestChordSegmentIndex(
        trackSpec,
        splineSample.x,
        splineSample.y,
      ) !== splineSample.segmentIndex
    ) {
      continue;
    }

    const owningSegment = trackSpec.segments[splineSample.segmentIndex]!;
    const chordMidpointX = (owningSegment.startX + owningSegment.endX) / 2;
    const chordMidpointY = (owningSegment.startY + owningSegment.endY) / 2;
    const chordDeviationWorld = Math.hypot(
      splineSample.x - chordMidpointX,
      splineSample.y - chordMidpointY,
    );

    if (chordDeviationWorld > bestDeviationWorld) {
      bestDeviationWorld = chordDeviationWorld;
      bestSample = splineSample;
    }
  }

  return bestSample;
}

function resolveBestControllerSample(
  trackSpec: TrackSpec,
  splineSamples: readonly TestSplineSample[],
): TestSplineSample {
  let bestSample = splineSamples[0]!;
  let bestAngularGapRadians = -Infinity;

  for (const splineSample of splineSamples) {
    if (
      splineSample.sampleIndexWithinSegment < 4 ||
      splineSample.sampleIndexWithinSegment > 13 ||
      resolveNearestChordSegmentIndex(
        trackSpec,
        splineSample.x,
        splineSample.y,
      ) !== splineSample.segmentIndex
    ) {
      continue;
    }

    const owningSegment = trackSpec.segments[splineSample.segmentIndex]!;
    const distanceToChordEndpointWorld = Math.hypot(
      owningSegment.endX - splineSample.x,
      owningSegment.endY - splineSample.y,
    );

    if (distanceToChordEndpointWorld <= 18) {
      continue;
    }

    const splineFrame = resolveSampleFrame(
      splineSamples,
      splineSample.globalIndex,
    );
    const chordEndpointHeadingRadians = Math.atan2(
      owningSegment.endY - splineSample.y,
      owningSegment.endX - splineSample.x,
    );
    const angularGapRadians = Math.abs(
      wrapAngleToMinusPiPi(
        chordEndpointHeadingRadians - splineFrame.tangentHeadingRadians,
      ),
    );

    if (angularGapRadians > bestAngularGapRadians) {
      bestAngularGapRadians = angularGapRadians;
      bestSample = splineSample;
    }
  }

  return bestSample;
}

function resolveNearestChordSegmentIndex(
  trackSpec: TrackSpec,
  worldX: number,
  worldY: number,
): number {
  let bestSegmentIndex = 0;
  let bestDistanceWorld = Number.POSITIVE_INFINITY;

  for (const [segmentIndex, segment] of trackSpec.segments.entries()) {
    const chordMidpointX = (segment.startX + segment.endX) / 2;
    const chordMidpointY = (segment.startY + segment.endY) / 2;
    const distanceToChordMidpointWorld = Math.hypot(
      chordMidpointX - worldX,
      chordMidpointY - worldY,
    );

    if (distanceToChordMidpointWorld < bestDistanceWorld) {
      bestDistanceWorld = distanceToChordMidpointWorld;
      bestSegmentIndex = segmentIndex;
    }
  }

  return bestSegmentIndex;
}

function resolveNearestSplineSampleIndex(
  worldX: number,
  worldY: number,
  splineSamples: readonly TestSplineSample[],
): number {
  let bestSampleIndex = 0;
  let bestDistanceWorld = Number.POSITIVE_INFINITY;

  for (const splineSample of splineSamples) {
    const distanceToSampleWorld = Math.hypot(
      splineSample.x - worldX,
      splineSample.y - worldY,
    );

    if (distanceToSampleWorld < bestDistanceWorld) {
      bestDistanceWorld = distanceToSampleWorld;
      bestSampleIndex = splineSample.globalIndex;
    }
  }

  return bestSampleIndex;
}
