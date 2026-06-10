import type { SplineSample, TrackSegment } from './track.generator.types';

type WorldPoint = {
  readonly x: number;
  readonly y: number;
};

/** Number of Catmull-Rom samples emitted for each control segment. */
export const TRACK_SPLINE_SAMPLES_PER_SEGMENT = 18;

/** Tangent-space frame resolved for one sampled spline point. */
export interface SplineSampleFrame {
  readonly tangentHeadingRadians: number;
  readonly normalX: number;
  readonly normalY: number;
}

/**
 * Builds the shared Catmull-Rom lane-center samples for a closed-loop track.
 *
 * @param segments - Ordered closed-loop control segments.
 * @returns Stable spline samples shared by the renderer and controllers.
 */
export function buildTrackSplineSamples(
  segments: readonly TrackSegment[],
): readonly SplineSample[] {
  if (segments.length === 0) {
    return [];
  }

  const controlPoints = segments.map((segment) => ({
    x: segment.startX,
    y: segment.startY,
  }));
  const splineSamples: SplineSample[] = [];

  for (const [segmentIndex, segment] of segments.entries()) {
    const previousPoint = controlPoints.at(
      (segmentIndex - 1 + controlPoints.length) % controlPoints.length,
    )!;
    const startPoint = controlPoints[segmentIndex]!;
    const endPoint = controlPoints[(segmentIndex + 1) % controlPoints.length]!;
    const nextPoint = controlPoints[(segmentIndex + 2) % controlPoints.length]!;
    const nextWidth =
      segments[(segmentIndex + 1) % segments.length]?.width ?? segment.width;

    for (
      let sampleIndexWithinSegment = 0;
      sampleIndexWithinSegment < TRACK_SPLINE_SAMPLES_PER_SEGMENT;
      sampleIndexWithinSegment++
    ) {
      const interpolationFactor =
        sampleIndexWithinSegment / TRACK_SPLINE_SAMPLES_PER_SEGMENT;
      const sampledPoint = resolveCatmullRomPoint(
        previousPoint,
        startPoint,
        endPoint,
        nextPoint,
        interpolationFactor,
      );

      splineSamples.push({
        ...sampledPoint,
        width:
          segment.width + (nextWidth - segment.width) * interpolationFactor,
        segmentIndex,
        sampleIndexWithinSegment,
        globalIndex: splineSamples.length,
      });
    }
  }

  return splineSamples;
}

/**
 * Resolves the local tangent frame for one sampled spline point.
 *
 * @param splineSamples - Ordered closed-loop spline samples.
 * @param sampleIndex - Global index of the focal sample.
 * @returns Tangent heading plus the unit left normal.
 */
export function resolveSplineSampleFrame(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): SplineSampleFrame {
  if (splineSamples.length === 0) {
    return {
      tangentHeadingRadians: 0,
      normalX: 0,
      normalY: 1,
    };
  }

  const wrappedSampleIndex = wrapSplineSampleIndex(
    splineSamples.length,
    sampleIndex,
  );
  const previousSample = splineSamples.at(
    wrapSplineSampleIndex(splineSamples.length, wrappedSampleIndex - 1),
  )!;
  const nextSample =
    splineSamples[
      wrapSplineSampleIndex(splineSamples.length, wrappedSampleIndex + 1)
    ]!;
  const tangentDeltaX = nextSample.x - previousSample.x;
  const tangentDeltaY = nextSample.y - previousSample.y;
  const tangentLength = Math.hypot(tangentDeltaX, tangentDeltaY) || 1;
  const tangentUnitX = tangentDeltaX / tangentLength;
  const tangentUnitY = tangentDeltaY / tangentLength;

  return {
    tangentHeadingRadians: Math.atan2(tangentDeltaY, tangentDeltaX),
    normalX: -tangentUnitY,
    normalY: tangentUnitX,
  };
}

function resolveCatmullRomPoint(
  previousPoint: WorldPoint,
  startPoint: WorldPoint,
  endPoint: WorldPoint,
  nextPoint: WorldPoint,
  interpolationFactor: number,
): WorldPoint {
  const interpolationFactorSquared = interpolationFactor * interpolationFactor;
  const interpolationFactorCubed =
    interpolationFactorSquared * interpolationFactor;

  return {
    x:
      0.5 *
      (2 * startPoint.x +
        (-previousPoint.x + endPoint.x) * interpolationFactor +
        (2 * previousPoint.x -
          5 * startPoint.x +
          4 * endPoint.x -
          nextPoint.x) *
          interpolationFactorSquared +
        (-previousPoint.x + 3 * startPoint.x - 3 * endPoint.x + nextPoint.x) *
          interpolationFactorCubed),
    y:
      0.5 *
      (2 * startPoint.y +
        (-previousPoint.y + endPoint.y) * interpolationFactor +
        (2 * previousPoint.y -
          5 * startPoint.y +
          4 * endPoint.y -
          nextPoint.y) *
          interpolationFactorSquared +
        (-previousPoint.y + 3 * startPoint.y - 3 * endPoint.y + nextPoint.y) *
          interpolationFactorCubed),
  };
}

function wrapSplineSampleIndex(
  sampleCount: number,
  sampleIndex: number,
): number {
  return ((sampleIndex % sampleCount) + sampleCount) % sampleCount;
}
