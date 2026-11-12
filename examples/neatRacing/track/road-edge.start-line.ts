/**
 * The GREEN 2-hex start line: the ALONG-TRACK classification seam of the
 * neatRacing hexagonal neon road-edge painter (plan slice P5-S27-2,
 * AC-104).
 *
 * THE START LINE IS AN ALONG-TRACK QUESTION, NOT A CROSS-TRACK ONE: the
 * road-edge painter classifies every honeycomb tile by its distance PAST the
 * track edge (the green/yellow/red border rings), so the start/finish line -
 * which lies ON the drivable surface - has no home in that channel. This
 * module owns the missing axis: the forward ARC LENGTH along the published
 * closed spline polyline, measured from the start/finish plane.
 *
 * THE MEASURE IS THE ARC LENGTH, NEVER THE SEGMENT-0 CHORD: the track is a
 * CLOSED loop, so a chord-relative measure spreads the start-line tiles over
 * the whole lap (~570-810 m) instead of the first few meters. The arc is
 * measured FORWARD from `splineSamples[0]` (segment 0's start point - the
 * plane the lap-line detector counts the crossing on), so the band is the
 * FIRST 4.0 m of the lap and is NOT centred on the line.
 *
 * THE BAND IS A NEW CHANNEL BESIDE THE COLORWAY, NEVER A FIFTH COLORWAY:
 * the EXPORTED `TileColorway` surface stays at its four members ('road' |
 * 'red' | 'yellow' | 'green'), because the 'green' colorway is the BORDER
 * ring beyond the track edge and every one of its tiles must sit OUTSIDE
 * the roadway - a road tile can never carry it.
 *
 * THE WORLD/GENERATOR MIRROR: the generator plane (x, y) is mirrored into
 * the world plane by `worldZ = -y`, the same handedness the slab rasterizer
 * and the road-edge painter use.
 *
 * Determinism: every export is a pure function of the published spline
 * samples and the point it is given - no RNG, no clock, no hidden state.
 */

import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';

/**
 * One projected position of a world point on the closed spline polyline.
 */
export interface StartLineProjection {
  /** Forward arc length along the closed polyline from sample 0, in meters. */
  readonly arcM: number;
  /** Signed lateral offset from the projected centreline point, in meters. */
  readonly signedLateralM: number;
}

/**
 * Projects a world point onto the nearest point of the CLOSED spline
 * polyline and measures its position along the lap.
 *
 * The closest point of the SEGMENT that owns the nearest sample wins: the
 * factor along the segment is clamped to [0, 1], and the candidate wins by
 * the SMALLEST squared distance (the FIRST minimum wins a tie, so the
 * measure is stable).
 *
 * @param samples - The published closed-loop spline samples.
 * @param worldX - World x of the projected point.
 * @param worldZ - World z of the projected point (`worldZ = -y`, the same
 *   handedness the slab rasterizer uses).
 * @returns The forward arc length and the signed lateral offset.
 */
export function projectOntoClosedSpline(
  samples: readonly SplineSample[],
  worldX: number,
  worldZ: number,
): StartLineProjection {
  // The generator plane (x, y) is mirrored into the world plane by
  // `worldZ = -y`.
  const pointX = worldX;
  const pointY = -worldZ;

  let bestDistanceSquared = Number.POSITIVE_INFINITY;
  let bestArcM = 0;
  let bestSignedLateralM = 0;
  let arcAtSegmentStartM = 0;

  for (let index = 0; index < samples.length; index += 1) {
    const start = samples[index]!;
    const end = samples[(index + 1) % samples.length]!;
    const tangentX = end.x - start.x;
    const tangentY = end.y - start.y;
    const segmentLengthM = Math.hypot(tangentX, tangentY);

    let projectionFactor = 0;
    if (segmentLengthM > 0) {
      projectionFactor =
        ((pointX - start.x) * tangentX + (pointY - start.y) * tangentY) /
        (segmentLengthM * segmentLengthM);
      projectionFactor = Math.min(1, Math.max(0, projectionFactor));
    }

    const closestX = start.x + projectionFactor * tangentX;
    const closestY = start.y + projectionFactor * tangentY;
    const offsetX = pointX - closestX;
    const offsetY = pointY - closestY;
    const distanceSquared = offsetX * offsetX + offsetY * offsetY;

    if (distanceSquared < bestDistanceSquared) {
      bestDistanceSquared = distanceSquared;
      bestArcM = arcAtSegmentStartM + projectionFactor * segmentLengthM;
      // The sign of the local tangent x offset cross product picks the side.
      bestSignedLateralM =
        segmentLengthM > 0
          ? (tangentX * offsetY - tangentY * offsetX) / segmentLengthM
          : 0;
    }

    arcAtSegmentStartM += segmentLengthM;
  }

  return { arcM: bestArcM, signedLateralM: bestSignedLateralM };
}

/**
 * The number of hex tiles the start-line band is DEEP along the track axis.
 *
 * TWO hexes deep - the AC-104 depth. The count is named so the depth can
 * never drift away from the band the packet asks for.
 */
const START_LINE_BAND_HEX_DEPTH_COUNT = 2;

/**
 * The forward depth of the start-line band for one hex tile diameter: TWO
 * hexes deep.
 *
 * The depth is DERIVED from the hex diameter instead of being spelled out,
 * so the band can never drift away from the tile size it is painted on.
 *
 * @param hexDiameterM - The corner-to-corner diameter of one hex tile, in
 *   meters.
 * @returns The forward depth of the band, in meters.
 */
export function resolveStartLineBandDepthM(hexDiameterM: number): number {
  return START_LINE_BAND_HEX_DEPTH_COUNT * hexDiameterM;
}

/**
 * Whether an along-track arc position falls inside the start-line band.
 *
 * The band runs FORWARD from the start/finish plane (arc 0) and is NOT
 * centred on it: the lap-line detector counts the crossing ON the plane, so
 * the paint sits in front of it, on the first metres of the lap.
 *
 * @param arcM - Forward arc length from the start/finish plane, in meters.
 * @param bandDepthM - The forward depth of the band, in meters.
 * @returns True when the position lies inside the band.
 */
export function isWithinStartLineBand(
  arcM: number,
  bandDepthM: number,
): boolean {
  return arcM >= 0 && arcM <= bandDepthM;
}
