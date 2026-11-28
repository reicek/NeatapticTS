/**
 * The GREEN 2-hex start line: the ALONG-TRACK classification seam of the
 * neatRacing hexagonal neon road-edge painter.
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
 * the roadway - a road tile can never carry it. The band paints ROAD tiles
 * green (reusing the border's green fill and edge tokens) and marks them
 * with a separate flag, so the two greens never touch: the band lies ON the
 * ribbon, the border ring lies BEYOND it.
 *
 * THE WORLD/GENERATOR MIRROR: the generator plane (x, y) is mirrored into
 * the world plane by `worldZ = -y`, the same handedness the slab rasterizer
 * and the road-edge painter use.
 *
 * Two questions classify a tile, and they never answer the same question:
 *
 * ```mermaid
 * flowchart LR
 *     Tile["One honeycomb tile"] --> Cross{"How far PAST<br/>the track edge?"}
 *     Cross -->|"on the ribbon"| Road["colorway 'road'"]
 *     Cross -->|"0-2 m past"| GreenRing["colorway 'green'"]
 *     Cross -->|"2-4 m past"| YellowRing["colorway 'yellow'"]
 *     Cross -->|"4-6 m past"| RedRing["colorway 'red'"]
 *     Tile --> Along{"How far FORWARD<br/>along the lap?"}
 *     Along -->|"arc 0 to 4.0 m"| Band["start-line band tile<br/>(keeps colorway 'road')"]
 *
 *     classDef boundary fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:2px;
 *     classDef active fill:#03111f,stroke:#00ff66,color:#d8ffe9,stroke-width:2px;
 *     classDef spotlight fill:#2a1029,stroke:#ff5cff,color:#ffe6ff,stroke-width:3px;
 *
 *     class Tile,Cross,Along boundary;
 *     class Road,GreenRing,YellowRing,RedRing boundary;
 *     class Band active;
 * ```
 *
 * The along-track measure is a projection onto the closed polyline:
 *
 * ```mermaid
 * flowchart TB
 *     Point["World point (x, z)"] --> Mirror["Generator plane (x, -z)"]
 *     Mirror --> Segment["Nearest closed-polyline segment"]
 *     Segment --> Clamp["Projection factor clamped to 0..1"]
 *     Clamp --> Arc["arc = segment offset + factor x segment length"]
 *     Arc --> Inside{"arc inside the band?"}
 *     Inside -->|"yes"| BandTile["start-line band tile"]
 *     Inside -->|"no"| PlainTile["plain road tile"]
 * ```
 *
 * Further reading: [Arc length](https://en.wikipedia.org/wiki/Arc_length) for
 * the measure this module walks, and
 * [Spline](https://en.wikipedia.org/wiki/Spline) for the polyline it walks
 * along.
 *
 * Determinism: every export is a pure function of the published spline
 * samples and the point it is given - no RNG, no clock, no hidden state.
 */

import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';

/**
 * One projected position of a world point on the closed spline polyline.
 *
 * The pair answers BOTH questions the road-edge painter asks about a
 * tile: `arcM` is the along-track position (the start-line band reads it)
 * and `signedLateralM` is the cross-track offset from the centreline (the
 * lateral sign is the tangent's right-hand side).
 *
 * @example
 * ```ts
 * const projection = projectOntoClosedSpline(splineSamples, worldX, worldZ);
 * console.log(projection.arcM); // metres forward from the start/finish plane
 * console.log(projection.signedLateralM); // metres left/right of the centreline
 * ```
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
 *
 * @example
 * ```ts
 * const projection = projectOntoClosedSpline(splineSamples, worldX, worldZ);
 * const bandDepthM = resolveStartLineBandDepthM(2);
 * if (isWithinStartLineBand(projection.arcM, bandDepthM)) {
 *   // the point lies on the green start-line band
 * }
 * ```
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
 * TWO hexes deep - the band the start/finish stripe is painted with. The
 * count is named so the depth can never drift away from the band the
 * roadway expects.
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
 *
 * @example
 * ```ts
 * resolveStartLineBandDepthM(2); // 4 - a 2 m hex tile paints a 4 m band
 * ```
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
 *
 * @example
 * ```ts
 * const bandDepthM = resolveStartLineBandDepthM(2);
 * isWithinStartLineBand(0, bandDepthM); // true - the start/finish plane itself
 * isWithinStartLineBand(3, bandDepthM); // true - inside the band
 * isWithinStartLineBand(9, bandDepthM); // false - past the band
 * ```
 */
export function isWithinStartLineBand(
  arcM: number,
  bandDepthM: number,
): boolean {
  return arcM >= 0 && arcM <= bandDepthM;
}
