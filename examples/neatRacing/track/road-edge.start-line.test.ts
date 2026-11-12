/**
 * ============================================================================
 * P5-S27-1 (plan step 27, Phase 5) - THE RACE START: the GREEN 2-hex start
 * line - the RED-S27-1 lane (the start-line band classification).
 *
 * AC tags: AC-104 (the green 2-hex start line: a GREEN band TWO hexes DEEP
 * along the track axis - 2 x NEATRACING_ROAD_EDGE_HEX_DIAMETER_M = 4.0 m -
 * spanning the FULL road width), GAP-5-01 (the start line has NO renderer:
 * every road tile of the hex lattice is painted with the plain ROAD fill
 * tokens today, so there is NO green band anywhere on the track).
 *
 * THE OBSERVABLE IS THE RENDERED FILL COLOUR - A CHANNEL-INDEPENDENT VALUE,
 * NEVER THE NAME OF A NEW FIELD. The band is a NEW along-track
 * classification, so this lane must NOT pin `tile.colorway === 'green'`:
 * the colorway channel is the BORDER-LINE channel (the green/yellow/red
 * rings BEYOND the track edge), and the EXISTING road-edge pin
 * (`road-edge.test.ts` -> 'steps border-line tiles outward in green, yellow,
 * red bands') requires EVERY 'green' colorway tile to sit OUTSIDE the
 * roadway (`0 < distancePastEdgeM <= 2`) - a road tile, which lies INSIDE
 * the roadway, can never carry it. So the lane pins the RENDERED FILL
 * COLOUR TOKEN instead: the lattice emits SIX fill triangles per tile and
 * every one of them starts at the tile CENTRE vertex
 * (`[tile.x, NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M, tile.z]`), and the
 * triangle factory gives all three vertices the SAME colour - so that
 * centre vertex's colour IS the tile's rendered fill colour. The GREEN
 * FILL TOKEN (`NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR`) is the value the
 * band must render; the ROAD FILL TOKEN (`NEATRACING_ROAD_EDGE_FILL_COLOR`)
 * is the value the plain road keeps. (The EDGE/stroke colour is
 * deliberately NOT pinned: a shared edge between a band tile and a
 * neighbouring road tile has its colour resolved by precedence, so the
 * stroke colour is NOT a stable observable.)
 *
 * THE COLOURWAY-UNTOUCHED INVARIANT (a RUNTIME stability guard): every
 * strict road-core tile inside the start window keeps `colorway ===
 * 'road'`. The band rides on a NEW along-track classification, so the
 * EXPORTED `TileColorway` surface stays at its FOUR members ('road' |
 * 'red' | 'yellow' | 'green') and the EXISTING road-edge green-border pins
 * stay green. The four-member union itself is pinned by a COMPILE-TIME
 * TYPE-LEVEL assertion (a union's member count is ERASED at runtime, so
 * NO runtime lane can observe it; a fifth member must break the
 * `npx tsc -p tsconfig.neatRacing.json --noEmit` lane instead).
 *
 * GREEN TARGET (04-implementing, slice P5-S27-2 / AC-104, must make exactly
 * these pass):
 * - the road tiles whose along-track ARC LENGTH falls inside the 2-hex start
 *   window (the FORWARD 4.0 m from the start/finish plane) RENDER the GREEN
 *   fill token,
 * - the band is TWO hexes DEEP (its far half is green too) and it spans the
 *   FULL road width (it reaches BOTH sides of the centreline),
 * - the plain ROAD band OUTSIDE the start window still renders the ROAD fill
 *   token, and
 * - the road tiles keep their 'road' colorway (the colourway-untouched
 *   invariant above).
 *
 * THE LANE IS RUNTIME-ONLY: every runtime assertion probes the EXISTING
 * road-edge module (the hex lattice) and the EXISTING slab surface seam (the
 * road/curb classification), so every red is a NAMED RUNTIME failure and
 * NEVER an import failure. The only type-level import is a TYPE-ONLY import
 * (`import type { TileColorway }`), which is ERASED at compile time.
 *
 * THE ALONG-TRACK MEASURE IS THE SPLINE-POLYLINE ARC LENGTH: never the
 * segment-0 CHORD projection - the track is a CLOSED loop, so a
 * chord-relative measure spreads the start-line tiles over the whole lap
 * (~570-810 m) instead of the first few meters. The arc is measured FORWARD
 * from `splineSamples[0]` (segment 0's start point - the start/finish plane),
 * so the start window is [0, 4.0] m and NOT centred on the start line.
 *
 * THE POPULATION IS THE STRICT ROAD CORE: only the cells the slab surface
 * seam classifies as 'road' (NOT the curb band), so the lane holds for an
 * implementation that paints the core ALONE and for one that paints the core
 * PLUS the curb.
 *
 * Determinism: the fixture is the canonical `{ seed: 42, layoutVersion: 1,
 * sizeBucket: 'small' }` tuple; no RNG, no `Date.now()`, no wall-clock.
 *
 * THE MEASURED BASELINE (this lane, today - the numbers the green lane
 * compares against): the lattice carries 9174 tiles, of which 5549 are
 * ROAD-BAND tiles (the 'road' AND 'curb' surfaces) and 4946 are STRICT
 * ROAD-CORE tiles (the 'road' surface - the tiles the module paints with
 * the plain ROAD fill token today); the closed spline polyline is 810.86 m
 * long. The arc windows hold 23 strict road-core tiles in the CORE window
 * [0.5, 3.5] m, 7 in the DEPTH window [2.5, 3.5] m and 28 in the OUTSIDE
 * window [8, 12] m - every one of them rendering the PLAIN ROAD fill token
 * today, and NOT ONE of them rendering the GREEN fill token.
 * ============================================================================
 *
 * Contract tests for the Phase-5 GREEN 2-hex start line painted by the
 * neatRacing hexagonal neon road-edge module (plan slice P5-S27-1, the
 * RED-S27-1 lane of step 27).
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { SplineSample } from '../../racing_curriculum/track/track.generator.types';
import { resolveTrackEdgeProfile } from './track.edge';
import { classifyTrackSlabSurfaceAt } from './track.slabs';
import {
  buildRoadEdgeHexLattice,
  NEATRACING_ROAD_EDGE_FILL_COLOR,
  NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR,
} from './road-edge';
import type { TileColorway } from './road-edge';

/** Canonical determinism tuple reused from the track slab builder. */
const FIXTURE_TRACK_BUILD_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'small',
} as const;

/** The generated track spec shared by the lattice and the surface seam. */
const FIXTURE_TRACK_SPEC = generateTrack(FIXTURE_TRACK_BUILD_INPUT);

/**
 * The published edge profile, resolved ONCE for the whole fixture (the
 * classification seam rebuilds it per point when it is omitted).
 */
const FIXTURE_EDGE_PROFILE = resolveTrackEdgeProfile(
  FIXTURE_TRACK_SPEC.segments.slice(),
);

/** The hex lattice under test - built ONCE for the whole lane. */
const FIXTURE_LATTICE = buildRoadEdgeHexLattice(FIXTURE_TRACK_BUILD_INPUT);

/**
 * COMPILE-TIME exact-union assertion: `IsExact<T, U>` is true only when the
 * two unions are MUTUALLY assignable, so a FIFTH `TileColorway` member (or a
 * dropped one) turns the pin FALSE - and a false pin is a TYPE ERROR, so the
 * `npx tsc -p tsconfig.neatRacing.json --noEmit` lane rejects it. The union's
 * member count is ERASED at runtime, so this pin is NEVER a runtime
 * observation (it is deliberately NOT re-cast as one).
 */
type IsExact<T, U> = [T] extends [U] ? ([U] extends [T] ? true : false) : false;

/** The type-level pin over the EXPORTED `TileColorway` surface. */
const TILE_COLORWAY_PIN: IsExact<
  TileColorway,
  'road' | 'red' | 'yellow' | 'green'
> = true;

/** One projected position of a world point on the closed spline polyline. */
interface SplineProjection {
  /** Forward arc length along the closed polyline from sample 0, in meters. */
  readonly arcM: number;
  /** Signed lateral offset from the projected centreline point, in meters. */
  readonly signedLateralM: number;
}

/** One 0-255 RGB colour token. */
type RgbColor = readonly [number, number, number];

/** One strict road-core lattice tile with its measured along-track position. */
interface RoadCoreTile {
  /** Forward arc length from the start/finish plane, in meters. */
  readonly arcM: number;
  /** Signed lateral offset from the centreline, in meters. */
  readonly signedLateralM: number;
  /** The colorway the lattice classifies this tile with today. */
  readonly colorway: TileColorway;
  /** The RENDERED fill colour of this tile (the paint colour a viewer sees). */
  readonly renderedFillColor: RgbColor;
}

/** A closed along-track arc window, in meters from the start/finish plane. */
interface ArcWindow {
  /** Inclusive lower bound of the window, in meters. */
  readonly min: number;
  /** Inclusive upper bound of the window, in meters. */
  readonly max: number;
}

/**
 * The CORE of the 2-hex start window: the tiles whose center is at least half
 * a hex away from either edge of the [0, 4.0] m band, so the contract holds
 * for a center-based AND for an extent-based band implementation.
 */
const START_BAND_CORE_WINDOW: ArcWindow = { min: 0.5, max: 3.5 };

/** The FAR hex of the 2-hex start window: the second hex, DEEP along the axis. */
const START_BAND_DEPTH_WINDOW: ArcWindow = { min: 2.5, max: 3.5 };

/** A window well OUTSIDE the start band - the plain-road stability guard. */
const OUTSIDE_BAND_WINDOW: ArcWindow = { min: 8, max: 12 };

/** Signed lateral offset (in meters) the band must reach on BOTH road sides. */
const FULL_ROAD_WIDTH_LATERAL_M = 1.5;

/**
 * Projects a world point onto the nearest point of the CLOSED spline polyline
 * and measures its position along the lap.
 *
 * The along-track measure is the ARC LENGTH along the polyline, measured
 * FORWARD from `splineSamples[0]` (segment 0's start point - the
 * start/finish plane), so `arcM` stays in `[0, loopLength)`. NEVER the
 * segment-0 CHORD projection: the track is a closed loop, so a chord-relative
 * measure spreads the start-line tiles over the whole lap.
 *
 * @param samples - The published closed-loop spline samples.
 * @param worldX - World x of the projected point.
 * @param worldZ - World z of the projected point (`worldZ = -y`, the same
 *   handedness the slab rasterizer uses).
 * @returns The forward arc length and the signed lateral offset.
 */
const projectOntoClosedSpline = (
  samples: readonly SplineSample[],
  worldX: number,
  worldZ: number,
): SplineProjection => {
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
};

/** The world key of one lattice tile centre. */
const tileCentreKey = (centreX: number, centreZ: number): string =>
  `${centreX},${centreZ}`;

/** The value key of one 0-255 RGB colour token. */
const colorKeyOf = (color: RgbColor): string =>
  `${color[0]},${color[1]},${color[2]}`;

/**
 * Compares two 0-255 RGB colour tokens BY VALUE - the tokens are compared
 * component by component, never by reference.
 *
 * @param left - The first colour token.
 * @param right - The second colour token.
 * @returns True when both tokens carry the same RGB components.
 */
const isSameColor = (left: RgbColor, right: RgbColor): boolean =>
  left[0] === right[0] && left[1] === right[1] && left[2] === right[2];

/** The value key of the plain ROAD fill token. */
const ROAD_FILL_COLOR_KEY = colorKeyOf(NEATRACING_ROAD_EDGE_FILL_COLOR);

/** The value key of the GREEN band fill token. */
const GREEN_FILL_COLOR_KEY = colorKeyOf(NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR);

/**
 * Resolves the VISIBLE fill colour of one tile centre.
 *
 * The band paint is laid OVER the plain road paint, so a centre that
 * carries BOTH colours renders the GREEN band token - the colour a viewer
 * actually sees. Any other colour set resolves to its first colour, so an
 * UNEXPECTED colour surfaces in the failure message instead of passing
 * silently.
 *
 * @param colorKeys - The value keys of every fill colour at that centre.
 * @param colorByKey - One representative colour per value key.
 * @returns The rendered (visible) fill colour of the centre.
 */
const resolveRenderedFillColor = (
  colorKeys: ReadonlySet<string>,
  colorByKey: ReadonlyMap<string, RgbColor>,
): RgbColor => {
  if (colorKeys.has(GREEN_FILL_COLOR_KEY)) {
    return NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR;
  }
  if (colorKeys.has(ROAD_FILL_COLOR_KEY)) {
    return NEATRACING_ROAD_EDGE_FILL_COLOR;
  }

  const [firstColorKey] = colorKeys;
  if (firstColorKey === undefined) {
    return NEATRACING_ROAD_EDGE_FILL_COLOR;
  }
  return colorByKey.get(firstColorKey) ?? NEATRACING_ROAD_EDGE_FILL_COLOR;
};

/**
 * The RENDERED fill colour of every lattice tile centre, keyed by the
 * tile centre's world (x, z).
 *
 * The lattice emits SIX fill triangles per tile and every one of them
 * starts at the tile CENTRE vertex
 * (`[tile.x, NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M, tile.z]`); because the
 * triangle factory gives all three vertices the SAME colour, that centre
 * vertex's colour IS the tile's rendered fill colour. The map is therefore
 * a CHANNEL-INDEPENDENT observable: it reads the rendered VALUE, never the
 * NAME of a field.
 *
 * @returns The rendered fill colour per tile centre.
 */
const buildRenderedFillColorsByTileCentre = (): ReadonlyMap<
  string,
  RgbColor
> => {
  const colorKeysByCentre = new Map<string, Set<string>>();
  const colorByKey = new Map<string, RgbColor>();

  for (const triangle of FIXTURE_LATTICE.fillTriangles) {
    const centreVertex = triangle.vertices[0];
    const centreKey = tileCentreKey(
      centreVertex.position[0],
      centreVertex.position[2],
    );
    const colorKey = colorKeyOf(centreVertex.color);

    if (!colorByKey.has(colorKey)) {
      colorByKey.set(colorKey, centreVertex.color);
    }

    const colorKeys = colorKeysByCentre.get(centreKey);
    if (colorKeys) {
      colorKeys.add(colorKey);
    } else {
      colorKeysByCentre.set(centreKey, new Set([colorKey]));
    }
  }

  const renderedByCentre = new Map<string, RgbColor>();
  for (const [centreKey, colorKeys] of colorKeysByCentre) {
    renderedByCentre.set(
      centreKey,
      resolveRenderedFillColor(colorKeys, colorByKey),
    );
  }
  return renderedByCentre;
};

/** The rendered fill colour of every lattice tile centre. */
const FIXTURE_RENDERED_FILL_COLORS = buildRenderedFillColorsByTileCentre();

/**
 * The RENDERED fill colour of one lattice tile centre.
 *
 * @param centreX - World x of the tile centre.
 * @param centreZ - World z of the tile centre.
 * @returns The rendered fill colour of that tile.
 */
const renderedFillColorOf = (centreX: number, centreZ: number): RgbColor =>
  FIXTURE_RENDERED_FILL_COLORS.get(tileCentreKey(centreX, centreZ)) ??
  NEATRACING_ROAD_EDGE_FILL_COLOR;

/**
 * The STRICT ROAD-CORE tiles of the lattice: the cells the slab surface
 * seam classifies as 'road' (NOT the curb band).
 *
 * The population is the SURFACE classification, NOT the colorway, so it
 * stays the SAME set before and after the green band lands (a
 * colorway-keyed population would silently shrink once the band lands).
 * The strict core is the population BOTH a core-only and a
 * core-plus-curb implementation satisfy.
 */
const collectRoadCoreTiles = (): readonly RoadCoreTile[] =>
  FIXTURE_LATTICE.tiles
    .filter((tile) => {
      const { surface } = classifyTrackSlabSurfaceAt(
        tile.x,
        tile.z,
        FIXTURE_TRACK_SPEC.segments,
        FIXTURE_EDGE_PROFILE,
      );
      return surface === 'road';
    })
    .map((tile) => {
      const { arcM, signedLateralM } = projectOntoClosedSpline(
        FIXTURE_TRACK_SPEC.splineSamples,
        tile.x,
        tile.z,
      );
      return {
        arcM,
        signedLateralM,
        colorway: tile.colorway,
        renderedFillColor: renderedFillColorOf(tile.x, tile.z),
      };
    });

/** The strict road-core tiles whose along-track arc falls inside the window. */
const selectArcWindow = (
  tiles: readonly RoadCoreTile[],
  window: ArcWindow,
): readonly RoadCoreTile[] =>
  tiles.filter((tile) => tile.arcM >= window.min && tile.arcM <= window.max);

describe('P5-S27-1 (RED-S27-1) the green 2-hex start line across the road', () => {
  describe('the 2-hex start window (the along-track band on the closed spline polyline)', () => {
    it('GUARD (NON-VACUITY): the 2-hex start window really carries strict road-core tiles', () => {
      const coreWindowTiles = selectArcWindow(
        collectRoadCoreTiles(),
        START_BAND_CORE_WINDOW,
      );

      expect(coreWindowTiles.length).toBeGreaterThan(0);
    });

    it('paints the start-line band in the GREEN fill tokens on the road tiles inside the 2-hex start window', () => {
      const coreWindowTiles = selectArcWindow(
        collectRoadCoreTiles(),
        START_BAND_CORE_WINDOW,
      );
      const nonGreenTiles = coreWindowTiles.filter(
        (tile) =>
          !isSameColor(
            tile.renderedFillColor,
            NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR,
          ),
      );

      expect(nonGreenTiles.map((tile) => tile.renderedFillColor)).toEqual([]);
    });

    it('paints the start line TWO hexes DEEP along the track axis', () => {
      const depthWindowTiles = selectArcWindow(
        collectRoadCoreTiles(),
        START_BAND_DEPTH_WINDOW,
      );
      const nonGreenTiles = depthWindowTiles.filter(
        (tile) =>
          !isSameColor(
            tile.renderedFillColor,
            NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR,
          ),
      );

      expect(nonGreenTiles.map((tile) => tile.renderedFillColor)).toEqual([]);
    });

    it('paints the start line across the FULL road width', () => {
      // THE BAND TILES ARE THE GREEN-PAINTED ONES: today the band does not
      // exist at all, so the band tile set of the core window is EMPTY and
      // this width clause is red BY ASSERTION (it can never pass vacuously -
      // the guard above pins a NON-EMPTY strict road-core population).
      const bandTiles = selectArcWindow(
        collectRoadCoreTiles(),
        START_BAND_CORE_WINDOW,
      ).filter((tile) =>
        isSameColor(
          tile.renderedFillColor,
          NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR,
        ),
      );
      const lateralOffsetsM = bandTiles.map((tile) => tile.signedLateralM);

      expect({
        reachesLeftRoadSide: lateralOffsetsM.some(
          (offsetM) => offsetM <= -FULL_ROAD_WIDTH_LATERAL_M,
        ),
        reachesRightRoadSide: lateralOffsetsM.some(
          (offsetM) => offsetM >= FULL_ROAD_WIDTH_LATERAL_M,
        ),
      }).toEqual({
        reachesLeftRoadSide: true,
        reachesRightRoadSide: true,
      });
    });

    it('keeps the plain ROAD band OUTSIDE the start window', () => {
      const outsideWindowTiles = selectArcWindow(
        collectRoadCoreTiles(),
        OUTSIDE_BAND_WINDOW,
      );
      const nonRoadTiles = outsideWindowTiles.filter(
        (tile) =>
          !isSameColor(tile.renderedFillColor, NEATRACING_ROAD_EDGE_FILL_COLOR),
      );

      expect(nonRoadTiles.map((tile) => tile.renderedFillColor)).toEqual([]);
    });

    it('leaves the road tiles colourway classification untouched', () => {
      // THE COLOURWAY-UNTOUCHED INVARIANT: the band is a NEW along-track
      // classification, so the road tiles keep their 'road' colorway and the
      // EXISTING road-edge green-border pins stay green.
      const coreWindowTiles = selectArcWindow(
        collectRoadCoreTiles(),
        START_BAND_CORE_WINDOW,
      );
      const recoloredTiles = coreWindowTiles.filter(
        (tile) => tile.colorway !== 'road',
      );

      expect(recoloredTiles.map((tile) => tile.colorway)).toEqual([]);
    });
  });

  describe('the EXPORTED TileColorway surface (the COMPILE-TIME type-level pin)', () => {
    it('keeps the exported TileColorway union at EXACTLY its four members', () => {
      expect(TILE_COLORWAY_PIN).toBe(true);
    });
  });
});
