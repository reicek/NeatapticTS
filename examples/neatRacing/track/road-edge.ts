/**
 * Hexagonal neon road-edge flat-paint builder for the neatRacing example
 * (plan slice P1-road-edge-hex, Step 14C).
 *
 * The roadway floor is paved edge-to-edge with a continuous honeycomb of
 * large 2 m diameter flat hexagonal tiles. Each tile is a bright neon edge
 * stroke around a one-shade darker same-hue fill; shared edges are stroked
 * exactly once, so neighbors join into one continuous hexagonal grid with no
 * corner gaps and no double borders. The default hue is the existing
 * cyan/teal palette; beyond the track edge, single-file border lines appear in
 * red, yellow, then green (~6 m past the edge total). All paint — darker fill
 * and bright stroke — shares one small positive y offset so it sits just above
 * the slab top face as a single flat plane. The darker fill ends at the inner
 * radius and the bright stroke covers the inner-to-outer radius band, so the
 * matte slab below never shows through as a black outline. There are no
 * vertical pillars or side faces.
 *
 * THE START LINE IS AN ALONG-TRACK BAND: the GREEN 2-hex start line (plan
 * slice P5-S27-2, AC-104) is painted across the roadway in the EXISTING
 * green fill + edge tokens. It rides on a NEW along-track classification
 * (`./road-edge.start-line` - the forward arc length along the published
 * closed spline polyline), so the EXPORTED `TileColorway` surface stays at
 * its FOUR members: the band tiles keep their 'road' colorway and the
 * green/yellow/red border rings beyond the track edge stay untouched.
 *
 * The honeycomb lattice is derived from the same track geometry that builds
 * the 0.5 m slab ribbon, so the paint aligns with the ribbon and its edge
 * markers rather than being a separate approximation. The sampled lattice
 * window is measured from the RIBBON itself (the centerline AABB grown by
 * the widest published half width) plus the full 6 m border band, so the
 * tri-color band stays COMPLETE around the convex corners, where the ribbon
 * cap reaches past the slab raster bounds.
 *
 * THE EDGE IS NOT THIS MODULE'S TO DECIDE: the band is measured from the
 * NAMED asymmetric edge published by the edge seam (`./track.edge`) - the
 * widened outer half-width on the outer side of a corner, the inner
 * half-width on the inner side - so the band starts where the road actually
 * ends. This module derives NO half-width of its own and keeps only its OWN
 * ratified policy: the ladder constants and their CUMULATIVE outer
 * semantics are untouched, only the base the bands are measured from moves.
 *
 * Determinism: {@link buildRoadEdgeHexVoxels} is a pure function of the same
 * `{ seed, layoutVersion, sizeBucket }` tuple used by the slab builder. Two
 * calls with the same tuple produce byte-identical `JSON.stringify` output on
 * the same runtime.
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import {
  isWithinStartLineBand,
  projectOntoClosedSpline,
  resolveStartLineBandDepthM,
} from './road-edge.start-line';
import { resolveTrackEdgeProfile } from './track.edge';
import type { TrackEdgeProfile } from './track.edge';
import {
  classifyTrackSlabSurfaceAt,
  type TrackSlabBuildInput,
} from './track.slabs';

/**
 * Corner-to-corner diameter of one flat hex tile, in meters. The roadway is
 * paved with a honeycomb of these tiles; the same 2 m cell is used for the
 * single-file red/yellow/green border lines beyond the track edge.
 */
export const NEATRACING_ROAD_EDGE_HEX_DIAMETER_M = 2.0;

/**
 * Circumradius of one flat hex tile: half of the corner-to-corner diameter.
 */
export const NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M =
  NEATRACING_ROAD_EDGE_HEX_DIAMETER_M / 2;

/**
 * Inradius that separates the bright edge stroke from the darker fill. The
 * stroke width is derived from this value so every shared edge is stroked
 * exactly once. The darker fill ends at this radius and the bright stroke
 * covers the inner-to-outer radius band on the same paint plane, so the
 * near-black matte slab below never shows through as a black outline (OBS-23).
 */
export const NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M = 0.75;

/**
 * Small positive world-up offset of the flat paint above the road surface,
 * in meters. Keeps the paint from z-fighting with the slab top face at y = 0.
 * Both the darker fill and the bright stroke share this height; there is no
 * separate stroke y offset (single-plane paint, OBS-23).
 */
export const NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M = 0.02;

/** Darker teal roadway fill color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_FILL_COLOR: readonly [
  number,
  number,
  number,
] = [20, 120, 120];

/** Bright cyan roadway edge color (0-255 RGB). Its Rec.709 luma exceeds the bloom threshold. */
export const NEATRACING_ROAD_EDGE_EDGE_COLOR: readonly [
  number,
  number,
  number,
] = [60, 255, 255];

/** Darker red border-line fill color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_RED_FILL_COLOR: readonly [
  number,
  number,
  number,
] = [160, 30, 30];

/** Bright red border-line edge color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_RED_EDGE_COLOR: readonly [
  number,
  number,
  number,
] = [255, 60, 60];

/** Darker yellow border-line fill color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_YELLOW_FILL_COLOR: readonly [
  number,
  number,
  number,
] = [160, 160, 30];

/** Bright yellow border-line edge color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_YELLOW_EDGE_COLOR: readonly [
  number,
  number,
  number,
] = [255, 255, 60];

/** Darker green border-line fill color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR: readonly [
  number,
  number,
  number,
] = [30, 160, 30];

/** Bright green border-line edge color (0-255 RGB). */
export const NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR: readonly [
  number,
  number,
  number,
] = [60, 255, 60];

/** One interleaved vertex: world position plus 0-255 RGB color. */
export interface RoadEdgeHexVertex {
  readonly position: readonly [number, number, number];
  readonly color: readonly [number, number, number];
}

/** One triangle of the hex road-edge mesh. */
export interface RoadEdgeHexTriangle {
  readonly vertices: readonly [
    RoadEdgeHexVertex,
    RoadEdgeHexVertex,
    RoadEdgeHexVertex,
  ];
}

/** Output mesh: an ordered list of colored triangles ready for VBO upload. */
export interface RoadEdgeHexMesh {
  readonly triangles: readonly RoadEdgeHexTriangle[];
}

/** One shared or boundary edge of the honeycomb lattice. */
export interface RoadEdgeHexLatticeEdge {
  /** Axial coordinates of the owning tile. */
  readonly q: number;
  /** Axial coordinates of the owning tile. */
  readonly r: number;
  /** Axial coordinates of the neighboring tile, or `null` for a boundary edge. */
  readonly neighborQ: number | null;
  /** Axial coordinates of the neighboring tile, or `null` for a boundary edge. */
  readonly neighborR: number | null;
  /** 0-255 RGB stroke color assigned to the shared edge. */
  readonly color: readonly [number, number, number];
  /** First endpoint of the edge segment (outer hex vertex). */
  readonly p1: readonly [number, number, number];
  /** Second endpoint of the edge segment (outer hex vertex). */
  readonly p2: readonly [number, number, number];
}

/** One tile of the honeycomb lattice. */
export interface RoadEdgeHexLatticeTile {
  /** Axial coordinate q. */
  readonly q: number;
  /** Axial coordinate r. */
  readonly r: number;
  /** World-space center x. */
  readonly x: number;
  /** World-space center z. */
  readonly z: number;
  /** Colorway classification. */
  readonly colorway: TileColorway;
  /**
   * Whether this tile carries the GREEN start-line band paint.
   *
   * The band is a NEW ALONG-TRACK classification, so it rides BESIDE the
   * border colorway and never inside it: a band tile KEEPS its 'road'
   * colorway and the EXPORTED `TileColorway` surface stays at its four
   * members.
   */
  readonly startLineBand: boolean;
}

/** Complete deterministic honeycomb lattice including fill and stroke geometry. */
export interface RoadEdgeHexLattice {
  readonly tiles: readonly RoadEdgeHexLatticeTile[];
  readonly fillTriangles: readonly RoadEdgeHexTriangle[];
  readonly edgeTriangles: readonly RoadEdgeHexTriangle[];
  readonly edges: readonly RoadEdgeHexLatticeEdge[];
}

/** Synthetic type alias for the public input tuple. */
export type RoadEdgeHexBuildInput = TrackSlabBuildInput;

/**
 * Tile colorway emitted for one honeycomb cell.
 *
 * `road` covers the drivable ribbon and its curb band; the three border
 * colorways name the successive rings painted beyond the track edge.
 */
export type TileColorway = 'road' | 'red' | 'yellow' | 'green';

const HEX_VERTEX_COUNT = 6;
const HEX_ANGLE_STEP = Math.PI / 3;
const HEX_APOTHEM_M =
  NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M * (Math.sqrt(3) / 2);

/**
 * Half-width of the shared edge stroke. The stroke is centered on each
 * tile boundary and extends inward exactly to the stroke boundary radius,
 * so adjacent tiles meet with a single continuous line and no overlap.
 */
const HEX_STROKE_HALF_WIDTH_M =
  HEX_APOTHEM_M - NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M;

/**
 * CUMULATIVE outward distance from the track edge, in meters, at which the
 * green band ENDS. The green band therefore spans 0-2 m past the edge.
 *
 * These three limits are CUMULATIVE outer bounds, not per-band widths: each
 * one is measured from the track edge, so the yellow band is 2 m wide (2-4 m
 * past the edge) and the red band is 2 m wide (4-6 m past the edge). Do not
 * read them as the width of their own band.
 */
export const ROAD_EDGE_BORDER_GREEN_M = 2;

/**
 * CUMULATIVE outward distance from the track edge, in meters, at which the
 * yellow band ENDS. The yellow band therefore spans 2-4 m past the edge.
 *
 * This is a cumulative outer bound measured from the track edge, NOT the
 * width of the yellow band itself (the yellow band is 2 m wide).
 */
export const ROAD_EDGE_BORDER_YELLOW_M = 4;

/**
 * CUMULATIVE outward distance from the track edge, in meters, at which the
 * red band ENDS. The red band therefore spans 4-6 m past the edge.
 *
 * This is a cumulative outer bound measured from the track edge, NOT the
 * width of the red band itself (the red band is 2 m wide). It is also the
 * OUTERMOST painted distance: no tile is emitted beyond it.
 */
export const ROAD_EDGE_BORDER_RED_M = 6;

/**
 * The forward depth of the GREEN start-line band along the track axis, in
 * meters: TWO hexes deep (2 x NEATRACING_ROAD_EDGE_HEX_DIAMETER_M = 4.0 m).
 *
 * The depth is DERIVED from the hex diameter, so the band can never drift
 * away from the tile size it is painted on.
 */
const START_LINE_BAND_DEPTH_M = resolveStartLineBandDepthM(
  NEATRACING_ROAD_EDGE_HEX_DIAMETER_M,
);

/** Precedence used to choose the visible color of a shared edge. */
const COLORWAY_PRECEDENCE: Readonly<Record<TileColorway, number>> = {
  road: 0,
  green: 1,
  yellow: 2,
  red: 3,
};

/**
 * Precedence of the GREEN start-line band paint.
 *
 * The band lives INSIDE the roadway (its tiles keep the 'road' colorway), so
 * it outranks the plain road paint - the band owns the shared edges it
 * touches and the stripe keeps a crisp outline - and it stays BELOW the
 * border ladder, so the green/yellow/red rings never lose a shared edge to
 * the start line.
 */
const START_LINE_BAND_PRECEDENCE = 0.5;

/**
 * Axial neighbor directions for a flat-topped hex grid. Each entry maps a
 * direction to the two outer-vertex indices that form the shared edge on that
 * side of the tile.
 */
const HEX_NEIGHBOR_DIRECTIONS: readonly {
  readonly dq: number;
  readonly dr: number;
  readonly edge: readonly [number, number];
}[] = [
  { dq: 1, dr: 0, edge: [0, 1] },
  { dq: 1, dr: -1, edge: [5, 0] },
  { dq: 0, dr: -1, edge: [4, 5] },
  { dq: -1, dr: 0, edge: [3, 4] },
  { dq: -1, dr: 1, edge: [2, 3] },
  { dq: 0, dr: 1, edge: [1, 2] },
];

/**
 * Builds the flat hexagonal neon road-edge paint mesh for the deterministic
 * track tuple.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns Colored triangle mesh of 2 m flat hex tiles paving the roadway
 *   floor and the red/yellow/green border lines beyond the track edge.
 */
export function buildRoadEdgeHexVoxels(
  input: RoadEdgeHexBuildInput,
): RoadEdgeHexMesh {
  const lattice = buildRoadEdgeHexLattice(input);

  return {
    triangles: [...lattice.fillTriangles, ...lattice.edgeTriangles],
  };
}

/**
 * Builds the complete honeycomb lattice for the deterministic track tuple.
 *
 * Exported as a test seam so contract tests can verify edge-to-edge tiling,
 * single-stroke shared edges, and border-line ordering without
 * reverse-engineering geometry from overlapping triangle vertices.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @returns Lattice with tiles, fill/edge triangles, and a canonical edge list.
 */
export function buildRoadEdgeHexLattice(
  input: TrackSlabBuildInput,
): RoadEdgeHexLattice {
  const spec = generateTrack(input);
  // The edge profile is resolved ONCE for the whole lattice: it is the
  // published width profile every cell measures the road through.
  const edgeProfile = resolveTrackEdgeProfile(spec.segments.slice());

  const tiles = collectHoneycombTiles(spec, edgeProfile);
  const tileMap = new Map<string, RoadEdgeHexLatticeTile>();
  for (const tile of tiles) {
    tileMap.set(tileKey(tile.q, tile.r), tile);
  }

  const fillTriangles: RoadEdgeHexTriangle[] = [];
  const edgeTriangles: RoadEdgeHexTriangle[] = [];
  const edges: RoadEdgeHexLatticeEdge[] = [];

  for (const tile of tiles) {
    appendTileFillTriangles(tile, fillTriangles);

    for (const { dq, dr, edge } of HEX_NEIGHBOR_DIRECTIONS) {
      const neighborQ = tile.q + dq;
      const neighborR = tile.r + dr;
      const neighbor = tileMap.get(tileKey(neighborQ, neighborR));

      // Canonical ownership: emit a shared edge only from the lexicographically
      // smaller tile. Boundary edges are always emitted by the lone tile.
      const isOwner =
        !neighbor ||
        tile.q < neighborQ ||
        (tile.q === neighborQ && tile.r < neighborR);
      if (!isOwner) {
        continue;
      }

      const edgeColor = resolveEdgeColor(tile, neighbor ?? null);
      const [p1, p2] = tileOuterEdgeVertices(tile, edge[0], edge[1]);

      edges.push({
        q: tile.q,
        r: tile.r,
        neighborQ,
        neighborR,
        color: edgeColor,
        p1,
        p2,
      });

      appendEdgeStrokeTriangles(p1, p2, edgeColor, edgeTriangles);
    }
  }

  return {
    tiles,
    fillTriangles,
    edgeTriangles,
    edges,
  };
}

/**
 * Builds a flat-topped axial honeycomb grid covering the track ribbon plus
 * the full ~6 m border band, classifies each cell center with the same
 * track-edge logic that paints the ribbon, and returns the cells that should
 * emit tiles.
 *
 * The sampled window is measured from the RIBBON (the union of the
 * per-segment capsules), NOT from the slab raster bounds: the slab grid
 * spans the centerline AABB, so at a convex corner the ribbon cap - and with
 * it the whole border band - reaches past that grid. A window derived from
 * the slab bounds therefore cut the red shell off at the outer corners and
 * left them with an incomplete marker set.
 *
 * @param spec - Generated track spec.
 * @param edgeProfile - The edge profile published by the edge seam for the
 *   ribbon being painted.
 * @returns The cells that should emit tiles.
 */
function collectHoneycombTiles(
  spec: ReturnType<typeof generateTrack>,
  edgeProfile: TrackEdgeProfile,
): RoadEdgeHexLatticeTile[] {
  const outerR = NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M;
  const bounds = measureTrackBorderBandBounds(spec, edgeProfile);

  // One cell of slack on every side keeps the window inclusive of the
  // outermost band cells.
  const pad = 1;
  const qMin = Math.floor(bounds.minX / (outerR * 1.5)) - pad;
  const qMax = Math.ceil(bounds.maxX / (outerR * 1.5)) + pad;

  const tiles: RoadEdgeHexLatticeTile[] = [];
  const seen = new Set<string>();

  for (let q = qMin; q <= qMax; q++) {
    const z0 = (Math.sqrt(3) / 2) * outerR * q;
    const rMin = Math.floor((bounds.minZ - z0) / (outerR * Math.sqrt(3))) - pad;
    const rMax = Math.ceil((bounds.maxZ - z0) / (outerR * Math.sqrt(3))) + pad;

    for (let r = rMin; r <= rMax; r++) {
      const [x, z] = axialToWorld(q, r);
      const { surface, distancePastEdgeM } = classifyTrackSlabSurfaceAt(
        x,
        z,
        spec.segments,
        edgeProfile,
      );

      let colorway: TileColorway | null = null;
      if (surface === 'road' || surface === 'curb') {
        colorway = 'road';
      } else if (distancePastEdgeM <= ROAD_EDGE_BORDER_GREEN_M) {
        colorway = 'green';
      } else if (distancePastEdgeM <= ROAD_EDGE_BORDER_YELLOW_M) {
        colorway = 'yellow';
      } else if (distancePastEdgeM <= ROAD_EDGE_BORDER_RED_M) {
        colorway = 'red';
      }

      if (!colorway) {
        continue;
      }

      const key = tileKey(q, r);
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);

      // THE START LINE IS AN ALONG-TRACK CLASSIFICATION: it rides beside the
      // border colorway, so the road tiles keep their 'road' colorway and
      // the EXPORTED `TileColorway` surface stays at its four members.
      const startLineBand =
        colorway === 'road' &&
        isWithinStartLineBand(
          projectOntoClosedSpline(spec.splineSamples, x, z).arcM,
          START_LINE_BAND_DEPTH_M,
        );

      tiles.push({ q, r, x, z, colorway, startLineBand });
    }
  }

  // Sort deterministically so JSON.stringify output is stable across runs.
  tiles.sort((a, b) => {
    if (a.q !== b.q) return a.q - b.q;
    if (a.r !== b.r) return a.r - b.r;
    return a.colorway.localeCompare(b.colorway);
  });

  return tiles;
}

/**
 * Measures the world-space rectangle that has to be sampled so the tri-color
 * border band is COMPLETE around the whole ribbon.
 *
 * The ribbon is the union of the per-segment capsules (each chord grown by
 * its PUBLISHED half width - the widened outer half-width on the outer side
 * of a corner, the inner half-width on the inner side), so the ribbon AABB
 * is the centerline AABB grown by the WIDEST published half width. The
 * border band reaches 6 m past that ribbon, and one hex radius of margin is
 * added so the outermost tile geometry stays inside the sampled rectangle.
 *
 * The world z axis mirrors the generator y axis (`worldZ = -y`), the same
 * handedness the slab builder uses.
 *
 * A spec with NO segments has no ribbon to grow, so the helper returns an
 * inverted (empty) rectangle instead of an unbounded one: the caller then
 * samples no cell at all and emits no tile. This keeps the helper total for
 * degenerate specs - it never yields NaN bounds and never throws.
 *
 * @param spec - Generated track spec.
 * @param edgeProfile - The edge profile published by the edge seam for the
 *   ribbon being painted.
 * @returns World-space bounds of the sampled rectangle, in meters.
 */
function measureTrackBorderBandBounds(
  spec: ReturnType<typeof generateTrack>,
  edgeProfile: TrackEdgeProfile,
): {
  readonly minX: number;
  readonly maxX: number;
  readonly minZ: number;
  readonly maxZ: number;
} {
  if (spec.segments.length === 0) {
    return {
      minX: Number.POSITIVE_INFINITY,
      maxX: Number.NEGATIVE_INFINITY,
      minZ: Number.POSITIVE_INFINITY,
      maxZ: Number.NEGATIVE_INFINITY,
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  // The WIDEST published half width of the ribbon: the outer half-width of
  // a corner is the widest one, so the window still covers the whole
  // asymmetric ribbon and the band around it.
  let maxHalfWidthM = 0;

  for (const segment of spec.segments) {
    minX = Math.min(minX, segment.startX, segment.endX);
    maxX = Math.max(maxX, segment.startX, segment.endX);
    minY = Math.min(minY, segment.startY, segment.endY);
    maxY = Math.max(maxY, segment.startY, segment.endY);
  }

  for (const chord of edgeProfile.chords) {
    maxHalfWidthM = Math.max(maxHalfWidthM, chord.outerHalfWidthM);
  }

  const marginM =
    maxHalfWidthM +
    ROAD_EDGE_BORDER_RED_M +
    NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M;

  return {
    minX: minX - marginM,
    maxX: maxX + marginM,
    minZ: -maxY - marginM,
    maxZ: -minY + marginM,
  };
}

/**
 * Builds a 2 m honeycomb grid aligned to the track ribbon geometry,
 * classifies each cell center with the same track-edge logic that paints the
 * ribbon, and returns every cell that should emit a roadway or border-line
 * tile.
 *
 * Exported as a test seam so contract tests can verify lattice alignment
 * without reverse-engineering centers from overlapping triangle vertices.
 */
export function collectRoadEdgeTileCenters(input: TrackSlabBuildInput): Array<{
  readonly x: number;
  readonly z: number;
  readonly colorway: TileColorway;
}> {
  const lattice = buildRoadEdgeHexLattice(input);

  return lattice.tiles.map(({ x, z, colorway }) => ({
    x,
    z,
    colorway,
  }));
}

/**
 * Converts flat-topped axial coordinates `(q, r)` to world-space `(x, z)`.
 */
function axialToWorld(q: number, r: number): readonly [number, number] {
  const outerR = NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M;
  return [
    outerR * 1.5 * q,
    outerR * ((Math.sqrt(3) / 2) * q + Math.sqrt(3) * r),
  ] as const;
}

function tileKey(q: number, r: number): string {
  return `${q},${r}`;
}

/**
 * Resolves the fill+edge color pair for a tile colorway.
 */
function colorwayToColors(
  colorway: TileColorway,
): readonly [
  readonly [number, number, number],
  readonly [number, number, number],
] {
  if (colorway === 'red') {
    return [
      NEATRACING_ROAD_EDGE_RED_FILL_COLOR,
      NEATRACING_ROAD_EDGE_RED_EDGE_COLOR,
    ];
  }
  if (colorway === 'yellow') {
    return [
      NEATRACING_ROAD_EDGE_YELLOW_FILL_COLOR,
      NEATRACING_ROAD_EDGE_YELLOW_EDGE_COLOR,
    ];
  }
  if (colorway === 'green') {
    return [
      NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR,
      NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR,
    ];
  }
  return [NEATRACING_ROAD_EDGE_FILL_COLOR, NEATRACING_ROAD_EDGE_EDGE_COLOR];
}

/**
 * The FILL color of one tile.
 *
 * The start-line band paints in the EXISTING green fill token, so the band
 * reads as a green stripe across the racing line without a new palette.
 */
function tileFillColor(
  tile: RoadEdgeHexLatticeTile,
): readonly [number, number, number] {
  return tile.startLineBand
    ? NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR
    : colorwayToColors(tile.colorway)[0];
}

/**
 * The EDGE (stroke) color of one tile.
 */
function tileEdgeColor(
  tile: RoadEdgeHexLatticeTile,
): readonly [number, number, number] {
  return tile.startLineBand
    ? NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR
    : colorwayToColors(tile.colorway)[1];
}

/**
 * The paint PRECEDENCE of one tile: the border ladder rank of its colorway,
 * or the start-line band rank when the tile carries the band paint.
 */
function tilePaintPrecedence(tile: RoadEdgeHexLatticeTile): number {
  return tile.startLineBand
    ? START_LINE_BAND_PRECEDENCE
    : COLORWAY_PRECEDENCE[tile.colorway];
}

/**
 * Chooses the visible color for a shared edge. The paint with higher
 * precedence wins, matching the border-line ordering (road → red → yellow →
 * green) with the start-line band sitting between the plain road paint and
 * the border ladder. Boundary edges use the tile's own edge color.
 */
function resolveEdgeColor(
  tile: RoadEdgeHexLatticeTile,
  neighborTile: RoadEdgeHexLatticeTile | null,
): readonly [number, number, number] {
  const edgeColor = tileEdgeColor(tile);
  if (!neighborTile) {
    return edgeColor;
  }

  return tilePaintPrecedence(tile) >= tilePaintPrecedence(neighborTile)
    ? edgeColor
    : tileEdgeColor(neighborTile);
}

/**
 * Appends the six fill triangles for one tile to the output list.
 */
function appendTileFillTriangles(
  tile: RoadEdgeHexLatticeTile,
  out: RoadEdgeHexTriangle[],
): void {
  const paintY = NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M;
  // Single-plane paint (OBS-23): the darker fill ends exactly at the inner
  // radius and abuts the bright stroke ring on the same y plane.
  const innerR = NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M;
  const fillColor = tileFillColor(tile);
  const center: readonly [number, number, number] = [tile.x, paintY, tile.z];
  const inner = hexVertices(tile.x, tile.z, innerR, paintY);

  for (let i = 0; i < HEX_VERTEX_COUNT; i++) {
    const next = (i + 1) % HEX_VERTEX_COUNT;
    out.push(makeTriangle(center, inner[i]!, inner[next]!, fillColor));
  }
}

/**
 * Appends two triangles that form a thin stroke centered on the edge segment
 * `p1-p2`. The stroke is extended by a half-width perpendicular offset and
 * by half the stroke width along the edge direction so adjacent stroke pieces
 * meet cleanly at hex vertices without leaving corner gaps.
 */
function appendEdgeStrokeTriangles(
  p1: readonly [number, number, number],
  p2: readonly [number, number, number],
  color: readonly [number, number, number],
  out: RoadEdgeHexTriangle[],
): void {
  const halfWidth = HEX_STROKE_HALF_WIDTH_M;
  const dx = p2[0] - p1[0];
  const dz = p2[2] - p1[2];
  const length = Math.hypot(dx, dz);
  if (length === 0) {
    return;
  }

  const ux = dx / length;
  const uz = dz / length;
  const nx = -uz * halfWidth;
  const nz = ux * halfWidth;

  const a: readonly [number, number, number] = [
    p1[0] - ux * halfWidth,
    p1[1],
    p1[2] - uz * halfWidth,
  ];
  const b: readonly [number, number, number] = [
    p2[0] + ux * halfWidth,
    p2[1],
    p2[2] + uz * halfWidth,
  ];

  const aOut: readonly [number, number, number] = [a[0] + nx, a[1], a[2] + nz];
  const aIn: readonly [number, number, number] = [a[0] - nx, a[1], a[2] - nz];
  const bOut: readonly [number, number, number] = [b[0] + nx, b[1], b[2] + nz];
  const bIn: readonly [number, number, number] = [b[0] - nx, b[1], b[2] - nz];

  out.push(makeTriangle(aOut, aIn, bOut, color));
  out.push(makeTriangle(bOut, aIn, bIn, color));
}

/**
 * Computes the two outer vertices that form an edge of a tile.
 */
function tileOuterEdgeVertices(
  tile: RoadEdgeHexLatticeTile,
  vertexIndex0: number,
  vertexIndex1: number,
): readonly [
  readonly [number, number, number],
  readonly [number, number, number],
] {
  const paintY = NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M;
  const outer = hexVertices(
    tile.x,
    tile.z,
    NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M,
    paintY,
  );
  return [outer[vertexIndex0]!, outer[vertexIndex1]!];
}

/**
 * Computes the six flat-topped hex vertices at the given radius and height.
 */
function hexVertices(
  centerX: number,
  centerZ: number,
  radius: number,
  y: number,
): readonly (readonly [number, number, number])[] {
  const result: Array<readonly [number, number, number]> = [];

  for (let i = 0; i < HEX_VERTEX_COUNT; i++) {
    const angle = i * HEX_ANGLE_STEP;
    result.push([
      centerX + radius * Math.cos(angle),
      y,
      centerZ + radius * Math.sin(angle),
    ] as const);
  }

  return result;
}

function makeTriangle(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
  color: readonly [number, number, number],
): RoadEdgeHexTriangle {
  return {
    vertices: [
      { position: a, color },
      { position: b, color },
      { position: c, color },
    ] as const,
  };
}
