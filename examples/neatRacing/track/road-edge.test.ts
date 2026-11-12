/**
 * ============================================================================
 * P1-outer-edge-red (plan step 14U) - the ASYMMETRIC EDGE MODEL - the BAND
 * lane (RED-3).
 *
 * AC tags: AC-095 (the asymmetric edge - the named, curvature-derived
 * inner/outer half-width pair, the outer one growing with the local corner
 * measure), AC-064 (the tri-colour band fill and its CUMULATIVE outer
 * semantics retained) + AC-080 (the ladder cut-off - the ladder stays a
 * PAINT ladder), AC-090 (the user gate is NOT claimed - this lane closes
 * nothing and records no verdict).
 *
 * RED-3 (DISCRIMINATING - the lane FAILS today on its own assertion): the
 * tri-colour band starts at the ASYMMETRIC OUTER EDGE on a corner (the band
 * is painted from the widened outer edge, NOT from the retired symmetric
 * half-width), the CUMULATIVE outer semantics are UNCHANGED (each band still
 * ends at its own cumulative limit and the ladder order is unchanged) and
 * the ladder STAYS A PAINT LADDER (the band is never a physical bound - the
 * containment bound is a DIFFERENT thing and the two are never conflated).
 * The contract opens with a NON-VACUITY GUARD: the probed corner MUST really
 * carry a painted band past the widened edge, because a zero-width band
 * would satisfy the starts-at clause vacuously. The probe is RUNTIME-ONLY:
 * it reads the EXISTING band paint and the EXISTING surface classification,
 * so the lane is red BY ASSERTION and never by an import failure.
 *
 * The drift space (the measured outer extension) is a NAMED EXPORT owned by
 * the implementing slice: THIS contract NEVER carries its number - the
 * band base is compared RELATIVE to the retired symmetric half-width.
 *
 * Standing constraints (plan: zero per-frame allocation (P9) and the
 * capacity limits maxNodes 8000 / maxConnections 32000, AC-029): this RED
 * lane adds no per-frame allocation and no node/connection growth - it only
 * PROBES the existing readers.
 *
 * ============================================================================
 *
 * Contract tests for the Neat Racing hexagonal neon road-edge flat paint
 * module (slice P1-road-edge-hex, Step 14C).
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Exports the named pure builder {@link buildRoadEdgeHexVoxels} taking the
 *   same determinism tuple `{ seed, layoutVersion, sizeBucket }` used by the
 *   track slab builder.
 * - Returns a {@link RoadEdgeHexMesh} of colored triangles whose vertices all
 *   sit at the same small positive paint height (flat floor paint), never as
 *   vertical pillars and never below the transparent ground.
 * - Each hex tile is 2 m corner-to-corner and has a darker fill color with a
 *   slightly brighter same-hue edge color so the top hexagon reads as a
 *   neon-lit frame.
 * - The roadway floor carries cyan/teal tiles that cover the road core and
 *   the curb band; beyond the track edge there are single-hex-wide red, yellow,
 *   and green border-line tiles.
 * - Tiles sit on a true flat-topped honeycomb lattice: edge-sharing neighbors
 *   are spaced `sqrt(3) * R`, every shared edge is stroked exactly once, and
 *   boundary edges are still stroked so the pavement reads as one continuous
 *   hexagonal grid with no corner gaps and no double borders.
 * - The builder is deterministic: two calls with the same tuple produce the
 *   same serialized mesh on the same runtime.
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';
import { classifyTrackSlabSurfaceAt } from './track.slabs';
import {
  NEATRACING_ROAD_EDGE_HEX_DIAMETER_M,
  NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M,
  NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M,
  NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M,
  NEATRACING_ROAD_EDGE_FILL_COLOR,
  NEATRACING_ROAD_EDGE_EDGE_COLOR,
  NEATRACING_ROAD_EDGE_RED_FILL_COLOR,
  NEATRACING_ROAD_EDGE_RED_EDGE_COLOR,
  NEATRACING_ROAD_EDGE_YELLOW_FILL_COLOR,
  NEATRACING_ROAD_EDGE_YELLOW_EDGE_COLOR,
  NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR,
  NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR,
  buildRoadEdgeHexLattice,
  buildRoadEdgeHexVoxels,
  collectRoadEdgeTileCenters,
  ROAD_EDGE_BORDER_GREEN_M,
  ROAD_EDGE_BORDER_YELLOW_M,
  ROAD_EDGE_BORDER_RED_M,
  type RoadEdgeHexLattice,
  type RoadEdgeHexMesh,
  type TileColorway,
} from './road-edge';

/** Canonical determinism tuple reused from the track slab builder. */
const FIXTURE_TRACK_BUILD_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'small',
} as const;

/** Pre-built track spec for surface classification tests. */
const FIXTURE_TRACK_SPEC = generateTrack(FIXTURE_TRACK_BUILD_INPUT);

/** Serializes a hex mesh for byte-identity comparison. */
const serializeHexMesh = (mesh: RoadEdgeHexMesh): string =>
  JSON.stringify(mesh);

/** Rec709 luminance of a 0-255 RGB color. */
const lumaOf = (color: readonly [number, number, number]): number => {
  const [r, g, b] = color;
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
};

/** Builds the honeycomb lattice once for the test fixture. */
const buildFixtureLattice = (): RoadEdgeHexLattice =>
  buildRoadEdgeHexLattice(FIXTURE_TRACK_BUILD_INPUT);

describe('buildRoadEdgeHexVoxels (P1-road-edge-hex flat paint)', () => {
  it('exports the named diameter, offset, radius, and color constants', () => {
    expect(NEATRACING_ROAD_EDGE_HEX_DIAMETER_M).toBe(2);
    expect(NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M).toBe(1);
    expect(NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M).toBeGreaterThan(0);
    expect(NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M).toBeGreaterThan(
      NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M,
    );
    expect(NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M).toBeGreaterThan(0);
  });

  it('returns a mesh of colored triangles for the default track tuple', () => {
    const mesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);

    expect(mesh.triangles.length).toBeGreaterThan(0);
    expect(mesh.triangles[0].vertices.length).toBe(3);
  });

  it('keeps every vertex at the flat paint height and never below ground', () => {
    const mesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);
    const ys = mesh.triangles
      .flatMap((triangle) => triangle.vertices)
      .map((vertex) => vertex.position[1]);

    expect(ys.every((y) => y >= -1e-9)).toBe(true);
    // Single-plane paint (OBS-23): the darker fill and the bright edge stroke
    // share the same small positive paint height; there is no separate stroke
    // y offset to create coplanar depth interleaving or a dark seam.
    for (const y of ys) {
      expect(
        Math.abs(y - NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M),
      ).toBeLessThanOrEqual(1e-9);
    }
    expect(ys).toContain(NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M);
    const yMin = ys.reduce(function (acc, v) {
      return Math.min(acc, v);
    }, Infinity);
    const yMax = ys.reduce(function (acc, v) {
      return Math.max(acc, v);
    }, -Infinity);
    const yRange = yMax - yMin;
    expect(yRange).toBeLessThanOrEqual(1e-9);
  });

  it('uses brighter edge colors than their same-hue fill colors', () => {
    expect(lumaOf(NEATRACING_ROAD_EDGE_EDGE_COLOR)).toBeGreaterThan(
      lumaOf(NEATRACING_ROAD_EDGE_FILL_COLOR),
    );
    expect(lumaOf(NEATRACING_ROAD_EDGE_RED_EDGE_COLOR)).toBeGreaterThan(
      lumaOf(NEATRACING_ROAD_EDGE_RED_FILL_COLOR),
    );
    expect(lumaOf(NEATRACING_ROAD_EDGE_YELLOW_EDGE_COLOR)).toBeGreaterThan(
      lumaOf(NEATRACING_ROAD_EDGE_YELLOW_FILL_COLOR),
    );
    expect(lumaOf(NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR)).toBeGreaterThan(
      lumaOf(NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR),
    );
  });

  it('contains no near-black strokes between the hex fill and bright edge ring', () => {
    const mesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);
    const darkThreshold = 10 / 255;

    for (const triangle of mesh.triangles) {
      for (const vertex of triangle.vertices) {
        const [r, g, b] = vertex.color;
        const maxChannel = Math.max(r, g, b);
        expect(maxChannel).toBeGreaterThan(darkThreshold);
      }
    }
  });

  it('includes roadway, red, yellow, and green colors in the generated triangles', () => {
    const mesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);
    const colors = new Set(
      mesh.triangles
        .flatMap((triangle) => triangle.vertices)
        .map((vertex) => JSON.stringify(vertex.color)),
    );

    expect(colors.has(JSON.stringify(NEATRACING_ROAD_EDGE_FILL_COLOR))).toBe(
      true,
    );
    expect(colors.has(JSON.stringify(NEATRACING_ROAD_EDGE_EDGE_COLOR))).toBe(
      true,
    );
    expect(
      colors.has(JSON.stringify(NEATRACING_ROAD_EDGE_RED_FILL_COLOR)),
    ).toBe(true);
    expect(
      colors.has(JSON.stringify(NEATRACING_ROAD_EDGE_YELLOW_FILL_COLOR)),
    ).toBe(true);
    expect(
      colors.has(JSON.stringify(NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR)),
    ).toBe(true);
  });

  it('lays tiles on a true honeycomb lattice with edge-sharing neighbors at sqrt(3) * R', () => {
    const lattice = buildFixtureLattice();
    const tileSet = new Set(lattice.tiles.map((tile) => `${tile.q},${tile.r}`));
    const outerR = NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M;
    const expectedNeighborSpacing = Math.sqrt(3) * outerR;
    const directions = [
      [1, 0],
      [1, -1],
      [0, -1],
      [-1, 0],
      [-1, 1],
      [0, 1],
    ] as const;

    expect(lattice.tiles.length).toBeGreaterThan(0);

    for (const tile of lattice.tiles) {
      let neighbors = 0;
      for (const [dq, dr] of directions) {
        const neighborQ = tile.q + dq;
        const neighborR = tile.r + dr;
        if (tileSet.has(`${neighborQ},${neighborR}`)) {
          neighbors += 1;
        }
      }
      expect(neighbors).toBeLessThanOrEqual(6);
    }

    // Sample one edge-sharing pair to confirm the world-space spacing.
    const first = lattice.tiles[0]!;
    const neighbor = directions
      .map(([dq, dr]) =>
        lattice.tiles.find(
          (tile) => tile.q === first.q + dq && tile.r === first.r + dr,
        ),
      )
      .find(Boolean);
    expect(neighbor).toBeDefined();
    if (neighbor) {
      const distance = Math.hypot(first.x - neighbor.x, first.z - neighbor.z);
      expect(distance).toBeCloseTo(expectedNeighborSpacing, 6);
    }
  });

  it('covers the road core and curb band with roadway tiles', () => {
    const centers = collectRoadEdgeTileCenters(FIXTURE_TRACK_BUILD_INPUT);
    const roadCenters = centers.filter(({ colorway }) => colorway === 'road');

    expect(roadCenters.length).toBeGreaterThan(0);

    for (const { x, z } of roadCenters) {
      const { surface } = classifyTrackSlabSurfaceAt(
        x,
        z,
        FIXTURE_TRACK_SPEC.segments,
      );
      expect(surface === 'road' || surface === 'curb').toBe(true);
    }
  });

  it('steps border-line tiles outward in green, yellow, red bands', () => {
    const centers = collectRoadEdgeTileCenters(FIXTURE_TRACK_BUILD_INPUT);
    const colors = new Set<string>();

    for (const { x, z, colorway } of centers) {
      if (colorway === 'road') continue;
      const { distancePastEdgeM } = classifyTrackSlabSurfaceAt(
        x,
        z,
        FIXTURE_TRACK_SPEC.segments,
      );

      if (colorway === 'green') {
        expect(distancePastEdgeM).toBeGreaterThan(0);
        expect(distancePastEdgeM).toBeLessThanOrEqual(2);
      } else if (colorway === 'yellow') {
        expect(distancePastEdgeM).toBeGreaterThan(2);
        expect(distancePastEdgeM).toBeLessThanOrEqual(4);
      } else if (colorway === 'red') {
        expect(distancePastEdgeM).toBeGreaterThan(4);
        expect(distancePastEdgeM).toBeLessThanOrEqual(6);
      }

      colors.add(colorway);
    }

    expect(colors.has('green')).toBe(true);
    expect(colors.has('yellow')).toBe(true);
    expect(colors.has('red')).toBe(true);
  });

  it('strokes every shared edge exactly once and keeps boundary edges', () => {
    const lattice = buildFixtureLattice();
    const incidentCount = new Map<string, number>();

    for (const edge of lattice.edges) {
      const ownerKey = `${edge.q},${edge.r}`;
      incidentCount.set(ownerKey, (incidentCount.get(ownerKey) ?? 0) + 1);
      if (edge.neighborQ !== null && edge.neighborR !== null) {
        const neighborKey = `${edge.neighborQ},${edge.neighborR}`;
        incidentCount.set(
          neighborKey,
          (incidentCount.get(neighborKey) ?? 0) + 1,
        );
      }
    }

    for (const tile of lattice.tiles) {
      const key = `${tile.q},${tile.r}`;
      expect(incidentCount.get(key)).toBe(6);
    }
  });

  it('uses the outer border colorway at road→border transitions', () => {
    const lattice = buildFixtureLattice();

    for (const edge of lattice.edges) {
      if (edge.neighborQ === null || edge.neighborR === null) continue;
      const owner = lattice.tiles.find(
        (tile) => tile.q === edge.q && tile.r === edge.r,
      );
      const neighbor = lattice.tiles.find(
        (tile) => tile.q === edge.neighborQ && tile.r === edge.neighborR,
      );
      if (!owner || !neighbor) continue;

      const isTransition = owner.colorway !== neighbor.colorway;
      if (isTransition) {
        // The higher-precedence (outermost) colorway owns the shared edge
        // color: green innermost, yellow middle, red outermost (OBS-12).
        const precedence: Record<string, number> = {
          road: 0,
          green: 1,
          yellow: 2,
          red: 3,
        };
        const expectedColorway =
          precedence[owner.colorway] >= precedence[neighbor.colorway]
            ? owner.colorway
            : neighbor.colorway;
        const expectedEdgeColor =
          expectedColorway === 'red'
            ? NEATRACING_ROAD_EDGE_RED_EDGE_COLOR
            : expectedColorway === 'yellow'
              ? NEATRACING_ROAD_EDGE_YELLOW_EDGE_COLOR
              : expectedColorway === 'green'
                ? NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR
                : NEATRACING_ROAD_EDGE_EDGE_COLOR;
        expect(edge.color).toEqual(expectedEdgeColor);
      }
    }
  });

  it('builds a byte-identical mesh for the same tuple (same runtime)', () => {
    const firstMesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);
    const secondMesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);

    expect(serializeHexMesh(secondMesh)).toBe(serializeHexMesh(firstMesh));
  });

  it('changes the mesh when the seed changes', () => {
    const firstMesh = buildRoadEdgeHexVoxels(FIXTURE_TRACK_BUILD_INPUT);
    const otherMesh = buildRoadEdgeHexVoxels({
      ...FIXTURE_TRACK_BUILD_INPUT,
      seed: 43,
    });

    expect(serializeHexMesh(otherMesh)).not.toBe(serializeHexMesh(firstMesh));
  });
});

/**
 * Fixture for the outer-corner border-marker contract (slice
 * P1-track-outer-corner-markers). The current implementation is known to miss
 * a red tile within the exterior bisector cone at vertex 2.
 */
const OUTER_CORNER_FIXTURE_INPUT = {
  seed: 0,
  layoutVersion: 1,
  sizeBucket: 'small',
} as const;

/**
 * Largest tolerated gap between a point on the exterior bisector and the
 * nearest tile of the colorway that point should be painted with. The hex
 * lattice covering radius is 1 m, and a band boundary can cut through a hex
 * whose center falls in the neighbouring colorway, so a fully painted band
 * leaves up to about 2 m of slack.
 *
 * MEASURED with the CONE-SCOPED nearest-tile search this lane uses (18 m
 * radius, 60° half-angle around the corner vertex):
 * - seed 0 / small (the fixture the lane runs on): worst gap 1.446 m;
 * - the seven 'small'-bucket seeds 0, 1, 2, 3, 7, 11 and 42: worst gap
 *   1.736 m (seed 3);
 * - the seven 'medium'-bucket seeds of that same list: worst gap 2.039 m
 *   (seed 0).
 *
 * 3 m therefore keeps about 1.7x headroom over the worst gap of the seven
 * 'small' fixtures and about 1.4x over the 'medium' ones, so the lane stays
 * discriminating (a truncated band leaves NO tile of the expected colorway
 * inside the cone at all, which the lane reports as an infinite gap) without
 * pinning the exact lattice phase.
 *
 * KNOWN LIMITATION: the 18 m cone only reaches the band on the 'small' and
 * 'medium' buckets. On the 'large' bucket the band starts more than 18 m
 * away from the corner vertex, so the cone cannot see it (measured worst
 * scoped gap 7.921 m, plus walk points with no visible tile of the expected
 * colorway at all). This lane therefore does NOT guard the 'large' bucket.
 */
const BAND_CONTINUITY_TOLERANCE_M = 3;

/** Radial step of the exterior-bisector walk, in meters. */
const BAND_CONTINUITY_STEP_M = 0.5;

/** Outward reach of the exterior-bisector walk, in meters. */
const BAND_CONTINUITY_REACH_M = 18;

/**
 * Radius of the outer-corner marker cone, in meters. The marker lane and the
 * continuity lane share this radius, so both look at the same corner
 * neighbourhood.
 */
const OUTER_CORNER_CONE_RADIUS_M = 18;

/**
 * Half-angle of the outer-corner marker cone, in radians (60 degrees).
 */
const OUTER_CORNER_CONE_HALF_ANGLE_RAD = Math.PI / 3;

/**
 * Returns the colorway a ground point must be painted with, or `null` when
 * the point is outside the border band and stays unpainted.
 *
 * The band thresholds are the PRODUCTION constants, so this expectation can
 * never drift from the ladder the painter actually applies.
 */
const expectedBandColorway = (
  surface: string,
  distancePastEdgeM: number,
): TileColorway | null => {
  if (surface === 'road' || surface === 'curb') {
    return 'road';
  }
  if (distancePastEdgeM <= ROAD_EDGE_BORDER_GREEN_M) {
    return 'green';
  }
  if (distancePastEdgeM <= ROAD_EDGE_BORDER_YELLOW_M) {
    return 'yellow';
  }
  if (distancePastEdgeM <= ROAD_EDGE_BORDER_RED_M) {
    return 'red';
  }
  return null;
};

/** 2-D cross product (z component) of vectors a and b. */
const crossZ = (ax: number, az: number, bx: number, bz: number): number =>
  ax * bz - az * bx;

/** Normalizes a 2-D vector, returning a zero vector if the input is zero. */
const normalize2 = (
  x: number,
  z: number,
): { readonly x: number; readonly z: number } => {
  const len = Math.hypot(x, z);
  if (len === 0) {
    return { x: 0, z: 0 };
  }
  return { x: x / len, z: z / len };
};

/** Angle between two unit vectors, in radians. */
const angleBetween = (
  ax: number,
  az: number,
  bx: number,
  bz: number,
): number => {
  const dot = ax * bx + az * bz;
  return Math.acos(Math.min(1, Math.max(-1, dot)));
};

/** Returns true when the closed polygon is ordered clockwise. */
const isClockwisePolygon = (
  vertices: ReadonlyArray<{ readonly x: number; readonly z: number }>,
): boolean => {
  let signedArea = 0;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i]!;
    const b = vertices[(i + 1) % vertices.length]!;
    signedArea += a.x * b.z - b.x * a.z;
  }
  return signedArea < 0;
};

/**
 * Computes the exterior-angle bisector direction at vertex `i`. For a convex
 * (outer) corner the bisector points away from the track interior.
 */
const exteriorBisectorAt = (
  vertices: ReadonlyArray<{ readonly x: number; readonly z: number }>,
  i: number,
  clockwise: boolean,
): { readonly x: number; readonly z: number } => {
  const prev = vertices[(i - 1 + vertices.length) % vertices.length]!;
  const curr = vertices[i]!;
  const next = vertices[(i + 1) % vertices.length]!;

  const inDir = normalize2(curr.x - prev.x, curr.z - prev.z);
  const outDir = normalize2(next.x - curr.x, next.z - curr.z);

  // For a counter-clockwise polygon the interior is to the left of each edge,
  // so the outward normal is the right normal. For a clockwise polygon the
  // interior is to the right, so the outward normal is the left normal.
  const outwardNormal = clockwise
    ? { x: -inDir.z, z: inDir.x }
    : { x: inDir.z, z: -inDir.x };
  const outwardNormal2 = clockwise
    ? { x: -outDir.z, z: outDir.x }
    : { x: outDir.z, z: -outDir.x };

  return normalize2(
    outwardNormal.x + outwardNormal2.x,
    outwardNormal.z + outwardNormal2.z,
  );
};

/**
 * Collects the honeycomb tiles that lie inside the marker cone of one outer
 * corner: at most {@link OUTER_CORNER_CONE_RADIUS_M} away from the corner
 * vertex and at most {@link OUTER_CORNER_CONE_HALF_ANGLE_RAD} away from the
 * exterior bisector. The surviving tiles are grouped per colorway.
 *
 * The continuity lane searches this corner neighbourhood (the same scope the
 * sibling marker lane uses) instead of every tile of the track, so the
 * nearest-tile search stays local to the corner under test.
 */
const collectCornerConeTiles = (
  corner: { readonly x: number; readonly z: number },
  bisector: { readonly x: number; readonly z: number },
  tiles: ReadonlyArray<{
    readonly x: number;
    readonly z: number;
    readonly colorway: TileColorway;
  }>,
): Map<TileColorway, Array<{ readonly x: number; readonly z: number }>> => {
  const scoped = new Map<
    TileColorway,
    Array<{ readonly x: number; readonly z: number }>
  >([
    ['road', []],
    ['green', []],
    ['yellow', []],
    ['red', []],
  ]);

  for (const tile of tiles) {
    const dx = tile.x - corner.x;
    const dz = tile.z - corner.z;
    if (Math.hypot(dx, dz) > OUTER_CORNER_CONE_RADIUS_M) {
      continue;
    }
    const direction = normalize2(dx, dz);
    const angle = angleBetween(
      direction.x,
      direction.z,
      bisector.x,
      bisector.z,
    );
    if (angle > OUTER_CORNER_CONE_HALF_ANGLE_RAD) {
      continue;
    }
    scoped.get(tile.colorway)?.push({ x: tile.x, z: tile.z });
  }

  return scoped;
};

describe('outer-corner border markers', () => {
  const centers = collectRoadEdgeTileCenters(OUTER_CORNER_FIXTURE_INPUT);
  const outerCornerSpec = generateTrack(OUTER_CORNER_FIXTURE_INPUT);
  const vertices = outerCornerSpec.segments.map((segment) => ({
    x: segment.startX,
    z: -segment.startY,
  }));
  const clockwise = isClockwisePolygon(vertices);

  for (let i = 0; i < vertices.length; i++) {
    const prev = vertices[(i - 1 + vertices.length) % vertices.length]!;
    const curr = vertices[i]!;
    const next = vertices[(i + 1) % vertices.length]!;
    const turn = crossZ(
      curr.x - prev.x,
      curr.z - prev.z,
      next.x - curr.x,
      next.z - curr.z,
    );
    const isOuter = clockwise ? turn < 0 : turn > 0;
    if (!isOuter) {
      continue;
    }

    it(
      `has green, yellow, and red border tiles within 18 m and 60° of ` +
        `the exterior bisector at outer corner ${i} (${curr.x.toFixed(2)}, ${curr.z.toFixed(2)})`,
      () => {
        const bisector = exteriorBisectorAt(vertices, i, clockwise);
        const matched = centers.filter((tile) => {
          const dx = tile.x - curr.x;
          const dz = tile.z - curr.z;
          const distance = Math.hypot(dx, dz);
          if (distance > OUTER_CORNER_CONE_RADIUS_M) {
            return false;
          }
          const direction = normalize2(dx, dz);
          const angle = angleBetween(
            direction.x,
            direction.z,
            bisector.x,
            bisector.z,
          );
          return angle <= OUTER_CORNER_CONE_HALF_ANGLE_RAD;
        });

        expect(matched.some((tile) => tile.colorway === 'green')).toBe(true);
        expect(matched.some((tile) => tile.colorway === 'yellow')).toBe(true);
        expect(matched.some((tile) => tile.colorway === 'red')).toBe(true);
      },
    );

    it(
      `paints a CONTINUOUS band along the exterior bisector of outer ` +
        `corner ${i} (${curr.x.toFixed(2)}, ${curr.z.toFixed(2)})`,
      () => {
        const bisector = exteriorBisectorAt(vertices, i, clockwise);
        const coneTiles = collectCornerConeTiles(curr, bisector, centers);
        const gaps: string[] = [];

        for (
          let radius = 0;
          radius <= BAND_CONTINUITY_REACH_M;
          radius += BAND_CONTINUITY_STEP_M
        ) {
          const pointX = curr.x + bisector.x * radius;
          const pointZ = curr.z + bisector.z * radius;
          const { surface, distancePastEdgeM } = classifyTrackSlabSurfaceAt(
            pointX,
            pointZ,
            outerCornerSpec.segments,
          );
          const expected = expectedBandColorway(surface, distancePastEdgeM);
          if (!expected) {
            continue;
          }

          let nearest = Number.POSITIVE_INFINITY;
          for (const tile of coneTiles.get(expected) ?? []) {
            const distance = Math.hypot(tile.x - pointX, tile.z - pointZ);
            nearest = Math.min(nearest, distance);
          }

          if (nearest > BAND_CONTINUITY_TOLERANCE_M) {
            gaps.push(
              `r=${radius.toFixed(1)} m expected=${expected} ` +
                `nearest=${nearest.toFixed(2)} m`,
            );
          }
        }

        expect(gaps).toEqual([]);
      },
    );
  }
});

/**
 * P1-outer-edge-red (plan step 14U) - the ASYMMETRIC EDGE band contract
 * (RED-3).
 *
 * The band lane probes the EXISTING readers only: `collectRoadEdgeTileCenters`
 * (the band paint) and `classifyTrackSlabSurfaceAt` (the band measurement).
 * Nothing here imports a module that does not exist yet, so the lane is red
 * BY ASSERTION and never by an import failure.
 */
describe('P1-outer-edge-red band contract (RED-3): the band starts at the ASYMMETRIC OUTER EDGE', () => {
  /**
   * The signed turn angle (radians) at the START vertex of the chord at
   * `index`; positive is a left turn (counter-clockwise).
   */
  const turnAngleAt = (
    segments: readonly TrackSegment[],
    index: number,
  ): number => {
    const previous = segments[(index - 1 + segments.length) % segments.length]!;
    const current = segments[index]!;
    const incoming = Math.atan2(
      current.startY - previous.startY,
      current.startX - previous.startX,
    );
    const outgoing = Math.atan2(
      current.endY - current.startY,
      current.endX - current.startX,
    );
    let delta = outgoing - incoming;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;
    return delta;
  };

  /**
   * The distance from a GENERATOR-plane point to a chord (the clamped
   * projection - the same measure the production rasterizer uses).
   */
  const distanceToChord = (
    pointX: number,
    pointY: number,
    segment: TrackSegment,
  ): number => {
    const chordX = segment.endX - segment.startX;
    const chordY = segment.endY - segment.startY;
    const chordLengthSquared = chordX * chordX + chordY * chordY;
    if (chordLengthSquared <= 0) {
      return Math.hypot(pointX - segment.startX, pointY - segment.startY);
    }
    const projection =
      ((pointX - segment.startX) * chordX +
        (pointY - segment.startY) * chordY) /
      chordLengthSquared;
    const clamped = Math.min(1, Math.max(0, projection));
    return Math.hypot(
      pointX - (segment.startX + clamped * chordX),
      pointY - (segment.startY + clamped * chordY),
    );
  };

  /**
   * The signed area of the closed chord loop in the generator plane
   * (positive means the loop runs counter-clockwise).
   */
  const signedArea = (segments: readonly TrackSegment[]): number => {
    let area = 0;
    for (const segment of segments) {
      area += segment.startX * segment.endY - segment.endX * segment.startY;
    }
    return area / 2;
  };

  const segments = FIXTURE_TRACK_SPEC.segments;
  const turnAngles = segments.map((_, index) => turnAngleAt(segments, index));
  const tightestIndex = turnAngles.reduce(
    (best, value, index) =>
      Math.abs(value) > Math.abs(turnAngles[best]!) ? index : best,
    0,
  );
  const tightestSegment = segments[tightestIndex]!;
  const tightestTurnDeg = (turnAngles[tightestIndex]! * 180) / Math.PI;
  /** The RETIRED symmetric half width of the probed chord (meters). */
  const retiredHalfWidthM = tightestSegment.width / 2;
  const counterClockwise = signedArea(segments) > 0;
  const tightestChordLengthM = Math.hypot(
    tightestSegment.endX - tightestSegment.startX,
    tightestSegment.endY - tightestSegment.startY,
  );
  const tightestHeadingX =
    (tightestSegment.endX - tightestSegment.startX) / tightestChordLengthM;
  const tightestHeadingY =
    (tightestSegment.endY - tightestSegment.startY) / tightestChordLengthM;

  /**
   * The RETIRED symmetric band base of a tile: the smallest distance past the
   * symmetric half width over every chord. This is the measurement base the
   * band uses TODAY.
   */
  const retiredBaseM = (tileX: number, tileZ: number): number => {
    const generatorY = -tileZ;
    let base = Number.POSITIVE_INFINITY;
    for (const segment of segments) {
      const candidate =
        distanceToChord(tileX, generatorY, segment) - segment.width / 2;
      if (candidate < base) {
        base = candidate;
      }
    }
    return base;
  };

  /**
   * The INDEX of the BINDING chord of a tile: the chord the retired band
   * base is measured from (the arg-min of the distance past that chord's own
   * half width - the nearest chord, not the nearest centreline point).
   */
  const bindingSegmentIndex = (tileX: number, tileZ: number): number => {
    const generatorY = -tileZ;
    let bindingIndex = 0;
    let bindingValue = Number.POSITIVE_INFINITY;
    for (let index = 0; index < segments.length; index += 1) {
      const segment = segments[index]!;
      const candidate =
        distanceToChord(tileX, generatorY, segment) - segment.width / 2;
      if (candidate < bindingValue) {
        bindingValue = candidate;
        bindingIndex = index;
      }
    }
    return bindingIndex;
  };

  /** The PRODUCTION band measurement base of a tile (the band paint reads it). */
  const productionPastEdgeM = (tileX: number, tileZ: number): number =>
    classifyTrackSlabSurfaceAt(tileX, tileZ, segments).distancePastEdgeM;

  const allTiles = collectRoadEdgeTileCenters(FIXTURE_TRACK_BUILD_INPUT);

  /**
   * The tiles that lie BEYOND the retired symmetric half width of the probed
   * corner AND on its OUTER side - the population a widened outer edge would
   * push inward (the drift room is added on the OUTER side of the curve only).
   */
  const outerBandTiles = allTiles.filter((tile) => {
    const generatorY = -tile.z;
    const distance = distanceToChord(tile.x, generatorY, tightestSegment);
    if (distance <= retiredHalfWidthM) {
      return false;
    }
    const cross =
      tightestHeadingX * (generatorY - tightestSegment.startY) -
      tightestHeadingY * (tile.x - tightestSegment.startX);
    // The outer side of a counter-clockwise loop is the RIGHT of the heading
    // (a negative cross product); the handedness flips with the loop.
    return counterClockwise ? cross < 0 : cross > 0;
  });

  /** The painted (green/yellow/red) subset of the outer population. */
  const paintedOuterTiles = outerBandTiles.filter(
    (tile) => tile.colorway !== 'road',
  );

  /**
   * The outer population RESTRICTED to the tiles whose BINDING chord IS the
   * probed corner chord (the chord their retired band base is measured
   * from). For those tiles the corner chord IS their measurement base, so the
   * widening of that chord MUST move their base inward one-for-one - a
   * partial re-base (re-basing only some of the chords) can NOT satisfy the
   * discriminating clause on this population alone.
   */
  const cornerBoundOuterTiles = outerBandTiles.filter(
    (tile) => bindingSegmentIndex(tile.x, tile.z) === tightestIndex,
  );

  /**
   * RED-3 (DISCRIMINATING).
   *
   * The band is re-based on the ASYMMETRIC OUTER EDGE: its measurement base
   * moves inward from the RETIRED symmetric half width to the widened outer
   * edge of the corner, so the drift room is added on the OUTER side of the
   * curve only. The ladder CONSTANTS and their CUMULATIVE outer semantics
   * are untouched by this move - only the BASE the bands are measured from
   * changes.
   *
   * The POPULATION is the outer tiles whose BINDING chord IS the probed
   * corner chord: for those tiles the corner chord IS their measurement
   * base, so widening it MUST move their base inward one-for-one - a partial
   * re-base (re-basing only some of the chords) can NOT satisfy the clause.
   *
   * The contract carries NO drift number: the band base is compared
   * RELATIVE to the retired symmetric half width, so the lane states WHERE
   * the base sits without pinning HOW FAR the outer edge moves.
   */
  it('RED-3 (DISCRIMINATING): the band starts at the ASYMMETRIC OUTER EDGE on a corner (its measurement base moved inward from the retired symmetric half-width)', () => {
    // The NON-VACUITY GUARD comes FIRST (the Step 14Q shape): the probed
    // corner MUST really carry a painted band past the widened edge, because
    // a zero-width band would satisfy the starts-at clause VACUOUSLY.
    expect(paintedOuterTiles.length).toBeGreaterThan(0);
    expect(Math.abs(tightestTurnDeg)).toBeGreaterThanOrEqual(45);
    // The corner-bound population (the tiles whose BINDING chord IS the
    // probed corner chord) must not be empty either.
    expect(cornerBoundOuterTiles.length).toBeGreaterThan(0);

    const baseDeltas = cornerBoundOuterTiles.map(
      (tile) =>
        productionPastEdgeM(tile.x, tile.z) - retiredBaseM(tile.x, tile.z),
    );
    const minimumBaseDeltaM = Math.min(...baseDeltas);
    const minimumRetiredBaseM = Math.min(
      ...cornerBoundOuterTiles.map((tile) => retiredBaseM(tile.x, tile.z)),
    );

    // TODAY the band base IS the retired symmetric half width, so the
    // minimum delta is exactly 0 and this assertion fails.
    expect(minimumBaseDeltaM).toBeLessThan(-1e-6);
    // The band must not start INSIDE the retired edge: the probed tiles
    // really are past it, so the re-base moves the base and not the tiles.
    expect(minimumRetiredBaseM).toBeGreaterThan(0);
  });

  /**
   * RED-3 (PRESERVATION - the CUMULATIVE outer semantics are UNCHANGED).
   *
   * Re-basing the band on the asymmetric outer edge does NOT touch the
   * ladder: the three limits keep their values and keep their cumulative
   * meaning (green ends at 2 m, yellow at 4 m, red at 6 m past the edge -
   * each limit measured from the edge, not the width of its own band).
   */
  it('RED-3 (PRESERVATION - the CUMULATIVE outer semantics are UNCHANGED): each band still ends at its own cumulative limit and the ladder order is unchanged', () => {
    expect(ROAD_EDGE_BORDER_GREEN_M).toBe(2);
    expect(ROAD_EDGE_BORDER_YELLOW_M).toBe(4);
    expect(ROAD_EDGE_BORDER_RED_M).toBe(6);
  });

  /**
   * RED-3 (PRESERVATION - the ladder ORDER is unchanged, per tile).
   *
   * Every painted tile of the probed corner still carries the colourway the
   * ladder gives for its CURRENT past-edge distance, and no painted tile
   * reaches past the outermost cumulative limit (6 m).
   */
  it('RED-3 (PRESERVATION - the ladder ORDER is unchanged): every painted tile of the probed corner matches the ladder applied to its current past-edge distance', () => {
    const mismatches: string[] = [];

    for (const tile of paintedOuterTiles) {
      const pastEdgeM = productionPastEdgeM(tile.x, tile.z);
      const expectedColorway =
        pastEdgeM <= ROAD_EDGE_BORDER_GREEN_M
          ? 'green'
          : pastEdgeM <= ROAD_EDGE_BORDER_YELLOW_M
            ? 'yellow'
            : pastEdgeM <= ROAD_EDGE_BORDER_RED_M
              ? 'red'
              : null;
      if (expectedColorway !== tile.colorway) {
        mismatches.push(
          `(${tile.x}, ${tile.z}) painted=${tile.colorway} ` +
            `expected=${String(expectedColorway)} pastEdge=${pastEdgeM.toFixed(4)} m`,
        );
      }
      if (pastEdgeM > ROAD_EDGE_BORDER_RED_M) {
        mismatches.push(
          `(${tile.x}, ${tile.z}) painted=${tile.colorway} ` +
            `pastEdge=${pastEdgeM.toFixed(4)} m is past the outermost limit`,
        );
      }
    }

    expect(mismatches).toEqual([]);
  });

  /**
   * RED-3 (PRESERVATION - the ladder STAYS A PAINT LADDER).
   *
   * The band only MARKS the apron beyond the drivable edge: every border
   * tile sits on a NON-drivable surface, and the drivable surface itself is
   * always painted as roadway, never as a border band. The CONTAINMENT
   * BOUND is a DIFFERENT thing (it is re-based in the collision lane): the
   * band is never derived from a bound constant and a bound is never derived
   * from a band constant.
   */
  it('RED-3 (PRESERVATION - the ladder STAYS A PAINT LADDER): the border ladder never paints the drivable surface and never becomes a physical bound', () => {
    const borderTileSurfaces = allTiles
      .filter((tile) => tile.colorway !== 'road')
      .map(
        (tile) => classifyTrackSlabSurfaceAt(tile.x, tile.z, segments).surface,
      );
    const drivableBorderTiles = allTiles.filter((tile) => {
      if (tile.colorway === 'road') {
        return false;
      }
      const surface = classifyTrackSlabSurfaceAt(
        tile.x,
        tile.z,
        segments,
      ).surface;
      return surface === 'road' || surface === 'curb';
    });

    expect(
      borderTileSurfaces.every(
        (surface) => surface === 'grass' || surface === 'air',
      ),
    ).toBe(true);
    expect(drivableBorderTiles.length).toBe(0);

    // The ladder is measured from the EDGE and never from the centreline: the
    // painted tiles of the probed corner sit about 9.8 m to about 18.1 m from
    // the corner CHORD's CENTRELINE - far beyond the 6 m outer limit - yet
    // they ARE painted, so the band and the containment bound are never
    // conflated.
    const maximumCenterlineDistanceM = Math.max(
      ...paintedOuterTiles.map((tile) =>
        distanceToChord(tile.x, -tile.z, tightestSegment),
      ),
    );
    expect(maximumCenterlineDistanceM).toBeGreaterThan(ROAD_EDGE_BORDER_RED_M);
  });
});
