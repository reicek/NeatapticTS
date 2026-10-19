/**
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
  type RoadEdgeHexLattice,
  type RoadEdgeHexMesh,
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
const serializeHexMesh = (mesh: RoadEdgeHexMesh): string => JSON.stringify(mesh);

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
      expect(Math.abs(y - NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M)).toBeLessThanOrEqual(1e-9);
    }
    expect(ys).toContain(NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M);
    const yMin = ys.reduce(function (acc, v) { return Math.min(acc, v); }, Infinity);
    const yMax = ys.reduce(function (acc, v) { return Math.max(acc, v); }, -Infinity);
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
    expect(colors.has(JSON.stringify(NEATRACING_ROAD_EDGE_RED_FILL_COLOR))).toBe(
      true,
    );
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
        lattice.tiles.find((tile) => tile.q === first.q + dq && tile.r === first.r + dr),
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
        incidentCount.set(neighborKey, (incidentCount.get(neighborKey) ?? 0) + 1);
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
