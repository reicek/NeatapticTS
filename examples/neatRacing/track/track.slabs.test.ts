/**
 * Contract tests for the Neat Racing track slab builder (slice P1-S7-A;
 * AC-017, spec §6.7 LD-25, §7.3.1 merge contract, TP-19, TP-16).
 *
 * Red phase: this suite must fail while `./track.slabs` does not exist
 * ("Cannot find module './track.slabs'" under ts-jest diagnostics); a green
 * suite during the red phase would be a contract defect. The plan packet
 * red_test wording ("slab pitch, bank angles, and boost pads") was
 * reconciled in plan batch 80: the spec (source of truth) defines no track
 * bank-angle or boost-pad surface feature for P1-S7-A, so the assertions
 * below pin the normative AC-017 clauses — slab pitch, road/curb/grass slab
 * palette, greedy merge, tuple determinism, and draw-distance extent.
 *
 * Determinism rung: same-runtime byte-identical via deterministic
 * serialization (reproducibility Level 2, ordered-deterministic); quad
 * coordinates are 0.5 m lattice multiples (exact binary halves), so float
 * serialization is stable. Not claimed: cross-environment byte equality.
 *
 * Fixtures and the no-tolerance lattice check: `buildTrackSlabs` is pure,
 * so the "large" fixture mesh is built once at module scope; the lattice
 * snap check uses exact remainder equality because every 0.5 m multiple is
 * exactly representable in binary, and a tolerance would weaken the coarse
 * grid contract. Bucket "large" is required by the TP-16 extent clause: the
 * recycled closed-loop generator (spec §11) spans well past
 * `NEATRACING_DRAW_DISTANCE_M` (320 m).
 *
 * Green contract for the implementer: `./track.slabs` exports the named
 * pitch constant `NEATRACING_TRACK_SLAB_PITCH_M` (0.5), the pure
 * `buildTrackSlabs(input)` builder taking the determinism tuple
 * `{ seed, layoutVersion, sizeBucket }` with `sizeBucket: string`, and a
 * `TrackSlabMesh` whose `quads: readonly VoxelQuad[]` follow the §7.3.1
 * record shape in world meters (`{ position: [x, y, z], normalAxis,
 * width, height, paletteIndex }`, position = min corner, widths are
 * 0.5 m multiples) and whose `palette` is the track-only slab palette with
 * `type: 'road' | 'curb' | 'grass'` surface labels.
 */

import { describe, expect, it } from '@jest/globals';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import { NEATRACING_DRAW_DISTANCE_M } from '../engine/passes/grid.pass';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import { buildTrackSlabs, NEATRACING_TRACK_SLAB_PITCH_M } from './track.slabs';
import type { TrackSlabMesh } from './track.slabs';

/** Quad record of the fixture mesh (§7.3.1 `VoxelQuad` shape, world meters). */
type FixtureSlabQuad = TrackSlabMesh['quads'][number];

/**
 * Canonical P1-S7-A determinism tuple; the literals ARE the fixture
 * assertion target (seed 42 per determinism default, version 1, bucket
 * "large" for the TP-16 extent margin).
 */
const FIXTURE_TRACK_BUILD_INPUT = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/** Same fixture tuple with a different seed (determinism negative differentiator). */
const OTHER_SEED_TRACK_BUILD_INPUT = { ...FIXTURE_TRACK_BUILD_INPUT, seed: 43 };

/**
 * Fixture track mesh built once at module scope: the builder is pure and
 * the mesh immutable, and the "large" loop lattice spans millions of cells,
 * so sharing one build across the suite keeps runtime bounded. Built
 * eagerly so a red run fails in this very module while `./track.slabs` is
 * missing.
 */
const FIXTURE_TRACK_MESH: TrackSlabMesh = buildTrackSlabs(
  FIXTURE_TRACK_BUILD_INPUT,
);

/** Lattice axes scanned by the pitch-snap check. */
const LATTICE_AXES = [0, 1, 2] as const;

/**
 * §7.3.1 tangent axes per normal axis: `width` extends along the first
 * tangent and `height` along the second (fixed quad record mapping).
 */
const TANGENT_AXES_BY_NORMAL_AXIS: Readonly<
  Record<FixtureSlabQuad['normalAxis'], readonly [number, number]>
> = {
  0: [1, 2],
  1: [0, 2],
  2: [0, 1],
};

/**
 * Serializes a slab mesh for byte-identity comparison: deterministic JSON
 * over construction-ordered records with exactly representable coordinates.
 */
const serializeTrackSlabMesh = (mesh: TrackSlabMesh): string =>
  JSON.stringify(mesh);

/**
 * Collects quads whose corner, size, or span is off the 0.5 m slab lattice.
 * Exact remainder equality is safe (0.5 m multiples are binary-exact).
 */
const collectOffLatticeQuads = (
  mesh: TrackSlabMesh,
): readonly FixtureSlabQuad[] =>
  mesh.quads.filter(
    (quad) =>
      LATTICE_AXES.some(
        (axis) => quad.position[axis] % NEATRACING_TRACK_SLAB_PITCH_M !== 0,
      ) ||
      quad.width % NEATRACING_TRACK_SLAB_PITCH_M !== 0 ||
      quad.height % NEATRACING_TRACK_SLAB_PITCH_M !== 0,
  );

/**
 * Measures the world-space axis spans [x, y, z] of a slab mesh by expanding
 * an AABB over every quad record (extent clause of AC-017 / TP-16).
 */
const measureMeshAxisSpansM = (
  mesh: TrackSlabMesh,
): readonly [number, number, number] => {
  const minM = [
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const maxM = [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];
  for (const quad of mesh.quads) {
    const [widthAxis, heightAxis] =
      TANGENT_AXES_BY_NORMAL_AXIS[quad.normalAxis];
    const extentByAxis: Readonly<Record<number, number>> = {
      [widthAxis]: quad.width,
      [heightAxis]: quad.height,
    };
    for (const axis of [widthAxis, heightAxis] as const) {
      minM[axis] = Math.min(minM[axis], quad.position[axis]);
      maxM[axis] = Math.max(
        maxM[axis],
        quad.position[axis] + extentByAxis[axis],
      );
    }
  }
  return LATTICE_AXES.map((axis) => maxM[axis] - minM[axis]) as [
    number,
    number,
    number,
  ];
};

describe('buildTrackSlabs (spec §6.7 LD-25, AC-017)', () => {
  it('exposes the named 0.5 m track slab pitch (TP-19)', () => {
    expect(NEATRACING_TRACK_SLAB_PITCH_M).toBe(0.5);
  });

  it('keeps the track slab pitch separate from the car voxel world scale (G12)', () => {
    expect(NEATRACING_TRACK_SLAB_PITCH_M).not.toBe(NEATRACING_VOXEL_WORLD_SIZE);
  });

  it('snaps every slab quad onto the 0.5 m slab lattice', () => {
    const offLatticeQuads = collectOffLatticeQuads(FIXTURE_TRACK_MESH);
    expect(offLatticeQuads).toEqual([]);
  });

  it('labels the separate slab palette with road, curb, and grass surface types', () => {
    const surfaceTypes = new Set(
      FIXTURE_TRACK_MESH.palette.map((entry) => entry.type),
    );
    expect(Array.from(surfaceTypes).sort()).toEqual(['curb', 'grass', 'road']);
  });

  it('draws every slab quad from a valid track slab palette entry', () => {
    const paletteSize = FIXTURE_TRACK_MESH.palette.length;
    const quadsWithInvalidPaletteIndex = FIXTURE_TRACK_MESH.quads.filter(
      (quad) => quad.paletteIndex < 0 || quad.paletteIndex >= paletteSize,
    );
    expect(quadsWithInvalidPaletteIndex).toEqual([]);
  });

  it('emits normative §7.3.1 quad records with positive spans', () => {
    const malformedQuads = FIXTURE_TRACK_MESH.quads.filter(
      (quad) =>
        LATTICE_AXES.every((axis) => quad.normalAxis !== axis) ||
        !(quad.width > 0) ||
        !(quad.height > 0),
    );
    expect(malformedQuads).toEqual([]);
  });

  it('greedy-merges adjacent same-surface runs into multi-cell quads (§7.3.1)', () => {
    const hasMergedRun = FIXTURE_TRACK_MESH.quads.some(
      (quad) =>
        quad.width > NEATRACING_TRACK_SLAB_PITCH_M ||
        quad.height > NEATRACING_TRACK_SLAB_PITCH_M,
    );
    expect(hasMergedRun).toBe(true);
  });

  it('builds a byte-identical mesh for the same tuple (same runtime)', () => {
    const rebuiltMesh = buildTrackSlabs(FIXTURE_TRACK_BUILD_INPUT);
    expect(serializeTrackSlabMesh(rebuiltMesh)).toBe(
      serializeTrackSlabMesh(FIXTURE_TRACK_MESH),
    );
  });

  it('changes the mesh when the seed changes (determinism is data-driven)', () => {
    const otherSeedMesh = buildTrackSlabs(OTHER_SEED_TRACK_BUILD_INPUT);
    expect(serializeTrackSlabMesh(otherSeedMesh)).not.toBe(
      serializeTrackSlabMesh(FIXTURE_TRACK_MESH),
    );
  });

  it('covers the TP-16 fog draw distance with the slab mesh extent', () => {
    const axisSpansM = measureMeshAxisSpansM(FIXTURE_TRACK_MESH);
    const widestAxisSpanM = Math.max(...axisSpansM);
    expect(widestAxisSpanM).toBeGreaterThanOrEqual(NEATRACING_DRAW_DISTANCE_M);
  });

  it('mirrors the generator ground y into the scene z axis (world z = -y handedness)', () => {
    // The GL scene maps the top-down math plane (x, y) with y flipped: world
    // z = -y (Step 14B round-3 handedness fix). The slab lattice must mirror
    // the generator's ground-plane y the same way so the ribbon and the car
    // occupy the same world region.
    const spec = generateTrack(FIXTURE_TRACK_BUILD_INPUT);
    let spanMinY = Number.POSITIVE_INFINITY;
    let spanMaxY = Number.NEGATIVE_INFINITY;
    for (const segment of spec.segments) {
      spanMinY = Math.min(spanMinY, segment.startY, segment.endY);
      spanMaxY = Math.max(spanMaxY, segment.startY, segment.endY);
    }

    let slabMinZ = Number.POSITIVE_INFINITY;
    let slabMaxZ = Number.NEGATIVE_INFINITY;
    for (const quad of FIXTURE_TRACK_MESH.quads) {
      // Axis-2 faces sit on a fixed z plane; axes 0 and 1 extend z by height.
      const extentOnZ = quad.normalAxis === 2 ? 0 : quad.height;
      slabMinZ = Math.min(slabMinZ, quad.position[2]);
      slabMaxZ = Math.max(slabMaxZ, quad.position[2] + extentOnZ);
    }

    // The floor/ceil lattice rounding may widen the span by one cell per side.
    const toleranceM = NEATRACING_TRACK_SLAB_PITCH_M + 1e-9;
    expect(slabMinZ).toBeLessThanOrEqual(-spanMaxY + toleranceM);
    expect(slabMaxZ).toBeGreaterThanOrEqual(-spanMinY - toleranceM);

    const spanCenterY = (spanMinY + spanMaxY) / 2;
    const spanCenterZ = (slabMinZ + slabMaxZ) / 2;
    expect(Math.abs(spanCenterZ + spanCenterY)).toBeLessThanOrEqual(toleranceM);

    // The road surface under a segment midpoint must sit at (x, -y) in the
    // mirrored world, so the car (whose scene z mirrors y identically) drives
    // ON the ribbon rather than beside a flipped copy of it.
    const segment = spec.segments[0];
    const midX = (segment.startX + segment.endX) / 2;
    const midGeneratorY = (segment.startY + segment.endY) / 2;
    const mirroredWorldZ = -midGeneratorY;
    const roadUnderMirrorPoint = FIXTURE_TRACK_MESH.quads.some(
      (quad) =>
        quad.normalAxis === 1 &&
        FIXTURE_TRACK_MESH.palette[quad.paletteIndex]?.type === 'road' &&
        quad.position[0] <= midX &&
        midX <= quad.position[0] + quad.width &&
        quad.position[2] <= mirroredWorldZ &&
        mirroredWorldZ <= quad.position[2] + quad.height,
    );
    expect(roadUnderMirrorPoint).toBe(true);
  });
});
