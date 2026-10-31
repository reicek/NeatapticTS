/**
 * ============================================================================
 * P1-outer-edge-red (plan step 14U) - the ASYMMETRIC EDGE MODEL - the
 * GEOMETRY lane (RED-1 + RED-2).
 *
 * AC tags: AC-095 (the asymmetric edge - the named, curvature-derived
 * inner/outer half-width pair, the outer one growing with the local corner
 * measure), AC-064 + AC-080 (the band re-base reconciliation), AC-090 (the
 * user gate is NOT claimed - this lane closes nothing and records no
 * verdict).
 *
 * RED-1 (DISCRIMINATING - the lane FAILS today on its own assertion): the
 * road is WIDER on the OUTER side of a tight corner than on the INNER side
 * of the SAME corner, and the corner's outer width is STRICTLY GREATER than
 * the outer width of a straight. The probe is RUNTIME-ONLY: it probes the
 * SURFACE CLASSIFICATION of a point that sits AT the retired symmetric
 * half-width on the OUTER side of the corner, and that point MUST classify
 * as ROAD (today it classifies as CURB, so the lane is red BY ASSERTION and
 * never by an import failure). The contract opens with a NON-VACUITY GUARD:
 * the probed corner MUST really be a tight corner and the probed straight
 * MUST really be straight.
 *
 * RED-2 (LABELLED PRESERVATION GUARD - NOT A RED and never counted as one):
 * the INNER side of the SAME tight corner classifies EXACTLY as it does
 * today (the road never narrows on the inside) and the road CENTRELINE does
 * not move (the classification stays symmetric about the centreline on a
 * straight). Both clauses hold today; they are rails that must stay green.
 *
 * The drift space (the measured outer extension) is a NAMED EXPORT owned by
 * the implementing slice: THIS contract NEVER carries its number - the probe
 * sits AT the retired half width, which is a property of the FIXTURE.
 *
 * Standing constraints (plan: zero per-frame allocation (P9) and the
 * capacity limits maxNodes 8000 / maxConnections 32000, AC-029): this RED
 * lane adds no per-frame allocation and no node/connection growth - it only
 * PROBES the existing readers.
 *
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
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';
import { NEATRACING_DRAW_DISTANCE_M } from '../engine/passes/grid.pass';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import {
  buildTrackSlabs,
  classifyTrackSlabSurfaceAt,
  NEATRACING_TRACK_SLAB_PITCH_M,
} from './track.slabs';
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

/**
 * The asymmetric-edge fixture track: a closed counter-clockwise loop with
 * FOUR 90 degree corners (P0, P3, P4, P7) and COLLINEAR vertices elsewhere,
 * so the long straight S1 (P1->P2) really has a ZERO turn at both of its
 * ends - the real generated track turns 44-73 degrees at EVERY vertex, so no
 * real straight exists to probe. The coordinates are the GENERATOR plane
 * (the same plane the production rasterizer mirrors into world z).
 */
const CORNER_FIXTURE_VERTICES: readonly (readonly [number, number])[] = [
  [-90, -30],
  [-60, -30],
  [60, -30],
  [90, -30],
  [90, 30],
  [60, 30],
  [-60, 30],
  [-90, 30],
];

/** The fixture chords (generator plane: startX/startY/endX/endY + width). */
const CORNER_FIXTURE_SEGMENTS: readonly TrackSegment[] =
  CORNER_FIXTURE_VERTICES.map((vertex, index) => {
    const next =
      CORNER_FIXTURE_VERTICES[(index + 1) % CORNER_FIXTURE_VERTICES.length]!;
    return {
      startX: vertex[0],
      startY: vertex[1],
      endX: next[0],
      endY: next[1],
      width: 8,
    };
  });

/** The retired (symmetric) half width of the fixture chords, in meters. */
const RETIRED_HALF_WIDTH_M = 4;

/** The index of the fixture chord that STARTS at the probed corner (P3). */
const PROBE_CORNER_SEGMENT_INDEX = 3;

/** The index of the fixture chord that is the probed straight (P1->P2). */
const PROBE_STRAIGHT_SEGMENT_INDEX = 1;

/** The probed corner vertex (P3), in the generator plane. */
const PROBE_CORNER_X = 90;
const PROBE_CORNER_Y = -30;

/** The midpoint of the probed straight (S1), in the generator plane. */
const PROBE_STRAIGHT_MID_X = 0;
const PROBE_STRAIGHT_MID_Y = -30;

/** How far along the corner chord the two corner probes sit (meters). */
const PROBE_ALONG_CHORD_M = 4;

/**
 * The OUTER probe: the point that sits exactly ON the retired road edge OF
 * THE CORNER CHORD (S3, the chord that STARTS at the probed corner P3) on
 * that chord's OUTER side - the fixture loop is counter-clockwise, so the
 * OUTER side of the north-heading chord is EAST (+x) - and 4 m along the
 * chord from the corner vertex. It is the MIRROR of the INNER probe about
 * that chord's line: the SAME chord, the SAME distance from it (the retired
 * half width), and the only difference is the SIDE.
 *
 * It is deliberately NOT the corner's outer bisector: a bisector probe is
 * EQUIDISTANT from the two chords that meet at the corner, so a reader that
 * resolves the NEAREST chord first tie-breaks the corner away (the incoming
 * straight S2, whose corner measure is 0) and the lane could never go
 * green. On the corner chord itself the nearest chord is UNAMBIGUOUS.
 */
const PROBE_OUTER_X = PROBE_CORNER_X + RETIRED_HALF_WIDTH_M;
const PROBE_OUTER_Y = PROBE_CORNER_Y + PROBE_ALONG_CHORD_M;

/**
 * The INNER probe: the MIRROR of the OUTER probe about the SAME corner
 * chord's line - the same chord, the same retired half width from it - on
 * the INNER side, where the road must NEVER narrow.
 */
const PROBE_INNER_X = PROBE_CORNER_X - RETIRED_HALF_WIDTH_M;
const PROBE_INNER_Y = PROBE_CORNER_Y + PROBE_ALONG_CHORD_M;

/**
 * The STRAIGHT's outer probe: the point half a curb width out from the
 * probed straight's midpoint, on the outer side of the straight.
 */
const PROBE_STRAIGHT_OUTER_X = PROBE_STRAIGHT_MID_X;
const PROBE_STRAIGHT_OUTER_Y = PROBE_STRAIGHT_MID_Y - RETIRED_HALF_WIDTH_M;

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
 * Surface classification of a GENERATOR-plane point: the production
 * rasterizer works in the world frame, where world z mirrors the generator
 * y, so the probe is mirrored here.
 */
const surfaceAt = (
  generatorX: number,
  generatorY: number,
): { readonly surface: string; readonly distancePastEdgeM: number } =>
  classifyTrackSlabSurfaceAt(
    generatorX,
    -generatorY,
    CORNER_FIXTURE_SEGMENTS,
  );

/**
 * Pure FIXTURE geometry: the Euclidean distance from a generator-plane point
 * to a fixture chord (clamped nearest-point projection). It measures the
 * FIXTURE's own geometry, independently of the production edge model, so a
 * probe's distance to the RETIRED road edge stays provable after the
 * production outer side widens.
 */
const distanceToFixtureChord2D = (
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
    ((pointX - segment.startX) * chordX + (pointY - segment.startY) * chordY) /
    chordLengthSquared;
  const clamped = Math.min(1, Math.max(0, projection));
  return Math.hypot(
    pointX - (segment.startX + clamped * chordX),
    pointY - (segment.startY + clamped * chordY),
  );
};

describe('P1-outer-edge-red (plan step 14U) - the asymmetric edge model (RED-1 + RED-2)', () => {
  it('RED-1 (DISCRIMINATING): the road is WIDER on the OUTER side of the tight corner than on the INNER side of the SAME corner and than the outer side of a straight', () => {
    // Arrange (NON-VACUITY GUARD): the fixture's OWN turn angles, measured
    // on its chords. The NON-VACUITY GUARD comes FIRST (the ratified Step
    // 14Q shape): the probed corner MUST really be a tight corner and the
    // probed straight MUST really be straight, or the relational clause
    // below is satisfied VACUOUSLY.
    const cornerTurnDeg =
      Math.abs(
        (turnAngleAt(CORNER_FIXTURE_SEGMENTS, PROBE_CORNER_SEGMENT_INDEX) *
          180) /
          Math.PI,
      );
    const straightEndpointTurnDegAbs = [
      Math.abs(
        (turnAngleAt(CORNER_FIXTURE_SEGMENTS, PROBE_STRAIGHT_SEGMENT_INDEX) *
          180) /
          Math.PI,
      ),
      Math.abs(
        (turnAngleAt(
          CORNER_FIXTURE_SEGMENTS,
          PROBE_STRAIGHT_SEGMENT_INDEX + 1,
        ) *
          180) /
          Math.PI,
      ),
    ];
    // The OUTER probe must sit EXACTLY ON the retired (symmetric) road
    // edge. The distance is measured on the FIXTURE chords, NOT through the
    // production reader, so the guard stays true once the production outer
    // side widens (the drift space is never carried by this contract).
    const outerProbeChordDistanceM = Math.min(
      ...CORNER_FIXTURE_SEGMENTS.map((segment) =>
        distanceToFixtureChord2D(PROBE_OUTER_X, PROBE_OUTER_Y, segment),
      ),
    );

    expect(cornerTurnDeg).toBeGreaterThanOrEqual(45);
    expect(Math.max(...straightEndpointTurnDegAbs)).toBeLessThan(1e-9);
    expect(outerProbeChordDistanceM).toBeCloseTo(RETIRED_HALF_WIDTH_M, 9);

    // Arrange: three probes that ALL sit AT the retired symmetric half
    // width (4 m) - the OUTER probe on the OUTER side of the corner chord
    // S3 (4 m along that chord from the corner vertex), the INNER probe on
    // the INNER side of that SAME chord, and the straight's outer probe half
    // a curb width out from the straight's midpoint. The drift space is
    // added on the OUTER side of the curve ONLY, so the retired half width
    // on the outer side of a tight corner must still be ROAD.
    const outerProbe = surfaceAt(PROBE_OUTER_X, PROBE_OUTER_Y);
    const innerProbe = surfaceAt(PROBE_INNER_X, PROBE_INNER_Y);
    const straightOuterProbe = surfaceAt(
      PROBE_STRAIGHT_OUTER_X,
      PROBE_STRAIGHT_OUTER_Y,
    );

    // Act + Assert: the order is [the OUTER probe of the tight corner, the
    // INNER probe of the SAME corner, the outer probe of the straight]. The
    // FIRST entry is the DISCRIMINATING clause: the outer probe sits AT the
    // retired half width on the corner chord's outer side and MUST classify
    // as ROAD (today it classifies as CURB - this is the RED).
    expect([
      outerProbe.surface,
      innerProbe.surface,
      straightOuterProbe.surface,
    ]).toEqual(['road', 'curb', 'curb']);
  });

  it('RED-2 (LABELLED PRESERVATION GUARD - NOT A RED): the INNER side of the SAME tight corner classifies exactly as it does today', () => {
    // Arrange
    const innerProbe = surfaceAt(PROBE_INNER_X, PROBE_INNER_Y);

    // Act + Assert: NOT A RED - this clause HOLDS today and must stay green:
    // the road never NARROWS on the inside of a corner, so the inner side
    // of the tight corner still classifies as CURB at the retired half
    // width, exactly on the retired edge.
    expect(innerProbe.surface).toBe('curb');
    expect(innerProbe.distancePastEdgeM).toBeCloseTo(0, 9);
  });

  it('RED-2 (LABELLED PRESERVATION GUARD - NOT A RED): the road CENTRELINE does not move (the classification stays symmetric about the centreline on the straight)', () => {
    // Arrange: mirrored probes 3 m either side of the straight's centreline
    // (still road) and at the retired half width, 4 m either side (curb).
    const innerSideSurface = surfaceAt(
      PROBE_STRAIGHT_MID_X,
      PROBE_STRAIGHT_MID_Y + 3,
    ).surface;
    const outerSideSurface = surfaceAt(
      PROBE_STRAIGHT_MID_X,
      PROBE_STRAIGHT_MID_Y - 3,
    ).surface;
    const innerEdgeSurface = surfaceAt(
      PROBE_STRAIGHT_MID_X,
      PROBE_STRAIGHT_MID_Y + RETIRED_HALF_WIDTH_M,
    ).surface;
    const outerEdgeSurface = surfaceAt(
      PROBE_STRAIGHT_MID_X,
      PROBE_STRAIGHT_MID_Y - RETIRED_HALF_WIDTH_M,
    ).surface;

    // Act + Assert: NOT A RED - this clause HOLDS today and must stay green:
    // the centreline does not move, so the classification stays symmetric
    // about it on a straight.
    expect(innerSideSurface).toBe('road');
    expect(outerSideSurface).toBe('road');
    expect([innerEdgeSurface, outerEdgeSurface]).toEqual(['curb', 'curb']);
  });
});
