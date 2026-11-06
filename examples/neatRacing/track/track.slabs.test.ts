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
import {
  classifyTrackSurface,
  NEATRACING_TRACK_CURB_WIDTH_M,
  NEATRACING_TRACK_GRASS_APRON_WIDTH_M,
} from '../physics/collision.utils';
import { createInitialPhysicsCarState } from '../physics/physics.step.service';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import {
  collectRoadEdgeTileCenters,
  ROAD_EDGE_BORDER_GREEN_M,
} from './road-edge';
import {
  distanceToTrackChordM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAt,
} from './track.edge';
import type { TrackEdgeProfile } from './track.edge';
import {
  buildTrackSlabs,
  buildTrackSlabSurfaceGrid,
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

/**
 * The MEASURED post-unclip OFFSET (meters) between the painted slab span's
 * z centre and the mirrored centreline span's centre, for the canonical
 * fixture tuple.
 *
 * BASIS (the Step 14U2 MEASURED clip report and the `P1-raster-unclip-impl`
 * measurement): the painted mesh's z extent MOVED from the CLIPPED
 * [-270, 235.5] - the retired centreline-only lattice extent, which cut the
 * band off at the centreline AABB - to the TRUE band extent [-287, 254], so
 * the painted centre moved from -17.25 to -16.5 while the negated
 * centreline-span centre stays at -17.237. The offset is therefore the
 * BAND'S OWN ASYMMETRY about the centreline AABB: the topmost chord is
 * narrower than the widest chord, so the band reaches 17.43 m past the top
 * of the AABB but 19.09 m past its bottom.
 *
 * The painted cell set is IMPLEMENTATION-INDEPENDENT (the cell-centre grid
 * is identical for ANY multiple-of-0.5 origin - the phase is fixed at 0.5 m
 * - and the classifier is ANALYTIC), so EVERY implementation that satisfies
 * AC-099 produces exactly this offset.
 *
 * The assertion PINS this value with a 9-digit closeness - about nine orders
 * of magnitude TIGHTER than the retired one-pitch (0.5 m) bound it replaces
 * - so the re-pin is a MEASURED value and NEVER a loosened tolerance.
 */
const PAINTED_BAND_CENTRE_OFFSET_M = 0.7369999999999806;

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
    // RE-PINNED (Step 14U2, `P1-raster-unclip-repin`): the rasterized domain
    // is no longer CLIPPED to the centreline AABB, so the painted band sits
    // ASYMMETRICALLY about it - the offset is the MEASURED value, pinned.
    expect(Math.abs(spanCenterZ + spanCenterY)).toBeCloseTo(
      PAINTED_BAND_CENTRE_OFFSET_M,
      9,
    );

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

/**
 * ============================================================================
 * P1-raster-unclip-red (plan step 14U2) - the AABB RASTER UNCLIP
 * (RED-5 + RED-6 + RED-7).
 *
 * AC tags: AC-099 (the rasterized domain CONTAINS the widened road - the
 * additive per-side growth and the raster-domain probe), AC-064 + AC-080
 * (the tri-colour band is NOT disturbed by the unclip), AC-090 (the user
 * gate is NOT claimed - this lane closes nothing and records NO verdict).
 *
 * THE DEFECT: the road lattice is sized from the CENTRELINE AABB (the
 * axis-aligned bounding box of the centreline segment ENDPOINTS, measured by
 * `measureSegmentSpanAabb` in `track.slabs.ts`), so every ribbon cell whose
 * published outer half-width reaches PAST that AABB is NEVER RASTERIZED and
 * the Step 14U outer widening stays INVISIBLE in the painted mesh - which is
 * exactly the USER report (NO visible fix).
 *
 * THE PROBE IS THE RASTER, NOT THE CLASSIFIER: `classifyTrackSlabSurfaceAt`
 * is ANALYTIC and is NOT clipped (it classifies ANY point from the published
 * half-widths), so a classifier probe would be GREEN today and would be a
 * VACUOUS lane. Every clause below reads the RASTERIZED DOMAIN (the grid
 * and the painted mesh) - the surface the USER sees.
 *
 * RED-5 (DISCRIMINATING - the lane FAILS today on its own assertion): the
 * rasterized grid CONTAINS the outer corner probe AND the rasterized cell AT
 * the probe is ROAD. On the current build the probe falls OUTSIDE the
 * rasterized domain (the cell does not exist), so the lane is RED BY
 * ASSERTION and NEVER by an import failure. The contract OPENS with a
 * NON-VACUITY GUARD in the Step 14Q style, asserted BEFORE the relational
 * clause: the probed corner MUST REALLY carry a published widening, the
 * probe MUST REALLY fall OUTSIDE the current centreline AABB and the probe
 * MUST REALLY fall INSIDE the published road region on that chord.
 *
 * RED-6 (LABELLED PRESERVATION GUARD - NOT A RED and NEVER counted as one):
 * the INNER side of the SAME corner classifies EXACTLY as it does today, the
 * road CENTRELINE does not move, and the tri-colour band still starts at the
 * PUBLISHED OUTER edge and NOT at the lattice border. All three clauses HOLD
 * today; they are rails that must stay green.
 *
 * RED-7 (DISCRIMINATING - the lane FAILS today on its own assertion): the
 * PAINTED MESH carries a ROAD quad covering the SAME probe point - without
 * one the widened corner is INVISIBLE, which is the USER's report.
 *
 * NO NUMBER IS CARRIED INTO THIS CONTRACT: the probe is RESOLVED AT RUNTIME
 * from the published seam (the corner, the side, the offset and the
 * rasterized-cell indices are all DERIVED, never hard-coded), and every
 * number the implementing slice uses is MEASURED and RECORDED in the plan
 * (RED-8).
 * ============================================================================
 */

/** The fixture tuple the unclip lane probes (the canonical P1-S7-A tuple). */
const UNCLIP_FIXTURE_INPUT = FIXTURE_TRACK_BUILD_INPUT;

/** The fixture spec (the generator plane) resolved once at module scope. */
const UNCLIP_SPEC = generateTrack(UNCLIP_FIXTURE_INPUT);

/** The PUBLISHED edge profile of the fixture loop (the seam the lane reads). */
const UNCLIP_PROFILE: TrackEdgeProfile = resolveTrackEdgeProfile(
  UNCLIP_SPEC.segments.slice(),
);

/**
 * The CURRENT rasterized surface grid: the lattice the clip sizes TODAY
 * (the surface under test). Built at module scope so the grid and the mesh
 * are read from the SAME deterministic tuple.
 */
const UNCLIP_GRID: ReturnType<typeof buildTrackSlabSurfaceGrid> =
  buildTrackSlabSurfaceGrid(UNCLIP_FIXTURE_INPUT);

/**
 * The CENTRELINE AABB of the fixture loop (the box the CURRENT clip sizes
 * the lattice from): the axis-aligned bounding box of the centreline segment
 * ENDPOINTS, in the generator plane. The lane measures it ITSELF and never
 * reads it from the production lattice, so the guard survives the clip
 * changing.
 */
const measureCentrelineAabbM = (): {
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
} => {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const segment of UNCLIP_PROFILE.segments) {
    minX = Math.min(minX, segment.startX, segment.endX);
    maxX = Math.max(maxX, segment.startX, segment.endX);
    minY = Math.min(minY, segment.startY, segment.endY);
    maxY = Math.max(maxY, segment.startY, segment.endY);
  }
  return { minX, maxX, minY, maxY };
};

/**
 * The OUTER normal of a chord in the GENERATOR plane: the loop-global outer
 * side, read from the PUBLISHED `outerSideSign` (NEVER from the local turn
 * sign, which flips at a straight and is meaningless there).
 */
const outerNormalOfChord = (
  segment: Readonly<TrackSegment>,
  outerSideSign: 1 | -1,
): readonly [number, number] => {
  const chordX = segment.endX - segment.startX;
  const chordY = segment.endY - segment.startY;
  const chordLengthM = Math.hypot(chordX, chordY);
  return outerSideSign === 1
    ? [-chordY / chordLengthM, chordX / chordLengthM]
    : [chordY / chordLengthM, -chordX / chordLengthM];
};

/** The resolved outer corner probe: a point on the OUTER side of a corner. */
interface OuterCornerProbe {
  /** The index of the probed chord (the tightest qualifying corner chord). */
  readonly chordIndex: number;
  /** How far along the chord the probe sits (0 = the corner vertex). */
  readonly alongFraction: number;
  /** The distance from the chord to the probe, in meters. */
  readonly distanceM: number;
  /** The probe in the GENERATOR plane. */
  readonly generatorX: number;
  readonly generatorY: number;
}

/**
 * THE PROBE: the point that sits HALF the published outer half-width out
 * from a corner chord's START (corner) vertex, on that chord's OUTER side.
 *
 * It is RESOLVED AT RUNTIME from the PUBLISHED seam - the tightest chord
 * that carries a published widening and whose outer side leaves the
 * centreline AABB wins - so NO measured number (no corner index, no offset,
 * no cell index) is carried into the contract text. The offset is HALF the
 * published outer half-width, so the probe sits well inside the published
 * road region (the published road core reaches the outer half-width minus
 * the curb width) with room for the rasterizer's cell-centre sampling.
 */
const resolveOuterCornerProbe = (): OuterCornerProbe | null => {
  const aabb = measureCentrelineAabbM();
  const chordsByTightness = UNCLIP_PROFILE.chords
    .map((chord, chordIndex) => ({ chord, chordIndex }))
    .sort((left, right) =>
      right.chord.cornerMeasureDeg !== left.chord.cornerMeasureDeg
        ? right.chord.cornerMeasureDeg - left.chord.cornerMeasureDeg
        : left.chordIndex - right.chordIndex,
    );

  for (const { chord, chordIndex } of chordsByTightness) {
    // A chord that carries NO published widening can never carry the probe.
    if (chord.outerHalfWidthM <= chord.innerHalfWidthM) {
      continue;
    }
    const segment = UNCLIP_PROFILE.segments[chordIndex]!;
    const [normalX, normalY] = outerNormalOfChord(
      segment,
      UNCLIP_PROFILE.outerSideSign,
    );
    for (const alongFraction of [0, 0.5, 1] as const) {
      const baseX =
        segment.startX + (segment.endX - segment.startX) * alongFraction;
      const baseY =
        segment.startY + (segment.endY - segment.startY) * alongFraction;
      const distanceM = chord.outerHalfWidthM / 2;
      const generatorX = baseX + normalX * distanceM;
      const generatorY = baseY + normalY * distanceM;
      const leavesCentrelineAabb =
        generatorX < aabb.minX ||
        generatorX > aabb.maxX ||
        generatorY < aabb.minY ||
        generatorY > aabb.maxY;
      if (!leavesCentrelineAabb) {
        continue;
      }
      const chordDistanceM = distanceToTrackChordM(
        segment,
        generatorX,
        generatorY,
      );
      if (
        chordDistanceM >=
        chord.outerHalfWidthM - NEATRACING_TRACK_CURB_WIDTH_M
      ) {
        continue;
      }
      if (
        resolveTrackEdgeSideAt(
          UNCLIP_PROFILE,
          chordIndex,
          generatorX,
          generatorY,
        ) !== 'outer'
      ) {
        continue;
      }
      return { chordIndex, alongFraction, distanceM, generatorX, generatorY };
    }
  }
  return null;
};

/**
 * The probe the lane asserts on. A fixture that exposes NO qualifying
 * corner is a FIXTURE defect (not a contract failure), so it throws with an
 * explicit message instead of asserting a vacuous relation.
 */
const requireOuterCornerProbe = (): OuterCornerProbe => {
  const probe = resolveOuterCornerProbe();
  if (probe === null) {
    throw new Error(
      'P1-raster-unclip-red: the fixture loop exposes NO outer corner probe (a chord that carries a published outer widening AND whose outer side leaves the centreline AABB).',
    );
  }
  return probe;
};

describe('P1-raster-unclip-red (plan step 14U2) - the AABB raster unclip (RED-5 + RED-6 + RED-7)', () => {
  it('RED-5 (NON-VACUITY GUARD): the probed corner REALLY carries a published outer widening and the probe REALLY falls OUTSIDE the current centreline clip box and INSIDE the published road region', () => {
    // Arrange (NON-VACUITY GUARD - the Step 14Q shape, asserted FIRST, so
    // the discriminating clause can never be satisfied VACUOUSLY).
    const probe = requireOuterCornerProbe();
    const chord = UNCLIP_PROFILE.chords[probe.chordIndex]!;
    const segment = UNCLIP_PROFILE.segments[probe.chordIndex]!;
    const aabb = measureCentrelineAabbM();
    const leavesCentrelineAabb =
      probe.generatorX < aabb.minX ||
      probe.generatorX > aabb.maxX ||
      probe.generatorY < aabb.minY ||
      probe.generatorY > aabb.maxY;
    const chordDistanceM = distanceToTrackChordM(
      segment,
      probe.generatorX,
      probe.generatorY,
    );
    const publishedSurface = classifyTrackSlabSurfaceAt(
      probe.generatorX,
      -probe.generatorY,
      UNCLIP_SPEC.segments,
      UNCLIP_PROFILE,
    ).surface;

    // Act + Assert: the published outer half-width MUST be STRICTLY GREATER
    // than the published inner half-width on the probed corner (the corner
    // REALLY carries a widening), the probe MUST REALLY fall OUTSIDE the
    // current centreline AABB (a probe inside it would satisfy the clause
    // vacuously) and it MUST REALLY fall INSIDE the published road region on
    // that chord (the distance to that chord is strictly less than the
    // published outer half-width minus the curb width).
    expect([
      chord.outerHalfWidthM > chord.innerHalfWidthM,
      leavesCentrelineAabb,
      chordDistanceM < chord.outerHalfWidthM - NEATRACING_TRACK_CURB_WIDTH_M,
      publishedSurface,
    ]).toEqual([true, true, true, 'road']);
  });

  it('RED-5 (DISCRIMINATING): the rasterized domain CONTAINS the outer corner probe and the rasterized cell AT it is ROAD', () => {
    // Arrange: the probe in the WORLD frame (the frame the rasterizer
    // paints in) and the rasterized cell that contains it.
    const probe = requireOuterCornerProbe();
    const worldX = probe.generatorX;
    const worldZ = -probe.generatorY;
    const cellX = Math.floor(
      (worldX - UNCLIP_GRID.origin[0]) / NEATRACING_TRACK_SLAB_PITCH_M,
    );
    const cellZ = Math.floor(
      (worldZ - UNCLIP_GRID.origin[2]) / NEATRACING_TRACK_SLAB_PITCH_M,
    );
    const insideRasterizedDomain =
      cellX >= 0 &&
      cellX < UNCLIP_GRID.cellsX &&
      cellZ >= 0 &&
      cellZ < UNCLIP_GRID.cellsZ;
    const cellLabel = UNCLIP_GRID.rows[cellZ]?.[cellX] ?? 'absent';

    // Act + Assert: the RASTER (the clipped surface the USER sees) must
    // CONTAIN the probe and paint the cell AT it as ROAD. On the current
    // build the probe falls OUTSIDE the rasterized domain (the cell does not
    // exist), so this lane is RED BY ASSERTION - NEVER by an import
    // failure and NEVER by a wrong label.
    expect([insideRasterizedDomain, cellLabel]).toEqual([true, 'road']);
  });

  it('RED-6 (LABELLED PRESERVATION GUARD - NOT A RED): the INNER side of the SAME probed corner classifies EXACTLY as it does today and the road never narrows inside', () => {
    // Arrange: the MIRROR of the probe about the probed chord's line - the
    // SAME chord, the SAME distance from it - on the INNER side, which faces
    // the loop centre and was already INSIDE the current centreline AABB.
    const probe = requireOuterCornerProbe();
    const chord = UNCLIP_PROFILE.chords[probe.chordIndex]!;
    const segment = UNCLIP_PROFILE.segments[probe.chordIndex]!;
    const [normalX, normalY] = outerNormalOfChord(
      segment,
      UNCLIP_PROFILE.outerSideSign,
    );
    const baseX =
      segment.startX + (segment.endX - segment.startX) * probe.alongFraction;
    const baseY =
      segment.startY + (segment.endY - segment.startY) * probe.alongFraction;
    const innerGeneratorX = baseX - normalX * probe.distanceM;
    const innerGeneratorY = baseY - normalY * probe.distanceM;
    const innerSurface = classifyTrackSlabSurfaceAt(
      innerGeneratorX,
      -innerGeneratorY,
      UNCLIP_SPEC.segments,
      UNCLIP_PROFILE,
    ).surface;
    const innerCellX = Math.floor(
      (innerGeneratorX - UNCLIP_GRID.origin[0]) / NEATRACING_TRACK_SLAB_PITCH_M,
    );
    const innerCellZ = Math.floor(
      (-innerGeneratorY - UNCLIP_GRID.origin[2]) / NEATRACING_TRACK_SLAB_PITCH_M,
    );
    const innerCellLabel = UNCLIP_GRID.rows[innerCellZ]?.[innerCellX] ?? 'absent';

    // Act + Assert: NOT A RED - this clause HOLDS today and must stay
    // green: the road never NARROWS on the inside of a corner, so the inner
    // side of the probed corner still classifies as ROAD, the rasterized
    // cell AT it still says ROAD, and the published inner half-width still
    // does not exceed the published outer half-width.
    expect([
      innerSurface,
      innerCellLabel,
      chord.innerHalfWidthM <= chord.outerHalfWidthM,
    ]).toEqual(['road', 'road', true]);
  });

  it('RED-6 (LABELLED PRESERVATION GUARD - NOT A RED): the road CENTRELINE does not move (the probed chord centreline stays ROAD and strictly inside the published edge)', () => {
    // Arrange: the centreline counterpart of the probe - the probed chord's
    // START (corner) vertex, the point the probe is measured FROM.
    const probe = requireOuterCornerProbe();
    const segment = UNCLIP_PROFILE.segments[probe.chordIndex]!;
    const baseX =
      segment.startX + (segment.endX - segment.startX) * probe.alongFraction;
    const baseY =
      segment.startY + (segment.endY - segment.startY) * probe.alongFraction;
    const centrelineClassification = classifyTrackSlabSurfaceAt(
      baseX,
      -baseY,
      UNCLIP_SPEC.segments,
      UNCLIP_PROFILE,
    );

    // Act + Assert: NOT A RED - this clause HOLDS today and must stay green:
    // the centreline does not move, so the centreline itself still
    // classifies as ROAD and still lies STRICTLY INSIDE the published edge.
    expect([
      centrelineClassification.surface,
      centrelineClassification.distancePastEdgeM < 0,
    ]).toEqual(['road', true]);
  });

  it('RED-6 (LABELLED PRESERVATION GUARD - NOT A RED): the tri-colour band still starts at the PUBLISHED OUTER edge and NOT at the lattice border', () => {
    // Arrange: the band tiles on the OUTER side of the probed chord, each
    // measured against that chord's PUBLISHED outer edge (the band surface
    // is NOT re-derived from the lattice border).
    const probe = requireOuterCornerProbe();
    const chord = UNCLIP_PROFILE.chords[probe.chordIndex]!;
    const segment = UNCLIP_PROFILE.segments[probe.chordIndex]!;
    const pastPublishedEdgeM: number[] = [];
    for (const tile of collectRoadEdgeTileCenters(UNCLIP_FIXTURE_INPUT)) {
      if (tile.colorway !== 'green') {
        continue;
      }
      const classification = classifyTrackSlabSurfaceAt(
        tile.x,
        tile.z,
        UNCLIP_SPEC.segments,
        UNCLIP_PROFILE,
      );
      if (
        classification.surface === 'road' ||
        classification.surface === 'curb'
      ) {
        continue;
      }
      if (
        resolveTrackEdgeSideAt(
          UNCLIP_PROFILE,
          probe.chordIndex,
          tile.x,
          -tile.z,
        ) !== 'outer'
      ) {
        continue;
      }
      pastPublishedEdgeM.push(
        distanceToTrackChordM(segment, tile.x, -tile.z) -
          chord.outerHalfWidthM,
      );
    }
    const closestToPublishedEdgeM = Math.min(...pastPublishedEdgeM);

    // Act + Assert: NOT A RED - this clause HOLDS today and must stay
    // green: the band base IS the published outer edge (the innermost band
    // tile on the outer side sits AT that edge, within the green band's own
    // width) and the band never paints INSIDE the published edge. A raster
    // that grows must not silently move the band base - that would be a
    // second, competing edge.
    expect([
      pastPublishedEdgeM.length > 0,
      closestToPublishedEdgeM >= -1e-9,
      closestToPublishedEdgeM <= ROAD_EDGE_BORDER_GREEN_M,
    ]).toEqual([true, true, true]);
  });

  it('RED-7 (DISCRIMINATING): the painted slab mesh carries a ROAD quad covering the SAME outer corner probe', () => {
    // Arrange: the SAME probe RED-5 measures, read in the PAINTED mesh -
    // the surface the USER sees.
    const probe = requireOuterCornerProbe();
    const worldX = probe.generatorX;
    const worldZ = -probe.generatorY;
    const coveringRoadQuad = FIXTURE_TRACK_MESH.quads.find(
      (quad) =>
        quad.normalAxis === 1 &&
        FIXTURE_TRACK_MESH.palette[quad.paletteIndex]?.type === 'road' &&
        quad.position[0] <= worldX &&
        worldX <= quad.position[0] + quad.width &&
        quad.position[2] <= worldZ &&
        worldZ <= quad.position[2] + quad.height,
    );

    // Act + Assert: without a painted ROAD quad there the widened corner is
    // INVISIBLE - exactly the USER report. On the current build NO road
    // quad covers the probe (the cell is outside the raster), so this lane
    // is RED BY ASSERTION and NEVER by an import failure.
    expect(coveringRoadQuad !== undefined).toBe(true);
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14W (slice `P1-sand-red`): the ONE SURFACE MODEL lane (RED-5).
 *
 * AC tags: AC-094 (b - the off-track band is DRIVABLE and SAND-LIKE),
 * AC-080 (the tri-colour ladder stays a PAINT ladder: the band REUSES the
 * EXISTING 'grass' member, so the surface union stays road | curb | grass
 * and NO new member is introduced), AC-090 (the user gate is NOT claimed -
 * this lane closes nothing and records no verdict).
 *
 * THE CLASSIFICATION THIS SLICE RECORDS: MODEL - the off-track band must
 * be the SAME surface for the RASTERIZER (the painted mesh) and for the
 * PHYSICS classifier (the surface the tick drives on). The two readers
 * read the SAME physical point through DIFFERENT frames (the rasterizer
 * reads the WORLD frame, where world z mirrors the generator y; the
 * physics classifier reads the GENERATOR plane the tick drives in), so
 * the lane compares them across that mirror - a drift between them is a
 * MODEL defect, never a paint defect.
 *
 * RED-5 (DISCRIMINATING - the lane FAILS today on its own assertion): the
 * physics classifier marks the off-track band as the DRIVABLE off-road
 * surface. TODAY the band classifies as the NON-drivable grass, so the
 * lane FAILS on its own assertion. The NON-VACUITY GUARD comes FIRST (the
 * Step 14Q style): the probe MUST really lie on the band (a probe inside
 * the road would satisfy the clause VACUOUSLY) and the two probed models
 * MUST really be DIFFERENT readers (comparing one reader with itself
 * would satisfy the agreement clause VACUOUSLY).
 *
 * The band width is the EXISTING named grass-apron constant (AC-094 (d)),
 * so the probe margin is expressed through
 * `NEATRACING_TRACK_GRASS_APRON_WIDTH_M` and NEVER through a band-paint
 * constant.
 * ------------------------------------------------------------------------- */

/**
 * The band probe's world x (m): the midpoint of the fixture loop's probed
 * straight, so the nearest chord of the probe is UNAMBIGUOUS.
 */
const BAND_PROBE_X_M = PROBE_STRAIGHT_MID_X;

/**
 * The band probe's lateral offset from the centreline (m): the drivable
 * edge plus HALF the named grass-apron width, so the probe is the MIDDLE
 * of the off-track band.
 */
const BAND_PROBE_LATERAL_M =
  RETIRED_HALF_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M / 2;

/**
 * The band probe's generator-plane y (m): the probed straight's midpoint
 * pushed OUTWARD (the loop is counter-clockwise, so the outer side of the
 * bottom chord is -y) by the band probe's lateral offset.
 */
const BAND_PROBE_GENERATOR_Y_M =
  PROBE_STRAIGHT_MID_Y - BAND_PROBE_LATERAL_M;

/**
 * The band probe's world z (m): the rasterizer's frame mirrors the
 * generator y (`world z = -y`), so the SAME physical point is read by
 * both readers.
 */
const BAND_PROBE_WORLD_Z_M = -BAND_PROBE_GENERATOR_Y_M;

/**
 * The ONE SURFACE MODEL contracts (Step 14W, slice `P1-sand-red`;
 * AC-094, AC-080).
 */
describe('the ONE SURFACE MODEL (Step 14W, slice P1-sand-red)', () => {
  // [RED-14W-ONE-SURFACE-MODEL]
  it('RED-5 (DISCRIMINATING): the band is the SAME surface in the PHYSICS classifier and in the RASTERIZER, and the band is DRIVABLE (the physics model marks the off-track band as the drivable off-road surface)', () => {
    // Arrange: the SAME physical point read by the TWO readers - the
    // RASTERIZER (the WORLD frame) and the PHYSICS classifier (the
    // GENERATOR plane the tick drives in).
    const raster = classifyTrackSlabSurfaceAt(
      BAND_PROBE_X_M,
      BAND_PROBE_WORLD_Z_M,
      CORNER_FIXTURE_SEGMENTS,
    );
    const physics = classifyTrackSurface(
      {
        ...createInitialPhysicsCarState(42),
        x: BAND_PROBE_X_M,
        y: BAND_PROBE_GENERATOR_Y_M,
      },
      CORNER_FIXTURE_SEGMENTS,
    );

    // The NON-VACUITY GUARD comes FIRST: the probe MUST really lie on the
    // off-track band (a probe inside the road would satisfy the clause
    // VACUOUSLY) - the rasterizer paints it off-road and its MEASURED
    // distance past the edge is inside the named apron width.
    expect([
      raster.surface,
      raster.distancePastEdgeM > 0,
      raster.distancePastEdgeM <= NEATRACING_TRACK_GRASS_APRON_WIDTH_M,
    ]).toEqual(['grass', true, true]);

    // ... and the two probed models MUST really be DIFFERENT readers
    // (comparing one reader with itself would satisfy the agreement
    // clause VACUOUSLY): the rasterizer publishes the MEASURED distance
    // past the edge, the physics classifier publishes the DRIVABILITY.
    expect([
      'distancePastEdgeM' in raster,
      'isDrivable' in raster,
      'isDrivable' in physics,
      'distancePastEdgeM' in physics,
    ]).toEqual([true, false, true, false]);

    // RELATIONAL CLAUSE (DISCRIMINATING): the physics classifier marks the
    // band as the DRIVABLE off-road surface. TODAY the band classifies as
    // the NON-drivable grass, so this lane FAILS HERE (by assertion, never
    // by an import failure).
    expect(physics.isDrivable).toBe(true);

    // RAIL: both readers name the SAME EXISTING surface member for that
    // point - the band REUSES the 'grass' member (the surface union stays
    // road | curb | grass, so NO new member was introduced) and the band
    // is not road-only.
    expect([physics.surface, physics.isRoadOnly]).toEqual(['grass', false]);
  });
});
