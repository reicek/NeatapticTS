/**
 * Track slab builder for the neatRacing example (plan slice P1-S7-A,
 * spec section 6.7 "Track rendering" / decision LD-25).
 *
 * The track renders as an axis-aligned slab ribbon rasterized onto a
 * 0.5-meter lattice (TP-19) at ground scale, distinct from the voxel-car
 * world scale (G12: {@link NEATRACING_TRACK_SLAB_PITCH_M} is 0.5 m while
 * `NEATRACING_VOXEL_WORLD_SIZE` is a much finer per-car lattice size).
 *
 * Rasterization reuses the deterministic closed-loop generator already
 * shipped in `examples/racing_curriculum` (spec section 11.1 recycling
 * inventory), meshes the resulting slab lattice with the shared greedy
 * mesher (`../voxel/greedy-mesher`), and converts every quad back into
 * world meters with a 0.5 m multiple minimum-corner position so slab
 * floors, walls, and the ground plane meet exactly with no z-fighting
 * seams.
 *
 * Determinism: {@link buildTrackSlabs} is a pure function of
 * `{ seed, layoutVersion, sizeBucket }` — two calls with the same input
 * produce byte-identical `JSON.stringify` output in the same runtime
 * (ordered emission from the shared mesher, no runtime-random state).
 *
 * Scope note (evidence batch 80 rescoping): the spec defines no bank or
 * boost-pad track feature, so the slab builder emits road, curb, and
 * grass surfaces only.
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type {
  TrackSegment,
  TrackSpec,
} from '../../racing_curriculum/track/track.generator.types';
import { meshVoxelModel } from '../voxel/greedy-mesher';
import type { VoxelMesh, VoxelQuad } from '../voxel/greedy-mesher';
import type { DecodedVoxelModel } from '../voxel/model.decode.utils';
import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  VOXEL_EMPTY_PALETTE_INDEX,
} from '../voxel/model.schema';
import type { VoxelPaletteEntry } from '../voxel/model.schema';

/** World-space pitch of every track slab, in meters (spec TP-19). */
export const NEATRACING_TRACK_SLAB_PITCH_M = 0.5;

/** Lattice rows along world-up: one slab row of 0.5 m track thickness. */
const TRACK_SLAB_STACK_ROWS = 1;

/**
 * World-up origin of the slab lattice, in meters: the single lattice row
 * spans world y in [-0.5, 0), so the road's top face plane (lattice y = 1)
 * maps to world y = 0 — flush with the ground plane.
 */
const TRACK_SLAB_ORIGIN_Y_M = -0.5;

/** Width of the curb band on each side of the road core, in meters. */
const TRACK_SLAB_CURB_WIDTH_M = 1.0;

/** Width of the grass apron beyond the curb on each side, in meters. */
const TRACK_SLAB_GRASS_APRON_WIDTH_M = 3.0;

/** Internal model name for the synthesized slab lattice. */
const TRACK_SLAB_MODEL_NAME = 'neatracing-track-slabs';

/** Lattice palette index for the road core (must equal the palette position). */
const TRACK_ROAD_SURFACE_INDEX = 1;

/** Lattice palette index for the curb band (must equal the palette position). */
const TRACK_CURB_SURFACE_INDEX = 2;

/** Lattice palette index for the grass apron (must equal the palette position). */
const TRACK_GRASS_SURFACE_INDEX = 3;

const TRACK_ROAD_COLOR: readonly [number, number, number] = [64, 64, 68];
const TRACK_CURB_COLOR: readonly [number, number, number] = [214, 44, 44];
const TRACK_GRASS_COLOR: readonly [number, number, number] = [46, 156, 70];

/** Opaque alpha byte for every track slab surface (fully solid slabs). */
const TRACK_SLAB_OPAQUE_ALPHA = 255;

/** Glow byte for track slabs: none of the three surfaces bloom. */
const TRACK_SLAB_SURFACE_GLOW = 0;

const TRACK_ROAD_PALETTE_ENTRY: VoxelPaletteEntry = [
  ...TRACK_ROAD_COLOR,
  TRACK_SLAB_OPAQUE_ALPHA,
  TRACK_SLAB_SURFACE_GLOW,
];
const TRACK_CURB_PALETTE_ENTRY: VoxelPaletteEntry = [
  ...TRACK_CURB_COLOR,
  TRACK_SLAB_OPAQUE_ALPHA,
  TRACK_SLAB_SURFACE_GLOW,
];
const TRACK_GRASS_PALETTE_ENTRY: VoxelPaletteEntry = [
  ...TRACK_GRASS_COLOR,
  TRACK_SLAB_OPAQUE_ALPHA,
  TRACK_SLAB_SURFACE_GLOW,
];

/** Surface labels exported with the slab palette (no EMPTY sentinel). */
export type TrackSlabSurfaceType = 'road' | 'curb' | 'grass';

/**
 * One labeled track slab material. Array position is the quad
 * `paletteIndex` in {@link TrackSlabMesh.quads}; the EMPTY sentinel from
 * the lattice palette is intentionally not exported.
 */
export interface TrackSlabPaletteEntry {
  /** Surface label. */
  readonly type: TrackSlabSurfaceType;
  /** Display color bytes `[r, g, b]`, 0-255 each. */
  readonly color: readonly [number, number, number];
}

/** Mesh of the track: labeled surface palette plus world-meter quads. */
export interface TrackSlabMesh {
  /** Exactly the road / curb / grass entries, in this order. */
  readonly palette: readonly TrackSlabPaletteEntry[];
  /** §7.3.1 quads scaled into world meters at 0.5 m pitch. */
  readonly quads: readonly VoxelQuad[];
}

/** Deterministic track build input mirrored from the recycled generator. */
export interface TrackSlabBuildInput {
  /** Base deterministic seed (for example 42). */
  readonly seed: number;
  /** Layout version mixed into the seed for reproducible revisions. */
  readonly layoutVersion: number;
  /** Track size bucket (`small` | `medium` | `large`). */
  readonly sizeBucket: string;
}

/** World-meter origin of the slab lattice per axis [x, y, z]. */
type TrackSlabWorldOrigin = readonly [number, number, number];

/** Rasterized slab lattice paired with its world-space origin. */
interface TrackSlabLattice {
  /** Decoded-model-shaped lattice ready for the shared greedy mesher. */
  readonly model: DecodedVoxelModel;
  /** World-meter position of lattice cell [0, 0, 0]. */
  readonly origin: TrackSlabWorldOrigin;
}

/** Closed interval span of the track segments in the ground plane. */
interface TrackSpanAabb {
  readonly minX: number;
  readonly maxX: number;
  readonly minZ: number;
  readonly maxZ: number;
}

/**
 * Distance from a ground-plane point to the segment's centerline chord.
 *
 * `startY`/`endY` on a {@link TrackSegment} are ground-plane coordinates
 * (the generator works in the x/y plane). The slab lattice rasterizes in
 * generator ground coordinates with `pointZ` on the same y axis; the mirror
 * into the rendered scene (world z = -y) happens at the lattice origin,
 * where each painted cell's query point negates the lattice z (see
 * {@link paintTrackSlabSlices}).
 *
 * @param pointX - Ground-plane x of the sample point.
 * @param pointZ - Ground-plane y-axis coordinate of the sample point (the
 *   generator's y, in generator plane meters).
 * @param segment - Segment whose centerline chord is measured against.
 * @returns Euclidean distance to the clamped nearest chord point.
 *
 * @example
 * ```ts
 * const { generateTrack } = require('../../racing_curriculum/track/track.generator');
 * distanceToSegment2D(0, 0, generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' }).segments[0]);
 * ```
 */
function distanceToSegment2D(
  pointX: number,
  pointZ: number,
  segment: TrackSegment,
): number {
  const chordX = segment.endX - segment.startX;
  const chordZ = segment.endY - segment.startY;
  const chordLengthSquared = chordX * chordX + chordZ * chordZ;

  if (chordLengthSquared <= 0) {
    return Math.hypot(pointX - segment.startX, pointZ - segment.startY);
  }

  const projection =
    ((pointX - segment.startX) * chordX + (pointZ - segment.startY) * chordZ) /
    chordLengthSquared;
  const clampedProjection = Math.min(1, Math.max(0, projection));
  const nearestX = segment.startX + clampedProjection * chordX;
  const nearestZ = segment.startY + clampedProjection * chordZ;

  return Math.hypot(pointX - nearestX, pointZ - nearestZ);
}

/**
 * Classifies one ground-plane point into a lattice surface palette index.
 *
 * Bands are measured from the centerline of the nearest segment: road
 * core, then curb, then grass apron, then air. When a point sits in the
 * widening of one segment but inside the road core of another, the
 * highest-priority surface wins (road before curb before grass), keeping
 * junctions and overlapping corner ribbons drivable.
 *
 * @param pointX - Ground-plane x of the sample point.
 * @param pointZ - Ground-plane z of the sample point.
 * @param segments - Closed-loop segments defining the ribbon.
 * @returns Lattice palette index: 1 road, 2 curb, 3 grass, 0 air.
 *
 * @example
 * ```ts
 * resolveSurfaceIndexAt(0, 0, spec.segments); // 1 -> road core
 * ```
 */
function resolveSurfaceIndexAt(
  pointX: number,
  pointZ: number,
  segments: readonly TrackSegment[],
): number {
  let inRoadCore = false;
  let inCurbBand = false;
  let inGrassApron = false;

  for (const segment of segments) {
    const halfWidth = segment.width / 2;
    const distance = distanceToSegment2D(pointX, pointZ, segment);

    if (distance <= halfWidth - TRACK_SLAB_CURB_WIDTH_M) {
      inRoadCore = true;
    } else if (distance <= halfWidth) {
      inCurbBand = true;
    } else if (distance <= halfWidth + TRACK_SLAB_GRASS_APRON_WIDTH_M) {
      inGrassApron = true;
    }
  }

  if (inRoadCore) {
    return TRACK_ROAD_SURFACE_INDEX;
  }
  if (inCurbBand) {
    return TRACK_CURB_SURFACE_INDEX;
  }
  if (inGrassApron) {
    return TRACK_GRASS_SURFACE_INDEX;
  }
  return VOXEL_EMPTY_PALETTE_INDEX;
}

/**
 * Rasterizes the ribbon into one `Uint8Array` slice per lattice z column.
 *
 * The slab lattice has a single body row (`dims[1] === 1`), so each slice
 * is a one-row `Uint8Array` indexed by lattice x (`slices[z][x]`), with
 * cell centers sampled at half-pitch offsets for a symmetric margin.
 *
 * The lattice z column mirrors the generator y (`worldZ = -y`, the scene
 * handedness seam shared with the frame composer's car pose and camera):
 * `originZ` is the mirrored world-meter z of lattice cell z = 0, and each
 * cell's generator-plane query point negates the world z back into
 * generator y before the surface classification runs.
 *
 * @param cellsX - Lattice width in 0.5 m cells.
 * @param cellsZ - Lattice depth in 0.5 m cells.
 * @param originX - World-meter x of lattice cell x = 0.
 * @param originZ - Mirrored world-meter z of lattice cell z = 0 (the
 *   query point per row is the generator y at -worldZ).
 * @param segments - Closed-loop segments defining the ribbon.
 * @returns One painted slice per lattice z, in ascending z order.
 *
 * @example
 * ```ts
 * paintTrackSlabSlices(4, 4, -1, -1, spec.segments);
 * // -> [Uint8Array(4), Uint8Array(4), Uint8Array(4), Uint8Array(4)]
 * ```
 */
function paintTrackSlabSlices(
  cellsX: number,
  cellsZ: number,
  originX: number,
  originZ: number,
  segments: readonly TrackSegment[],
): Uint8Array[] {
  const slices: Uint8Array[] = [];

  for (let sliceZ = 0; sliceZ < cellsZ; sliceZ++) {
    const slice = new Uint8Array(cellsX);

    for (let cellX = 0; cellX < cellsX; cellX++) {
      const pointX = originX + (cellX + 0.5) * NEATRACING_TRACK_SLAB_PITCH_M;
      const worldZ = originZ + (sliceZ + 0.5) * NEATRACING_TRACK_SLAB_PITCH_M;
      const pointZ = -worldZ;
      slice[cellX] = resolveSurfaceIndexAt(pointX, pointZ, segments);
    }

    slices.push(slice);
  }

  return slices;
}

/**
 * Measures the ground-plane AABB spanned by every segment endpoint, in
 * generator ground coordinates (x, generator y).
 *
 * @param spec - Generated track spec.
 * @returns Span over all segment start/end corners (`minZ`/`maxZ` are the
 *   generator y-axis span; the scene mirror negates these at the origin).
 *
 * @example
 * ```ts
 * const { minX, maxX } = measureSegmentSpanAabb(spec);
 * ```
 */
function measureSegmentSpanAabb(spec: TrackSpec): TrackSpanAabb {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (const segment of spec.segments) {
    minX = Math.min(minX, segment.startX, segment.endX);
    maxX = Math.max(maxX, segment.startX, segment.endX);
    minZ = Math.min(minZ, segment.startY, segment.endY);
    maxZ = Math.max(maxZ, segment.startY, segment.endY);
  }

  return { minX, maxX, minZ, maxZ };
}

/**
 * Builds the slab lattice palette: the RESERVED EMPTY sentinel at index 0
 * followed by road, curb, then grass. Lattice indices {@link
 * TRACK_ROAD_SURFACE_INDEX}-{@link TRACK_GRASS_SURFACE_INDEX} must stay
 * aligned with these positions.
 *
 * @returns The four-entry lattice palette handed to the shared mesher.
 *
 * @example
 * ```ts
 * buildTrackSlabLatticePalette(); // [EMPTY, road, curb, grass]
 * ```
 */
function buildTrackSlabLatticePalette(): VoxelPaletteEntry[] {
  return [
    [...VOXEL_EMPTY_PALETTE_ENTRY] as VoxelPaletteEntry,
    TRACK_ROAD_PALETTE_ENTRY,
    TRACK_CURB_PALETTE_ENTRY,
    TRACK_GRASS_PALETTE_ENTRY,
  ];
}

/**
 * Rasterizes the closed-loop ribbon into a `DecodedVoxelModel` lattice on
 * the world grid, snapping the origin to the 0.5 m grid so every painted
 * cell and every produced quad maps to exact 0.5 m world coordinates. The
 * lattice z axis mirrors the generator ground y (`worldZ = -y`), matching
 * the scene handedness the frame composer applies to the car pose and the
 * chase camera.
 *
 * @param spec - Generated track spec (already validated by the generator).
 * @returns Lattice model ready for {@link meshVoxelModel}, plus origin.
 *
 * @example
 * ```ts
 * const lattice = synthesizeTrackSlabLattice(spec);
 * meshVoxelModel(lattice.model); // greedy-meshed lattice quads
 * ```
 */
function synthesizeTrackSlabLattice(spec: TrackSpec): TrackSlabLattice {
  const span = measureSegmentSpanAabb(spec);
  const pitch = NEATRACING_TRACK_SLAB_PITCH_M;
  const originX = Math.floor(span.minX / pitch) * pitch;
  const mirroredMinZ = -span.maxZ;
  const mirroredMaxZ = -span.minZ;
  const originZ = Math.floor(mirroredMinZ / pitch) * pitch;
  const cellsX = Math.max(1, Math.ceil((span.maxX - originX) / pitch));
  const cellsZ = Math.max(1, Math.ceil((mirroredMaxZ - originZ) / pitch));

  const model: DecodedVoxelModel = {
    name: TRACK_SLAB_MODEL_NAME,
    dims: [cellsX, TRACK_SLAB_STACK_ROWS, cellsZ],
    symmetric: false,
    palette: buildTrackSlabLatticePalette(),
    slices: paintTrackSlabSlices(
      cellsX,
      cellsZ,
      originX,
      originZ,
      spec.segments,
    ),
  };

  return {
    model,
    origin: [originX, TRACK_SLAB_ORIGIN_Y_M, originZ],
  };
}

/**
 * Converts lattice quads into world-meter track slabs.
 *
 * Positions become 0.5 m multiple minimum corners
 * (`origin + lattice * pitch`), extents scale by the pitch, the normal
 * axis passes through unchanged, and the lattice palette index shifts by
 * one so index 0 addresses the first exported labeled palette entry
 * (skipping the internal EMPTY sentinel).
 *
 * @param latticeMesh - Merged quads from the shared mesher (cell units).
 * @param origin - World-meter origin of lattice cell [0, 0, 0].
 * @returns Quads in world meters, preserving deterministic emission order.
 *
 * @example
 * ```ts
 * const quads = toWorldSlabQuads(meshVoxelModel(lattice.model), lattice.origin);
 * quads[0].position; // world-meter minimum corner at 0.5 m multiples
 * ```
 */
function toWorldSlabQuads(
  latticeMesh: VoxelMesh,
  origin: TrackSlabWorldOrigin,
): VoxelQuad[] {
  const pitch = NEATRACING_TRACK_SLAB_PITCH_M;

  return latticeMesh.quads.map((quad): VoxelQuad => ({
    position: [
      origin[0] + quad.position[0] * pitch,
      origin[1] + quad.position[1] * pitch,
      origin[2] + quad.position[2] * pitch,
    ],
    normalAxis: quad.normalAxis,
    width: quad.width * pitch,
    height: quad.height * pitch,
    paletteIndex: quad.paletteIndex - 1,
  }));
}

/**
 * Exported track surface palette: exactly the labeled road, curb, and
 * grass entries (array position doubles as the quad `paletteIndex`).
 */
const TRACK_SLAB_PALETTE: readonly TrackSlabPaletteEntry[] = [
  { type: 'road', color: TRACK_ROAD_COLOR },
  { type: 'curb', color: TRACK_CURB_COLOR },
  { type: 'grass', color: TRACK_GRASS_COLOR },
];

/**
 * Builds the deterministic track slab mesh for a generateTrack fixture.
 *
 * Pipeline: recycled closed-loop generator (spec section 11) ->
 * 0.5 m lattice rasterization at ground scale (TP-19) -> shared greedy
 * mesher (dogfooding LD-28) -> world-meter quad conversion. Two calls
 * with the same input produce byte-identical meshes (AC-017 Level 2).
 *
 * @param input - Deterministic build input
 *   `{ seed, layoutVersion, sizeBucket }` matching the shared generator.
 * @returns Track slab mesh with a labeled road/curb/grass palette and
 *   world-meter quads whose positions and extents are 0.5 m multiples.
 * @example
 * ```ts
 * const mesh = buildTrackSlabs({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 * mesh.palette.map((entry) => entry.type); // ['road', 'curb', 'grass']
 * mesh.quads[0]; // { position, normalAxis, width, height, paletteIndex }
 * ```
 */
export function buildTrackSlabs(input: TrackSlabBuildInput): TrackSlabMesh {
  const spec = generateTrack(input);
  const lattice = synthesizeTrackSlabLattice(spec);
  const latticeMesh = meshVoxelModel(lattice.model);

  return {
    palette: TRACK_SLAB_PALETTE,
    quads: toWorldSlabQuads(latticeMesh, lattice.origin),
  };
}
