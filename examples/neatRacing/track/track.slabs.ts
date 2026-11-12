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
 * THE EDGE IS NOT THIS MODULE'S TO DECIDE: the road edge is the NAMED
 * asymmetric pair published by the edge seam (`./track.edge`) - an inner
 * half-width (the road never narrows) and an outer half-width (the inner
 * one plus the corner-measured drift room). This module MEASURES the road
 * through that seam - the published half-width pair, the loop-global side
 * assignment and the canonical chord distance - and keeps only its OWN
 * ratified band policy (the min-past-edge UNION over every chord, road
 * before curb before grass). It derives NO half-width of its own, so the
 * rasterizer, the tri-colour band paint and the containment bound can never
 * disagree about how WIDE the road is.
 *
 * THE RASTERIZED DOMAIN FOLLOWS THE PUBLISHED WIDTHS: the lattice the
 * ribbon is painted on is the CENTRELINE AABB grown by the WIDEST PUBLISHED
 * half-width on EVERY side plus the NAMED grass-apron width, so the widened
 * corner is rasterized instead of being clipped away by a centreline-only
 * rectangle. The growth is ADDITIVE and DOMAIN-ONLY: it moves no road edge,
 * no centreline and no band, and it is resolved ONCE per build.
 *
 * THE BAND GEOMETRY IS PUBLISHED HERE, NOT MIRRORED: this module owns the
 * two band widths of the tri-colour band - the curb band
 * ({@link NEATRACING_TRACK_CURB_WIDTH_M}) and the grass apron
 * ({@link NEATRACING_TRACK_GRASS_APRON_WIDTH_M}) - and the physics surface
 * classifier and the ONE containment bound read them from HERE (through the
 * re-export in `../physics/collision.utils`), so the rasterizer and the
 * physics model share ONE band boundary (AC-094 (e)) instead of mirroring a
 * value that could drift apart. The apron width is the width of the
 * OFF-TRACK BAND: the band stays PAINTED as off-road (the paint ladder is
 * unchanged - the palette values, their order and their cumulative
 * semantics are untouched) and only the PHYSICAL meaning of the drivable
 * edge changes.
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
import {
  distanceToTrackChordMAtWorldZ,
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAtWorldZ,
} from './track.edge';
import type { TrackEdgeProfile } from './track.edge';
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

/**
 * Width of the curb band on each side of the road core, in meters - the
 * PUBLISHED curb-band width of the tri-colour band.
 *
 * THE BAND GEOMETRY IS PUBLISHED HERE, NOT MIRRORED: this module owns the
 * two band widths of the tri-colour band (the curb band and the grass
 * apron), and the physics surface classifier and the ONE containment bound
 * read them from HERE (through the re-export in
 * `../physics/collision.utils`), so the rasterizer and the physics model
 * can never disagree about where a band ends.
 */
export const NEATRACING_TRACK_CURB_WIDTH_M = 1.0;

/**
 * Width of the grass apron beyond the curb on each side, in meters - the
 * PUBLISHED width of the OFF-TRACK BAND (AC-094 (e)).
 *
 * The apron IS the off-track band: it stays PAINTED as off-road (the paint
 * ladder is unchanged - the palette values, their order and their
 * cumulative semantics are untouched), and it is also the band the physics
 * surface classifier reads as the DRIVABLE off-road surface and the ONE
 * containment bound measures to (the band outer edge). Both read THIS
 * constant through the re-export in `../physics/collision.utils`, so the
 * rasterizer and the physics model share ONE band boundary.
 */
export const NEATRACING_TRACK_GRASS_APRON_WIDTH_M = 3.0;

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

/**
 * Closed interval span of the RASTERIZED track domain on the world ground
 * plane: the world x axis and the MIRRORED world z axis (the scene
 * handedness the rasterizer paints in).
 */
interface TrackDomainAabb {
  /** Minimum world-meter x of the rasterized domain. */
  readonly minX: number;
  /** Maximum world-meter x of the rasterized domain. */
  readonly maxX: number;
  /** Minimum mirrored world-meter z of the rasterized domain. */
  readonly minZ: number;
  /** Maximum mirrored world-meter z of the rasterized domain. */
  readonly maxZ: number;
}

/**
 * Classifies one ground-plane point into a lattice surface palette index.
 *
 * Bands are measured from the centerline of every chord, through the edge
 * seam: the half-width of a chord is the PUBLISHED half-width of the side
 * the point sits on (the inner half-width on the inner side, the widened
 * outer half-width on the outer side), so the road is WIDER on the outer
 * side of a corner and never narrower on the inner side. When a point sits
 * in the widening of one chord but inside the road core of another, the
 * highest-priority surface wins (road before curb before grass), keeping
 * junctions and overlapping corner ribbons drivable.
 *
 * @param pointX - World-frame x of the sample point.
 * @param worldZ - World-frame z of the sample point.
 * @param edgeProfile - The edge profile published by the edge seam for the
 *   ribbon being rasterized.
 * @returns Lattice palette index: 1 road, 2 curb, 3 grass, 0 air.
 *
 * @example
 * ```ts
 * resolveSurfaceIndexAt(0, 0, resolveTrackEdgeProfile(spec.segments)); // 1 -> road core
 * ```
 */
function resolveSurfaceIndexAt(
  pointX: number,
  worldZ: number,
  edgeProfile: TrackEdgeProfile,
): number {
  let inRoadCore = false;
  let inCurbBand = false;
  let inGrassApron = false;

  for (
    let segmentIndex = 0;
    segmentIndex < edgeProfile.chords.length;
    segmentIndex += 1
  ) {
    const chord = edgeProfile.chords[segmentIndex];
    const segment = edgeProfile.segments[segmentIndex];
    if (chord === undefined || segment === undefined) {
      continue;
    }

    const halfWidthM = resolveTrackEdgeHalfWidthM(
      chord,
      resolveTrackEdgeSideAtWorldZ(edgeProfile, segmentIndex, pointX, worldZ),
    );
    const distance = distanceToTrackChordMAtWorldZ(segment, pointX, worldZ);

    if (distance <= halfWidthM - NEATRACING_TRACK_CURB_WIDTH_M) {
      inRoadCore = true;
    } else if (distance <= halfWidthM) {
      inCurbBand = true;
    } else if (distance <= halfWidthM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M) {
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
 * @param edgeProfile - The edge profile published by the edge seam for the
 *   ribbon being rasterized.
 * @returns One painted slice per lattice z, in ascending z order.
 *
 * @example
 * ```ts
 * paintTrackSlabSlices(4, 4, -1, -1, resolveTrackEdgeProfile(spec.segments));
 * // -> [Uint8Array(4), Uint8Array(4), Uint8Array(4), Uint8Array(4)]
 * ```
 */
function paintTrackSlabSlices(
  cellsX: number,
  cellsZ: number,
  originX: number,
  originZ: number,
  edgeProfile: TrackEdgeProfile,
): Uint8Array[] {
  const slices: Uint8Array[] = [];

  for (let sliceZ = 0; sliceZ < cellsZ; sliceZ++) {
    const slice = new Uint8Array(cellsX);

    for (let cellX = 0; cellX < cellsX; cellX++) {
      const pointX = originX + (cellX + 0.5) * NEATRACING_TRACK_SLAB_PITCH_M;
      const worldZ = originZ + (sliceZ + 0.5) * NEATRACING_TRACK_SLAB_PITCH_M;
      slice[cellX] = resolveSurfaceIndexAt(pointX, worldZ, edgeProfile);
    }

    slices.push(slice);
  }

  return slices;
}

/**
 * Resolves the WIDEST PUBLISHED half-width of the ribbon: the largest
 * published OUTER half-width over every chord of the edge profile.
 *
 * The outer half-width of a corner is the widest one, and the inner
 * half-width never exceeds it (the road never widens inwards), so this
 * value bounds the width of EVERY chord band on BOTH sides.
 *
 * @param edgeProfile - The edge profile published by the edge seam.
 * @returns The widest published half-width, in meters (0 for an empty loop).
 *
 * @example
 * ```ts
 * resolveWidestPublishedHalfWidthM(resolveTrackEdgeProfile(spec.segments));
 * ```
 */
function resolveWidestPublishedHalfWidthM(
  edgeProfile: TrackEdgeProfile,
): number {
  let widestHalfWidthM = 0;

  for (const chord of edgeProfile.chords) {
    widestHalfWidthM = Math.max(widestHalfWidthM, chord.outerHalfWidthM);
  }

  return widestHalfWidthM;
}

/**
 * Measures the RASTERIZED DOMAIN of the closed loop: the CENTRELINE AABB
 * grown by the WIDEST PUBLISHED half-width on EVERY side, plus the NAMED
 * grass-apron width.
 *
 * THE GROWTH IS ADDITIVE AND DOMAIN-ONLY: the road itself is NOT widened
 * here - the CENTRELINE, the chord count, the spline samples and the
 * published half-widths are untouched - only the RECTANGLE the rasterizer
 * samples is grown, so the widened corner (the published outer half-width)
 * is rasterized instead of being clipped away.
 *
 * THE GROWTH IS AXIS-ALIGNED: a lattice rectangle cannot grow on ONE side
 * of a chord, so the outer/inner distinction is NOT applied to this
 * rectangle - it is applied to the PROBE and to the WIDENING MEASUREMENT
 * (the loop-global outer side), never to the lattice bounds.
 *
 * The apron width is the SAME constant the band policy measures from
 * ({@link NEATRACING_TRACK_GRASS_APRON_WIDTH_M}), so the rasterized domain
 * contains every cell the band policy can classify: no classified surface
 * cell is dropped at the border.
 *
 * @param spec - Generated track spec.
 * @param edgeProfile - The edge profile published by the edge seam for the
 *   ribbon being rasterized.
 * @returns World-space bounds of the rasterized domain, in meters (the z
 *   bounds are MIRRORED world z, the scene handedness the rasterizer
 *   paints in).
 *
 * @example
 * ```ts
 * const domain = measureRasterizedDomainAabb(spec, resolveTrackEdgeProfile(spec.segments));
 * domain.maxX - domain.minX; // the rasterized width, in meters
 * ```
 */
function measureRasterizedDomainAabb(
  spec: TrackSpec,
  edgeProfile: TrackEdgeProfile,
): TrackDomainAabb {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const segment of spec.segments) {
    minX = Math.min(minX, segment.startX, segment.endX);
    maxX = Math.max(maxX, segment.startX, segment.endX);
    minY = Math.min(minY, segment.startY, segment.endY);
    maxY = Math.max(maxY, segment.startY, segment.endY);
  }

  const growthM =
    resolveWidestPublishedHalfWidthM(edgeProfile) +
    NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

  return {
    minX: minX - growthM,
    maxX: maxX + growthM,
    minZ: -(maxY + growthM),
    maxZ: -(minY - growthM),
  };
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
 * The sampled rectangle is the RASTERIZED DOMAIN (the centreline AABB grown
 * by the widest published half-width plus the grass apron), so the widened
 * corner is inside the lattice instead of being clipped away by a
 * centreline-only rectangle.
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
  // The edge profile is resolved ONCE for the whole lattice: it is the
  // published width profile every cell measures the road through AND the
  // source of the WIDEST published half-width the rasterized domain is
  // grown by. Nothing here is resolved per cell or per tick.
  const edgeProfile = resolveTrackEdgeProfile(spec.segments.slice());
  const domain = measureRasterizedDomainAabb(spec, edgeProfile);
  const pitch = NEATRACING_TRACK_SLAB_PITCH_M;
  const originX = Math.floor(domain.minX / pitch) * pitch;
  const originZ = Math.floor(domain.minZ / pitch) * pitch;
  const cellsX = Math.max(1, Math.ceil((domain.maxX - originX) / pitch));
  const cellsZ = Math.max(1, Math.ceil((domain.maxZ - originZ) / pitch));

  const model: DecodedVoxelModel = {
    name: TRACK_SLAB_MODEL_NAME,
    dims: [cellsX, TRACK_SLAB_STACK_ROWS, cellsZ],
    symmetric: false,
    palette: buildTrackSlabLatticePalette(),
    slices: paintTrackSlabSlices(cellsX, cellsZ, originX, originZ, edgeProfile),
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
 * Surface label for one cell of the track slab lattice, including the
 * internal air/empty sentinel so callers can reason about boundaries.
 */
export type TrackSlabSurfaceCell = 'air' | TrackSlabSurfaceType;

/**
 * Classification of an arbitrary ground-plane point, including both the
 * priority-winning surface label and the distance past the nearest track edge
 * so consumers can place border geometry consistently with the ribbon.
 */
export interface TrackSlabSurfaceClassification {
  /** Priority-winning surface label at the query point. */
  readonly surface: TrackSlabSurfaceCell;
  /**
   * Signed distance from the query point to the nearest segment's road edge
   * (positive means outside the roadway, negative means inside the half-width
   * envelope). Always defined so consumers can place border geometry beyond the
   * grass apron consistently with the ribbon.
   */
  readonly distancePastEdgeM: number;
}

/**
 * Classifies an arbitrary ground-plane point using the same band logic that
 * paints the 0.5 m slab lattice. This lets other builders (for example the
 * road-edge flat paint) emit geometry whose cells sit exactly on the ribbon's
 * road/curb/grass boundaries without duplicating classification logic.
 *
 * THE EDGE IS THE SEAM'S: the half-width of every chord is the PUBLISHED
 * half-width of the side the point sits on (the inner half-width on the
 * inner side, the widened outer half-width on the outer side), measured
 * through the edge seam - this reader derives NO half-width of its own. The
 * band policy stays THIS reader's: the min-past-edge UNION over every chord
 * with the road-before-curb-before-grass priority.
 *
 * @param pointX - World-frame x of the sample point.
 * @param pointZ - World-frame z of the sample point. The edge seam mirrors
 *   the world z back into the generator plane (`worldZ = -y`), the same
 *   handedness the slab rasterizer uses.
 * @param segments - Closed-loop segments defining the ribbon.
 * @param edgeProfile - The edge profile published by the edge seam for those
 *   segments. It is OPTIONAL: a caller that classifies many points of the
 *   same ribbon passes the profile it resolved ONCE, so the published width
 *   profile is not rebuilt per point. When it is omitted the profile is
 *   resolved from `segments` - a caller that passes a profile MUST pass one
 *   resolved from the SAME segments.
 * @returns Surface label plus signed distance past the nearest track edge.
 *   `distancePastEdgeM` is always defined: it is negative inside the roadway,
 *   zero on the curb outer edge, and positive outside.
 */
export function classifyTrackSlabSurfaceAt(
  pointX: number,
  pointZ: number,
  segments: readonly TrackSegment[],
  edgeProfile: TrackEdgeProfile = resolveTrackEdgeProfile(segments.slice()),
): TrackSlabSurfaceClassification {
  let inRoadCore = false;
  let inCurbBand = false;
  let inGrassApron = false;
  let minDistancePastEdgeM = Number.POSITIVE_INFINITY;

  for (
    let segmentIndex = 0;
    segmentIndex < edgeProfile.chords.length;
    segmentIndex += 1
  ) {
    const chord = edgeProfile.chords[segmentIndex];
    const segment = edgeProfile.segments[segmentIndex];
    if (chord === undefined || segment === undefined) {
      continue;
    }

    const halfWidthM = resolveTrackEdgeHalfWidthM(
      chord,
      resolveTrackEdgeSideAtWorldZ(edgeProfile, segmentIndex, pointX, pointZ),
    );
    const distance = distanceToTrackChordMAtWorldZ(segment, pointX, pointZ);
    const pastEdge = distance - halfWidthM;

    if (pastEdge < minDistancePastEdgeM) {
      minDistancePastEdgeM = pastEdge;
    }

    if (distance <= halfWidthM - NEATRACING_TRACK_CURB_WIDTH_M) {
      inRoadCore = true;
    } else if (distance <= halfWidthM) {
      inCurbBand = true;
    } else if (distance <= halfWidthM + NEATRACING_TRACK_GRASS_APRON_WIDTH_M) {
      inGrassApron = true;
    }
  }

  if (inRoadCore) {
    return { surface: 'road', distancePastEdgeM: minDistancePastEdgeM };
  }
  if (inCurbBand) {
    return { surface: 'curb', distancePastEdgeM: minDistancePastEdgeM };
  }
  if (inGrassApron) {
    return { surface: 'grass', distancePastEdgeM: minDistancePastEdgeM };
  }
  return { surface: 'air', distancePastEdgeM: minDistancePastEdgeM };
}

/**
 * Typed 2-D grid of track slab surface labels aligned to the same 0.5 m
 * lattice used by {@link buildTrackSlabs}. This lets other builders emit
 * geometry whose cells sit exactly on the ribbon's road/curb/grass boundaries
 * without duplicating classification logic.
 */
export interface TrackSlabSurfaceGrid {
  /** World-meter position of lattice cell [0, 0, 0]. */
  readonly origin: readonly [number, number, number];
  /** World-space pitch of every cell, in meters. */
  readonly pitch: number;
  /** Number of lattice cells along world x. */
  readonly cellsX: number;
  /** Number of lattice cells along world z. */
  readonly cellsZ: number;
  /**
   * Surface labels indexed `rows[z][x]` matching the lattice handedness
   * (`worldZ = -y`).
   */
  readonly rows: readonly (readonly TrackSlabSurfaceCell[])[];
}

/**
 * Maps an internal lattice palette index to the exported surface label.
 */
function surfaceIndexToType(index: number): TrackSlabSurfaceCell {
  if (index === TRACK_ROAD_SURFACE_INDEX) {
    return 'road';
  }
  if (index === TRACK_CURB_SURFACE_INDEX) {
    return 'curb';
  }
  if (index === TRACK_GRASS_SURFACE_INDEX) {
    return 'grass';
  }
  return 'air';
}

/**
 * Builds a typed surface grid for the deterministic track tuple, sharing the
 * same 0.5 m lattice rasterization used by the slab mesh. The returned grid
 * origin, pitch, and dimensions are exactly those the slab builder uses, so
 * consumers can place geometry on the same cell centers.
 *
 * @param input - Deterministic build input
 *   `{ seed, layoutVersion, sizeBucket }` matching the shared generator.
 * @returns Typed surface grid with road, curb, grass, and air cells.
 * @example
 * ```ts
 * const grid = buildTrackSlabSurfaceGrid({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
 * grid.rows[0][0]; // 'air' | 'road' | 'curb' | 'grass'
 * ```
 */
export function buildTrackSlabSurfaceGrid(
  input: TrackSlabBuildInput,
): TrackSlabSurfaceGrid {
  const spec = generateTrack(input);
  const lattice = synthesizeTrackSlabLattice(spec);
  const pitch = NEATRACING_TRACK_SLAB_PITCH_M;

  const rows: TrackSlabSurfaceCell[][] = lattice.model.slices.map((slice) =>
    Array.from(slice, surfaceIndexToType),
  );

  return {
    origin: lattice.origin,
    pitch,
    cellsX: lattice.model.dims[0],
    cellsZ: lattice.model.dims[2],
    rows,
  };
}

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
