/**
 * Contract tests for the Neat Racing greedy mesher (slice P1-S3-B; AC-009,
 * spec §7.3.1).
 *
 * These tests pin the normative mesher behavior on canonical fixture models:
 * the greedy merge key (palette index + normal axis + face orientation), the
 * pinned quad record `{ position, normalAxis, width, height, paletteIndex }`,
 * interior-face culling (zero interior faces on adjacent solids), greedy
 * quad-count reduction (merged face area exactly tiles every exposed body
 * face), glass semantics (fully transparent glow voxels emit no body quads
 * and do not occlude body faces), and the per-voxel color/emissive lookup
 * with the LD-23 black exception (black never shines regardless of glow).
 */
import { describe, expect, it } from '@jest/globals';
import type { DecodedVoxelModel } from './model.decode.utils';
import type { VoxelPaletteEntry } from './model.schema';
import {
  VOXEL_OPAQUE_ALPHA_MIN,
  meshVoxelModel,
  resolveVoxelShade,
  type VoxelQuad,
} from './greedy-mesher';

/** Palette index of the all-zero EMPTY sentinel entry. */
const PALETTE_INDEX_EMPTY = 0;
/** Palette index of an opaque red body entry (no glow). */
const PALETTE_INDEX_RED = 1;
/** Palette index of an opaque blue accent entry with glow 40. */
const PALETTE_INDEX_ACCENT = 2;
/** Palette index of a black body entry with glow 200 (LD-23: never shines). */
const PALETTE_INDEX_BLACK_GLOW = 3;
/** Palette index of a fully transparent glass entry with glow 180. */
const PALETTE_INDEX_GLASS = 4;

/**
 * Fixture palette covering every material class the mesher must discriminate:
 * EMPTY sentinel, opaque body, glowing accent, black body, and glow glass.
 */
const FIXTURE_PALETTE: VoxelPaletteEntry[] = [
  [0, 0, 0, 0, 0],
  [255, 0, 0, 255, 0],
  [0, 0, 255, 255, 40],
  [0, 0, 0, 255, 200],
  [200, 230, 255, 0, 180],
];

/** A lone solid voxel exposes exactly one face per direction. */
const SINGLE_VOXEL_QUAD_COUNT = 6;

/** Two adjacent solids share exactly one interior face pair. */
const DOMINO_SHARED_PLANE = 1;

/** Two-tone domino: every face pair stays unmerged across the palette split. */
const TWO_TONE_DOMINO_QUAD_COUNT = 10;

/** A body voxel next to glass keeps all six faces exposed (glass is air). */
const GLASS_NEIGHBOR_BODY_QUAD_COUNT = 6;

/** L-shaped fixture (voxels (0,0,0), (1,0,0), (0,0,1)): 10 merged quads. */
const L_SHAPE_QUAD_COUNT = 10;

/**
 * Canonical quad records for a 2x2x2 solid cube, in deterministic emission
 * order (ascending axis, ascending face plane): one merged 2x2 quad per
 * face direction, all twelve interior faces culled.
 */
const SOLID_CUBE_EXPECTED_QUADS: readonly VoxelQuad[] = [
  {
    position: [0, 0, 0],
    normalAxis: 0,
    width: 2,
    height: 2,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [2, 0, 0],
    normalAxis: 0,
    width: 2,
    height: 2,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 0, 0],
    normalAxis: 1,
    width: 2,
    height: 2,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 2, 0],
    normalAxis: 1,
    width: 2,
    height: 2,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 0, 0],
    normalAxis: 2,
    width: 2,
    height: 2,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 0, 2],
    normalAxis: 2,
    width: 2,
    height: 2,
    paletteIndex: PALETTE_INDEX_RED,
  },
];

/**
 * Canonical quad records for a lone solid voxel in emission order (ascending
 * axis, ascending face plane): six unit quads, one per face direction.
 */
const SINGLE_VOXEL_EXPECTED_QUADS: readonly VoxelQuad[] = [
  {
    position: [0, 0, 0],
    normalAxis: 0,
    width: 1,
    height: 1,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [1, 0, 0],
    normalAxis: 0,
    width: 1,
    height: 1,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 0, 0],
    normalAxis: 1,
    width: 1,
    height: 1,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 1, 0],
    normalAxis: 1,
    width: 1,
    height: 1,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 0, 0],
    normalAxis: 2,
    width: 1,
    height: 1,
    paletteIndex: PALETTE_INDEX_RED,
  },
  {
    position: [0, 0, 1],
    normalAxis: 2,
    width: 1,
    height: 1,
    paletteIndex: PALETTE_INDEX_RED,
  },
];

/**
 * The six face-direction neighbor deltas shared by the exposed-face oracle.
 */
const FACE_DELTAS: readonly (readonly [number, number, number])[] = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

/**
 * Builds a decoded model fixture by painting each lattice cell through
 * `paintVoxel`. Bypasses `decodeVoxelModel` on purpose: the decoder pipeline
 * is unit-tested by model.decode.utils.test.ts and this suite isolates the
 * mesher contract.
 *
 * @param dims - Grid dimensions as [width, height, depth]; depth = slice count.
 * @param paintVoxel - Maps a lattice cell (x, y, z) to its palette index.
 * @returns A decoded model over the fixture palette with the painted cells.
 */
function createDecodedModel(
  dims: readonly [number, number, number],
  paintVoxel: (x: number, y: number, z: number) => number,
): DecodedVoxelModel {
  const slices = Array.from({ length: dims[2] }, (_unusedSlice, z) => {
    const cells = new Uint8Array(dims[0] * dims[1]);
    for (let y = 0; y < dims[1]; y++) {
      for (let x = 0; x < dims[0]; x++) {
        cells[x + y * dims[0]] = paintVoxel(x, y, z);
      }
    }
    return cells;
  });
  return {
    name: 'fixture-model',
    dims: [dims[0], dims[1], dims[2]],
    symmetric: false,
    palette: FIXTURE_PALETTE,
    slices,
  };
}

/**
 * Sums merged quad areas (width x height per quad) so merged output can be
 * compared against the naive exposed-face count.
 *
 * @param quads - Quads of a meshed model.
 * @returns Total covered face area in unit-face squares.
 */
function totalQuadArea(quads: readonly VoxelQuad[]): number {
  return quads.reduce((area, quad) => area + quad.width * quad.height, 0);
}

/**
 * Naive per-voxel exposed-face oracle: counts the six-neighbor boundary of
 * every body voxel (alpha >= VOXEL_OPAQUE_ALPHA_MIN), treating glass and
 * EMPTY neighbors as exposed air exactly like the mesher must.
 *
 * @param model - Decoded model to probe.
 * @returns Number of exposed unit faces across all body voxels.
 */
function countExposedFaces(model: DecodedVoxelModel): number {
  const isBodyCell = (x: number, y: number, z: number): boolean => {
    const inGrid =
      x >= 0 &&
      y >= 0 &&
      z >= 0 &&
      x < model.dims[0] &&
      y < model.dims[1] &&
      z < model.dims[2];
    if (!inGrid) {
      return false;
    }
    const index = model.slices[z][x + y * model.dims[0]];
    return model.palette[index][3] >= VOXEL_OPAQUE_ALPHA_MIN;
  };
  let exposed = 0;
  for (let z = 0; z < model.dims[2]; z++) {
    for (let y = 0; y < model.dims[1]; y++) {
      for (let x = 0; x < model.dims[0]; x++) {
        if (!isBodyCell(x, y, z)) {
          continue;
        }
        for (const [dx, dy, dz] of FACE_DELTAS) {
          if (!isBodyCell(x + dx, y + dy, z + dz)) {
            exposed++;
          }
        }
      }
    }
  }
  return exposed;
}

describe('greedy mesher (AC-009, spec 7.3.1)', () => {
  describe('quad record contract', () => {
    it('meshes a single solid voxel into the six canonical unit quads', () => {
      const model = createDecodedModel([1, 1, 1], () => PALETTE_INDEX_RED);
      const mesh = meshVoxelModel(model);
      expect(mesh.quads.length).toBe(SINGLE_VOXEL_QUAD_COUNT);
      expect(mesh.quads).toEqual(SINGLE_VOXEL_EXPECTED_QUADS);
    });

    it('echoes the model dims on the mesh', () => {
      const model = createDecodedModel([2, 1, 2], () => PALETTE_INDEX_EMPTY);
      const mesh = meshVoxelModel(model);
      expect(mesh.dims).toEqual([2, 1, 2]);
    });
  });

  describe('greedy quad reduction and interior-face culling', () => {
    it('meshes a 2x2x2 solid cube into six merged face quads in canonical order', () => {
      const model = createDecodedModel([2, 2, 2], () => PALETTE_INDEX_RED);
      const mesh = meshVoxelModel(model);
      expect(mesh.quads).toEqual(SOLID_CUBE_EXPECTED_QUADS);
    });

    it('emits no quad at the shared plane of two adjacent solids', () => {
      const model = createDecodedModel([2, 1, 1], () => PALETTE_INDEX_RED);
      const mesh = meshVoxelModel(model);
      const interiorPlaneQuads = mesh.quads.filter(
        (quad) =>
          quad.normalAxis === 0 && quad.position[0] === DOMINO_SHARED_PLANE,
      );
      expect(interiorPlaneQuads).toHaveLength(0);
    });

    it('reduces an L-shaped fixture to ten quads tiling its fourteen exposed faces', () => {
      const model = createDecodedModel([2, 1, 2], (x, _y, z) =>
        z === 0 || x === 0 ? PALETTE_INDEX_RED : PALETTE_INDEX_EMPTY,
      );
      const mesh = meshVoxelModel(model);
      expect(mesh.quads.length).toBe(L_SHAPE_QUAD_COUNT);
      expect(totalQuadArea(mesh.quads)).toBe(countExposedFaces(model));
    });
  });

  describe('merge key: palette index + normal axis + orientation', () => {
    it('does not merge adjacent faces of different palette indices', () => {
      const model = createDecodedModel([2, 1, 1], (x) =>
        x === 0 ? PALETTE_INDEX_RED : PALETTE_INDEX_ACCENT,
      );
      const mesh = meshVoxelModel(model);
      expect(mesh.quads.length).toBe(TWO_TONE_DOMINO_QUAD_COUNT);
    });

    it('does not merge opposite face orientations on the same plane', () => {
      const model = createDecodedModel([2, 2, 1], (x, y) =>
        (x === 0 && y === 0) || (x === 1 && y === 1)
          ? PALETTE_INDEX_RED
          : PALETTE_INDEX_EMPTY,
      );
      const mesh = meshVoxelModel(model);
      const sharedPlaneQuads = mesh.quads.filter(
        (quad) =>
          quad.normalAxis === 0 && quad.position[0] === DOMINO_SHARED_PLANE,
      );
      expect(sharedPlaneQuads).toEqual([
        {
          position: [1, 0, 0],
          normalAxis: 0,
          width: 1,
          height: 1,
          paletteIndex: PALETTE_INDEX_RED,
        },
        {
          position: [1, 1, 0],
          normalAxis: 0,
          width: 1,
          height: 1,
          paletteIndex: PALETTE_INDEX_RED,
        },
      ]);
    });
  });

  describe('glass semantics (spec 7.2)', () => {
    it('emits no body quads for fully transparent glow voxels', () => {
      const model = createDecodedModel([1, 1, 1], () => PALETTE_INDEX_GLASS);
      const mesh = meshVoxelModel(model);
      expect(mesh.quads).toHaveLength(0);
    });

    it('keeps body faces exposed against glass neighbors', () => {
      const model = createDecodedModel([2, 1, 1], (x) =>
        x === 0 ? PALETTE_INDEX_RED : PALETTE_INDEX_GLASS,
      );
      const mesh = meshVoxelModel(model);
      expect(mesh.quads.length).toBe(GLASS_NEIGHBOR_BODY_QUAD_COUNT);
      expect(
        mesh.quads.every((quad) => quad.paletteIndex === PALETTE_INDEX_RED),
      ).toBe(true);
      const faceAgainstGlass = mesh.quads.filter(
        (quad) =>
          quad.normalAxis === 0 && quad.position[0] === DOMINO_SHARED_PLANE,
      );
      expect(faceAgainstGlass).toHaveLength(1);
    });
  });

  describe('per-voxel color/emissive lookup (LD-23)', () => {
    it('forces emissive to zero for black palette entries regardless of glow', () => {
      const shade = resolveVoxelShade(
        FIXTURE_PALETTE[PALETTE_INDEX_BLACK_GLOW],
      );
      expect(shade).toEqual({ color: [0, 0, 0], alpha: 255, emissive: 0 });
    });

    it('resolves color, alpha, and glow verbatim for non-black entries', () => {
      const accent = resolveVoxelShade(FIXTURE_PALETTE[PALETTE_INDEX_ACCENT]);
      expect(accent).toEqual({ color: [0, 0, 255], alpha: 255, emissive: 40 });
      const glass = resolveVoxelShade(FIXTURE_PALETTE[PALETTE_INDEX_GLASS]);
      expect(glass).toEqual({
        color: [200, 230, 255],
        alpha: 0,
        emissive: 180,
      });
    });
  });
});
