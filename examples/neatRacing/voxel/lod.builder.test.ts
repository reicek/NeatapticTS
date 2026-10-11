/**
 * Contract tests for the Neat Racing LOD coarse-tier builder (slice P1-S3-D;
 * AC-011, spec §7.3.3, green-only TDD).
 *
 * These tests pin every normative §7.3.3 rule on small fixture models, one
 * single-expectation test per rule: the 128³ → 64³ tier shape, empty-block
 * and solids-decide semantics (rules 1-2), verbatim single-index
 * pass-through including the LD-23 black glow on disk (rule 4), multi-index
 * color averaging with half-up rounding and 5-bit quantization (rule 5),
 * glow averaged over non-black contributors only (rule 6), the derived-black
 * glow exception (rule 7), the derived-alpha glass snap (rule 8), dedup of
 * identical derived tuples (rule 9a), the named odd-dims rejection, and the
 * >256-entry palette cap with merge-to-nearest fallback (rule 9b).
 *
 * Fixtures are built as {@link DecodedVoxelModel} literals via
 * {@link createDecodedModel} — bypassing the RLE decoder on purpose so this
 * suite isolates the builder contract (the decoder pipeline is unit-tested
 * by model.decode.utils.test.ts).
 *
 * Tracker-shorthand note: the tracker's "lod32/lod8" AC wording is retired
 * drift (spec §7.3.3 is normative: the only derived tier is the 64-cubed
 * grid from 2x2x2 merges; a 32-cubed tier is default NOT built per the
 * LD-19 gate), so no lod32/lod8 tests exist by design.
 */
import { describe, expect, it } from '@jest/globals';
import type { DecodedVoxelModel } from './model.decode.utils';
import {
  VOXEL_PALETTE_COMPONENT_MAX,
  type VoxelPaletteEntry,
} from './model.schema';
import {
  VOXEL_LOD_PALETTE_ENTRY_CAP,
  VoxelLodDimsError,
  buildLodModel,
} from './lod.builder';

/** Palette index of the all-zero EMPTY sentinel entry. */
const PALETTE_INDEX_EMPTY = 0;

/**
 * Builds a decoded model fixture by painting each lattice cell through
 * `paintVoxel`. Bypasses `decodeVoxelModel` on purpose: the decoder pipeline
 * is unit-tested by model.decode.utils.test.ts and this suite isolates the
 * LOD builder contract.
 *
 * @param dims - Source grid dimensions as [width, height, depth].
 * @param palette - Source palette; index 0 must be the EMPTY sentinel.
 * @param paintVoxel - Maps a lattice cell (x, y, z) to its palette index.
 * @returns A decoded model over the given palette with the painted cells.
 */
function createDecodedModel(
  dims: readonly [number, number, number],
  palette: readonly VoxelPaletteEntry[],
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
    name: 'lod-fixture',
    dims: [dims[0], dims[1], dims[2]],
    symmetric: false,
    palette,
    slices,
  };
}

describe('buildLodModel (spec §7.3.3, AC-011)', () => {
  it('derives the coarse lod64 tier shape from the normative lod128 car grid', () => {
    const model = createDecodedModel(
      [128, 128, 128],
      [[0, 0, 0, 0, 0]],
      () => PALETTE_INDEX_EMPTY,
    );
    const result = buildLodModel(model);
    expect(result.dims).toEqual([64, 64, 64]);
  });

  it('rule 2a: an all-empty block writes the EMPTY sentinel (no geometry)', () => {
    const model = createDecodedModel(
      [2, 2, 2],
      [[0, 0, 0, 0, 0]],
      () => PALETTE_INDEX_EMPTY,
    );
    const result = buildLodModel(model);
    expect(result.slices[0][0]).toBe(0);
  });

  it('rule 2b: solids decide a partially empty block (empties are ignored)', () => {
    // Palette: index 0 EMPTY, index 1 unused red, index 2 opaque blue.
    const blueIndex = 2;
    const model = createDecodedModel(
      [2, 2, 2],
      [
        [0, 0, 0, 0, 0],
        [255, 0, 0, 255, 0],
        [0, 0, 255, 255, 0],
      ],
      // Cells j = z*4 + y*2 + x with j < 5 hold blue; j 5..7 stay empty, so
      // the 3 empty cells are ignored entirely and blue decides the block.
      (x, y, z) => (z * 4 + y * 2 + x < 5 ? blueIndex : PALETTE_INDEX_EMPTY),
    );
    const result = buildLodModel(model);
    expect(result.slices[0][0]).toBe(blueIndex);
  });

  it('rule 4: a single-index block passes its palette entry through verbatim (black glow kept on disk)', () => {
    // Palette: index 1 is a legal black body entry with glow 200 (LD-23).
    const blackGlowIndex = 1;
    const model = createDecodedModel(
      [2, 2, 2],
      [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 255, 200],
      ],
      () => blackGlowIndex,
    );
    const result = buildLodModel(model);
    expect(result.palette[result.slices[0][0]]).toEqual([0, 0, 0, 255, 200]);
  });

  it('rule 5: a multi-index block averages non-black colors half-up onto the 5-bit grid', () => {
    const redIndex = 1;
    const blueIndex = 2;
    const model = createDecodedModel(
      [2, 2, 2],
      [
        [0, 0, 0, 0, 0],
        [255, 0, 0, 255, 0], // index 1: red
        [0, 0, 255, 255, 0], // index 2: blue
      ],
      (_x, _y, z) => (z === 0 ? redIndex : blueIndex),
    );
    const result = buildLodModel(model);
    // r: 255*4/8 = 127.5 → half-up 128 → level round(128*31/255) = 16 →
    // byte round(16*255/31) = 132; b symmetric → 132; alpha 255; glow 0.
    expect(result.palette[result.slices[0][0]]).toEqual([132, 0, 132, 255, 0]);
  });

  it('rule 6: glow averages over non-black contributors only (LD-23)', () => {
    const blackGlowIndex = 1;
    const whiteIndex = 2;
    const model = createDecodedModel(
      [2, 2, 2],
      [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255], // index 1: black, glow 255 (excluded from averages)
        [255, 255, 255, 255, 15], // index 2: white, glow 15
      ],
      (_x, _y, z) => (z === 0 ? blackGlowIndex : whiteIndex),
    );
    const result = buildLodModel(model);
    // Color 255 over the 4 non-black cells; glow 15 (135 if the black glow
    // wrongly leaked in); alpha 255 over all 8 contributors.
    expect(result.palette[result.slices[0][0]]).toEqual([
      255, 255, 255, 255, 15,
    ]);
  });

  it('rule 7: a derived (0, 0, 0) color forces glow to 0 (black exception)', () => {
    const blackGlowHighIndex = 1;
    const blackGlowLowIndex = 2;
    const model = createDecodedModel(
      [2, 2, 2],
      [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 255, 100], // index 1: black, glow 100
        [0, 0, 0, 255, 50], // index 2: black, glow 50
      ],
      (_x, _y, z) => (z === 0 ? blackGlowHighIndex : blackGlowLowIndex),
    );
    const result = buildLodModel(model);
    // All-black multi-index block: no non-black contributors → color
    // (0, 0, 0), and the derived black exception forces glow to 0.
    expect(result.palette[result.slices[0][0]]).toEqual([0, 0, 0, 255, 0]);
  });

  it('rule 8: derived alpha below the threshold snaps to the glass-glow entry', () => {
    const redGlassIndex = 1;
    const blueGlassIndex = 2;
    const model = createDecodedModel(
      [2, 2, 2],
      [
        [0, 0, 0, 0, 0],
        [255, 0, 0, 0, 50], // index 1: red glass, glow 50
        [0, 0, 100, 100, 10], // index 2: translucent blue, glow 10
      ],
      (_x, _y, z) => (z === 0 ? redGlassIndex : blueGlassIndex),
    );
    const result = buildLodModel(model);
    // Alpha (0*4 + 100*4)/8 = 50 < 128 → 0; r 127.5 → 128 → 132;
    // b 50 → level round(50*31/255) = 6 → byte 49; glow (50*4+10*4)/8 = 30:
    // a glass-glow entry (averaged color kept, alpha 0, glow kept).
    expect(result.palette[result.slices[0][0]]).toEqual([132, 0, 49, 0, 30]);
  });

  it('rule 9a: identical derived tuples dedup to one appended palette entry', () => {
    const redIndex = 1;
    const blueIndex = 2;
    const model = createDecodedModel(
      [4, 4, 4],
      [
        [0, 0, 0, 0, 0],
        [255, 0, 0, 255, 0], // index 1: red
        [0, 0, 255, 255, 0], // index 2: blue
      ],
      // Only blocks (0,0,0) and (1,0,0) are painted, each 4 red + 4 blue;
      // both derive the identical [132, 0, 132, 255, 0] tuple.
      (x, y, z) => {
        const block = (x >> 1) + (y >> 1) * 2 + (z >> 1) * 4;
        if (block > 1) {
          return PALETTE_INDEX_EMPTY;
        }
        return (z & 1) === 0 ? redIndex : blueIndex;
      },
    );
    const result = buildLodModel(model);
    // 3 source entries + exactly one derived entry shared by both blocks.
    expect(result.palette).toHaveLength(4);
  });

  it('rejects odd input dims with VoxelLodDimsError', () => {
    const model = createDecodedModel(
      [3, 4, 4],
      [[0, 0, 0, 0, 0]],
      () => PALETTE_INDEX_EMPTY,
    );
    expect(() => buildLodModel(model)).toThrow(VoxelLodDimsError);
  });

  it('rule 9b: the palette caps at 256 entries and the overflow block merges to the nearest entry', () => {
    const model = createDecodedModel(
      [20, 20, 20],
      buildCapFixturePalette(),
      paintCapFixtureVoxel,
    );
    const result = buildLodModel(model);
    // (a) The derived palette saturates at the hard 256-entry cap.
    expect(result.palette).toHaveLength(VOXEL_LOD_PALETTE_ENTRY_CAP);
    // (b) The deliberate overflow block (k = 999, output cell (9, 9, 9))
    // derives (8, 8, 0, 255, 0) — absent from the dedup cache by the
    // green-channel argument in the fixture docs — so the cap-full palette
    // merges it to the nearest entry: the index-9 seed (RGB distance 0).
    expect(result.slices[9][CAP_OVERFLOW_OUTPUT_CELL]).toBe(
      CAP_MERGE_TARGET_INDEX,
    );
  });
});

/** Output width/height/depth of the cap fixture (20³ / 2 = 10³). */
const CAP_FIXTURE_OUT_DIMS = 10;

/** Total block ordinals in the cap fixture (10³ = 1_000 blocks). */
const CAP_FIXTURE_BLOCK_COUNT = 1_000;

/** Ordinal of the deliberate overflow block: the last processed block. */
const CAP_OVERFLOW_BLOCK = CAP_FIXTURE_BLOCK_COUNT - 1;

/** Palette index of the merge-to-nearest seed entry (RGB 8,8,0, glow 200). */
const CAP_MERGE_TARGET_INDEX = 9;

/** Palette index of the overflow block's high-red seed entry. */
const CAP_OVERFLOW_INDEX_RED = 10;

/** Palette index of the overflow block's high-green seed entry. */
const CAP_OVERFLOW_INDEX_GREEN = 11;

/**
 * Output cell of the deliberate overflow block in slice 9: block 999 =
 * (9, 9, 9) writes output cell (x=9, y=9) → `9 + 9 * 10 = 99`.
 */
const CAP_OVERFLOW_OUTPUT_CELL = 9 + 9 * CAP_FIXTURE_OUT_DIMS;

/**
 * Deterministic source palette for the >256 cap fixture (rule 9b).
 *
 * Entries 1-9 follow the plan's seed recipe `[(k-1)*28, 255-(k-1)*28,
 * ((k-1)*13) % 256, 255, (k-1)*5]` with one documented deviation: the blue
 * channel is quadratic (`7s² + 13s` for s = k-1). The plan's linear blue
 * keeps every entry on a single line through (r, g, b, glow) space (r and
 * g are exact opposites: r + g = 255 per entry), so every 8-cell mix
 * collapses to a function of the single weighted-sum parameter
 * M = Σ(k-1)·countₖ plus the index-9 count — exhaustive enumeration of all
 * 12_870 eight-cell multisets proves that construction yields only 230
 * distinct derived tuples, under the 244 appends needed to reach the
 * 256-entry cap. The quadratic blue adds a second independent parameter
 * (Σ(k-1)²·countₖ), lifting the same enumeration to 1_637 distinct tuples.
 * Index 9 is overridden to the merge-target seed [8, 8, 0, 255, 200];
 * indices 10/11 are reserved for the deliberate overflow block and are
 * never emitted by the block generator.
 *
 * @returns The 12-entry source palette (EMPTY + 9 seeds + 2 overflow seeds).
 */
function buildCapFixturePalette(): VoxelPaletteEntry[] {
  const palette: VoxelPaletteEntry[] = [[0, 0, 0, 0, 0]];
  for (let seed = 1; seed <= 9; seed++) {
    const s = seed - 1;
    palette.push([
      s * 28,
      VOXEL_PALETTE_COMPONENT_MAX - s * 28,
      (s * s * 7 + s * 13) % VOXEL_PALETTE_COMPONENT_MAX,
      VOXEL_PALETTE_COMPONENT_MAX,
      s * 5,
    ]);
  }
  palette[CAP_MERGE_TARGET_INDEX] = [8, 8, 0, VOXEL_PALETTE_COMPONENT_MAX, 200];
  palette.push([16, 0, 0, VOXEL_PALETTE_COMPONENT_MAX, 0]);
  palette.push([0, 16, 0, VOXEL_PALETTE_COMPONENT_MAX, 0]);
  return palette;
}

/**
 * Maps (block, cell) to a source palette index 1..9 for the cap fixture.
 * Affine generators (`(k + j) % 9`, any fixed multipliers) collapse: an
 * arithmetic progression mod 9 yields at most 9 distinct multisets, and the
 * first non-affine candidates measured 18-36 distinct multisets — far under
 * the 244 appends needed. This generator adds two k-dependent cross terms
 * whose periods in k are coprime-ish — `trunc(k²·j/13)` has period
 * 9·13 = 117 and `trunc(k·j²/29)` has period 9·29 = 261 — so the multiset
 * pattern repeats only after 9·13·29 = 3_393 blocks, well past the 999
 * derived blocks. Deterministic simulation over k = 0..998 measured:
 * 824 distinct block multisets, every block multi-index, the derived
 * palette reaching the 256-entry cap at block k = 349, and the overflow
 * tuple (8, 8, 0, 255, 0) never derived — any multi-index block has at
 * least one non-index-9 cell contributing green ≥ 59 with the rest ≥ 8
 * each, so its green sum is ≥ 59 + 7·8 = 115 and the quantized green byte
 * is ≥ 16, never 8.
 *
 * @param block - Block ordinal 0..998.
 * @param cell - Cell ordinal 0..7 within the block.
 * @returns Source palette index in 1..9 (never 0, never 10/11).
 */
function capFixtureBlockIndex(block: number, cell: number): number {
  return (
    ((block * 7 +
      cell * 5 +
      Math.trunc((block * block * cell) / 13) +
      Math.trunc((block * cell * cell) / 29)) %
      9) +
    1
  );
}

/**
 * Paints the 20-cubed cap fixture grid. Block k = bx + by·10 + bz·100
 * (bx = x >> 1 etc.) fills its 8 cells via {@link capFixtureBlockIndex}; the
 * last block (k = 999) is overridden to the deliberate overflow mix — 4
 * cells of index 10 ([16, 0, 0, 255, 0]) + 4 cells of index 11
 * ([0, 16, 0, 255, 0]) — whose derived tuple is r = halfUp(64, 8) = 8 →
 * level 1 → byte 8 (and g = 8, b = 0, glow 0, alpha 255): exactly
 * (8, 8, 0, 255, 0), RGB-identical to the index-9 seed but with glow
 * 0 ≠ 200, so the full tuple misses the dedup cache, the palette is at its
 * cap, and the builder must merge to the nearest entry (index 9, squared
 * RGB distance 0).
 *
 * @param x - Lattice x in 0..19.
 * @param y - Lattice y in 0..19.
 * @param z - Lattice z in 0..19.
 * @returns Source palette index for the cell.
 */
function paintCapFixtureVoxel(x: number, y: number, z: number): number {
  const block =
    (x >> 1) +
    (y >> 1) * CAP_FIXTURE_OUT_DIMS +
    (z >> 1) * CAP_FIXTURE_OUT_DIMS * CAP_FIXTURE_OUT_DIMS;
  const cell = (z & 1) * 4 + (y & 1) * 2 + (x & 1);
  if (block === CAP_OVERFLOW_BLOCK) {
    return cell < 4 ? CAP_OVERFLOW_INDEX_RED : CAP_OVERFLOW_INDEX_GREEN;
  }
  return capFixtureBlockIndex(block, cell);
}
