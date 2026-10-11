/**
 * LOD coarse-tier builder for the Neat Racing voxel pipeline (spec §7.3.3;
 * slice P1-S3-D, AC-011, green-only TDD).
 *
 * Derives the coarse LOD tier from a decoded voxel model by merging
 * 2x2x2 fine-grid blocks into single coarse cells ({@link buildLodModel}).
 * The normative car case is the 128-cubed grid (`lod128` per
 * lod.distance.utils.ts) reduced to the 64-cubed tier (`lod64`); the same
 * builder serves the 32-cubed wheel grids (→ 16-cubed) and future track
 * chunks, per spec §7.3.4 ("the LOD builder machinery — derived palettes,
 * merge rules, §7.3.3 — is scoped to serve those as well").
 *
 * Tier reconciliation: the tracker's "lod32/lod8" AC shorthand is retired
 * drift. Spec §7.3.3 (LD-19, locked; G2 — closed edge cases) is normative
 * and wins: the only derived tier is the 64-cubed grid built by merging
 * 2x2x2 blocks from the 128-cubed grid. A 32-cubed tier is default NOT
 * built (LD-19 gate: non-goal until a recorded quad-count estimate
 * justifies it); this module implements no lod32/lod8 tiers.
 *
 * The builder implements every §7.3.3 rule:
 *
 * 1-2. Contributing cells are palette indices ≠ EMPTY; empties are ignored
 *      entirely; an all-empty block writes EMPTY (no geometry).
 * 3.   Black cells (r === g === b === 0, LD-23) are solid geometry: they
 *      pass through verbatim under the single-index rule and are excluded
 *      ONLY from the color and glow averages, never from alpha.
 * 4.   A block whose contributing cells share one palette index references
 *      that source index verbatim (color, alpha, and glow unchanged; glow
 *      passes through on disk — render-time shine suppression stays in the
 *      mesher's LD-23 shade lookup, greedy-mesher.ts).
 * 5.   Multi-index color bytes average over non-black contributing cells,
 *      rounding integer half-up, and quantize onto the 5-bit grid.
 * 6.   Glow averages over the same non-black contributors (LD-23: black
 *      glow never leaks into derived shine).
 * 7.   All averages round half-up, and a derived (0, 0, 0) color forces
 *      glow to 0 (the black exception at derive time), covering black+black
 *      mixes and legitimate dark mixes.
 * 8.   The derived alpha snaps to 0 (glass-glow entry: averaged color and
 *      glow kept, alpha 0) below the threshold, else 255 (opaque).
 * 9.   Identical derived tuples dedup (first = lowest index wins) with
 *      color channels quantized BEFORE dedup, under a hard 256-entry cap
 *      whose overflow merges to the nearest entry by squared RGB distance
 *      (tie-break lowest index; EMPTY index 0 is never a merge target).
 *
 * Output is a plain {@link DecodedVoxelModel} — derived palette plus new
 * slice grids — so the existing mesher and bin codec consume it unchanged.
 * `symmetric` is always false in the output: mirror-X is already
 * materialized in decoded grids, so the derived grid carries full width.
 *
 * @module
 */

import {
  VOXEL_EMPTY_PALETTE_INDEX,
  VOXEL_PALETTE_COMPONENT_MAX,
  VoxelModelError,
  type VoxelPaletteEntry,
} from './model.schema';
import type { DecodedVoxelModel } from './model.decode.utils';

/**
 * Merge-block edge per axis for the LOD tier (spec §7.3.3 rule 1; LD-19
 * locked). The coarse tier merges exactly 2x2x2 fine-grid blocks into one
 * output cell, so output dims are input dims / 2 on every axis.
 */
export const VOXEL_LOD_MERGE_BLOCK_SIZE = 2;

/**
 * Quantization levels per color channel of the derived palette (spec
 * §7.3.3 rule 9; G2/P4-F6): color channels are quantized onto the 5-bit
 * grid before dedup so near-identical mixes collapse to one entry.
 */
export const VOXEL_LOD_QUANTIZATION_LEVELS = 32;

/**
 * Highest 5-bit quantization level (= {@link VOXEL_LOD_QUANTIZATION_LEVELS}
 * - 1; kept derived so the pair stays in sync).
 */
export const VOXEL_LOD_QUANTIZATION_MAX_LEVEL =
  VOXEL_LOD_QUANTIZATION_LEVELS - 1;

/**
 * Hard cap on derived palette entries (spec §7.3.3 rule 9): palette indices
 * live in the Uint8 index space 0-255 (index 0 is the EMPTY sentinel), so
 * a derived tuple arriving at the cap merges to the nearest existing entry
 * instead of appending.
 */
export const VOXEL_LOD_PALETTE_ENTRY_CAP = 256;

/**
 * Derived-alpha snap point (spec §7.3.3 rule 8): an averaged alpha below
 * this threshold snaps the entry to glass (alpha 0, color and glow kept);
 * at or above it the entry snaps to opaque (alpha 255).
 */
export const VOXEL_LOD_ALPHA_SNAP_THRESHOLD = 128;

/** Derived alpha for the glass-glow entry (averaged color and glow kept). */
export const VOXEL_LOD_ALPHA_GLASS = 0;

/** Derived alpha for the opaque entry. */
export const VOXEL_LOD_ALPHA_OPAQUE = 255;

/**
 * Glow forced onto derived entries whose quantized color is (0, 0, 0)
 * (spec §7.3.3 rule 7, the black exception at derive time, LD-23).
 */
export const VOXEL_LOD_DERIVED_BLACK_GLOW = 0;

/** Lowest palette index eligible as a merge target (index 0 is EMPTY). */
const LOD_MERGE_TARGET_INDEX_MIN = 1;

/**
 * The eight source-cell offsets of one 2x2x2 merge block in deterministic
 * order: cell j of a block sits at [j & 1, (j >> 1) & 1, (j >> 2) & 1]
 * within the block (x fastest, then y, then z).
 */
const LOD_BLOCK_CELL_OFFSETS: readonly (readonly [number, number, number])[] = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
];

/** Working palette plus dedup state for one {@link buildLodModel} call. */
interface LodPaletteState {
  /** Derived palette; starts as a copy of the source palette (append-only). */
  readonly palette: VoxelPaletteEntry[];
  /** Dedup cache keyed by the full `r,g,b,a,glow` tuple of an entry. */
  readonly dedupByKey: Map<string, number>;
}

/**
 * Named rejection when any input dim is not divisible by
 * {@link VOXEL_LOD_MERGE_BLOCK_SIZE} (spec §7.3.3 rule 1: only even-dim
 * grids split into whole 2x2x2 blocks).
 *
 * @example
 * ```ts
 * try {
 *   buildLodModel(decodedModel); // dims [3, 4, 4]
 * } catch (error) {
 *   // VoxelLodDimsError with code 'voxel-lod/dims'
 * }
 * ```
 */
export class VoxelLodDimsError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-lod/dims', message);
  }
}

/**
 * Derives the coarse LOD tier from a decoded voxel model (spec §7.3.3).
 * Deterministic: identical inputs always produce identical outputs, so
 * build outputs are byte-stable for the bin codec.
 *
 * The normative car case reduces the 128-cubed grid to the 64-cubed tier;
 * any even-dim grid is accepted, so the same builder also derives the
 * 16-cubed tier from 32-cubed wheel grids and serves track chunks.
 *
 * @param model - Decoded source model (palette indices and slice grids
 *   trusted as already validated by the decoder).
 * @returns The derived coarse model as a plain {@link DecodedVoxelModel}:
 *   dims halved per axis, `symmetric` false (mirror-X already materialized),
 *   a derived palette (source palette copied verbatim at indices 0..n, then
 *   appended or merged per rule 9), and new coarse slice grids so the
 *   existing mesher and bin codec consume it unchanged.
 * @throws {@link VoxelLodDimsError} when any input dim is not divisible by
 *   {@link VOXEL_LOD_MERGE_BLOCK_SIZE}.
 * @example
 * ```ts
 * import { decodeVoxelModel } from './model.decode.utils';
 * const lod = buildLodModel(decodeVoxelModel(carDocument)); // 128³ → 64³
 * const quads = meshVoxelModel(lod).quads; // consumes DecodedVoxelModel as-is
 * ```
 */
export function buildLodModel(model: DecodedVoxelModel): DecodedVoxelModel {
  // Step 1: validate every dim even (rule 1) and derive the coarse geometry.
  for (const dim of model.dims) {
    if (dim % VOXEL_LOD_MERGE_BLOCK_SIZE !== 0) {
      throw new VoxelLodDimsError(
        `LOD merge requires every dim divisible by ${VOXEL_LOD_MERGE_BLOCK_SIZE} (spec §7.3.3); got dims [${model.dims.join(', ')}]`,
      );
    }
  }
  const coarseWidth = model.dims[0] / VOXEL_LOD_MERGE_BLOCK_SIZE;
  const coarseHeight = model.dims[1] / VOXEL_LOD_MERGE_BLOCK_SIZE;
  const coarseDepth = model.dims[2] / VOXEL_LOD_MERGE_BLOCK_SIZE;

  // Step 2: seed palette state. The derived palette starts as a COPY of the
  // source palette (rule 4: single-index blocks reference their source index
  // verbatim with zero re-indexing; index 0 stays the EMPTY sentinel), and
  // the dedup cache is seeded from every source entry (first = lowest index
  // wins).
  const state: LodPaletteState = {
    palette: model.palette.slice(),
    dedupByKey: new Map<string, number>(),
  };
  model.palette.forEach((entry, index) => {
    const key = entry.join(',');
    if (!state.dedupByKey.has(key)) {
      state.dedupByKey.set(key, index);
    }
  });

  // Step 3: scan coarse cells in deterministic order (bz, by, bx ascending);
  // block (bx, by, bz) covers source cells [2bx..2bx+1, 2by..2by+1, 2bz..2bz+1].
  const coarseSlices: Uint8Array[] = [];
  for (let coarseZ = 0; coarseZ < coarseDepth; coarseZ++) {
    const coarseSlice = new Uint8Array(coarseWidth * coarseHeight);
    for (let coarseY = 0; coarseY < coarseHeight; coarseY++) {
      for (let coarseX = 0; coarseX < coarseWidth; coarseX++) {
        // Step 5: write the resolved index at `x + y * coarseWidth`,
        // slice z = bz (row-major, mirroring the decoder's grid layout).
        coarseSlice[coarseX + coarseY * coarseWidth] = resolveLodBlockIndex(
          model,
          state,
          coarseX,
          coarseY,
          coarseZ,
        );
      }
    }
    coarseSlices.push(coarseSlice);
  }

  return {
    name: model.name,
    dims: [coarseWidth, coarseHeight, coarseDepth],
    symmetric: false,
    palette: state.palette,
    slices: coarseSlices,
  };
}

/**
 * Resolves one coarse cell's palette index from its 2x2x2 source block in
 * two allocation-free passes over the eight cells: pass A classifies the
 * block (all-empty → EMPTY; single index → verbatim), pass B derives the
 * mixed entry (rules 3, 5-9).
 *
 * @param model - Decoded source model.
 * @param state - Palette + dedup state for this build call.
 * @param coarseX - Coarse cell x (block origin 2x).
 * @param coarseY - Coarse cell y (block origin 2y).
 * @param coarseZ - Coarse cell z (block origin 2z).
 * @returns The resolved palette index for the coarse cell.
 */
function resolveLodBlockIndex(
  model: DecodedVoxelModel,
  state: LodPaletteState,
  coarseX: number,
  coarseY: number,
  coarseZ: number,
): number {
  const originX = coarseX * VOXEL_LOD_MERGE_BLOCK_SIZE;
  const originY = coarseY * VOXEL_LOD_MERGE_BLOCK_SIZE;
  const originZ = coarseZ * VOXEL_LOD_MERGE_BLOCK_SIZE;

  // Pass A: classify the block over its contributing cells (rules 1-2:
  // empties are ignored entirely).
  let contributorCount = 0;
  let firstContributorIndex = 0;
  let allSameIndex = true;
  for (const [offsetX, offsetY, offsetZ] of LOD_BLOCK_CELL_OFFSETS) {
    const index = readModelCell(
      model,
      originX + offsetX,
      originY + offsetY,
      originZ + offsetZ,
    );
    if (index === VOXEL_EMPTY_PALETTE_INDEX) {
      continue;
    }
    contributorCount += 1;
    if (contributorCount === 1) {
      firstContributorIndex = index;
    } else if (index !== firstContributorIndex) {
      allSameIndex = false;
    }
  }
  // Rule 2a: an all-empty block writes EMPTY (no geometry).
  if (contributorCount === 0) {
    return VOXEL_EMPTY_PALETTE_INDEX;
  }
  // Rule 4: a single-index block references its source entry verbatim
  // (black included — the entry passes through unchanged, glow included on
  // disk; render-time shine suppression stays in the mesher's LD-23 shade
  // lookup). Single-index blocks never consult the dedup cache or the cap.
  if (allSameIndex) {
    return firstContributorIndex;
  }

  // Pass B: accumulate the rule sums. Alpha sums over ALL contributing
  // cells including black (rule 3: black is excluded ONLY from color and
  // glow averages); color and glow sums over non-black contributors only
  // (black detection by color bytes, LD-23, regardless of index).
  let alphaSum = 0;
  let redSum = 0;
  let greenSum = 0;
  let blueSum = 0;
  let glowSum = 0;
  let nonBlackCount = 0;
  for (const [offsetX, offsetY, offsetZ] of LOD_BLOCK_CELL_OFFSETS) {
    const index = readModelCell(
      model,
      originX + offsetX,
      originY + offsetY,
      originZ + offsetZ,
    );
    if (index === VOXEL_EMPTY_PALETTE_INDEX) {
      continue;
    }
    const entry = model.palette[index];
    alphaSum += entry[3];
    if (isBlackEntry(entry)) {
      continue;
    }
    nonBlackCount += 1;
    redSum += entry[0];
    greenSum += entry[1];
    blueSum += entry[2];
    glowSum += entry[4];
  }

  // Rules 5/6: half-up averages over non-black contributors; the explicit
  // nonBlackCount === 0 branch keeps all-black multi-index blocks NaN-free
  // (color (0, 0, 0), glow 0).
  const hasNonBlackContributors = nonBlackCount > 0;
  const derivedRed = hasNonBlackContributors
    ? quantizeChannel(roundHalfUp(redSum, nonBlackCount))
    : 0;
  const derivedGreen = hasNonBlackContributors
    ? quantizeChannel(roundHalfUp(greenSum, nonBlackCount))
    : 0;
  const derivedBlue = hasNonBlackContributors
    ? quantizeChannel(roundHalfUp(blueSum, nonBlackCount))
    : 0;
  const rawDerivedGlow = hasNonBlackContributors
    ? roundHalfUp(glowSum, nonBlackCount)
    : VOXEL_LOD_DERIVED_BLACK_GLOW;

  // Rule 8: derived alpha snaps to glass below the threshold and opaque at
  // or above it; averaged color and glow are kept either way.
  const averagedAlpha = roundHalfUp(alphaSum, contributorCount);
  const derivedAlpha =
    averagedAlpha < VOXEL_LOD_ALPHA_SNAP_THRESHOLD
      ? VOXEL_LOD_ALPHA_GLASS
      : VOXEL_LOD_ALPHA_OPAQUE;

  // Rule 7: the black exception at derive time — a derived (0, 0, 0) color
  // forces glow to 0, covering both black+black mixes and legitimate dark
  // mixes whose averaged color rounds to black.
  const derivedGlow =
    derivedRed === 0 && derivedGreen === 0 && derivedBlue === 0
      ? VOXEL_LOD_DERIVED_BLACK_GLOW
      : rawDerivedGlow;

  // Rule 9: dedup identical derived tuples; new tuples append below the
  // cap, overflow merges to the nearest existing entry.
  return resolveDerivedPaletteIndex(state, [
    derivedRed,
    derivedGreen,
    derivedBlue,
    derivedAlpha,
    derivedGlow,
  ]);
}

/**
 * Resolves a derived entry's palette index (rule 9): an identical tuple
 * reuses its cached index (first = lowest index wins); a new tuple appends
 * while the palette is under the cap; at the cap it merges to the nearest
 * existing entry and the merged index is cached so repeat tuples stay
 * stable.
 *
 * @param state - Palette + dedup state for this build call.
 * @param derived - Fully derived `[r, g, b, a, glow]` entry (quantized).
 * @returns The resolved palette index for the derived entry.
 */
function resolveDerivedPaletteIndex(
  state: LodPaletteState,
  derived: VoxelPaletteEntry,
): number {
  const key = derived.join(',');
  const cachedIndex = state.dedupByKey.get(key);
  if (cachedIndex !== undefined) {
    return cachedIndex;
  }
  if (state.palette.length < VOXEL_LOD_PALETTE_ENTRY_CAP) {
    const appendedIndex = state.palette.length;
    state.palette.push(derived);
    state.dedupByKey.set(key, appendedIndex);
    return appendedIndex;
  }
  const mergedIndex = findNearestPaletteIndex(
    state.palette,
    derived[0],
    derived[1],
    derived[2],
  );
  state.dedupByKey.set(key, mergedIndex);
  return mergedIndex;
}

/**
 * Finds the palette entry with the minimum squared RGB distance to a
 * derived color, the lowest index breaking ties (deterministic). The scan
 * starts at {@link LOD_MERGE_TARGET_INDEX_MIN} because index 0 is the EMPTY
 * sentinel and merging solid geometry into air would violate rule 3; the
 * merge path is only reachable at the {@link VOXEL_LOD_PALETTE_ENTRY_CAP}
 * cap, so an eligible index always exists.
 *
 * @param palette - Current derived palette (at the cap).
 * @param red - Derived red byte.
 * @param green - Derived green byte.
 * @param blue - Derived blue byte.
 * @returns Index of the nearest existing palette entry (never 0).
 */
function findNearestPaletteIndex(
  palette: readonly VoxelPaletteEntry[],
  red: number,
  green: number,
  blue: number,
): number {
  let nearestIndex = LOD_MERGE_TARGET_INDEX_MIN;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (
    let index = LOD_MERGE_TARGET_INDEX_MIN;
    index < palette.length;
    index++
  ) {
    const entry = palette[index];
    const deltaRed = entry[0] - red;
    const deltaGreen = entry[1] - green;
    const deltaBlue = entry[2] - blue;
    const distance =
      deltaRed * deltaRed + deltaGreen * deltaGreen + deltaBlue * deltaBlue;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  }
  return nearestIndex;
}

/**
 * Rounds sum/count to the nearest integer with .5 ties rounding up
 * (spec §7.3.3: all averages round half-up). Pure integer arithmetic over
 * non-negative byte-component sums, so there are no float edge cases.
 *
 * @param sum - Non-negative integer sum of the averaged component.
 * @param count - Positive contributor count (callers guard count 0).
 * @returns The half-up rounded integer average.
 */
function roundHalfUp(sum: number, count: number): number {
  return Math.floor((2 * sum + count) / (2 * count));
}

/**
 * Quantizes one averaged color byte onto the 5-bit grid (rule 9; G2/P4-F6),
 * mirroring the spec example `Math.round(v / 255 * 31) / 31 * 255` with the
 * final byte rounded to its grid point. `Math.round` never hits a .5 tie on
 * the level step for an integer byte value. Glow and alpha are NOT
 * quantized (the spec quantizes color channels only).
 *
 * @param value - Averaged color byte in 0-255.
 * @returns The quantized byte on the 5-bit grid.
 */
function quantizeChannel(value: number): number {
  const level = Math.round(
    (value / VOXEL_PALETTE_COMPONENT_MAX) * VOXEL_LOD_QUANTIZATION_MAX_LEVEL,
  );
  return Math.round(
    (level / VOXEL_LOD_QUANTIZATION_MAX_LEVEL) * VOXEL_PALETTE_COMPONENT_MAX,
  );
}

/**
 * Detects a black palette entry by its color bytes (LD-23), regardless of
 * its index, alpha, or glow.
 *
 * @param entry - Palette entry as [r, g, b, a, glow].
 * @returns True when the entry's color bytes are all zero.
 */
function isBlackEntry(entry: VoxelPaletteEntry): boolean {
  return entry[0] === 0 && entry[1] === 0 && entry[2] === 0;
}

/**
 * Reads the palette index at source lattice coordinates (row-major
 * `x + y * dims[0]` per depth slice), mirroring the decoder's grid layout.
 *
 * @param model - Decoded source model.
 * @param x - Lattice x in 0..dims[0] - 1.
 * @param y - Lattice y in 0..dims[1] - 1.
 * @param z - Lattice z in 0..dims[2] - 1.
 * @returns Palette index at the cell.
 */
function readModelCell(
  model: DecodedVoxelModel,
  x: number,
  y: number,
  z: number,
): number {
  return model.slices[z][x + y * model.dims[0]];
}
