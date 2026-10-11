/**
 * RLE hex-row and mirror-X decoding for Neat Racing voxel models (spec
 * section 7.2; slice P1-S3-A, AC-007 + AC-008).
 *
 * Row encoding: `<runLen:2hex><paletteIdx:2hex>` pairs; the runs of a row
 * must sum exactly to the declared row width — `dims[0]` cols for full
 * documents, `dims[0] / 2` stored cols when `meta.symmetric` is true.
 * Symmetric rows store the left half and are mirrored col c to col
 * `dims[0] - 1 - c` onto the full width.
 *
 * The decoder emits one row-major `Uint8Array` per slice sized exactly from
 * the schema header (`dims[0] x dims[1]` palette indices); rows keep their
 * storage order, so row 0 stays ground/down (section 7.2). Malformed rows
 * are rejected with the named errors from `model.schema.ts`.
 *
 * @module
 */

import {
  VoxelRowHexError,
  VoxelRowPaletteIndexError,
  VoxelRowSumError,
  validateVoxelModelDocument,
  type VoxelModelDocument,
  type VoxelPaletteEntry,
} from './model.schema';

/** Hex characters per RLE field; both `<runLen>` and `<paletteIdx>` use 2. */
export const VOXEL_RLE_FIELD_HEX_CHARS = 2;

/** Hex characters per full `<runLen><paletteIdx>` pair (2 fields). */
export const VOXEL_RLE_PAIR_HEX_CHARS = VOXEL_RLE_FIELD_HEX_CHARS * 2;

/** Shortest legal RLE run (section 7.2: run lengths are 1-255). */
export const VOXEL_RUN_LENGTH_MIN = 1;

/** Longest legal RLE run (two hex digits cap at 255). */
export const VOXEL_RUN_LENGTH_MAX = 255;

/** Divisor mapping the full width to the stored symmetric half (128 to 64). */
export const VOXEL_SYMMETRIC_HALF_DIVISOR = 2;

/**
 * Decoded voxel model — typed arrays recovered from a validated section 7.2
 * document.
 *
 * @property name - Model name echoed from `meta.name`.
 * @property dims - Grid dims `[width, height, depth]` echoed from `meta.dims`.
 * @property symmetric - Whether the source document stored half-width rows.
 * @property palette - Validated palette entries; index 0 is the EMPTY sentinel.
 * @property slices - One row-major `Uint8Array` per slice; `slices[z]` holds
 *   `dims[0] * dims[1]` palette indices (cell `x + y * dims[0]`), row 0
 *   remains ground/down exactly as stored (section 7.2).
 */
export interface DecodedVoxelModel {
  readonly name: string;
  readonly dims: readonly [number, number, number];
  readonly symmetric: boolean;
  readonly palette: readonly VoxelPaletteEntry[];
  readonly slices: readonly Uint8Array[];
}

/**
 * Decodes an on-disk model document into typed per-slice grids.
 *
 * The document is re-validated first so the decoder is the single normative
 * gate for section 7.2 conformance; then every RLE hex row is expanded into
 * palette indices. Symmetric documents have their stored half-rows mirrored
 * onto the full declared width (col c mirrors to col `dims[0] - 1 - c`,
 * LD-28).
 *
 * @param document - The on-disk model document (validated internally).
 * @returns The decoded model: one row-major `Uint8Array` of
 *   `dims[0] * dims[1]` palette indices per slice, sized exactly from the
 *   schema header.
 * @throws VoxelModelMetadataError when required metadata is missing or malformed.
 * @throws VoxelModelDimsError when dims components are not positive
 *   integers, the width is odd while symmetric, or slice/row counts
 *   contradict the declared dims.
 * @throws VoxelModelPaletteError when palette entries are malformed or index
 *   0 is not the EMPTY sentinel.
 * @throws VoxelRowHexError when a row holds invalid hex or a zero-length run.
 * @throws VoxelRowSumError when a row's runs do not sum to the stored width.
 * @throws VoxelRowPaletteIndexError when a row references a palette index
 *   outside the document's palette.
 *
 * @example
 * ```ts
 * const decoded = decodeVoxelModel(document);
 * const slice = decoded.slices[0];
 * const groundRowCol0 = slice[0]; // row 0, col 0 (ground row, front-left)
 * ```
 */
export function decodeVoxelModel(
  document: VoxelModelDocument,
): DecodedVoxelModel {
  // Step 1: validate the document (single normative section 7.2 gate).
  const validated = validateVoxelModelDocument(document);
  const { name, dims, symmetric } = validated.meta;

  // Step 2: compute the stored row width from the header.
  const storedWidth = symmetric
    ? dims[0] / VOXEL_SYMMETRIC_HALF_DIVISOR
    : dims[0];

  // Step 3: decode every slice into a typed row-major grid.
  const slices = validated.slices.map((rows) =>
    decodeVoxelSlice(
      rows,
      dims[0],
      storedWidth,
      symmetric,
      validated.palette.length,
    ),
  );
  return { name, dims, symmetric, palette: validated.palette, slices };
}

/**
 * Expands one RLE hex row into palette indices.
 *
 * Each `<runLen:2hex><paletteIdx:2hex>` pair fills `runLen` consecutive
 * cells with `paletteIdx`. The runs must sum exactly to `expectedWidth`
 * (64 stored cols for symmetric documents, 128 otherwise).
 *
 * @param row - RLE hex row string from the document.
 * @param expectedWidth - Number of cells the row must cover exactly.
 * @param paletteSize - Palette length; referenced indices must stay below it.
 * @returns A `Uint8Array` of `expectedWidth` palette indices.
 * @throws VoxelRowHexError on invalid hex fields or zero-length runs.
 * @throws VoxelRowPaletteIndexError when an index is outside the palette.
 * @throws VoxelRowSumError when the runs overshoot or undershoot
 *   `expectedWidth`.
 *
 * @example
 * ```ts
 * const cells = decodeVoxelRow('0a034000', 74, 5);
 * // cells[0..9] === 3 (10 voxels of palette index 3), cells[10..73] === 0
 * ```
 */
export function decodeVoxelRow(
  row: string,
  expectedWidth: number,
  paletteSize: number,
): Uint8Array {
  const runs = parseVoxelRuns(row, paletteSize);
  const cells = new Uint8Array(expectedWidth);
  let cursor = 0;
  for (const run of runs) {
    if (cursor + run.runLength > expectedWidth) {
      throw new VoxelRowSumError(
        `RLE runs must sum to the row width ${expectedWidth}; the row overshoots at cell ${cursor} with a run of ${run.runLength}`,
      );
    }
    cells.fill(run.paletteIndex, cursor, cursor + run.runLength);
    cursor += run.runLength;
  }
  if (cursor !== expectedWidth) {
    throw new VoxelRowSumError(
      `RLE runs must sum to the row width ${expectedWidth}, received ${cursor}`,
    );
  }
  return cells;
}

/** One decoded `<runLen><paletteIdx>` pair. */
interface VoxelRun {
  runLength: number;
  paletteIndex: number;
}

/** Exact shape of one 2-hex RLE field (case-insensitive). */
const VOXEL_HEX_FIELD_PATTERN = /^[0-9a-f]{2}$/i;

/**
 * Parses the run pairs of one row, rejecting invalid hex, zero-length runs,
 * and out-of-range palette indices.
 *
 * @param row - RLE hex row string.
 * @param paletteSize - Palette length; referenced indices must stay below it.
 * @returns The decoded run pairs in row order.
 * @throws VoxelRowHexError on invalid hex fields or zero-length runs.
 * @throws VoxelRowPaletteIndexError when an index is outside the palette.
 */
function parseVoxelRuns(row: string, paletteSize: number): VoxelRun[] {
  const runs: VoxelRun[] = [];
  for (
    let offset = 0;
    offset < row.length;
    offset += VOXEL_RLE_PAIR_HEX_CHARS
  ) {
    const runLength = parseHexField(row, offset);
    const paletteIndex = parseHexField(row, offset + VOXEL_RLE_FIELD_HEX_CHARS);
    if (runLength < VOXEL_RUN_LENGTH_MIN) {
      throw new VoxelRowHexError(
        `RLE run lengths must be ${VOXEL_RUN_LENGTH_MIN}-${VOXEL_RUN_LENGTH_MAX}; received a zero-length run at hex offset ${offset}`,
      );
    }
    if (paletteIndex >= paletteSize) {
      throw new VoxelRowPaletteIndexError(
        `palette index ${paletteIndex} is out of range for a ${paletteSize}-entry palette at hex offset ${offset}`,
      );
    }
    runs.push({ runLength, paletteIndex });
  }
  return runs;
}

/**
 * Parses one 2-hex field, rejecting anything that is not exactly two hex
 * digits (a bare `parseInt` would silently misparse values such as `1g`).
 *
 * @param row - RLE hex row string.
 * @param offset - Hex offset of the field within the row.
 * @returns The parsed field value.
 * @throws VoxelRowHexError when the field is not exactly 2 hex digits.
 */
function parseHexField(row: string, offset: number): number {
  const field = row.slice(offset, offset + VOXEL_RLE_FIELD_HEX_CHARS);
  if (!VOXEL_HEX_FIELD_PATTERN.test(field)) {
    throw new VoxelRowHexError(
      `RLE rows are strings of <runLen:2hex><paletteIdx:2hex> pairs; received invalid hex "${field}" at hex offset ${offset}`,
    );
  }
  return Number.parseInt(field, 16);
}

/**
 * Mirrors a stored half-row (cols 0-63) onto the full width, mapping col c
 * to col `fullWidth - 1 - c` (LD-28 mirror-X symmetry).
 *
 * @param half - The stored half-row of palette indices.
 * @param fullWidth - Full decoded width (`dims[0]`).
 * @returns The mirrored full-width row.
 */
function mirrorHalfRow(half: Uint8Array, fullWidth: number): Uint8Array {
  const full = new Uint8Array(fullWidth);
  for (let col = 0; col < half.length; col++) {
    full[col] = half[col];
    full[fullWidth - 1 - col] = half[col];
  }
  return full;
}

/**
 * Decodes one slice's rows into a row-major typed grid sized
 * `fullWidth * rows.length`, mirroring half-rows for symmetric documents.
 *
 * @param rows - The slice's RLE hex row strings in storage order.
 * @param fullWidth - Full decoded width (`dims[0]`).
 * @param storedWidth - Width each row must cover exactly (`dims[0] / 2`
 *   stored cols for symmetric documents, `dims[0]` otherwise).
 * @param symmetric - Whether stored half-rows must be mirrored.
 * @param paletteSize - Palette length; referenced indices must stay below it.
 * @returns The row-major typed grid for the slice.
 * @throws VoxelRowHexError / VoxelRowSumError / VoxelRowPaletteIndexError
 *   on malformed rows (see {@link decodeVoxelRow}).
 */
function decodeVoxelSlice(
  rows: readonly string[],
  fullWidth: number,
  storedWidth: number,
  symmetric: boolean,
  paletteSize: number,
): Uint8Array {
  const cells = new Uint8Array(fullWidth * rows.length);
  for (let row = 0; row < rows.length; row++) {
    const stored = decodeVoxelRow(rows[row], storedWidth, paletteSize);
    const full = symmetric ? mirrorHalfRow(stored, fullWidth) : stored;
    cells.set(full, row * fullWidth);
  }
  return cells;
}
