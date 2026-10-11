/**
 * On-disk voxel model schema for the Neat Racing voxel racer (spec section
 * 7.2, normative per LD-26 / section 2.4; slice P1-S3-A).
 *
 * A model is a single JSON document (for example
 * `assets/models/car-f1.model.json`) with three top-level fields:
 *
 * 1. `meta` — `{ name, dims: [width, height, depth], symmetric }`; dims are
 *    positive integers declaring the grid (normative car grid 128-cubed;
 *    wheels declare 32-cubed grids, section 7.4/TP-14).
 * 2. `palette` — `[r, g, b, a, glow]` byte entries. Index 0 is reserved as
 *    EMPTY (air; never rendered, never counted in averages) and must stay
 *    the all-zero sentinel. Black (`r === g === b === 0`) is a legal solid
 *    entry even with `glow > 0` (the black-is-solid exception, LD-23) — the
 *    schema accepts it, and shine suppression happens downstream in the
 *    mesher and LOD builder, not here.
 * 3. `slices` — one front-view slice per declared depth index (slice index =
 *    depth along the car's forward axis), each holding one row per declared
 *    height row by `dims[0]` cols; row 0 is ground/down. Rows are RLE hex
 *    strings of `<runLen:2hex><paletteIdx:2hex>` pairs; they are expanded by
 *    `model.decode.utils.ts`, which also mirrors the stored half-width when
 *    `meta.symmetric` is true.
 *
 * The module owns the format types, the named constants, and the named
 * validation errors. Validation is decoder-enforced and test-enforced
 * (AC-007/AC-008): malformed inputs are rejected with named errors.
 *
 * @module
 */

/**
 * Normative car-grid dimension along every axis (section 7.2/LD-19). This
 * constant describes the standard car asset size; validation itself accepts
 * any declared dims (section 7.2: reject `dims` ≠ 128³ *or the declared
 * dims*), for example the 32-cubed wheel grids of section 7.4/TP-14.
 */
export const VOXEL_GRID_DIMS = 128;

/** Inclusive lower bound for every declared grid axis (positive integers). */
export const VOXEL_DIMS_AXIS_MIN = 1;

/**
 * Modulus the declared width must satisfy when `meta.symmetric` is true:
 * mirror-X stores exactly half the width (LD-28), so the width must split
 * into two equal halves. Must stay in sync with
 * `VOXEL_SYMMETRIC_HALF_DIVISOR` in `model.decode.utils.ts`.
 */
export const VOXEL_SYMMETRIC_WIDTH_MODULUS = 2;

/** Number of components in one palette entry: `[r, g, b, a, glow]`. */
export const VOXEL_PALETTE_ENTRY_LENGTH = 5;

/** Inclusive upper bound for every palette component (byte range 0-255). */
export const VOXEL_PALETTE_COMPONENT_MAX = 255;

/** Palette index reserved as EMPTY (air; never rendered, section 7.2). */
export const VOXEL_EMPTY_PALETTE_INDEX = 0;

/**
 * The EMPTY sentinel palette entry that palette index 0 must hold exactly
 * (section 7.2: index 0 is reserved as EMPTY).
 */
export const VOXEL_EMPTY_PALETTE_ENTRY = [0, 0, 0, 0, 0] as const;

/**
 * One on-disk palette entry: `[r, g, b, a, glow]`, each component an integer
 * in the 0-255 byte range.
 *
 * - `glow` drives the bloom/shine pass (section 6.6).
 * - A fully transparent entry (`a === 0`) with `glow > 0` renders later as
 *   shining glass of that color (glow only, no body quads).
 * - Black (`r === g === b === 0`) carries no color and no shine regardless
 *   of `glow` (LD-23); the entry itself stays legal on disk.
 */
export type VoxelPaletteEntry = [
  r: number,
  g: number,
  b: number,
  a: number,
  glow: number,
];

/**
 * Model metadata header (`meta`).
 *
 * @property name - Model identifier (for example `car-f1`).
 * @property dims - Grid dims `[width, height, depth]` as positive integers
 *   declaring the grid; normative car models are the 128-cubed grid and
 *   wheels are 32-cubed (section 7.4/TP-14).
 * @property symmetric - Mirror-X flag: when true, rows store half the cols
 *   (runs sum to 64) and the loader mirrors cols 0-63 onto 64-127 (LD-28);
 *   when false, rows must cover all 128 cols.
 */
export interface VoxelModelMeta {
  name: string;
  dims: [number, number, number];
  symmetric: boolean;
}

/**
 * A full on-disk model document (section 7.2).
 *
 * @property meta - Metadata header.
 * @property palette - Palette entries; index 0 must be the EMPTY sentinel.
 * @property slices - One entry per declared depth index; each slice holds
 *   one RLE hex row string per declared height row.
 */
export interface VoxelModelDocument {
  meta: VoxelModelMeta;
  palette: VoxelPaletteEntry[];
  slices: string[][];
}

/**
 * Base class for every named voxel-model error (AC-007/AC-008).
 *
 * Carries a stable machine-readable `code` so callers (server endpoints, the
 * editor rebuild pipeline) can branch on rejection kinds without parsing
 * human-readable messages.
 */
export class VoxelModelError extends Error {
  /** Stable rejection code, for example `voxel-model/dims`. */
  readonly code: string;

  /**
   * @param code - Stable machine-readable rejection code.
   * @param message - Human-readable rejection reason.
   */
  constructor(code: string, message: string) {
    super(message);
    this.name = new.target.name;
    this.code = code;
  }
}

/**
 * Named rejection when required model metadata is missing or malformed
 * (section 7.2: "missing metadata" rejection category).
 */
export class VoxelModelMetadataError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-model/metadata', message);
  }
}

/**
 * Named rejection when `meta.dims` is not a positive-integer tuple, the
 * width is odd while `meta.symmetric` is true, or the slice/row counts
 * contradict the declared dims (section 7.2: "dims ≠ 128³ (or the declared
 * dims)" rejection category; section 7.4/TP-14 accepts 32-cubed wheels).
 */
export class VoxelModelDimsError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-model/dims', message);
  }
}

/**
 * Named rejection when palette entries are malformed (wrong shape, byte
 * range, or integer-ness) or palette index 0 is not the EMPTY sentinel.
 */
export class VoxelModelPaletteError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-model/palette', message);
  }
}

/**
 * Named rejection when an RLE row holds invalid hex (non-hex digits, short
 * fields) or a zero-length run (section 7.2: run lengths are 1-255).
 */
export class VoxelRowHexError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-model/row-hex', message);
  }
}

/**
 * Named rejection when a row's runs do not sum exactly to the declared row
 * width (`dims[0]` cols for full documents, `dims[0] / 2` stored cols for
 * symmetric documents).
 */
export class VoxelRowSumError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-model/row-sum', message);
  }
}

/**
 * Named rejection when a row references a palette index outside the
 * document's palette (section 7.2: "out-of-range palette indices").
 */
export class VoxelRowPaletteIndexError extends VoxelModelError {
  constructor(message: string) {
    super('voxel-model/row-palette-index', message);
  }
}

/**
 * Validates an untrusted model document against the section 7.2 schema
 * rules and returns the same content, narrowed to {@link VoxelModelDocument}.
 *
 * Row-level RLE rules (hex validity, run sums, palette index ranges) are
 * enforced by the decoder in `model.decode.utils.ts`; this validator owns
 * the document structure: metadata, dims, palette entries, and slice/row
 * counts.
 *
 * @param candidate - Untrusted parsed JSON value.
 * @returns The validated document content as a {@link VoxelModelDocument}.
 * @throws VoxelModelMetadataError when required metadata is missing or malformed.
 * @throws VoxelModelDimsError when dims components are not positive
 *   integers, the width is odd while symmetric, or slice/row counts
 *   contradict the declared dims.
 * @throws VoxelModelPaletteError when palette entries are malformed or index
 *   0 is not the EMPTY sentinel.
 *
 * @example
 * ```ts
 * const document = validateVoxelModelDocument(JSON.parse(rawJson));
 * console.log(document.meta.name);
 * ```
 */
export function validateVoxelModelDocument(
  candidate: unknown,
): VoxelModelDocument {
  if (!isRecord(candidate)) {
    throw new VoxelModelMetadataError(
      'model document must be a JSON object with meta, palette, and slices',
    );
  }
  // Step 1: validate the metadata header and the normative 128-cubed dims.
  const meta = validateMeta(candidate['meta']);
  // Step 2: validate the palette, including the EMPTY index-0 sentinel.
  const palette = validatePalette(candidate['palette']);
  // Step 3: validate slice/row counts against the declared dims.
  const slices = validateSlices(candidate['slices'], meta.dims);
  return { meta, palette, slices };
}

/**
 * Validates the `meta` header: object shape, non-empty `name`, boolean
 * `symmetric`, and the declared dims.
 *
 * @param candidate - Untrusted `meta` value.
 * @returns The validated metadata.
 * @throws VoxelModelMetadataError when fields are missing or malformed.
 * @throws VoxelModelDimsError when dims are not positive integers or the
 *   width is odd while symmetric.
 */
function validateMeta(candidate: unknown): VoxelModelMeta {
  if (!isRecord(candidate)) {
    throw new VoxelModelMetadataError(
      'meta must be an object with name, dims, and symmetric',
    );
  }
  const name = candidate['name'];
  if (typeof name !== 'string' || name.length === 0) {
    throw new VoxelModelMetadataError('meta.name must be a non-empty string');
  }
  const symmetric = candidate['symmetric'];
  if (typeof symmetric !== 'boolean') {
    throw new VoxelModelMetadataError('meta.symmetric must be a boolean');
  }
  const dims = validateMetaDims(candidate['dims']);
  validateSymmetricWidth(dims, symmetric);
  return { name, dims, symmetric };
}

/**
 * Validates `meta.dims` as a 3-number tuple of positive integers declaring
 * the document's grid (section 7.2: reject `dims` ≠ 128³ *or the declared
 * dims* — any positive-integer tuple is legal, e.g. the 32-cubed wheel
 * grids of section 7.4/TP-14; slice/row counts are checked separately
 * against these declared dims).
 *
 * @param candidate - Untrusted `dims` value.
 * @returns The validated dims tuple.
 * @throws VoxelModelMetadataError when dims is not a 3-number tuple.
 * @throws VoxelModelDimsError when any axis is not a positive integer.
 */
function validateMetaDims(candidate: unknown): [number, number, number] {
  if (!isDimsTuple(candidate)) {
    throw new VoxelModelMetadataError(
      'meta.dims must be [width, height, depth] numbers',
    );
  }
  const invalidAxis = candidate.find(
    (component) =>
      !Number.isInteger(component) || component < VOXEL_DIMS_AXIS_MIN,
  );
  if (invalidAxis !== undefined) {
    throw new VoxelModelDimsError(
      `meta.dims components must be positive integers (minimum ${VOXEL_DIMS_AXIS_MIN}), received [${candidate.join(', ')}]`,
    );
  }
  return candidate;
}

/**
 * Rejects odd widths on symmetric documents: mirror-X stores exactly half
 * the width (LD-28), so the declared width must split into two equal
 * integer halves.
 *
 * @param dims - The validated dims tuple.
 * @param symmetric - The validated `meta.symmetric` flag.
 * @throws VoxelModelDimsError when the width is odd while symmetric.
 */
function validateSymmetricWidth(
  dims: readonly [number, number, number],
  symmetric: boolean,
): void {
  if (symmetric && dims[0] % VOXEL_SYMMETRIC_WIDTH_MODULUS !== 0) {
    throw new VoxelModelDimsError(
      `meta.dims width ${dims[0]} must be even when meta.symmetric is true: mirror-X stores exactly half the width (spec 7.2/LD-28)`,
    );
  }
}

/**
 * Validates the palette array: present, non-empty, entries shaped as
 * integer byte tuples, and index 0 holding the EMPTY sentinel.
 *
 * @param candidate - Untrusted `palette` value.
 * @returns The validated palette entries.
 * @throws VoxelModelMetadataError when the palette is not an array.
 * @throws VoxelModelPaletteError when entries are malformed or index 0 is
 *   not the EMPTY sentinel.
 */
function validatePalette(candidate: unknown): VoxelPaletteEntry[] {
  if (!Array.isArray(candidate)) {
    throw new VoxelModelMetadataError(
      'palette must be an array of [r, g, b, a, glow] entries',
    );
  }
  validatePaletteEntries(candidate);
  validateEmptyPaletteEntry(candidate);
  // Every entry has been validated as a palette-entry tuple above.
  return candidate as VoxelPaletteEntry[];
}

/**
 * Validates every palette entry's shape and byte range.
 *
 * @param palette - The palette array to validate.
 * @throws VoxelModelPaletteError when the palette is empty or any entry is
 *   not the required number of integer components in the byte range.
 */
function validatePaletteEntries(palette: unknown[]): void {
  if (palette.length === 0) {
    throw new VoxelModelPaletteError(
      'palette must contain at least the EMPTY index-0 entry',
    );
  }
  for (const entry of palette) {
    if (!isPaletteEntry(entry)) {
      throw new VoxelModelPaletteError(
        `palette entries must be ${VOXEL_PALETTE_ENTRY_LENGTH} integer components [r, g, b, a, glow] in 0-${VOXEL_PALETTE_COMPONENT_MAX}, received ${JSON.stringify(entry)}`,
      );
    }
  }
}

/**
 * Validates that palette index 0 holds the EMPTY sentinel exactly
 * (section 7.2: index 0 is reserved as EMPTY; index 0 voxels never render
 * and never count in averages).
 *
 * @param palette - The palette array to validate.
 * @throws VoxelModelPaletteError when index 0 is not the all-zero entry.
 */
function validateEmptyPaletteEntry(palette: unknown[]): void {
  const emptyEntry = palette.at(VOXEL_EMPTY_PALETTE_INDEX);
  if (!isPaletteEntry(emptyEntry) || !isEmptyPaletteEntry(emptyEntry)) {
    throw new VoxelModelPaletteError(
      `palette index ${VOXEL_EMPTY_PALETTE_INDEX} is reserved as EMPTY and must be the all-zero entry [0, 0, 0, 0, 0]`,
    );
  }
}

/**
 * Validates the slices array: one slice per depth index, each slice holding
 * one row string per height row.
 *
 * @param candidate - Untrusted `slices` value.
 * @param dims - The validated dims tuple.
 * @returns The validated slice matrix of RLE hex row strings.
 * @throws VoxelModelMetadataError when slices is not an array of arrays of
 *   strings.
 * @throws VoxelModelDimsError when slice/row counts contradict the dims.
 */
function validateSlices(
  candidate: unknown,
  dims: readonly [number, number, number],
): string[][] {
  if (!Array.isArray(candidate)) {
    throw new VoxelModelMetadataError(
      'slices must be an array with one slice per depth index',
    );
  }
  if (candidate.length !== dims[2]) {
    throw new VoxelModelDimsError(
      `meta.dims declares ${dims[2]} slices along depth, received ${candidate.length}`,
    );
  }
  for (const slice of candidate) {
    validateSliceRows(slice, dims[1]);
  }
  // Every slice has been validated as an array of row strings above.
  return candidate as string[][];
}

/**
 * Validates one slice's rows: an array of exactly `rowCount` strings.
 *
 * @param candidate - Untrusted slice value.
 * @param rowCount - Row count declared by `meta.dims[1]`.
 * @throws VoxelModelMetadataError when the slice is not an array of strings.
 * @throws VoxelModelDimsError when the row count contradicts the dims.
 */
function validateSliceRows(candidate: unknown, rowCount: number): void {
  if (!Array.isArray(candidate)) {
    throw new VoxelModelMetadataError(
      'each slice must be an array of RLE hex row strings',
    );
  }
  if (candidate.length !== rowCount) {
    throw new VoxelModelDimsError(
      `meta.dims declares ${rowCount} rows per slice, received ${candidate.length}`,
    );
  }
  for (const row of candidate) {
    if (typeof row !== 'string') {
      throw new VoxelModelMetadataError(
        'each voxel row must be an RLE hex string of <runLen:2hex><paletteIdx:2hex> pairs',
      );
    }
  }
}

/**
 * Narrows an untrusted value to a plain JSON object record.
 *
 * @param value - Untrusted value.
 * @returns True when the value is a non-null, non-array object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Narrows an untrusted value to a 3-number dims tuple.
 *
 * @param value - Untrusted value.
 * @returns True when the value is an array of exactly three numbers.
 */
function isDimsTuple(value: unknown): value is [number, number, number] {
  return (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((component) => typeof component === 'number')
  );
}

/**
 * Narrows an untrusted value to a palette entry tuple of
 * {@link VOXEL_PALETTE_ENTRY_LENGTH} integer components in the 0-255 byte
 * range.
 *
 * @param value - Untrusted value.
 * @returns True when the value is a well-formed palette entry.
 */
function isPaletteEntry(value: unknown): value is VoxelPaletteEntry {
  return (
    Array.isArray(value) &&
    value.length === VOXEL_PALETTE_ENTRY_LENGTH &&
    value.every(
      (component) =>
        typeof component === 'number' &&
        Number.isInteger(component) &&
        component >= 0 &&
        component <= VOXEL_PALETTE_COMPONENT_MAX,
    )
  );
}

/**
 * Checks whether a palette entry is the EMPTY sentinel exactly.
 *
 * @param entry - The palette entry to check.
 * @returns True when every component matches
 *   {@link VOXEL_EMPTY_PALETTE_ENTRY}.
 */
function isEmptyPaletteEntry(entry: VoxelPaletteEntry): boolean {
  return VOXEL_EMPTY_PALETTE_ENTRY.every(
    (component, index) => component === entry[index],
  );
}
