/**
 * `.bin` LOD binary codec for the Neat Racing voxel racer (spec section
 * 7.3.2, normative; slice P1-S3-C, AC-010).
 *
 * Format: a magic + version + dims + palette-with-glow header followed by a
 * greedy-meshed quad array (position, normal axis, width/height, palette
 * index). The layout is little-endian and deterministic:
 *
 * | section    | encoding                                                                        |
 * | ---------- | ------------------------------------------------------------------------------- |
 * | magic      | `NXVB` as 4 ASCII bytes                                                          |
 * | version    | u32 LE, currently 1                                                              |
 * | dims       | u16 x 3 (`width`, `height`, `depth`)                                             |
 * | palette    | u16 entry count, then 5 bytes per entry (`r`, `g`, `b`, `a`, `glow`)             |
 * | quad count | u32 LE                                                                           |
 * | quads      | 12 bytes each: position u16 x 3, normalAxis u8, width u16, height u16, paletteIndex u8 |
 *
 * Round-trip guarantee (AC-010): writer output re-parsed by the loader and
 * re-written is byte-identical, so the build script
 * (`scripts/build-voxel-bins.mjs`) and the runtime loader share one format
 * without drift. The loader rejects bad magic, unknown versions, truncated
 * sections, and incoherent fields with named errors that mirror the
 * `model.schema.ts` class pattern (a stable `code` per subclass).
 *
 * `VoxelPaletteEntry` is imported as a type-only symbol from
 * `./model.schema`: the import is erased at runtime under native type
 * stripping, so no extensionless specifier survives and this module stays
 * self-contained for the Node build script.
 *
 * @module
 */

import type { VoxelPaletteEntry } from './model.schema';

/** Magic bytes prefixing every voxel bin (spec 7.3.2). */
export const VOXEL_BIN_MAGIC = 'NXVB';

/** Binary format version written by this codec (u32 LE). */
export const VOXEL_BIN_VERSION = 1;

/** Byte offset of the magic section. */
export const VOXEL_BIN_MAGIC_OFFSET = 0;

/** Byte offset of the u32 version field. */
export const VOXEL_BIN_VERSION_OFFSET = 4;

/** Byte offset of the u16 x 3 dims fields. */
export const VOXEL_BIN_DIMS_OFFSET = 8;

/** Byte offset of the u16 palette entry count field. */
export const VOXEL_BIN_PALETTE_COUNT_OFFSET = 14;

/** Fixed header size: magic + version + dims + palette count. */
export const VOXEL_BIN_HEADER_BYTES = 16;

/** Bytes per palette entry (`r`, `g`, `b`, `a`, `glow`). */
export const VOXEL_BIN_PALETTE_ENTRY_BYTES = 5;

/** Bytes of the u32 quad count field. */
export const VOXEL_BIN_QUAD_COUNT_BYTES = 4;

/** Bytes of one 12-byte quad record. */
export const VOXEL_BIN_QUAD_BYTES = 12;

/** Byte offset of the u8 normalAxis field inside one quad record. */
export const VOXEL_BIN_QUAD_NORMAL_AXIS_OFFSET = 6;

/** Byte offset of the u16 width field inside one quad record. */
export const VOXEL_BIN_QUAD_WIDTH_OFFSET = 7;

/** Byte offset of the u16 height field inside one quad record. */
export const VOXEL_BIN_QUAD_HEIGHT_OFFSET = 9;

/** Byte offset of the u8 paletteIndex field inside one quad record. */
export const VOXEL_BIN_QUAD_PALETTE_INDEX_OFFSET = 11;

/** Components per quad position (u16 x 3). */
export const VOXEL_BIN_POSITION_COMPONENTS = 3;

/** Bytes per u16 component (dims, positions, widths, heights). */
export const VOXEL_BIN_U16_COMPONENT_BYTES = 2;

/** Inclusive upper bound for u8 fields (palette components, paletteIndex). */
export const VOXEL_BIN_U8_MAX = 255;

/** Inclusive upper bound for u16 fields (dims, positions, extents, palette count). */
export const VOXEL_BIN_U16_MAX = 65535;

/** Inclusive lower bound for every dims component (positive integers). */
export const VOXEL_BIN_DIMS_COMPONENT_MIN = 1;

/** Inclusive lower bound for the palette entry count (the EMPTY sentinel). */
export const VOXEL_BIN_PALETTE_ENTRY_MIN = 1;

/** Inclusive lower bound for quad width/height extents. */
export const VOXEL_BIN_QUAD_EXTENT_MIN = 1;

/** Inclusive lower bound for the normalAxis field (x axis). */
export const VOXEL_BIN_NORMAL_AXIS_MIN = 0;

/** Inclusive upper bound for the normalAxis field (z axis). */
export const VOXEL_BIN_NORMAL_AXIS_MAX = 2;

/**
 * One greedy-meshed quad record (spec 7.3.2 body; AC-009 quad shape).
 *
 * @property position - Quad origin `[x, y, z]`, each component a u16.
 * @property normalAxis - Face normal axis: 0 = x, 1 = y, 2 = z (u8).
 * @property width - Quad extent along the first tangent axis (u16).
 * @property height - Quad extent along the second tangent axis (u16).
 * @property paletteIndex - Palette entry referenced by the quad (u8, must
 *   stay below the palette entry count).
 */
export interface VoxelBinQuad {
  readonly position: readonly [number, number, number];
  readonly normalAxis: number;
  readonly width: number;
  readonly height: number;
  readonly paletteIndex: number;
}

/**
 * In-memory voxel bin content (spec 7.3.2): dims, palette-with-glow, and
 * the greedy-meshed quad array.
 *
 * @property dims - Grid dims `[width, height, depth]`, each a positive u16.
 * @property palette - Palette entries; index 0 is the EMPTY sentinel.
 * @property quads - Greedy-meshed quad records.
 */
export interface VoxelBinModel {
  readonly dims: readonly [number, number, number];
  readonly palette: readonly VoxelPaletteEntry[];
  readonly quads: readonly VoxelBinQuad[];
}

/**
 * Base class for every named voxel-bin error (AC-010).
 *
 * Carries a stable machine-readable `code` so callers (the build script,
 * the runtime loader) can branch on rejection kinds without parsing
 * human-readable messages.
 */
export class VoxelBinCodecError extends Error {
  /** Stable rejection code, for example `voxel-bin/magic`. */
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
 * Named rejection when the leading bytes are not the `NXVB` magic
 * (spec 7.3.2: loader rejects bad magic).
 */
export class VoxelBinMagicError extends VoxelBinCodecError {
  constructor(message: string) {
    super('voxel-bin/magic', message);
  }
}

/**
 * Named rejection when the version field is not the supported format
 * version (spec 7.3.2: loader rejects bad version).
 */
export class VoxelBinVersionError extends VoxelBinCodecError {
  constructor(message: string) {
    super('voxel-bin/version', message);
  }
}

/**
 * Named rejection when the buffer ends before a declared section completes
 * (header, palette-with-glow, quad count, or quad records).
 */
export class VoxelBinTruncatedError extends VoxelBinCodecError {
  constructor(message: string) {
    super('voxel-bin/truncated', message);
  }
}

/**
 * Named rejection when a field value cannot be represented by, or is
 * incoherent with, the binary layout: dims/position/extent components
 * outside their u16 range, palette components outside the byte range, or a
 * quad referencing a normal axis or palette entry that does not exist.
 */
export class VoxelBinFieldError extends VoxelBinCodecError {
  constructor(message: string) {
    super('voxel-bin/field', message);
  }
}

/**
 * Encodes a voxel bin into the deterministic little-endian `NXVB` layout
 * (spec 7.3.2).
 *
 * Every field is validated against the binary layout caps first, so the
 * output is guaranteed to be re-loadable by {@link loadVoxelBin} and
 * byte-identical across repeated calls with equivalent content.
 *
 * @param model - The voxel bin content: dims, palette-with-glow, and quads.
 * @returns The encoded bytes sized exactly
 *   `16 + 5 x paletteCount + 4 + 12 x quadCount`.
 * @throws VoxelBinFieldError when any dims component is not a positive
 *   u16, the palette is empty or exceeds u16 entries, a palette component
 *   is outside the byte range, a quad position is negative or exceeds u16,
 *   a quad width/height is not a positive u16, the normal axis is outside
 *   0-2, or a quad palette index does not reference the palette.
 *
 * @example
 * ```ts
 * const bytes = writeVoxelBin({ dims: [4, 3, 2], palette, quads });
 * console.log(bytes.length); // 16 + 5 * palette.length + 4 + 12 * quads.length
 * ```
 */
export function writeVoxelBin(model: VoxelBinModel): Uint8Array {
  // Step 1: validate every field against the binary layout caps.
  validateBinModel(model);
  // Step 2: allocate the exact deterministic output size.
  const bytes = allocateBinBuffer(model);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // Step 3: write the fixed header (magic, version, dims, palette count).
  writeBinHeader(bytes, view, model);
  // Step 4: write the palette-with-glow entries.
  let offset = writeBinPalette(bytes, model.palette, VOXEL_BIN_HEADER_BYTES);
  // Step 5: write the u32 quad count and the 12-byte quad records.
  offset = writeBinQuadCount(view, model.quads.length, offset);
  for (const quad of model.quads) {
    offset = writeBinQuad(bytes, view, offset, quad);
  }
  return bytes;
}

/**
 * Parses a voxel bin produced by {@link writeVoxelBin} back into its
 * in-memory content (spec 7.3.2 loader; AC-010).
 *
 * The loader is the AC-010 gate: it rejects bad magic, unknown versions,
 * and any buffer that ends before a declared section completes. Re-writing
 * the parsed content is byte-identical to the input bytes.
 *
 * @param bytes - The encoded voxel bin (a `Uint8Array`; subarray views with
 *   a non-zero `byteOffset` are supported).
 * @returns The decoded dims, palette-with-glow, and quad records.
 * @throws VoxelBinTruncatedError when the buffer is shorter than the fixed
 *   header or ends before the palette, quad count, or quad records complete.
 * @throws VoxelBinMagicError when the leading bytes are not `NXVB`.
 * @throws VoxelBinVersionError when the version field is not supported.
 * @throws VoxelBinFieldError when a quad's normal axis is outside 0-2 or
 *   its palette index does not reference the palette.
 *
 * @example
 * ```ts
 * const model = loadVoxelBin(bytes);
 * console.log(model.quads.length);
 * ```
 */
export function loadVoxelBin(bytes: Uint8Array): VoxelBinModel {
  // Step 1: require the fixed header before any field can be read.
  assertBinHeaderPresent(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  // Step 2: verify magic + version (AC-010 loader gates).
  assertBinMagic(bytes);
  assertBinVersion(view);
  // Step 3: read dims + palette count, then the palette-with-glow entries.
  const dims = readBinDims(view);
  const paletteCount = view.getUint16(VOXEL_BIN_PALETTE_COUNT_OFFSET, true);
  assertBinPalettePresent(bytes, paletteCount);
  const palette = readBinPalette(bytes, paletteCount);
  // Step 4: read the u32 quad count, then every 12-byte quad record.
  const quadCountOffset =
    VOXEL_BIN_HEADER_BYTES + paletteCount * VOXEL_BIN_PALETTE_ENTRY_BYTES;
  assertBinQuadCountPresent(bytes, quadCountOffset);
  const quadCount = view.getUint32(quadCountOffset, true);
  const quads = readBinQuads(
    view,
    bytes.length,
    quadCountOffset + VOXEL_BIN_QUAD_COUNT_BYTES,
    quadCount,
    palette.length,
  );
  return { dims, palette, quads };
}

/**
 * Validates a full voxel bin model against the binary layout caps.
 *
 * @param model - The model to validate.
 * @throws VoxelBinFieldError on any out-of-layout field (see
 *   {@link writeVoxelBin}).
 */
function validateBinModel(model: VoxelBinModel): void {
  validateBinDims(model.dims);
  validateBinPalette(model.palette);
  for (const quad of model.quads) {
    validateBinQuad(quad, model.palette.length);
  }
}

/**
 * Validates the dims tuple: three positive-integer u16 components.
 *
 * @param dims - The dims tuple to validate.
 * @throws VoxelBinFieldError when any component is not a positive u16.
 */
function validateBinDims(dims: readonly [number, number, number]): void {
  for (let axis = 0; axis < VOXEL_BIN_POSITION_COMPONENTS; axis++) {
    assertBinIntegerField(
      `dims component ${axis}`,
      dims[axis],
      VOXEL_BIN_DIMS_COMPONENT_MIN,
      VOXEL_BIN_U16_MAX,
    );
  }
}

/**
 * Validates the palette: at least the EMPTY sentinel, at most u16 entries,
 * and every component an integer byte.
 *
 * @param palette - The palette to validate.
 * @throws VoxelBinFieldError when the palette is empty, exceeds u16
 *   entries, or holds a component outside the byte range.
 */
function validateBinPalette(palette: readonly VoxelPaletteEntry[]): void {
  if (palette.length < VOXEL_BIN_PALETTE_ENTRY_MIN) {
    throw new VoxelBinFieldError(
      'voxel bin palette must contain at least the EMPTY index-0 entry',
    );
  }
  if (palette.length > VOXEL_BIN_U16_MAX) {
    throw new VoxelBinFieldError(
      `voxel bin palette entry count ${palette.length} exceeds the u16 cap ${VOXEL_BIN_U16_MAX}`,
    );
  }
  for (let entryIndex = 0; entryIndex < palette.length; entryIndex++) {
    validateBinPaletteEntry(palette[entryIndex], entryIndex);
  }
}

/**
 * Validates one palette entry's five components as integer bytes.
 *
 * @param entry - The palette entry to validate.
 * @param entryIndex - Index of the entry (for the rejection message).
 * @throws VoxelBinFieldError when any component is not an integer byte.
 */
function validateBinPaletteEntry(
  entry: VoxelPaletteEntry,
  entryIndex: number,
): void {
  for (
    let component = 0;
    component < VOXEL_BIN_PALETTE_ENTRY_BYTES;
    component++
  ) {
    assertBinIntegerField(
      `palette entry ${entryIndex} component ${component}`,
      entry[component],
      0,
      VOXEL_BIN_U8_MAX,
    );
  }
}

/**
 * Validates one quad record against the binary layout and palette size.
 *
 * @param quad - The quad to validate.
 * @param paletteSize - Palette entry count; the quad's palette index must
 *   reference an existing entry.
 * @throws VoxelBinFieldError when any quad field is out of layout.
 */
function validateBinQuad(quad: VoxelBinQuad, paletteSize: number): void {
  validateBinQuadPosition(quad.position);
  assertBinIntegerField(
    'quad normalAxis',
    quad.normalAxis,
    VOXEL_BIN_NORMAL_AXIS_MIN,
    VOXEL_BIN_NORMAL_AXIS_MAX,
  );
  assertBinIntegerField(
    'quad width',
    quad.width,
    VOXEL_BIN_QUAD_EXTENT_MIN,
    VOXEL_BIN_U16_MAX,
  );
  assertBinIntegerField(
    'quad height',
    quad.height,
    VOXEL_BIN_QUAD_EXTENT_MIN,
    VOXEL_BIN_U16_MAX,
  );
  assertBinQuadPaletteIndex(quad.paletteIndex, paletteSize);
}

/**
 * Validates a quad position: three integer u16 components (non-negative).
 *
 * @param position - The quad origin to validate.
 * @throws VoxelBinFieldError when any component is negative, fractional,
 *   or exceeds the u16 cap.
 */
function validateBinQuadPosition(
  position: readonly [number, number, number],
): void {
  for (let axis = 0; axis < VOXEL_BIN_POSITION_COMPONENTS; axis++) {
    assertBinIntegerField(
      `quad position axis ${axis}`,
      position[axis],
      0,
      VOXEL_BIN_U16_MAX,
    );
  }
}

/**
 * Validates a quad's palette index: an integer byte below the palette size.
 *
 * @param paletteIndex - The quad's palette index.
 * @param paletteSize - Palette entry count.
 * @throws VoxelBinFieldError when the index is fractional, exceeds the u8
 *   cap, or does not reference an existing palette entry.
 */
function assertBinQuadPaletteIndex(
  paletteIndex: number,
  paletteSize: number,
): void {
  if (
    !Number.isInteger(paletteIndex) ||
    paletteIndex < 0 ||
    paletteIndex > VOXEL_BIN_U8_MAX ||
    paletteIndex >= paletteSize
  ) {
    throw new VoxelBinFieldError(
      `voxel bin quad paletteIndex ${paletteIndex} must be an integer in 0-${VOXEL_BIN_U8_MAX} and below the ${paletteSize}-entry palette`,
    );
  }
}

/**
 * Shared integer-field gate: throws one named field error covering range
 * and integer-ness for every binary field.
 *
 * @param fieldName - Field description used in the rejection message.
 * @param value - The field value to validate.
 * @param min - Inclusive lower bound of the field's binary encoding.
 * @param max - Inclusive upper bound of the field's binary encoding.
 * @throws VoxelBinFieldError when the value is not an integer within
 *   `min`-`max`.
 */
function assertBinIntegerField(
  fieldName: string,
  value: number,
  min: number,
  max: number,
): void {
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new VoxelBinFieldError(
      `voxel bin ${fieldName} must be an integer in ${min}-${max}, received ${value}`,
    );
  }
}

/**
 * Allocates the exact deterministic output size for a model:
 * header + palette + quad count + quad records.
 *
 * @param model - The validated model to size.
 * @returns The zero-initialized output buffer.
 */
function allocateBinBuffer(model: VoxelBinModel): Uint8Array {
  const paletteBytes = model.palette.length * VOXEL_BIN_PALETTE_ENTRY_BYTES;
  const quadBytes = model.quads.length * VOXEL_BIN_QUAD_BYTES;
  return new Uint8Array(
    VOXEL_BIN_HEADER_BYTES +
      paletteBytes +
      VOXEL_BIN_QUAD_COUNT_BYTES +
      quadBytes,
  );
}

/**
 * Writes the fixed header: `NXVB` magic, u32 version, u16 x 3 dims, and the
 * u16 palette entry count (all little-endian).
 *
 * @param bytes - The output buffer.
 * @param view - DataView over the output buffer.
 * @param model - The validated model providing dims and palette size.
 */
function writeBinHeader(
  bytes: Uint8Array,
  view: DataView,
  model: VoxelBinModel,
): void {
  for (let index = 0; index < VOXEL_BIN_MAGIC.length; index++) {
    bytes[VOXEL_BIN_MAGIC_OFFSET + index] = VOXEL_BIN_MAGIC.charCodeAt(index);
  }
  view.setUint32(VOXEL_BIN_VERSION_OFFSET, VOXEL_BIN_VERSION, true);
  view.setUint16(VOXEL_BIN_DIMS_OFFSET, model.dims[0], true);
  view.setUint16(
    VOXEL_BIN_DIMS_OFFSET + VOXEL_BIN_U16_COMPONENT_BYTES,
    model.dims[1],
    true,
  );
  view.setUint16(
    VOXEL_BIN_DIMS_OFFSET + VOXEL_BIN_U16_COMPONENT_BYTES * 2,
    model.dims[2],
    true,
  );
  view.setUint16(VOXEL_BIN_PALETTE_COUNT_OFFSET, model.palette.length, true);
}

/**
 * Writes the palette-with-glow entries (5 bytes each: `r`, `g`, `b`, `a`,
 * `glow`) starting at the header end.
 *
 * @param bytes - The output buffer.
 * @param palette - The validated palette entries.
 * @param startOffset - Offset of the first palette byte.
 * @returns The offset just past the palette section.
 */
function writeBinPalette(
  bytes: Uint8Array,
  palette: readonly VoxelPaletteEntry[],
  startOffset: number,
): number {
  let offset = startOffset;
  for (const entry of palette) {
    for (
      let component = 0;
      component < VOXEL_BIN_PALETTE_ENTRY_BYTES;
      component++
    ) {
      bytes[offset++] = entry[component];
    }
  }
  return offset;
}

/**
 * Writes the u32 little-endian quad count field.
 *
 * @param view - DataView over the output buffer.
 * @param quadCount - Number of quad records that follow.
 * @param offset - Offset of the quad count field.
 * @returns The offset just past the quad count field.
 */
function writeBinQuadCount(
  view: DataView,
  quadCount: number,
  offset: number,
): number {
  view.setUint32(offset, quadCount, true);
  return offset + VOXEL_BIN_QUAD_COUNT_BYTES;
}

/**
 * Writes one 12-byte quad record: position u16 x 3, normalAxis u8, width
 * u16, height u16, paletteIndex u8 (all little-endian).
 *
 * @param bytes - The output buffer (u8 fields).
 * @param view - DataView over the output buffer (u16 fields).
 * @param startOffset - Offset of the quad record.
 * @param quad - The validated quad to encode.
 * @returns The offset just past the quad record.
 */
function writeBinQuad(
  bytes: Uint8Array,
  view: DataView,
  startOffset: number,
  quad: VoxelBinQuad,
): number {
  let offset = startOffset;
  for (let axis = 0; axis < VOXEL_BIN_POSITION_COMPONENTS; axis++) {
    view.setUint16(offset, quad.position[axis], true);
    offset += VOXEL_BIN_U16_COMPONENT_BYTES;
  }
  bytes[offset++] = quad.normalAxis;
  view.setUint16(offset, quad.width, true);
  offset += VOXEL_BIN_U16_COMPONENT_BYTES;
  view.setUint16(offset, quad.height, true);
  offset += VOXEL_BIN_U16_COMPONENT_BYTES;
  bytes[offset++] = quad.paletteIndex;
  return offset;
}

/**
 * Rejects buffers too short for the fixed 16-byte header.
 *
 * @param bytes - The buffer to check.
 * @throws VoxelBinTruncatedError when the buffer holds fewer than the
 *   header bytes.
 */
function assertBinHeaderPresent(bytes: Uint8Array): void {
  if (bytes.length < VOXEL_BIN_HEADER_BYTES) {
    throw new VoxelBinTruncatedError(
      `voxel bin header needs ${VOXEL_BIN_HEADER_BYTES} bytes, buffer holds ${bytes.length}`,
    );
  }
}

/**
 * Rejects buffers whose leading bytes are not the `NXVB` magic.
 *
 * @param bytes - The buffer to check.
 * @throws VoxelBinMagicError when the magic bytes mismatch.
 */
function assertBinMagic(bytes: Uint8Array): void {
  const magic = String.fromCharCode(
    ...bytes.subarray(
      VOXEL_BIN_MAGIC_OFFSET,
      VOXEL_BIN_MAGIC_OFFSET + VOXEL_BIN_MAGIC.length,
    ),
  );
  if (magic !== VOXEL_BIN_MAGIC) {
    throw new VoxelBinMagicError(
      `voxel bin magic must be "${VOXEL_BIN_MAGIC}", received "${magic}"`,
    );
  }
}

/**
 * Rejects unknown format versions.
 *
 * @param view - DataView over the buffer.
 * @throws VoxelBinVersionError when the u32 version is not supported.
 */
function assertBinVersion(view: DataView): void {
  const version = view.getUint32(VOXEL_BIN_VERSION_OFFSET, true);
  if (version !== VOXEL_BIN_VERSION) {
    throw new VoxelBinVersionError(
      `voxel bin version must be ${VOXEL_BIN_VERSION}, received ${version}`,
    );
  }
}

/**
 * Reads the u16 x 3 dims fields as a tuple.
 *
 * @param view - DataView over the buffer.
 * @returns The decoded `[width, height, depth]` dims tuple.
 */
function readBinDims(view: DataView): [number, number, number] {
  return [
    view.getUint16(VOXEL_BIN_DIMS_OFFSET, true),
    view.getUint16(VOXEL_BIN_DIMS_OFFSET + VOXEL_BIN_U16_COMPONENT_BYTES, true),
    view.getUint16(
      VOXEL_BIN_DIMS_OFFSET + VOXEL_BIN_U16_COMPONENT_BYTES * 2,
      true,
    ),
  ];
}

/**
 * Rejects buffers that end before the declared palette section completes.
 *
 * @param bytes - The buffer to check.
 * @param paletteCount - Declared palette entry count.
 * @throws VoxelBinTruncatedError when the palette section is incomplete.
 */
function assertBinPalettePresent(
  bytes: Uint8Array,
  paletteCount: number,
): void {
  const paletteEnd =
    VOXEL_BIN_HEADER_BYTES + paletteCount * VOXEL_BIN_PALETTE_ENTRY_BYTES;
  if (bytes.length < paletteEnd) {
    throw new VoxelBinTruncatedError(
      `voxel bin palette section needs ${paletteEnd} bytes (${paletteCount} entries), buffer holds ${bytes.length}`,
    );
  }
}

/**
 * Reads the palette-with-glow entries (5 bytes each).
 *
 * @param bytes - The buffer holding the palette section.
 * @param paletteCount - Declared palette entry count (bounds-checked).
 * @returns The decoded palette entries.
 */
function readBinPalette(
  bytes: Uint8Array,
  paletteCount: number,
): VoxelPaletteEntry[] {
  const palette: VoxelPaletteEntry[] = [];
  let offset = VOXEL_BIN_HEADER_BYTES;
  for (let entryIndex = 0; entryIndex < paletteCount; entryIndex++) {
    palette.push([
      bytes[offset],
      bytes[offset + 1],
      bytes[offset + 2],
      bytes[offset + 3],
      bytes[offset + 4],
    ]);
    offset += VOXEL_BIN_PALETTE_ENTRY_BYTES;
  }
  return palette;
}

/**
 * Rejects buffers that end before the u32 quad count field completes.
 *
 * @param bytes - The buffer to check.
 * @param quadCountOffset - Offset of the quad count field.
 * @throws VoxelBinTruncatedError when the quad count field is incomplete.
 */
function assertBinQuadCountPresent(
  bytes: Uint8Array,
  quadCountOffset: number,
): void {
  const quadCountEnd = quadCountOffset + VOXEL_BIN_QUAD_COUNT_BYTES;
  if (bytes.length < quadCountEnd) {
    throw new VoxelBinTruncatedError(
      `voxel bin quad count needs ${quadCountEnd} bytes, buffer holds ${bytes.length}`,
    );
  }
}

/**
 * Reads every 12-byte quad record, rejecting any buffer that ends before a
 * record completes or any record with an incoherent field.
 *
 * @param view - DataView over the buffer.
 * @param bufferLength - Total buffer length (record bounds gate).
 * @param startOffset - Offset of the first quad record.
 * @param quadCount - Declared quad record count.
 * @param paletteSize - Palette entry count for index coherence.
 * @returns The decoded quad records.
 * @throws VoxelBinTruncatedError when a quad record is incomplete.
 * @throws VoxelBinFieldError when a quad's normal axis or palette index is
 *   incoherent (see {@link validateBinQuad}).
 */
function readBinQuads(
  view: DataView,
  bufferLength: number,
  startOffset: number,
  quadCount: number,
  paletteSize: number,
): VoxelBinQuad[] {
  const quads: VoxelBinQuad[] = [];
  let offset = startOffset;
  for (let quadIndex = 0; quadIndex < quadCount; quadIndex++) {
    const quadsEnd = offset + VOXEL_BIN_QUAD_BYTES;
    if (bufferLength < quadsEnd) {
      throw new VoxelBinTruncatedError(
        `voxel bin quad ${quadIndex} needs ${quadsEnd} bytes, buffer holds ${bufferLength}`,
      );
    }
    quads.push(readBinQuad(view, offset, paletteSize));
    offset += VOXEL_BIN_QUAD_BYTES;
  }
  return quads;
}

/**
 * Reads one 12-byte quad record and re-validates the fields whose binary
 * encoding can carry incoherent values (normal axis, palette index).
 *
 * @param view - DataView over the buffer.
 * @param offset - Offset of the quad record.
 * @param paletteSize - Palette entry count for index coherence.
 * @returns The decoded quad record.
 * @throws VoxelBinFieldError when the normal axis or palette index is
 *   incoherent (see {@link validateBinQuad}).
 */
function readBinQuad(
  view: DataView,
  offset: number,
  paletteSize: number,
): VoxelBinQuad {
  const position: [number, number, number] = [
    view.getUint16(offset, true),
    view.getUint16(offset + VOXEL_BIN_U16_COMPONENT_BYTES, true),
    view.getUint16(offset + VOXEL_BIN_U16_COMPONENT_BYTES * 2, true),
  ];
  const normalAxis = view.getUint8(offset + VOXEL_BIN_QUAD_NORMAL_AXIS_OFFSET);
  const width = view.getUint16(offset + VOXEL_BIN_QUAD_WIDTH_OFFSET, true);
  const height = view.getUint16(offset + VOXEL_BIN_QUAD_HEIGHT_OFFSET, true);
  const paletteIndex = view.getUint8(
    offset + VOXEL_BIN_QUAD_PALETTE_INDEX_OFFSET,
  );
  validateBinQuad(
    { position, normalAxis, width, height, paletteIndex },
    paletteSize,
  );
  return { position, normalAxis, width, height, paletteIndex };
}
