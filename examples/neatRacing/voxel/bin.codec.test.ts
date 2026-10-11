/**
 * Codec-contract tests for the Neat Racing `.bin` LOD binary format
 * (spec section 7.3.2, AC-010; slice P1-S3-C).
 *
 * Green-only TDD per the step packet: these tests pin the normative
 * section 7.3.2 behavior — the little-endian `NXVB` layout (magic,
 * version, dims, palette-with-glow, 12-byte quad records), deterministic
 * output, byte-identical write → load → write round-trips, and loader
 * rejection of bad magic, unknown versions, truncated sections, and
 * incoherent fields with the named codec errors.
 */
import { describe, expect, it } from '@jest/globals';
import {
  loadVoxelBin,
  writeVoxelBin,
  VOXEL_BIN_DIMS_OFFSET,
  VOXEL_BIN_HEADER_BYTES,
  VOXEL_BIN_MAGIC_OFFSET,
  VOXEL_BIN_NORMAL_AXIS_MAX,
  VOXEL_BIN_PALETTE_ENTRY_BYTES,
  VOXEL_BIN_QUAD_COUNT_BYTES,
  VOXEL_BIN_QUAD_NORMAL_AXIS_OFFSET,
  VOXEL_BIN_QUAD_PALETTE_INDEX_OFFSET,
  VOXEL_BIN_U8_MAX,
  VOXEL_BIN_U16_MAX,
  VOXEL_BIN_VERSION,
  VOXEL_BIN_VERSION_OFFSET,
  VoxelBinFieldError,
  VoxelBinMagicError,
  VoxelBinTruncatedError,
  VoxelBinVersionError,
  type VoxelBinModel,
  type VoxelBinQuad,
} from './bin.codec';
import type { VoxelPaletteEntry } from './model.schema';

/** Fixture dims for the four-quad-free reference model (u16 LE layout). */
const FIXTURE_DIMS: readonly [number, number, number] = [4, 3, 2];

/** Fixture palette: EMPTY sentinel, red, and a glow-tinted light blue. */
const FIXTURE_PALETTE: VoxelPaletteEntry[] = [
  [0, 0, 0, 0, 0],
  [255, 0, 0, 255, 0],
  [200, 230, 255, 0, 180],
];

/** The EMPTY sentinel palette entry used for oversize fixtures. */
const EMPTY_PALETTE_ENTRY: VoxelPaletteEntry = [0, 0, 0, 0, 0];

/** Fixture quads exercising both u16 position/extent fields and u8 fields. */
const FIXTURE_QUADS: readonly VoxelBinQuad[] = [
  { position: [1, 0, 1], normalAxis: 1, width: 2, height: 1, paletteIndex: 1 },
  { position: [3, 2, 0], normalAxis: 0, width: 1, height: 2, paletteIndex: 2 },
];

/** The reference fixture model shared by write and load tests. */
const FIXTURE_MODEL: VoxelBinModel = {
  dims: FIXTURE_DIMS,
  palette: FIXTURE_PALETTE,
  quads: FIXTURE_QUADS,
};

/** Expected little-endian magic + version bytes for the fixture. */
const EXPECTED_MAGIC_AND_VERSION = [
  0x4e,
  0x58,
  0x56,
  0x42, // 'N', 'X', 'V', 'B'
  1,
  0,
  0,
  0, // version 1, u32 LE
];

/** Expected little-endian dims + palette count bytes for the fixture. */
const EXPECTED_DIMS_AND_PALETTE_COUNT = [
  4,
  0,
  3,
  0,
  2,
  0, // dims [4, 3, 2], u16 LE x 3
  3,
  0, // palette count 3, u16 LE
];

/** Expected palette-with-glow bytes for the fixture (5 bytes per entry). */
const EXPECTED_PALETTE_BYTES = [
  0, 0, 0, 0, 0, 255, 0, 0, 255, 0, 200, 230, 255, 0, 180,
];

/** Expected quad count + both 12-byte quad records for the fixture. */
const EXPECTED_QUAD_BYTES = [
  2,
  0,
  0,
  0, // quad count 2, u32 LE
  1,
  0,
  0,
  0,
  1,
  0, // quad 1 position [1, 0, 1]
  1, // quad 1 normalAxis (y)
  2,
  0, // quad 1 width
  1,
  0, // quad 1 height
  1, // quad 1 paletteIndex
  3,
  0,
  2,
  0,
  0,
  0, // quad 2 position [3, 2, 0]
  0, // quad 2 normalAxis (x)
  1,
  0, // quad 2 width
  2,
  0, // quad 2 height
  2, // quad 2 paletteIndex
];

/** Byte value replacing the first magic byte for the magic rejection test. */
const CORRUPTED_FIRST_MAGIC_BYTE = 'X'.charCodeAt(0);

/** Version written into the version rejection fixture (unsupported). */
const CORRUPTED_VERSION = VOXEL_BIN_VERSION + 1;

/** Palette component value outside the u8 encoding range. */
const MALFORMED_PALETTE_COMPONENT = VOXEL_BIN_U8_MAX + 1;

/** Offsets of the fixture's variable sections, derived from codec constants. */
const PALETTE_OFFSET = VOXEL_BIN_HEADER_BYTES;
const QUAD_COUNT_OFFSET =
  PALETTE_OFFSET + FIXTURE_PALETTE.length * VOXEL_BIN_PALETTE_ENTRY_BYTES;
const QUADS_OFFSET = QUAD_COUNT_OFFSET + VOXEL_BIN_QUAD_COUNT_BYTES;

/**
 * Builds a fixture copy with its first quad patched for write-side
 * rejection tests (single-quad single-expect style).
 *
 * @param patch - Quad fields to override on the first fixture quad.
 * @returns The fixture model with the patched first quad.
 */
function modelWithQuad(patch: Partial<VoxelBinQuad>): VoxelBinModel {
  const [firstQuad, ...remainingQuads] = FIXTURE_QUADS;
  return {
    ...FIXTURE_MODEL,
    quads: [{ ...firstQuad, ...patch }, ...remainingQuads],
  };
}

describe('writeVoxelBin', () => {
  it('writes the NXVB magic and little-endian version 1', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(
      Array.from(bytes.subarray(VOXEL_BIN_MAGIC_OFFSET, VOXEL_BIN_DIMS_OFFSET)),
    ).toEqual(EXPECTED_MAGIC_AND_VERSION);
    expect(loadVoxelBin(bytes).dims).toEqual(FIXTURE_DIMS);
  });

  it('writes dims and palette count as little-endian u16 fields', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(
      Array.from(bytes.subarray(VOXEL_BIN_DIMS_OFFSET, VOXEL_BIN_HEADER_BYTES)),
    ).toEqual(EXPECTED_DIMS_AND_PALETTE_COUNT);
    expect(loadVoxelBin(bytes).quads.length).toBe(FIXTURE_QUADS.length);
  });

  it('writes palette-with-glow entries as 5-byte records', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(
      Array.from(bytes.subarray(PALETTE_OFFSET, QUAD_COUNT_OFFSET)),
    ).toEqual(EXPECTED_PALETTE_BYTES);
  });

  it('writes the quad count and 12-byte quad records', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(bytes.length).toBe(QUAD_COUNT_OFFSET + EXPECTED_QUAD_BYTES.length);
    expect(Array.from(bytes.subarray(QUAD_COUNT_OFFSET))).toEqual(
      EXPECTED_QUAD_BYTES,
    );
  });

  it('writes identical bytes across repeated calls', () => {
    expect(Array.from(writeVoxelBin(FIXTURE_MODEL))).toEqual(
      Array.from(writeVoxelBin(FIXTURE_MODEL)),
    );
  });

  it('rejects dims components outside the u16 layout with VoxelBinFieldError', () => {
    expect(() => writeVoxelBin({ ...FIXTURE_MODEL, dims: [0, 3, 2] })).toThrow(
      VoxelBinFieldError,
    );
    expect(() =>
      writeVoxelBin({ ...FIXTURE_MODEL, dims: [4, VOXEL_BIN_U16_MAX + 1, 2] }),
    ).toThrow(VoxelBinFieldError);
    expect(() =>
      writeVoxelBin({ ...FIXTURE_MODEL, dims: [4, 1.5, 2] }),
    ).toThrow(VoxelBinFieldError);
  });

  it('rejects empty and oversized palettes with VoxelBinFieldError', () => {
    expect(() => writeVoxelBin({ ...FIXTURE_MODEL, palette: [] })).toThrow(
      VoxelBinFieldError,
    );
    const oversizedPalette = Array.from(
      { length: VOXEL_BIN_U16_MAX + 1 },
      () => EMPTY_PALETTE_ENTRY,
    );
    expect(() =>
      writeVoxelBin({ ...FIXTURE_MODEL, palette: oversizedPalette }),
    ).toThrow(VoxelBinFieldError);
  });

  it('rejects palette components outside the byte range with VoxelBinFieldError', () => {
    const malformedPalette: VoxelPaletteEntry[] = [
      EMPTY_PALETTE_ENTRY,
      [MALFORMED_PALETTE_COMPONENT, 0, 0, 255, 0],
      FIXTURE_PALETTE[2],
    ];
    expect(() =>
      writeVoxelBin({ ...FIXTURE_MODEL, palette: malformedPalette }),
    ).toThrow(VoxelBinFieldError);
  });

  it('rejects quad positions outside the u16 layout with VoxelBinFieldError', () => {
    expect(() =>
      writeVoxelBin(modelWithQuad({ position: [-1, 0, 1] })),
    ).toThrow(VoxelBinFieldError);
    expect(() =>
      writeVoxelBin(modelWithQuad({ position: [VOXEL_BIN_U16_MAX + 1, 0, 1] })),
    ).toThrow(VoxelBinFieldError);
  });

  it('rejects invalid quad normal axes with VoxelBinFieldError', () => {
    expect(() =>
      writeVoxelBin(
        modelWithQuad({ normalAxis: VOXEL_BIN_NORMAL_AXIS_MAX + 1 }),
      ),
    ).toThrow(VoxelBinFieldError);
    expect(() => writeVoxelBin(modelWithQuad({ normalAxis: 0.5 }))).toThrow(
      VoxelBinFieldError,
    );
  });

  it('rejects non-positive quad extents with VoxelBinFieldError', () => {
    expect(() => writeVoxelBin(modelWithQuad({ width: 0 }))).toThrow(
      VoxelBinFieldError,
    );
    expect(() => writeVoxelBin(modelWithQuad({ height: 0 }))).toThrow(
      VoxelBinFieldError,
    );
  });

  it('rejects quad palette indices outside the palette with VoxelBinFieldError', () => {
    expect(() =>
      writeVoxelBin(modelWithQuad({ paletteIndex: FIXTURE_PALETTE.length })),
    ).toThrow(VoxelBinFieldError);
  });
});

describe('loadVoxelBin', () => {
  it('parses writer output into an equivalent model', () => {
    expect(loadVoxelBin(writeVoxelBin(FIXTURE_MODEL))).toEqual(FIXTURE_MODEL);
  });

  it('round-trips byte-identically through write, load, and write', () => {
    const firstBytes = writeVoxelBin(FIXTURE_MODEL);
    const reWritten = writeVoxelBin(loadVoxelBin(firstBytes));
    expect(Array.from(reWritten)).toEqual(Array.from(firstBytes));
  });

  it('round-trips a model with zero quads', () => {
    const zeroQuadModel: VoxelBinModel = { ...FIXTURE_MODEL, quads: [] };
    expect(loadVoxelBin(writeVoxelBin(zeroQuadModel))).toEqual(zeroQuadModel);
  });

  it('loads bins from subarray views with a non-zero byteOffset', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    const padded = new Uint8Array(bytes.length + 1);
    padded.set(bytes, 1);
    expect(loadVoxelBin(padded.subarray(1, 1 + bytes.length))).toEqual(
      FIXTURE_MODEL,
    );
  });

  it('rejects wrong magic bytes with VoxelBinMagicError', () => {
    const corrupted = structuredClone(writeVoxelBin(FIXTURE_MODEL));
    corrupted[VOXEL_BIN_MAGIC_OFFSET] = CORRUPTED_FIRST_MAGIC_BYTE;
    expect(() => loadVoxelBin(corrupted)).toThrow(VoxelBinMagicError);
  });

  it('rejects unknown versions with VoxelBinVersionError', () => {
    const corrupted = structuredClone(writeVoxelBin(FIXTURE_MODEL));
    new DataView(corrupted.buffer).setUint32(
      VOXEL_BIN_VERSION_OFFSET,
      CORRUPTED_VERSION,
      true,
    );
    expect(() => loadVoxelBin(corrupted)).toThrow(VoxelBinVersionError);
  });

  it('rejects a truncated header with VoxelBinTruncatedError', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(() =>
      loadVoxelBin(bytes.slice(0, VOXEL_BIN_HEADER_BYTES - 1)),
    ).toThrow(VoxelBinTruncatedError);
  });

  it('rejects a truncated palette section with VoxelBinTruncatedError', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(() => loadVoxelBin(bytes.slice(0, QUAD_COUNT_OFFSET - 1))).toThrow(
      VoxelBinTruncatedError,
    );
  });

  it('rejects a truncated quad count with VoxelBinTruncatedError', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(() => loadVoxelBin(bytes.slice(0, QUADS_OFFSET - 1))).toThrow(
      VoxelBinTruncatedError,
    );
  });

  it('rejects a truncated quad record with VoxelBinTruncatedError', () => {
    const bytes = writeVoxelBin(FIXTURE_MODEL);
    expect(() => loadVoxelBin(bytes.slice(0, bytes.length - 1))).toThrow(
      VoxelBinTruncatedError,
    );
  });

  it('rejects an out-of-range read-back normal axis with VoxelBinFieldError', () => {
    const corrupted = structuredClone(writeVoxelBin(FIXTURE_MODEL));
    corrupted[QUADS_OFFSET + VOXEL_BIN_QUAD_NORMAL_AXIS_OFFSET] =
      VOXEL_BIN_NORMAL_AXIS_MAX + 1;
    expect(() => loadVoxelBin(corrupted)).toThrow(VoxelBinFieldError);
  });

  it('rejects out-of-range read-back quad palette indices with VoxelBinFieldError', () => {
    const corrupted = structuredClone(writeVoxelBin(FIXTURE_MODEL));
    corrupted[QUADS_OFFSET + VOXEL_BIN_QUAD_PALETTE_INDEX_OFFSET] =
      FIXTURE_PALETTE.length;
    expect(() => loadVoxelBin(corrupted)).toThrow(VoxelBinFieldError);
  });
});
