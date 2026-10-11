/**
 * Decode-contract tests for the Neat Racing voxel model schema and decoder
 * (slice P1-S3-A; AC-007 section 7.2 decode rules + AC-008 mirror-X
 * symmetry).
 *
 * Green-only TDD per the step packet: these tests pin the normative
 * section 7.2 behavior — RLE hex expansion, row-0-is-ground storage order,
 * the black-is-solid exception, header-sized typed arrays, declared-dims
 * acceptance including the 32-cubed wheel grid (spec 7.4/TP-14), mirror-X
 * symmetry, and rejection of malformed inputs with the named schema errors.
 */
import { describe, expect, it } from '@jest/globals';
import {
  VOXEL_GRID_DIMS,
  VoxelModelDimsError,
  VoxelModelMetadataError,
  VoxelModelPaletteError,
  VoxelRowHexError,
  VoxelRowPaletteIndexError,
  VoxelRowSumError,
  validateVoxelModelDocument,
  type VoxelModelDocument,
  type VoxelModelMeta,
  type VoxelPaletteEntry,
} from './model.schema';
import { decodeVoxelModel, decodeVoxelRow } from './model.decode.utils';

/** Palette indices referenced by the fixture documents. */
const PALETTE_INDEX_EMPTY = 0;
const PALETTE_INDEX_RED = 1;
const PALETTE_INDEX_ACCENT = 2;
const PALETTE_INDEX_BLACK_GLOW = 3;
const PALETTE_INDEX_GLASS = 4;

/**
 * Fixture palette per section 7.2: EMPTY sentinel, solid red, glowing blue
 * accent, black-with-glow (the black-is-solid exception), and fully
 * transparent glow glass.
 */
const FIXTURE_PALETTE: VoxelPaletteEntry[] = [
  [0, 0, 0, 0, 0],
  [255, 0, 0, 255, 0],
  [0, 0, 255, 255, 40],
  [0, 0, 0, 255, 200],
  [200, 230, 255, 0, 180],
];

/** Spec section 7.2 example row: 10 voxels of palette index 3, then 64 empty. */
const SPEC_EXAMPLE_ROW = '0a034000';

/** Total cells covered by the spec example row (10 + 64). */
const SPEC_EXAMPLE_ROW_WIDTH = 74;

/** Palette index used by the spec example row. */
const SPEC_EXAMPLE_PALETTE_INDEX = 3;

/**
 * Builds one RLE hex row from `[runLength, paletteIndex]` pairs using the
 * `<runLen:2hex><paletteIdx:2hex>` encoding of section 7.2.
 *
 * @param runs - The run pairs in row order.
 * @returns The encoded RLE hex row string.
 */
function rleRow(
  runs: ReadonlyArray<readonly [runLength: number, paletteIndex: number]>,
): string {
  return runs
    .map(([runLength, paletteIndex]) => {
      const runHex = runLength.toString(16).padStart(2, '0');
      const indexHex = paletteIndex.toString(16).padStart(2, '0');
      return `${runHex}${indexHex}`;
    })
    .join('');
}

/** Full-width row of 128 EMPTY cells (`'8000'` in the spec example). */
const FULL_EMPTY_ROW = rleRow([[VOXEL_GRID_DIMS, PALETTE_INDEX_EMPTY]]);

/** Stored half-width row of 64 EMPTY cells for symmetric documents. */
const HALF_EMPTY_ROW = rleRow([[VOXEL_GRID_DIMS / 2, PALETTE_INDEX_EMPTY]]);

/** Wheel grid dims per spec 7.4/TP-14 (wheels default to a 32-cubed grid). */
const WHEEL_GRID_DIMS = 32;

/** Full-width row of 32 EMPTY cells for wheel documents. */
const WHEEL_EMPTY_ROW = rleRow([[WHEEL_GRID_DIMS, PALETTE_INDEX_EMPTY]]);

/**
 * Creates a valid fixture document whose every row is empty, with the given
 * symmetry flag; tests corrupt individual fields to drive rejection paths.
 *
 * @param symmetric - Whether the document stores 64-col half-rows.
 * @returns A valid fixture document.
 */
function createDocument(symmetric: boolean): VoxelModelDocument {
  const emptyRow = symmetric ? HALF_EMPTY_ROW : FULL_EMPTY_ROW;
  return {
    meta: {
      name: 'fixture-car',
      dims: [VOXEL_GRID_DIMS, VOXEL_GRID_DIMS, VOXEL_GRID_DIMS],
      symmetric,
    },
    palette: FIXTURE_PALETTE,
    slices: Array.from({ length: VOXEL_GRID_DIMS }, () =>
      Array.from({ length: VOXEL_GRID_DIMS }, () => emptyRow),
    ),
  };
}

/** Valid symmetric fixture (stored 64-col half-rows). */
function createSymmetricDocument(): VoxelModelDocument {
  return createDocument(true);
}

/** Valid full-width fixture (128-col rows, no mirroring). */
function createFullDocument(): VoxelModelDocument {
  return createDocument(false);
}

/**
 * Creates a valid 32-cubed wheel document whose every row is empty, per the
 * wheel format of spec 7.4/TP-14 (same compact JSON format as cars).
 *
 * @returns A valid wheel fixture document.
 */
function createWheelDocument(): VoxelModelDocument {
  return {
    meta: {
      name: 'fixture-wheel',
      dims: [WHEEL_GRID_DIMS, WHEEL_GRID_DIMS, WHEEL_GRID_DIMS],
      symmetric: false,
    },
    palette: FIXTURE_PALETTE,
    slices: Array.from({ length: WHEEL_GRID_DIMS }, () =>
      Array.from({ length: WHEEL_GRID_DIMS }, () => WHEEL_EMPTY_ROW),
    ),
  };
}

/**
 * Returns a copy of the document with one row replaced, keeping the rest of
 * the fixture untouched.
 *
 * @param document - The base document.
 * @param sliceIndex - Depth index of the slice to edit.
 * @param rowIndex - Height row to replace.
 * @param row - The replacement RLE hex row.
 * @returns A shallow copy with the single row replaced.
 */
function withRow(
  document: VoxelModelDocument,
  sliceIndex: number,
  rowIndex: number,
  row: string,
): VoxelModelDocument {
  return {
    ...document,
    slices: document.slices.map((rows, slice) =>
      slice === sliceIndex
        ? rows.map((existingRow, atRow) =>
            atRow === rowIndex ? row : existingRow,
          )
        : rows,
    ),
  };
}

/**
 * The valid fixture parts, fresh per call, for building raw documents.
 *
 * @returns The metadata, palette, and slices of a fresh full-width fixture.
 */
function fixtureParts(): {
  meta: VoxelModelMeta;
  palette: VoxelPaletteEntry[];
  slices: string[][];
} {
  const document = createFullDocument();
  return {
    meta: document.meta,
    palette: document.palette,
    slices: document.slices,
  };
}

/**
 * Builds an untyped document around the valid fixture parts so rejection
 * tests can corrupt individual fields without fighting the document types.
 *
 * @param overrides - Field overrides and top-level fields to drop.
 * @returns The raw untyped document.
 */
function rawDocument(
  overrides: {
    readonly meta?: Record<string, unknown>;
    readonly palette?: unknown;
    readonly slices?: unknown;
    readonly drop?: ReadonlyArray<'meta' | 'palette' | 'slices'>;
  } = {},
): Record<string, unknown> {
  const parts = fixtureParts();
  const document: Record<string, unknown> = {};
  const dropped = overrides.drop ?? [];
  if (!dropped.includes('meta')) {
    document['meta'] = {
      name: parts.meta.name,
      dims: parts.meta.dims,
      symmetric: parts.meta.symmetric,
      ...overrides.meta,
    };
  }
  if (!dropped.includes('palette')) {
    document['palette'] = overrides.palette ?? parts.palette;
  }
  if (!dropped.includes('slices')) {
    document['slices'] = overrides.slices ?? parts.slices;
  }
  return document;
}

describe('model.decode.utils (spec section 7.2; AC-007 + AC-008)', () => {
  describe('validateVoxelModelDocument', () => {
    it('accepts a valid document and returns its validated shape', () => {
      const document = createSymmetricDocument();
      const validated = validateVoxelModelDocument(document);
      expect(validated).toEqual(document);
    });

    it('rejects non-object documents', () => {
      expect(() => validateVoxelModelDocument(null)).toThrow(
        VoxelModelMetadataError,
      );
      expect(() => validateVoxelModelDocument(['nope'])).toThrow(
        VoxelModelMetadataError,
      );
    });

    it('rejects documents missing required top-level fields', () => {
      expect(() =>
        validateVoxelModelDocument(rawDocument({ drop: ['meta'] })),
      ).toThrow(VoxelModelMetadataError);
      expect(() =>
        validateVoxelModelDocument(rawDocument({ drop: ['palette'] })),
      ).toThrow(VoxelModelMetadataError);
      expect(() =>
        validateVoxelModelDocument(rawDocument({ drop: ['slices'] })),
      ).toThrow(VoxelModelMetadataError);
    });

    it('rejects meta without a name and without symmetric', () => {
      expect(() =>
        validateVoxelModelDocument(rawDocument({ meta: { name: undefined } })),
      ).toThrow(VoxelModelMetadataError);
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({ meta: { symmetric: undefined } }),
        ),
      ).toThrow(VoxelModelMetadataError);
    });

    it('rejects dims components that are not positive integers', () => {
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({ meta: { dims: [0, 128, 128] } }),
        ),
      ).toThrow(VoxelModelDimsError);
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({ meta: { dims: [-64, 128, 128] } }),
        ),
      ).toThrow(VoxelModelDimsError);
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({ meta: { dims: [128.5, 128, 128] } }),
        ),
      ).toThrow(VoxelModelDimsError);
    });

    it('rejects odd widths on symmetric documents', () => {
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({ meta: { dims: [63, 128, 128], symmetric: true } }),
        ),
      ).toThrow(VoxelModelDimsError);
    });

    it('accepts a 32-cubed wheel document declared by its header (spec 7.4/TP-14)', () => {
      const validated = validateVoxelModelDocument(createWheelDocument());
      expect(validated.meta.dims).toEqual([
        WHEEL_GRID_DIMS,
        WHEEL_GRID_DIMS,
        WHEEL_GRID_DIMS,
      ]);
    });

    it('rejects slice counts that do not match the declared depth', () => {
      const parts = fixtureParts();
      parts.slices.pop();
      expect(() =>
        validateVoxelModelDocument(rawDocument({ slices: parts.slices })),
      ).toThrow(VoxelModelDimsError);
    });

    it('rejects row counts that do not match the declared height', () => {
      const parts = fixtureParts();
      parts.slices[0].pop();
      expect(() =>
        validateVoxelModelDocument(rawDocument({ slices: parts.slices })),
      ).toThrow(VoxelModelDimsError);
    });

    it('rejects slices that are not arrays of RLE hex strings', () => {
      const sliceNotArray = rawDocument({
        slices: Array.from({ length: VOXEL_GRID_DIMS }, () => 'nope'),
      });
      expect(() => validateVoxelModelDocument(sliceNotArray)).toThrow(
        VoxelModelMetadataError,
      );
      const rowNotString = rawDocument({
        slices: Array.from({ length: VOXEL_GRID_DIMS }, (_value, rowIndex) =>
          rowIndex === 0
            ? 42
            : Array.from({ length: VOXEL_GRID_DIMS }, () => FULL_EMPTY_ROW),
        ),
      });
      expect(() => validateVoxelModelDocument(rowNotString)).toThrow(
        VoxelModelMetadataError,
      );
    });

    it('rejects an empty palette', () => {
      expect(() =>
        validateVoxelModelDocument(rawDocument({ palette: [] })),
      ).toThrow(VoxelModelPaletteError);
    });

    it('rejects malformed palette entries', () => {
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({
            palette: [
              [0, 0, 0, 0],
              [255, 0, 0, 255, 0],
            ],
          }),
        ),
      ).toThrow(VoxelModelPaletteError);
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({ palette: [[0, 0, 0, 0, 0], 'nope'] }),
        ),
      ).toThrow(VoxelModelPaletteError);
    });

    it('rejects palette components outside the 0-255 byte range', () => {
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({
            palette: [
              [0, 0, 0, 0, 0],
              [256, 0, 0, 255, 0],
            ],
          }),
        ),
      ).toThrow(VoxelModelPaletteError);
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({
            palette: [
              [0, 0, 0, 0, 0],
              [-1, 0, 0, 255, 0],
            ],
          }),
        ),
      ).toThrow(VoxelModelPaletteError);
    });

    it('rejects non-integer palette components', () => {
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({
            palette: [
              [0, 0, 0, 0, 0],
              [1.5, 0, 0, 255, 0],
            ],
          }),
        ),
      ).toThrow(VoxelModelPaletteError);
    });

    it('requires palette index 0 to be the EMPTY sentinel', () => {
      expect(() =>
        validateVoxelModelDocument(
          rawDocument({
            palette: [
              [10, 20, 30, 40, 50],
              [255, 0, 0, 255, 0],
            ],
          }),
        ),
      ).toThrow(VoxelModelPaletteError);
    });
  });

  describe('decodeVoxelModel', () => {
    it('decodes a model into a grid sized exactly from the schema header', () => {
      const decoded = decodeVoxelModel(createSymmetricDocument());
      expect(decoded.slices).toHaveLength(decoded.dims[2]);
      expect(decoded.slices[0].length).toBe(decoded.dims[0] * decoded.dims[1]);
      expect(decoded.slices.every((slice) => slice instanceof Uint8Array)).toBe(
        true,
      );
    });

    it('decodes a 32-cubed wheel model into header-sized typed arrays (spec 7.4/TP-14)', () => {
      const decoded = decodeVoxelModel(createWheelDocument());
      expect(decoded.dims).toEqual([
        WHEEL_GRID_DIMS,
        WHEEL_GRID_DIMS,
        WHEEL_GRID_DIMS,
      ]);
      expect(decoded.slices).toHaveLength(WHEEL_GRID_DIMS);
      expect(decoded.slices[0].length).toBe(WHEEL_GRID_DIMS * WHEEL_GRID_DIMS);
    });

    it('echoes the schema header onto the decoded result', () => {
      const decoded = decodeVoxelModel(createSymmetricDocument());
      expect(decoded.name).toBe('fixture-car');
      expect(decoded.symmetric).toBe(true);
      expect(decoded.palette).toEqual(FIXTURE_PALETTE);
    });

    it('expands RLE hex runs into palette indices at the full width', () => {
      const document = withRow(
        createFullDocument(),
        0,
        0,
        rleRow([
          [10, PALETTE_INDEX_ACCENT],
          [118, PALETTE_INDEX_EMPTY],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[0];
      expect(cells[9]).toBe(PALETTE_INDEX_ACCENT);
      expect(cells[10]).toBe(PALETTE_INDEX_EMPTY);
      expect(cells[VOXEL_GRID_DIMS - 1]).toBe(PALETTE_INDEX_EMPTY);
    });

    it('keeps row 0 as ground/down in storage order', () => {
      const document = withRow(
        withRow(
          createSymmetricDocument(),
          3,
          0,
          rleRow([[VOXEL_GRID_DIMS / 2, PALETTE_INDEX_RED]]),
        ),
        3,
        VOXEL_GRID_DIMS - 1,
        rleRow([[VOXEL_GRID_DIMS / 2, PALETTE_INDEX_ACCENT]]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[3];
      expect(cells[0]).toBe(PALETTE_INDEX_RED);
      expect(cells[(VOXEL_GRID_DIMS - 1) * VOXEL_GRID_DIMS]).toBe(
        PALETTE_INDEX_ACCENT,
      );
    });

    it('keeps black glow entries solid per the black-is-solid exception', () => {
      const document = withRow(
        createSymmetricDocument(),
        7,
        5,
        rleRow([
          [5, PALETTE_INDEX_EMPTY],
          [1, PALETTE_INDEX_BLACK_GLOW],
          [58, PALETTE_INDEX_EMPTY],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[7];
      expect(cells[5 * VOXEL_GRID_DIMS + 5]).toBe(PALETTE_INDEX_BLACK_GLOW);
    });

    it('carries fully transparent glow entries through as ordinary palette references', () => {
      const document = withRow(
        createSymmetricDocument(),
        1,
        10,
        rleRow([
          [2, PALETTE_INDEX_GLASS],
          [62, PALETTE_INDEX_EMPTY],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[1];
      expect(cells[10 * VOXEL_GRID_DIMS + 1]).toBe(PALETTE_INDEX_GLASS);
      expect(cells[10 * VOXEL_GRID_DIMS + (VOXEL_GRID_DIMS - 2)]).toBe(
        PALETTE_INDEX_GLASS,
      );
    });

    it('rejects rows whose runs sum to the wrong width', () => {
      const symmetricWithFullRow = withRow(
        createSymmetricDocument(),
        0,
        0,
        FULL_EMPTY_ROW,
      );
      const fullWithHalfRow = withRow(
        createFullDocument(),
        0,
        0,
        HALF_EMPTY_ROW,
      );
      expect(() => decodeVoxelModel(symmetricWithFullRow)).toThrow(
        VoxelRowSumError,
      );
      expect(() => decodeVoxelModel(fullWithHalfRow)).toThrow(VoxelRowSumError);
    });

    it('rejects rows that overshoot a narrower declared width ([64, 128, 128])', () => {
      const narrowWidth = rawDocument({ meta: { dims: [64, 128, 128] } });
      expect(() =>
        decodeVoxelModel(narrowWidth as unknown as VoxelModelDocument),
      ).toThrow(VoxelRowSumError);
    });

    it('rejects invalid hex fields in RLE rows', () => {
      const nonHex = withRow(createFullDocument(), 0, 0, 'zz00'.repeat(32));
      const shortField = withRow(createFullDocument(), 0, 0, '123');
      const zeroRun = withRow(createFullDocument(), 0, 0, '00ff'.repeat(32));
      expect(() => decodeVoxelModel(nonHex)).toThrow(VoxelRowHexError);
      expect(() => decodeVoxelModel(shortField)).toThrow(VoxelRowHexError);
      expect(() => decodeVoxelModel(zeroRun)).toThrow(VoxelRowHexError);
    });

    it('rejects out-of-range palette references', () => {
      const document = createFullDocument();
      document.palette = [
        [0, 0, 0, 0, 0],
        [255, 255, 255, 255, 0],
      ];
      const referencingIndex2 = withRow(
        document,
        0,
        0,
        rleRow([
          [1, 2],
          [127, PALETTE_INDEX_EMPTY],
        ]),
      );
      expect(() => decodeVoxelModel(referencingIndex2)).toThrow(
        VoxelRowPaletteIndexError,
      );
    });

    it('surfaces schema rejections when decoding malformed documents', () => {
      const malformed = rawDocument({ meta: { dims: [0, 128, 128] } });
      expect(() =>
        decodeVoxelModel(malformed as unknown as VoxelModelDocument),
      ).toThrow(VoxelModelDimsError);
    });
  });

  describe('mirror-X symmetry (AC-008)', () => {
    it('mirrors stored cols 0-63 across the full 128-col width', () => {
      const document = withRow(
        createSymmetricDocument(),
        2,
        4,
        rleRow([
          [5, PALETTE_INDEX_EMPTY],
          [1, PALETTE_INDEX_ACCENT],
          [58, PALETTE_INDEX_EMPTY],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[2];
      expect(cells[4 * VOXEL_GRID_DIMS + 5]).toBe(PALETTE_INDEX_ACCENT);
      expect(cells[4 * VOXEL_GRID_DIMS + (VOXEL_GRID_DIMS - 1 - 5)]).toBe(
        PALETTE_INDEX_ACCENT,
      );
      expect(cells[4 * VOXEL_GRID_DIMS + (VOXEL_GRID_DIMS - 1 - 4)]).toBe(
        PALETTE_INDEX_EMPTY,
      );
    });

    it('mirrors across the col-63/col-64 boundary', () => {
      const document = withRow(
        createSymmetricDocument(),
        0,
        0,
        rleRow([
          [63, PALETTE_INDEX_EMPTY],
          [1, PALETTE_INDEX_RED],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[0];
      expect(cells[VOXEL_GRID_DIMS / 2 - 1]).toBe(PALETTE_INDEX_RED);
      expect(cells[VOXEL_GRID_DIMS / 2]).toBe(PALETTE_INDEX_RED);
    });

    it('mirrors col 0 across to col 127', () => {
      const document = withRow(
        createSymmetricDocument(),
        0,
        0,
        rleRow([
          [1, PALETTE_INDEX_RED],
          [63, PALETTE_INDEX_EMPTY],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[0];
      expect(cells[0]).toBe(PALETTE_INDEX_RED);
      expect(cells[VOXEL_GRID_DIMS - 1]).toBe(PALETTE_INDEX_RED);
    });

    it('decodes full-width rows in non-symmetric models without mirroring', () => {
      const document = withRow(
        createFullDocument(),
        5,
        9,
        rleRow([
          [10, PALETTE_INDEX_ACCENT],
          [118, PALETTE_INDEX_EMPTY],
        ]),
      );
      const decoded = decodeVoxelModel(document);
      const cells = decoded.slices[5];
      expect(cells[9 * VOXEL_GRID_DIMS + 9]).toBe(PALETTE_INDEX_ACCENT);
      expect(cells[9 * VOXEL_GRID_DIMS + (VOXEL_GRID_DIMS - 10)]).toBe(
        PALETTE_INDEX_EMPTY,
      );
    });
  });

  describe('decodeVoxelRow', () => {
    it('returns a typed array of exactly the expected width', () => {
      const cells = decodeVoxelRow(
        FULL_EMPTY_ROW,
        VOXEL_GRID_DIMS,
        FIXTURE_PALETTE.length,
      );
      expect(cells.length).toBe(VOXEL_GRID_DIMS);
      expect(cells.every((cell) => cell === PALETTE_INDEX_EMPTY)).toBe(true);
    });

    it('expands the spec example pairs', () => {
      const cells = decodeVoxelRow(
        SPEC_EXAMPLE_ROW,
        SPEC_EXAMPLE_ROW_WIDTH,
        FIXTURE_PALETTE.length,
      );
      expect(cells[9]).toBe(SPEC_EXAMPLE_PALETTE_INDEX);
      expect(cells[10]).toBe(PALETTE_INDEX_EMPTY);
    });
  });
});
