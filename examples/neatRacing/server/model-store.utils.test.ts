/**
 * Store-contract tests for the Neat Racing model store (spec section 8.2,
 * AC-018 support + AC-007 support; slice P1-S5-A).
 *
 * RED contract per the step packet: these tests fail before the store exists
 * and pin the normative section 8.2 behavior — the S4 model-name shape
 * (`^[a-z0-9][a-z0-9-]{0,63}$`, checked before any path join so traversal
 * rejection is structural), strict S5 top-level key enumeration, the
 * section 7.2 schema gate delegated to `voxel/model.schema.ts` and
 * `voxel/model.decode.utils.ts`, atomic temp+rename saves, the S6 last-3
 * backup rotation (`model.json.bak1..bak3` inside a `.backups/` directory),
 * CRUD round-trips that never mutate the caller's payload, listing, and
 * deletion.
 *
 * Filesystem fixtures use tiny 2-cubed documents (the schema accepts any
 * declared dims, section 7.2) so every case is fast and deterministic inside
 * a per-test temp directory under the OS temp root.
 */
import { afterEach, describe, expect, it } from '@jest/globals';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  isVoxelModelName,
  listVoxelModelNames,
  deleteVoxelModel,
  readVoxelModelDocument,
  writeVoxelModelDocument,
  validateVoxelModelPayload,
  ModelStoreError,
  NEATRACING_MODEL_FILE_SUFFIX,
  NEATRACING_MODEL_STORE_BACKUP_DIR_NAME,
  NEATRACING_MODEL_STORE_BACKUP_COUNT,
  NEATRACING_MODEL_STORE_TEMP_SUFFIX,
} from './model-store.utils';
import {
  VoxelModelDimsError,
  VoxelModelPaletteError,
  VoxelModelMetadataError,
  VoxelRowHexError,
  VoxelRowPaletteIndexError,
  VoxelRowSumError,
} from '../voxel/model.schema';

/** Name list accepted by the S4 shape: lowercase, digits, hyphens. */
const ACCEPTED_NAMES: readonly string[] = ['car-f1', 'a', '0-x-9', 'wheel'];

/** Names that the S4 shape must reject structurally (no separators, no `..`, no `%`). */
const HOSTILE_NAMES: readonly string[] = [
  '../evil',
  'x/y',
  'x\\y',
  '%2e%2e',
  'Car',
  '.hidden',
  '-leading',
  'under_score',
  'x.y',
  '',
];

/** Longest name the S4 pattern permits: `1 + 63 = 64` characters. */
const NAME_LENGTH_MAX = 64;

/** Fixture dims: 2-cubed grid keeps row strings short and decodes fast. */
const FIXTURE_DIMS: readonly [number, number, number] = [2, 2, 2];

/** Full-width empty row for a 2-wide document: run 2 of palette index 0. */
const EMPTY_ROW_2 = '0200';

/** Full-width colored row for a 2-wide document: run 2 of palette index 1. */
const COLORED_ROW_2 = '0201';

/** Stored half-row (symmetric mirror-X, 2-wide grid stores 1 col) empty. */
const EMPTY_STOROW_1 = '0100';

/** Stored half-row (symmetric, 1 stored col) colored with palette index 1. */
const COLORED_STOROW_1 = '0101';

/** Per-test temp model directories created under one shared temp root. */
let tempRoot = '';

/**
 * Creates a fresh empty models directory for one test.
 *
 * @returns The absolute models directory path.
 */
function freshModelsDir(): string {
  return mkdtempSync(join(tempRoot, 'models-'));
}

/**
 * Builds a minimal valid 2-cubed document with every row fully empty.
 *
 * @param overridden - Field overrides merged over the base document.
 * @returns A schema-valid full-width document fixture.
 */
function makeEmptyDocument(
  overridden: Record<string, unknown> = {},
): Record<string, unknown> {
  const document: Record<string, unknown> = {
    meta: { name: 'car-f1', dims: FIXTURE_DIMS, symmetric: false },
    palette: [
      [0, 0, 0, 0, 0],
      [200, 230, 255, 255, 64],
    ],
    slices: [
      [EMPTY_ROW_2, EMPTY_ROW_2],
      [EMPTY_ROW_2, EMPTY_ROW_2],
    ],
    ...overridden,
  };
  return document;
}

/**
 * Resolves the on-disk file path a store would use for a model name.
 *
 * @param modelsDir - The models directory.
 * @param name - The model name.
 * @returns The absolute model document file path.
 */
function modelFilePath(modelsDir: string, name: string): string {
  return join(modelsDir, `${name}${NEATRACING_MODEL_FILE_SUFFIX}`);
}

/**
 * Resolves one backup slot path inside the `.backups/` directory.
 *
 * @param modelsDir - The models directory.
 * @param name - The model name.
 * @param slot - The backup slot (1 = most recent replaced version).
 * @returns The absolute backup file path.
 */
function backupSlotPath(modelsDir: string, name: string, slot: number): string {
  return join(
    modelsDir,
    NEATRACING_MODEL_STORE_BACKUP_DIR_NAME,
    `${name}${NEATRACING_MODEL_FILE_SUFFIX}.bak${slot}`,
  );
}

/**
 * Runs a store call expecting a named rejection and returns its code.
 *
 * Non-`ModelStoreError` failures are rethrown so genuine regressions do not
 * masquerade as expected rejections.
 *
 * @param run - The store call expected to reject.
 * @returns The rejection code, or `undefined` when the call did not reject.
 */
function captureModelStoreErrorCode(run: () => unknown): string | undefined {
  try {
    run();
    return undefined;
  } catch (error) {
    if (error instanceof ModelStoreError) {
      return error.code;
    }
    throw error;
  }
}

/**
 * Reads and parses a JSON document from disk.
 *
 * @param filePath - The JSON file path.
 * @returns The parsed JSON value.
 */
function readJson(filePath: string): unknown {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

describe('S4 model-name shape (before any join)', () => {
  it('accepts lowercase names with digits and hyphens', () => {
    expect(ACCEPTED_NAMES.every((name) => isVoxelModelName(name))).toBe(true);
  });

  it('rejects traversal and hostile name shapes structurally', () => {
    expect(HOSTILE_NAMES.every((name) => !isVoxelModelName(name))).toBe(true);
  });

  it('accepts the 64-character boundary name', () => {
    expect(isVoxelModelName('a'.repeat(NAME_LENGTH_MAX))).toBe(true);
  });

  it('rejects a 65-character name', () => {
    expect(isVoxelModelName('a'.repeat(NAME_LENGTH_MAX + 1))).toBe(false);
  });

  it('rejects non-string names', () => {
    expect(
      [undefined, null, 42].every(
        (name) => !isVoxelModelName(name as unknown as string),
      ),
    ).toBe(true);
  });
});

describe('CRUD list + read', () => {
  it('lists nothing for an absent or empty models directory', () => {
    const modelsDir = freshModelsDir();
    expect(listVoxelModelNames(modelsDir)).toStrictEqual([]);
    expect(
      listVoxelModelNames(join(modelsDir, 'does-not-exist')),
    ).toStrictEqual([]);
  });

  it('lists only valid model files sorted by name', () => {
    const modelsDir = freshModelsDir();
    writeFileSync(modelFilePath(modelsDir, 'b-model'), '{}');
    writeFileSync(modelFilePath(modelsDir, 'a-model'), '{}');
    writeFileSync(join(modelsDir, 'ignore.json'), '{}');
    writeFileSync(join(modelsDir, 'BadName.model.json'), '{}');
    expect(listVoxelModelNames(modelsDir)).toStrictEqual([
      'a-model',
      'b-model',
    ]);
  });

  it('round-trips a stored document without mutating the payload', () => {
    const modelsDir = freshModelsDir();
    const payload = makeEmptyDocument();
    const snapshot = structuredClone(payload);
    writeVoxelModelDocument(modelsDir, 'car-f1', payload);
    const stored = readVoxelModelDocument(modelsDir, 'car-f1');
    expect(stored).toEqual(payload);
    expect(payload).toEqual(snapshot);
  });

  it('reports a missing model read as not-found', () => {
    const modelsDir = freshModelsDir();
    expect(
      captureModelStoreErrorCode(() =>
        readVoxelModelDocument(modelsDir, 'ghost'),
      ),
    ).toBe('model-store/not-found');
  });

  it('rejects traversal-shaped names before any join on read', () => {
    const modelsDir = freshModelsDir();
    expect(
      captureModelStoreErrorCode(() =>
        readVoxelModelDocument(modelsDir, '../evil'),
      ),
    ).toBe('model-store/name');
  });
});

describe('atomic write + S6 backup rotation', () => {
  it('saves a new model with no temp-file residue', () => {
    const modelsDir = freshModelsDir();
    writeVoxelModelDocument(modelsDir, 'car-f1', makeEmptyDocument());
    expect(
      existsSync(
        `${modelFilePath(modelsDir, 'car-f1')}${NEATRACING_MODEL_STORE_TEMP_SUFFIX}`,
      ),
    ).toBe(false);
    expect(listVoxelModelNames(modelsDir)).toStrictEqual(['car-f1']);
  });

  it('preserves the replaced version in backup slot 1', () => {
    const modelsDir = freshModelsDir();
    const first = makeEmptyDocument({
      slices: [
        [COLORED_ROW_2, EMPTY_ROW_2],
        [EMPTY_ROW_2, EMPTY_ROW_2],
      ],
    });
    const second = makeEmptyDocument();
    writeVoxelModelDocument(modelsDir, 'car-f1', first);
    writeVoxelModelDocument(modelsDir, 'car-f1', second);
    expect(readJson(backupSlotPath(modelsDir, 'car-f1', 1))).toEqual(first);
    expect(readJson(modelFilePath(modelsDir, 'car-f1'))).toEqual(second);
  });

  it('keeps the last-3 rotation order so the oldest version drops', () => {
    const modelsDir = freshModelsDir();
    const versions = [
      makeEmptyDocument({
        slices: [
          [COLORED_ROW_2, EMPTY_ROW_2],
          [EMPTY_ROW_2, EMPTY_ROW_2],
        ],
      }),
      makeEmptyDocument({
        slices: [
          [EMPTY_ROW_2, COLORED_ROW_2],
          [EMPTY_ROW_2, EMPTY_ROW_2],
        ],
      }),
      makeEmptyDocument({
        slices: [
          [COLORED_ROW_2, COLORED_ROW_2],
          [EMPTY_ROW_2, EMPTY_ROW_2],
        ],
      }),
      makeEmptyDocument(),
    ];
    for (const version of versions) {
      writeVoxelModelDocument(modelsDir, 'car-f1', version);
    }
    expect(readJson(backupSlotPath(modelsDir, 'car-f1', 1))).toEqual(
      versions[2],
    );
    expect(readJson(backupSlotPath(modelsDir, 'car-f1', 2))).toEqual(
      versions[1],
    );
    expect(readJson(backupSlotPath(modelsDir, 'car-f1', 3))).toEqual(
      versions[0],
    );
  });

  it('holds exactly the named backup count on disk across saves', () => {
    const modelsDir = freshModelsDir();
    const versions = [
      makeEmptyDocument({
        slices: [
          [COLORED_ROW_2, EMPTY_ROW_2],
          [EMPTY_ROW_2, EMPTY_ROW_2],
        ],
      }),
      makeEmptyDocument({
        slices: [
          [EMPTY_ROW_2, COLORED_ROW_2],
          [EMPTY_ROW_2, EMPTY_ROW_2],
        ],
      }),
      makeEmptyDocument({
        slices: [
          [COLORED_ROW_2, COLORED_ROW_2],
          [EMPTY_ROW_2, EMPTY_ROW_2],
        ],
      }),
      makeEmptyDocument(),
    ];
    expect(NEATRACING_MODEL_STORE_BACKUP_COUNT).toBe(3);
    writeVoxelModelDocument(modelsDir, 'car-f1', makeEmptyDocument());
    for (const version of versions) {
      writeVoxelModelDocument(modelsDir, 'car-f1', version);
    }
    expect(readJson(modelFilePath(modelsDir, 'car-f1'))).toEqual(
      versions.at(-1),
    );
    const backupDir = join(modelsDir, NEATRACING_MODEL_STORE_BACKUP_DIR_NAME);
    expect(readdirSync(backupDir).sort()).toStrictEqual([
      `car-f1${NEATRACING_MODEL_FILE_SUFFIX}.bak1`,
      `car-f1${NEATRACING_MODEL_FILE_SUFFIX}.bak2`,
      `car-f1${NEATRACING_MODEL_FILE_SUFFIX}.bak3`,
    ]);
  });

  it('creates no backups for a first save', () => {
    const modelsDir = freshModelsDir();
    writeVoxelModelDocument(modelsDir, 'car-f1', makeEmptyDocument());
    expect(
      existsSync(join(modelsDir, NEATRACING_MODEL_STORE_BACKUP_DIR_NAME)),
    ).toBe(false);
    expect(
      captureModelStoreErrorCode(() =>
        readVoxelModelDocument(modelsDir, 'car-f1'),
      ),
    ).toBeUndefined();
  });

  it('rejects unknown top-level keys without writing (S5)', () => {
    const modelsDir = freshModelsDir();
    const original = makeEmptyDocument();
    writeVoxelModelDocument(modelsDir, 'car-f1', original);
    const forged = makeEmptyDocument({ evil: true });
    expect(
      captureModelStoreErrorCode(() =>
        writeVoxelModelDocument(modelsDir, 'car-f1', forged),
      ),
    ).toBe('model-store/top-level-keys');
    expect(readVoxelModelDocument(modelsDir, 'car-f1')).toEqual(original);
  });

  it('rejects rows whose runs do not sum to the declared width', () => {
    const modelsDir = freshModelsDir();
    const malformed = makeEmptyDocument({
      slices: [
        ['0101', EMPTY_ROW_2],
        [EMPTY_ROW_2, EMPTY_ROW_2],
      ],
    });
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', malformed),
    ).toThrow(VoxelRowSumError);
  });

  it('rejects rows with invalid hex characters', () => {
    const modelsDir = freshModelsDir();
    const malformed = makeEmptyDocument({
      slices: [
        ['zz00', EMPTY_ROW_2],
        [EMPTY_ROW_2, EMPTY_ROW_2],
      ],
    });
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', malformed),
    ).toThrow(VoxelRowHexError);
  });

  it('rejects palette indices outside the document palette', () => {
    const modelsDir = freshModelsDir();
    const malformed = makeEmptyDocument({
      slices: [
        ['0209', EMPTY_ROW_2],
        [EMPTY_ROW_2, EMPTY_ROW_2],
      ],
    });
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', malformed),
    ).toThrow(VoxelRowPaletteIndexError);
  });

  it('rejects slice counts contradicting meta.dims', () => {
    const modelsDir = freshModelsDir();
    const malformed = makeEmptyDocument({
      slices: [[EMPTY_ROW_2, EMPTY_ROW_2, EMPTY_ROW_2]],
    });
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', malformed),
    ).toThrow(VoxelModelDimsError);
  });

  it('rejects malformed palette entries without writing', () => {
    const modelsDir = freshModelsDir();
    const malformed = makeEmptyDocument({
      palette: [
        [0, 0, 0, 0, 0],
        [999, 0, 0, 0, 0],
      ],
    });
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', malformed),
    ).toThrow(VoxelModelPaletteError);
  });

  it('rejects documents without metadata without writing', () => {
    const modelsDir = freshModelsDir();
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', { palette: [], slices: [] }),
    ).toThrow(VoxelModelMetadataError);
  });

  it('never mutates a frozen payload during validation or save', () => {
    const modelsDir = freshModelsDir();
    const payload = Object.freeze(makeEmptyDocument());
    const snapshot = structuredClone(payload as Record<string, unknown>);
    expect(() =>
      writeVoxelModelDocument(modelsDir, 'car-f1', payload),
    ).not.toThrow();
    expect(payload).toEqual(snapshot);
    expect(readVoxelModelDocument(modelsDir, 'car-f1')).toEqual(snapshot);
  });
});

describe('symmetric stored half-width payloads', () => {
  it('stores and reads symmetric half-width documents without mutation', () => {
    const modelsDir = freshModelsDir();
    const payload = makeEmptyDocument({
      meta: { name: 'car-f1', dims: FIXTURE_DIMS, symmetric: true },
      slices: [
        [COLORED_STOROW_1, EMPTY_STOROW_1],
        [EMPTY_STOROW_1, EMPTY_STOROW_1],
      ],
    });
    const snapshot = structuredClone(payload);
    writeVoxelModelDocument(modelsDir, 'car-f1', payload);
    expect(readVoxelModelDocument(modelsDir, 'car-f1')).toEqual(payload);
    expect(payload).toEqual(snapshot);
  });
});

describe('delete', () => {
  it('removes the stored model file', () => {
    const modelsDir = freshModelsDir();
    writeVoxelModelDocument(modelsDir, 'car-f1', makeEmptyDocument());
    deleteVoxelModel(modelsDir, 'car-f1');
    expect(
      captureModelStoreErrorCode(() =>
        readVoxelModelDocument(modelsDir, 'car-f1'),
      ),
    ).toBe('model-store/not-found');
  });

  it('reports deleting a missing model as not-found', () => {
    const modelsDir = freshModelsDir();
    expect(
      captureModelStoreErrorCode(() => deleteVoxelModel(modelsDir, 'ghost')),
    ).toBe('model-store/not-found');
  });
});

describe('store validation gate (no save)', () => {
  it('returns the validated document for schema-valid payloads', () => {
    const payload = makeEmptyDocument();
    expect(validateVoxelModelPayload(payload)).toEqual(payload);
  });

  it('rejects non-object payloads with the strict-key error', () => {
    expect(
      [null, 'text', 3, []].every(
        (candidate) =>
          captureModelStoreErrorCode(() =>
            validateVoxelModelPayload(candidate),
          ) === 'model-store/top-level-keys',
      ),
    ).toBe(true);
  });

  it('rejects payloads with an unknown top-level key', () => {
    const forged = makeEmptyDocument({ extra: true });
    expect(
      captureModelStoreErrorCode(() => validateVoxelModelPayload(forged)),
    ).toBe('model-store/top-level-keys');
  });

  it('accepts symmetric half-width documents for the validate route', () => {
    const payload = makeEmptyDocument({
      meta: { name: 'car-f1', dims: FIXTURE_DIMS, symmetric: true },
      slices: [
        [COLORED_STOROW_1, EMPTY_STOROW_1],
        [EMPTY_STOROW_1, EMPTY_STOROW_1],
      ],
    });
    expect(validateVoxelModelPayload(payload)).toEqual(payload);
  });
});

// Durable temp root: created once at import, wiped and recreated per test.
tempRoot = mkdtempSync(join(tmpdir(), 'neatracing-model-store-'));

afterEach(() => {
  rmSync(tempRoot, { recursive: true, force: true });
  mkdirSync(tempRoot, { recursive: true });
});
