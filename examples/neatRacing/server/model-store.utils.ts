/**
 * File-system model store for the Neat Racing dev server (spec section 8.2,
 * AC-018 support + AC-007 support; slice P1-S5-A).
 *
 * Pure CRUD over `examples/neatRacing/assets/models/*.model.json` with the
 * section 8.2 hardening gates applied at the store boundary:
 *
 * - **S4 model-name shape** — `^[a-z0-9][a-z0-9-]{0,63}$` is checked on the
 *   decoded name *before any path join*, so `/`, `\`, `..`, and `%` can never
 *   reach the filesystem; traversal rejection is structural.
 * - **S5 strict payload keys** — accepted documents carry exactly the
 *   `meta`, `palette`, and `slices` top-level keys; anything else (including
 *   non-object payloads) is rejected with `model-store/top-level-keys`
 *   before a single byte is written.
 * - **Atomic save + S6 backup rotation** — writes go to a
 *   `<name>.model.json.tmp` sibling and are swapped in with a same-directory
 *   rename (atomic on both Windows and POSIX), then the replaced document is
 *   rotated through `<modelsDir>/.backups/<name>.model.json.bak1..3` so the
 *   last three versions survive an editor crash.
 * - **Section 7.2 schema gate** — every document read or written is validated
 *   by `voxel/model.schema.ts` and row-decoded by
 *   `voxel/model.decode.utils.ts`, so on-disk payloads round-trip the exact
 *   schema contract (AC-007) without ever mutating the caller's objects.
 *
 * Storage symmetry note: mirror-X symmetric documents store half-widths
 * (dims[0] / 2 hex columns per row); the store is format-agnostic about this
 * and delegates the width arithmetic to the row decoder.
 */
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import {
  validateVoxelModelDocument,
  VoxelModelError,
  type VoxelModelDocument,
} from '../voxel/model.schema';
import { decodeVoxelModel } from '../voxel/model.decode.utils';

/** Reserved file suffix for Neat Racing model documents (section 10.2). */
export const NEATRACING_MODEL_FILE_SUFFIX = '.model.json';

/**
 * S4 model-name shape: lowercase alphanumerics with inner hyphens, one
 * leading character, at most 64 characters total. The pattern cannot match
 * path separators, `%`, or leading dots/dashes, so name checks performed
 * before any `join` reject traversal payloads structurally (no `realpath`
 * dance needed).
 */
export const NEATRACING_MODEL_NAME_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;

/** Number of replaced versions kept per model in the backup directory (S6). */
export const NEATRACING_MODEL_STORE_BACKUP_COUNT = 3;

/** Directory (inside the models directory) holding rotated backup copies. */
export const NEATRACING_MODEL_STORE_BACKUP_DIR_NAME = '.backups';

/** Suffix appended to the model file while an atomic save is in flight. */
export const NEATRACING_MODEL_STORE_TEMP_SUFFIX = '.tmp';

/**
 * Stable store rejection codes, namespaced under `model-store/` to keep them
 * distinguishable from the `voxel-model/…` schema codes they complement.
 */
export type ModelStoreErrorCode =
  | 'model-store/name'
  | 'model-store/not-found'
  | 'model-store/serialize'
  | 'model-store/top-level-keys';

/** Strict top-level key whitelist for stored documents (S5). */
const REQUIRED_TOP_LEVEL_KEYS: readonly string[] = [
  'meta',
  'palette',
  'slices',
];

/**
 * Named store rejection: name-shape violations, missing files, serialization
 * failures, or S5 key-set violations. Extends `VoxelModelError` so server
 * handlers can catch the shared base class and forward `error.code` as an
 * HTTP status hint.
 */
export class ModelStoreError extends VoxelModelError {
  /**
   * @param code - Stable machine-readable store rejection code.
   * @param message - Human-readable rejection reason.
   */
  public constructor(code: ModelStoreErrorCode, message: string) {
    super(code, message);
  }
}

/**
 * Reports whether `name` satisfies the S4 model-name shape.
 *
 * Accepts `unknown` so dev-server handlers can gate untrusted input without
 * pre-narrowing; non-strings (including `undefined` from URL parsing) are
 * rejected, never thrown.
 *
 * @param name - Candidate model name from any source.
 * @returns `true` exactly when the name is store-safe.
 */
export function isVoxelModelName(name: unknown): boolean {
  return typeof name === 'string' && NEATRACING_MODEL_NAME_PATTERN.test(name);
}

/**
 * Builds the on-disk document path for a validated model name.
 *
 * The name is re-checked here (instead of trusting callers) because this is
 * also the join boundary: by the time any path is assembled, the name is
 * proven incapable of escaping the models directory.
 *
 * @param modelsDir - Directory holding the stored `*.model.json` documents.
 * @param name - Candidate model name (validated against the S4 shape).
 * @returns The absolute path of the model document.
 * @throws ModelStoreError with code `model-store/name` when the name fails
 * the S4 shape.
 */
export function resolveVoxelModelFilePath(
  modelsDir: string,
  name: unknown,
): string {
  if (!isVoxelModelName(name)) {
    throw new ModelStoreError(
      'model-store/name',
      `model name must satisfy ${NEATRACING_MODEL_NAME_PATTERN.toString()}, received: ${JSON.stringify(name)}`,
    );
  }
  return join(modelsDir, `${name as string}${NEATRACING_MODEL_FILE_SUFFIX}`);
}

/**
 * Lists the stored model names, sorted, ignoring foreign files.
 *
 * Only regular files whose stem satisfies the S4 shape are reported; helper
 * files (`*.json` payloads, temp files), symlinks (whose `Dirent` entries are
 * not `isFile()`), and subdirectories (including `.backups/`) are skipped.
 * A missing directory is not an error — an empty install simply has no
 * models.
 *
 * @param modelsDir - Directory holding the stored `*.model.json` documents.
 * @returns Valid model names in lexicographic order; `[]` when the directory
 * is missing or empty.
 */
export function listVoxelModelNames(modelsDir: string): string[] {
  let entries;
  try {
    entries = readdirSync(modelsDir, { withFileTypes: true });
  } catch (error) {
    if (
      error != null &&
      typeof error === 'object' &&
      'code' in error &&
      ((error as NodeJS.ErrnoException).code === 'ENOENT' ||
        (error as NodeJS.ErrnoException).code === 'ENOTDIR')
    ) {
      return [];
    }
    throw error;
  }
  const suffixLength = NEATRACING_MODEL_FILE_SUFFIX.length;
  return entries
    .filter(
      (entry) =>
        entry.isFile() &&
        entry.name.length > suffixLength &&
        entry.name.endsWith(NEATRACING_MODEL_FILE_SUFFIX),
    )
    .map((entry) => entry.name.slice(0, entry.name.length - suffixLength))
    .filter((name) => isVoxelModelName(name))
    .toSorted();
}

/**
 * Validates an untrusted payload against the S5 strict key set before the
 * section 7.2 schema gate runs.
 *
 * Non-records (including arrays) and payloads holding any key beyond the
 * strict `meta`/`palette`/`slices` enumeration are rejected up front so the
 * store never persists editor-forged envelopes. Schema and row-level errors
 * are then delegated to the shared validators and propagate unchanged.
 *
 * @param candidate - Untrusted JSON value, for example a PUT request body.
 * @returns The same document, narrowed to the section 7.2 schema type; the
 * input object is returned (not a clone) and never mutated.
 * @throws ModelStoreError with code `model-store/top-level-keys` when the
 * payload is not a plain record or carries unknown top-level keys.
 * @throws VoxelModelMetadataError, VoxelModelDimsError,
 * VoxelModelPaletteError, VoxelRowHexError, VoxelRowSumError, or
 * VoxelRowPaletteIndexError for section 7.2 schema or row violations.
 */
export function validateVoxelModelPayload(
  candidate: unknown,
): VoxelModelDocument {
  const isRecord =
    typeof candidate === 'object' &&
    candidate !== null &&
    !Array.isArray(candidate);
  if (!isRecord) {
    throw new ModelStoreError(
      'model-store/top-level-keys',
      `expected an object with keys ${REQUIRED_TOP_LEVEL_KEYS.join(', ')}, received: ${describePayload(candidate)}`,
    );
  }
  const payload = candidate as Record<string, unknown>;
  const foreignKeys = Object.keys(payload).filter(
    (key) => !REQUIRED_TOP_LEVEL_KEYS.includes(key),
  );
  if (foreignKeys.length > 0) {
    throw new ModelStoreError(
      'model-store/top-level-keys',
      `unknown top-level keys: ${foreignKeys.join(', ')}; expected only ${REQUIRED_TOP_LEVEL_KEYS.join(', ')}`,
    );
  }
  const document = validateVoxelModelDocument(payload);
  // Row-level gate: decode forces the full RLE/hex/palette walk so malformed
  // rows cannot be persisted even when the schema shape alone parses.
  decodeVoxelModel(document);
  return document;
}

/**
 * Reads and validates one stored model document.
 *
 * Order of gates: S4 name shape (`model-store/name`), file presence
 * (`model-store/not-found`), JSON parseability and schema/row validity
 * (section 7.2 errors propagate). The stored bytes are never modified.
 *
 * @param modelsDir - Directory holding the stored `*.model.json` documents.
 * @param name - Candidate model name (untrusted; shape-checked first).
 * @returns The validated section 7.2 document.
 * @throws ModelStoreError with code `model-store/name` or
 * `model-store/not-found`.
 * @throws Section 7.2 schema/row errors for corrupt stored payloads.
 */
export function readVoxelModelDocument(
  modelsDir: string,
  name: unknown,
): VoxelModelDocument {
  const filePath = resolveVoxelModelFilePath(modelsDir, name);
  let serialized;
  try {
    serialized = readFileSync(filePath, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new ModelStoreError(
        'model-store/not-found',
        `model not found: ${name as string}`,
      );
    }
    throw error;
  }
  let candidate;
  try {
    candidate = JSON.parse(serialized) as unknown;
  } catch (error) {
    throw new ModelStoreError(
      'model-store/serialize',
      `stored model is not valid JSON (${name as string}): ${(error as Error).message}`,
    );
  }
  return validateVoxelModelPayload(candidate) as VoxelModelDocument;
}

/**
 * Atomically saves a model document, rotating the previous version into the
 * S6 backup ring.
 *
 * Validation runs entirely before the models directory is touched, so
 * rejected payloads leave any stored document and the `.backups/` ring
 * untouched. The serialized document is written to a `.tmp` sibling and
 * swapped in with a same-directory rename (replace-existing on Windows,
 * POSIX `rename`), which is atomic against concurrent readers; only after a
 * successful swap is the pre-captured previous content rotated through
 * `bak1..3`, oldest first, so `bak1` always holds the immediately replaced
 * version. A first save (no previous content) creates no backup directory.
 *
 * @param modelsDir - Directory holding the stored `*.model.json` documents;
 * created on demand.
 * @param name - Candidate model name (untrusted; shape-checked first).
 * @param candidate - Untrusted JSON payload; never mutated on success or
 * rejection.
 * @throws ModelStoreError with code `model-store/name` or
 * `model-store/top-level-keys`.
 * @throws Section 7.2 schema/row errors for malformed payloads.
 * @throws The underlying file-system error when the swap itself fails (the
 * temp file is removed first).
 */
export function writeVoxelModelDocument(
  modelsDir: string,
  name: unknown,
  candidate: unknown,
): void {
  const document = validateVoxelModelPayload(candidate);
  const filePath = resolveVoxelModelFilePath(modelsDir, name);
  mkdirSync(modelsDir, { recursive: true });
  const previousContent = readStoredBytes(filePath);
  const serialized = `${JSON.stringify(document)}\n`;
  const tempPath = `${filePath}${NEATRACING_MODEL_STORE_TEMP_SUFFIX}`;
  writeFileSync(tempPath, serialized, 'utf8');
  try {
    renameSync(tempPath, filePath);
  } catch (error) {
    removeQuietly(tempPath);
    throw error;
  }
  if (previousContent !== undefined) {
    rotateModelBackups(modelsDir, name as string, previousContent);
  }
}

/**
 * Deletes one stored model document.
 *
 * Backup copies are intentionally kept: they are the crash-recovery net for
 * the editor and are only consumed by a human restore.
 *
 * @param modelsDir - Directory holding the stored `*.model.json` documents.
 * @param name - Candidate model name (untrusted; shape-checked first).
 * @throws ModelStoreError with code `model-store/name` or
 * `model-store/not-found` when the name is hostile or the file is missing.
 */
export function deleteVoxelModel(modelsDir: string, name: unknown): void {
  const filePath = resolveVoxelModelFilePath(modelsDir, name);
  try {
    unlinkSync(filePath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      throw new ModelStoreError(
        'model-store/not-found',
        `model not found: ${name as string}`,
      );
    }
    throw error;
  }
}

/**
 * Reads the stored bytes for pre-backup snapshotting.
 *
 * @param filePath - The model document path.
 * @returns The previous file content, or `undefined` when no document is
 * stored yet.
 */
function readStoredBytes(filePath: string): string | undefined {
  try {
    return readFileSync(filePath, 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      return undefined;
    }
    throw error;
  }
}

/**
 * Rotates the replaced document through the S6 backup ring.
 *
 * Shifts oldest-first (`bak2` into `bak3`, then `bak1` into `bak2`) so each
 * `rename` lands in a free slot, then writes the replaced content as the new
 * `bak1`. Renames within the same directory are atomic; a rename onto an
 * existing slot replaces it, which is exactly how the 4th-oldest version is
 * dropped once the ring is full.
 *
 * @param modelsDir - Directory holding the stored `*.model.json` documents.
 * @param name - Validated model name.
 * @param replacedContent - The pre-capture content of the replaced document.
 */
function rotateModelBackups(
  modelsDir: string,
  name: string,
  replacedContent: string,
): void {
  const backupDir = join(modelsDir, NEATRACING_MODEL_STORE_BACKUP_DIR_NAME);
  mkdirSync(backupDir, { recursive: true });
  const backupPath = (slot: number): string =>
    join(
      backupDir,
      `${name}${NEATRACING_MODEL_FILE_SUFFIX}.bak${String(slot)}`,
    );
  for (
    let slot = NEATRACING_MODEL_STORE_BACKUP_COUNT - 1;
    slot >= 1;
    slot -= 1
  ) {
    const source = backupPath(slot);
    if (existsSync(source)) {
      renameSync(source, backupPath(slot + 1));
    }
  }
  writeFileSync(backupPath(1), replacedContent, 'utf8');
}

/**
 * Removes a temp file, tolerating its absence.
 *
 * @param filePath - The temp file path.
 */
function removeQuietly(filePath: string): void {
  try {
    unlinkSync(filePath);
  } catch {
    // Removing an already-gone temp file is not a store failure.
  }
}

/**
 * Renders an untrusted payload for rejection messages without revealing more
 * than its JSON kind.
 *
 * @param candidate - The rejected payload.
 * @returns A short, deterministic description.
 */
function describePayload(candidate: unknown): string {
  if (candidate === null) {
    return 'null';
  }
  if (Array.isArray(candidate)) {
    return 'array';
  }
  return typeof candidate;
}
