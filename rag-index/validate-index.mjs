/**
 * @description Assert the health of `rag-index/data/turso-replica.sqlite`: minimum document and
 * chunk row counts, per-document freshness proof validity, and a per-family deletion sweep that
 * removes indexed rows for files no longer on disk. Emits the standard gate JSON contract
 * `{ ok, pass, documents, chunks, failures }` plus a per-family freshness map and writes the
 * per-family freshness manifest atomically. When a family has indexed files missing from disk
 * and its manifest `lastReindex` is older than the family's configured `max_age_ms`, a
 * warning-only deletion-sweep sanity signal is emitted on `warnings` (never a failure).
 *
 * @param {boolean} [--json]                - Emit JSON validation result.
 * @param {number}  [--min-documents <n>]   - Minimum expected document rows (default: 1).
 * @param {number}  [--min-chunks <n>]      - Minimum expected chunk rows (default: 1).
 * @param {string}  [--database <path>]     - Path to the SQLite database file (default: `rag-index/data/turso-replica.sqlite`).
 * @param {boolean} [--help]                - Show help and exit.
 *
 * @returns {void} Exits 0 when the index is healthy, 1 when any assertion fails.
 */
import { createClient } from '@libsql/client';
import fg from 'fast-glob';
import { existsSync, readFileSync } from 'node:fs';
import { rename, writeFile } from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  toRepoRelative,
  writeJsonOrText,
} from './cli-utils.mjs';
import { getFreshnessProof, isFreshDocument } from './freshness.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const DEFAULT_MIN_DOCUMENTS = 1;
const DEFAULT_MIN_CHUNKS = 1;
const REBUILD_COMMAND = 'node rag-index/build-index.mjs';
const MANIFEST_PATH = path.join(
  repoRoot,
  'rag-index',
  'data',
  'freshness-manifest.json',
);
const PLAN_MAX_SYNC_WAIT_MS = 60_000;

const DEFAULT_FAMILY_GATING = {
  readme: true,
  'ts-source': true,
  plan: true,
  demo: true,
  benchmark: true,
  'root-doc': true,
  skill: true,
  agent: true,
  'copilot-instructions': true,
  'completed-plan': false,
};

const DEFAULT_FAMILY_MAX_SYNC_WAIT_MS = {
  readme: 0,
  'ts-source': 0,
  plan: PLAN_MAX_SYNC_WAIT_MS,
  demo: 0,
  benchmark: 0,
  'root-doc': 0,
  'completed-plan': 0,
};

/**
 * Per-family `max_age_ms` sanity-signal defaults. `null` disables the signal
 * for a family; operators may configure a positive per-family value directly
 * in `freshness-manifest.json`, which the validator preserves across writes.
 * The signal is warning-only and applies to deleted/missing files only.
 */
const DEFAULT_FAMILY_MAX_AGE_MS = {
  readme: null,
  'ts-source': null,
  plan: null,
  demo: null,
  benchmark: null,
  'root-doc': null,
  skill: null,
  agent: null,
  'copilot-instructions': null,
  'completed-plan': null,
};
const STALE_FIX_HINT = `Stale paths detected. Run: ${REBUILD_COMMAND}`;
const MISSING_FIX_HINT = `Missing paths detected. Run: ${REBUILD_COMMAND}`;
const GENERIC_FIX_HINT = `Run: ${REBUILD_COMMAND}`;

const CORPUS_SOURCES = [
  { family: 'readme', patterns: ['src/**/README.md'] },
  {
    family: 'ts-source',
    patterns: ['src/**/*.ts'],
    ignore: ['src/**/*.d.ts', 'src/**/*.test.ts', 'src/**/*.spec.ts'],
  },
  {
    family: 'plan',
    patterns: ['plans/**/*.md'],
    ignore: ['plans/completed/**'],
  },
  { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
  { family: 'demo', patterns: ['examples/**/README.md', 'examples/**/*.ts'] },
  {
    family: 'benchmark',
    patterns: ['benchmarks/README.md', 'benchmarks/**/*.test.ts'],
  },
  {
    family: 'root-doc',
    patterns: ['README.md', 'STYLEGUIDE.md', 'CONTRIBUTING.md'],
  },
  {
    family: 'skill',
    patterns: ['.github/skills/**/SKILL.md'],
  },
  {
    family: 'agent',
    patterns: ['.github/agents/*.agent.md'],
  },
  {
    family: 'copilot-instructions',
    patterns: ['.github/copilot-instructions.md'],
  },
];

export async function discoverFamilyAssignments(sources = CORPUS_SOURCES) {
  const assignments = new Map();
  for (const source of sources) {
    const familyPaths = await discoverFamilyPaths(source);
    for (const filePath of familyPaths) {
      assignments.set(normalizeRepoPath(filePath), source.family);
    }
  }
  return assignments;
}

export function buildFamilyFresh(
  documents,
  freshnessChecks,
  familyAssignments = new Map(),
) {
  const familyFresh = {};
  for (const source of CORPUS_SOURCES) {
    familyFresh[source.family] = { fresh: true, stalePaths: [] };
  }

  const freshnessByPath = new Map(
    freshnessChecks.map((proof) => [proof.file_path, proof]),
  );

  for (const documentRow of documents) {
    const normalizedPath = normalizeRepoPath(documentRow.file_path);
    const family = familyAssignments.get(normalizedPath);
    if (!family || !familyFresh[family]) continue;

    const freshnessProof = freshnessByPath.get(documentRow.file_path);
    if (freshnessProof?.missing === true) {
      familyFresh[family].fresh = false;
      pushUnique(familyFresh[family].stalePaths, documentRow.file_path);
      continue;
    }

    if (freshnessProof && !isFreshDocument(documentRow, freshnessProof)) {
      familyFresh[family].fresh = false;
      pushUnique(familyFresh[family].stalePaths, documentRow.file_path);
    }
  }

  for (const source of CORPUS_SOURCES) {
    familyFresh[source.family].stalePaths.sort();
  }

  return familyFresh;
}

export function resolveGatedPass(
  familyFresh,
  gating = DEFAULT_FAMILY_GATING,
) {
  return Object.entries(gating).every(
    ([family, gated]) => !gated || familyFresh[family]?.fresh !== false,
  );
}

/**
 * Load and parse the per-family freshness manifest, tolerating absence,
 * corruption, and in-flight atomic writes.
 *
 * The validator is the single manifest writer (plan risk R7); this loader
 * surfaces operator-configured per-family `max_age_ms` budgets and the
 * per-family `lastReindex` timestamps recorded by the previous validation run.
 *
 * @param {string} [manifestPath] - Path to `freshness-manifest.json`
 *   (default: `rag-index/data/freshness-manifest.json`).
 * @returns {Object|null} The parsed manifest object, or `null` when the file
 *   is absent, unreadable, corrupt, or not a JSON object.
 */
export function loadFreshnessManifest(manifestPath = MANIFEST_PATH) {
  try {
    if (!existsSync(manifestPath)) return null;
    const parsed = JSON.parse(readFileSync(manifestPath, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * Evaluate the per-family `max_age_ms` deletion-sweep sanity signal.
 *
 * Secondary, warning-only signal for deleted/missing files: a family with
 * indexed files missing from disk whose manifest `lastReindex` is older than
 * its configured `max_age_ms` produces a warning suggesting a deletion sweep.
 * Families without missing files are never flagged, so the Phase 0 removal of
 * the age gate for unchanged files is not re-introduced. Warnings never affect
 * `pass`, `failures`, or `fixHint`.
 *
 * @param {Object} input - Evaluation inputs.
 * @param {string[]} input.missingPaths - Repo paths missing from disk (sorted).
 * @param {Map<string, string>} input.familyAssignments - Normalized repo path
 *   to family map.
 * @param {Object|null} [input.manifest] - Previously written freshness manifest.
 * @param {number} [input.now] - Evaluation clock in ms epoch; defaults to Date.now().
 * @returns {string[]} Deterministic (sorted) warning messages; empty when the
 *   signal is disabled or the manifest is absent/corrupt.
 */
function evaluateMaxAgeSanityWarnings({
  missingPaths,
  familyAssignments,
  manifest,
  now = Date.now(),
}) {
  const warnings = [];
  if (!manifest || typeof manifest !== 'object') return warnings;
  const families = manifest.families;
  if (!families || typeof families !== 'object') return warnings;

  const missingByFamily = new Map();
  for (const missingPath of missingPaths ?? []) {
    const family = familyAssignments.get(normalizeRepoPath(missingPath));
    if (!family) continue;
    if (!missingByFamily.has(family)) missingByFamily.set(family, []);
    missingByFamily.get(family).push(missingPath);
  }

  for (const [family, paths] of missingByFamily) {
    const maxAgeMs = families[family]?.max_age_ms;
    if (!Number.isFinite(maxAgeMs) || maxAgeMs <= 0) continue;
    const lastReindex = families[family]?.lastReindex;
    if (!Number.isFinite(lastReindex)) continue;
    const ageMs = now - lastReindex;
    if (ageMs <= maxAgeMs) continue;
    warnings.push(
      buildDeletionSweepWarning(family, paths.length, ageMs, maxAgeMs),
    );
  }

  return warnings.sort();
}

/**
 * Build the deterministic deletion-sweep warning message for one family.
 *
 * @param {string} family - Family with missing files and a stale manifest entry.
 * @param {number} missingCount - Number of indexed files missing from disk.
 * @param {number} ageMs - Age of the family manifest `lastReindex` in ms.
 * @param {number} maxAgeMs - Configured per-family `max_age_ms` budget in ms.
 * @returns {string} Warning message including the rebuild command hint.
 */
function buildDeletionSweepWarning(family, missingCount, ageMs, maxAgeMs) {
  return (
    `Deletion sweep suggested for family '${family}': ${missingCount} indexed ` +
    `file(s) missing from disk and manifest lastReindex age ${ageMs}ms ` +
    `exceeds max_age_ms ${maxAgeMs}. ${GENERIC_FIX_HINT}`
  );
}

/**
 * Resolve the per-family `max_age_ms` value written to the manifest.
 *
 * Operator-configured positive finite values are preserved across validator
 * writes; anything else (absent, null, zero, negative, non-numeric) normalizes
 * to `null`, which disables the sanity signal for the family.
 *
 * @param {Object|null} previousManifest - Previously written manifest.
 * @param {string} family - Family name.
 * @returns {number|null} Configured budget in ms, or `null` when disabled.
 */
function resolveConfiguredMaxAgeMs(previousManifest, family) {
  const value = previousManifest?.families?.[family]?.max_age_ms;
  return Number.isFinite(value) && value > 0 ? value : null;
}

export async function validateSemanticIndex(input) {
  input = /* istanbul ignore next -- defensive: input always provided in tests */ input ?? {};
  const documents = input.documents ?? [];
  const freshnessChecks = input.freshnessChecks ?? [];
  const minDocuments = Number(input.minDocuments ?? DEFAULT_MIN_DOCUMENTS);
  const minChunks = Number(input.minChunks ?? 0);
  const chunks = Number(input.chunks ?? documents.length);
  const familyAssignments =
    input.familyAssignments ?? /* istanbul ignore next -- assignments provided by validateDatabase */ new Map();
  const now = Number.isFinite(Number(input.now)) ? Number(input.now) : Date.now();
  const manifest = input.manifest ?? null;
  const failures = [];
  const stalePaths = [];
  const missingPaths = [];

  if (documents.length < minDocuments)
    failures.push(
      `Expected at least ${minDocuments} documents; found ${documents.length}.`,
    );
  if (chunks < minChunks)
    failures.push(`Expected at least ${minChunks} chunks; found ${chunks}.`);

  const familyFresh = buildFamilyFresh(
    documents,
    freshnessChecks,
    familyAssignments,
  );

  const freshnessByPath = new Map(
    freshnessChecks.map((proof) => [proof.file_path, proof]),
  );
  for (const documentRow of documents) {
    const normalizedPath = normalizeRepoPath(documentRow.file_path);
    const family = familyAssignments.get(normalizedPath);
    const freshnessProof = freshnessByPath.get(documentRow.file_path);

    if (freshnessProof?.missing === true) {
      pushUnique(stalePaths, documentRow.file_path);
      pushUnique(missingPaths, documentRow.file_path);
      if (family !== 'completed-plan') {
        failures.push(`Indexed file is missing: ${documentRow.file_path}.`);
      }
      continue;
    }

    if (freshnessProof && !isFreshDocument(documentRow, freshnessProof)) {
      pushUnique(stalePaths, documentRow.file_path);
      if (family !== 'completed-plan') {
        failures.push(`Stale freshness proof for ${documentRow.file_path}.`);
      }
    }
  }

  stalePaths.sort();
  missingPaths.sort();

  const warnings = evaluateMaxAgeSanityWarnings({
    missingPaths,
    familyAssignments,
    manifest,
    now,
  });

  return createValidationResult({
    failures,
    documents: documents.length,
    chunks,
    stalePaths,
    missingPaths,
    familyFresh,
    warnings,
  });
}

export async function validateDatabase(options) {
  options = /* istanbul ignore next -- defensive: options always provided in tests */ options ?? {};
  const familyAssignments =
    options.familyAssignments ??
    /* istanbul ignore next -- assignments discovered from disk in real runs */ await discoverFamilyAssignments();
  const previousManifest =
    options.manifest ??
    /* istanbul ignore next -- manifest read from disk in real runs */ loadFreshnessManifest();

  if (options.client) {
    await sweepDeletedPathsForAllFamilies(options.client);

    const docsResult = await options.client.execute({
      sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path',
      args: [],
    });
    const chunkCountResult = await options.client.execute({
      sql: 'SELECT COUNT(*) AS count FROM chunks',
      args: [],
    });
    const documents = docsResult.rows;
    const chunkCount = Number(chunkCountResult.rows[0].count);

    const freshnessChecks = await Promise.all(
      documents.map(async (documentRow) => {
        if (!isWithinRepoRoot(documentRow.file_path)) {
          return {
            file_path: documentRow.file_path,
            missing: true,
          };
        }

        const absolutePath = path.join(repoRoot, documentRow.file_path);

        try {
          return {
            file_path: documentRow.file_path,
            ...(await getFreshnessProof(absolutePath)),
          };
        } catch (error) {
          /* istanbul ignore next -- defensive: error always has expected shape in test fixtures */
          if (
            error &&
            typeof error === 'object' &&
            'code' in error &&
            error.code === 'ENOENT'
          ) {
            return {
              file_path: documentRow.file_path,
              missing: true,
            };
          }

          /* istanbul ignore next -- defensive: non-ENOENT errors from getFreshnessProof are re-thrown */
          throw error;
        }
      }),
    );

    return validateSemanticIndex({
      documents,
      freshnessChecks,
      chunks: chunkCount,
      minDocuments: options.minDocuments,
      minChunks: options.minChunks,
      familyAssignments,
      manifest: previousManifest,
    });
  }

  const databasePath = path.resolve(
    /* istanbul ignore next -- defensive: databasePath always provided in test calls */ (options.databasePath ?? defaultDatabasePath),
  );
  /* istanbul ignore else -- defensive: database not found path tested; real DB path not testable */
  if (!existsSync(databasePath)) {
    return createValidationResult({
      failures: [`Database not found: ${databasePath}`],
      documents: 0,
      chunks: 0,
    });
  }

  // Real database path — covered by client-injected tests above
  /* istanbul ignore next -- requires real SQLite database file, not suitable for unit tests */
  const database = createClient({ url: pathToFileURL(databasePath).href });

  /* istanbul ignore next -- requires real SQLite database */
  await sweepDeletedPathsForAllFamilies(database);

  /* istanbul ignore next -- requires real SQLite database */
  const docsResult = await database.execute({
    sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path',
    args: [],
  });
  /* istanbul ignore next -- requires real SQLite database */
  const chunkCountResult = await database.execute({
    sql: 'SELECT COUNT(*) AS count FROM chunks',
    args: [],
  });
  /* istanbul ignore next -- requires real SQLite database */
  const documents = docsResult.rows;
  /* istanbul ignore next -- requires real SQLite database */
  const chunkCount = Number(chunkCountResult.rows[0].count);

  /* istanbul ignore next -- requires real SQLite database */
  const freshnessChecks = await Promise.all(
    documents.map(async (documentRow) => {
      if (!isWithinRepoRoot(documentRow.file_path)) {
        return {
          file_path: documentRow.file_path,
          missing: true,
        };
      }

      const absolutePath = path.join(repoRoot, documentRow.file_path);

      try {
        return {
          file_path: documentRow.file_path,
          ...(await getFreshnessProof(absolutePath)),
        };
      } catch (error) {
        if (
          error &&
          typeof error === 'object' &&
          'code' in error &&
          error.code === 'ENOENT'
        ) {
          return {
            file_path: documentRow.file_path,
            missing: true,
          };
        }

        throw error;
      }
    }),
  );

  /* istanbul ignore next -- requires real SQLite database */
  const result = await validateSemanticIndex({
    documents,
    freshnessChecks,
    chunks: chunkCount,
    minDocuments: options.minDocuments,
    minChunks: options.minChunks,
    familyAssignments,
    manifest: previousManifest,
  });

  /* istanbul ignore next -- requires real SQLite database */
  await database.close();

  /* istanbul ignore next -- requires real SQLite database */
  await writeFreshnessManifest(
    result.family_fresh,
    MANIFEST_PATH,
    'validate-index.mjs',
    previousManifest,
  );

  /* istanbul ignore next -- requires real SQLite database */
  return result;
}

export async function sweepDeletedPathsForAllFamilies(client) {
  const deletedByFamily = [];
  for (const source of CORPUS_SOURCES) {
    const deleted = await sweepDeletedPaths(client, source.family);
    if (deleted.length > 0) deletedByFamily.push({ family: source.family, deleted });
  }
  return deletedByFamily;
}

export async function sweepDeletedPaths(client, family) {
  const source = CORPUS_SOURCES.find((s) => s.family === family);
  if (!source) return [];

  const diskPaths = await discoverFamilyPaths(source);
  const normalizedDisk = new Set(
    diskPaths.map(normalizeRepoPath).filter((p) => isWithinRepoRoot(p)),
  );

  const docsResult = await client.execute({
    sql: 'SELECT file_path FROM documents WHERE doc_family = ?',
    args: [family],
  });

  const toDeleteOriginal = [];
  const toDeleteNormalized = [];
  for (const row of docsResult.rows) {
    const storedPath = row.file_path;
    if (!isWithinRepoRoot(storedPath)) continue;
    const normalized = normalizeRepoPath(storedPath);
    if (!normalizedDisk.has(normalized)) {
      toDeleteOriginal.push(storedPath);
      toDeleteNormalized.push(normalized);
    }
  }

  if (toDeleteOriginal.length > 0) {
    if (typeof client.batch === 'function') {
      await client.batch(
        toDeleteOriginal.map((filePath) => ({
          sql: 'DELETE FROM documents WHERE file_path = ?',
          args: [filePath],
        })),
        'write',
      );
    } else {
      for (const filePath of toDeleteOriginal) {
        await client.execute({
          sql: 'DELETE FROM documents WHERE file_path = ?',
          args: [filePath],
        });
      }
    }
  }

  return toDeleteNormalized;
}

export async function discoverFamilyPaths(source) {
  return fg(source.patterns, {
    cwd: repoRoot,
    onlyFiles: true,
    ignore: source.ignore ?? [],
  });
}

function normalizeRepoPath(filePath) {
  return toRepoRelative(filePath)
    .replace(/\\+/g, '/')
    .replace(/\/+/g, '/');
}

function isWithinRepoRoot(filePath) {
  const resolved = path.resolve(repoRoot, filePath);
  const relative = path.relative(repoRoot, resolved);
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function createValidationResult({
  failures,
  documents,
  chunks,
  stalePaths = [],
  missingPaths = [],
  familyFresh,
  warnings = [],
}) {
  const family_fresh =
    familyFresh ?? buildEmptyFamilyFresh();
  const gatedPass = resolveGatedPass(family_fresh);
  const pass = failures.length === 0 && gatedPass;

  return {
    ok: pass,
    pass,
    failures,
    documents,
    chunks,
    stale_paths: stalePaths,
    missing_paths: missingPaths,
    warnings,
    family_fresh,
    fixHint: pass
      ? null
      : resolveFixHint({ stalePaths, missingPaths, failures }),
  };
}

function buildEmptyFamilyFresh() {
  const familyFresh = {};
  for (const source of CORPUS_SOURCES) {
    familyFresh[source.family] = { fresh: true, stalePaths: [] };
  }
  return familyFresh;
}

function resolveFixHint({ stalePaths, missingPaths, failures }) {
  if (missingPaths.length > 0) return MISSING_FIX_HINT;
  if (stalePaths.length > 0) return STALE_FIX_HINT;
  return /* istanbul ignore next -- defensive: failures.length > 0 is always true when this branch is reached */ (failures.length > 0 ? GENERIC_FIX_HINT : null);
}

/**
 * Write the per-family freshness manifest atomically (write-then-rename).
 *
 * Each family entry gains `max_age_ms`: operator-configured positive finite
 * values are preserved from `previousManifest`; anything else normalizes to
 * `null` (disabled). The validator is the single manifest writer (plan risk R7).
 *
 * @param {Object} familyFresh - Per-family freshness map from validation.
 * @param {string} [outputPath] - Manifest output path (default MANIFEST_PATH).
 * @param {string} [updatedBy] - Writer label recorded in the manifest.
 * @param {Object|null} [previousManifest] - Previously loaded manifest whose
 *   operator-configured `max_age_ms` budgets are preserved across the write.
 * @returns {Promise<Object>} The written manifest object.
 */
export async function writeFreshnessManifest(
  familyFresh,
  outputPath = MANIFEST_PATH,
  updatedBy = 'validate-index.mjs',
  previousManifest = null,
) {
  const now = Date.now();
  const families = {};
  for (const source of CORPUS_SOURCES) {
    const entry = familyFresh[source.family] ?? { fresh: true, stalePaths: [] };
    families[source.family] = {
      fresh: entry.fresh,
      stalePaths: [...entry.stalePaths].sort(),
      lastReindex: now,
      maxSyncWaitMs: DEFAULT_FAMILY_MAX_SYNC_WAIT_MS[source.family] ?? 0,
      max_age_ms:
        resolveConfiguredMaxAgeMs(previousManifest, source.family) ??
        DEFAULT_FAMILY_MAX_AGE_MS[source.family] ??
        null,
      gated: DEFAULT_FAMILY_GATING[source.family] ?? true,
    };
  }
  const manifest = {
    lastReindex: now,
    updatedBy,
    families,
  };

  const parentDir = path.dirname(outputPath);
  const randomSuffix = randomBytes(8).toString('hex');
  const tempPath = path.join(parentDir, `freshness-manifest.${randomSuffix}.tmp`);

  await writeFile(tempPath, JSON.stringify(manifest, null, 2));
  await rename(tempPath, outputPath);
  return manifest;
}

function pushUnique(values, value) {
  /* istanbul ignore else -- defensive: values only contains unique entries in test fixtures */
  if (!values.includes(value)) values.push(value);
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic index validator',
      usage:
        'node rag-index/validate-index.mjs [--json] [--min-documents 1] [--min-chunks 1]',
      options: [
        '--json                Emit JSON validation result',
        '--min-documents <n>   Minimum expected document rows (default: 1)',
        '--min-chunks <n>      Minimum expected chunk rows (default: 1)',
        '--database <path>     Path to SQLite database file (default: rag-index/data/turso-replica.sqlite)',
        '--help                Show this help',
      ],
    });
    return;
  }

  try {
    const result = await validateDatabase({
      minDocuments: args['min-documents'],
      minChunks: args['min-chunks'] ?? DEFAULT_MIN_CHUNKS,
      databasePath: args.database,
    });
    writeJsonOrText(result, Boolean(args.json), /* istanbul ignore next -- text formatter covered when writeJsonOrText is not mocked */ (payload) =>
      payload.pass
        ? `Semantic index valid: ${payload.documents} documents, ${payload.chunks} chunks`
        : `Semantic index invalid: ${payload.failures.join('; ')}`,
    );
    /* istanbul ignore else -- defensive: result.pass is always true in test invocations */
    if (!result.pass) process.exitCode = 1;
  } catch (error) {
    /* istanbul ignore next -- defensive: validateDatabase handles known errors, catch only for unexpected throws */
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

/* istanbul ignore next -- CLI entry point guard */
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
