/**
 * @description Lightweight session-start semantic-index refresh. Prevents the 24-hour
 * `indexed_at` staleness gate from drifting red for content that has not changed since
 * the last build. Runs in two passes:
 *
 * 1. **Touch pass** — for every document whose on-disk content is still fresh
 *    (mtime_ms, file_size, and sha256 all match the stored row) update `indexed_at`
 *    to NOW() with a single SQL UPDATE. No re-chunking, no FTS rewrite.
 * 2. **Incremental build pass** — delegates to `build-index.mjs` (without `--force`)
 *    so that content-changed or newly-added corpus files are indexed normally.
 *    Unchanged files are skipped by the build's freshness-proof check.
 *
 * Safe to run at session start every day; the touch pass is O(n) SQL reads + 1 batch
 * UPDATE, and the build pass is O(changed files) only.
 *
 * @param {boolean} [--json]             - Emit JSON summary.
 * @param {boolean} [--touch-only]       - Run only the touch pass; skip the build pass.
 * @param {string}  [--database <path>]  - Path to the SQLite database file (default: `data/turso-replica.sqlite`).
 * @param {boolean} [--help]             - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on fatal error.
 *
 * @example
 * ```sh
 * # recommended: run once at the start of every session
 * npm run index:session-start
 * ```
 */
import { createClient } from '@libsql/client';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseCliArgs, writeJsonOrText } from './cli-utils.mjs';
import { getFreshnessProof } from './freshness.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const SCRIPT_NAME = 'session-start-index.mjs';
const BUILD_INDEX_PATH = path.join(
  repoRoot,
  'scripts',
  'semantic-index',
  'build-index.mjs',
);

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

if (
  import.meta.url === pathToFileURL(process.argv[1]).href ||
  process.argv[1]?.endsWith(SCRIPT_NAME)
) {
  const args = parseCliArgs(process.argv.slice(2));

  if (args['help']) {
    printUsage();
    process.exit(0);
  }

  runSessionStartIndex({
    json: Boolean(args['json']),
    touchOnly: Boolean(args['touch-only']),
    databasePath: args['database'] ?? undefined,
  })
    .then((summary) => {
      writeJsonOrText(summary, Boolean(args['json']), formatSummaryText);
      process.exit(summary.fatalError ? 1 : 0);
    })
    .catch((error) => {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      if (args['json']) {
        console.log(
          JSON.stringify(
            { pass: false, ok: false, error: errorMessage },
            null,
            2,
          ),
        );
      } else {
        console.error(`session-start-index: fatal error: ${errorMessage}`);
      }
      process.exit(1);
    });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run the session-start index refresh.
 *
 * @param {object} [options]
 * @param {boolean} [options.touchOnly]    - Skip the build pass.
 * @param {string}  [options.databasePath] - Override the database path.
 * @param {boolean} [options.json]         - Emit JSON output (unused in library mode).
 * @returns {Promise<SessionStartSummary>}
 */
export async function runSessionStartIndex(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const startMs = Date.now();

  const summary = {
    databasePath,
    touched: 0,
    contentChanged: 0,
    onDiskMissing: 0,
    buildPassRan: false,
    buildPassExitCode: null,
    elapsedMs: 0,
    fatalError: null,
  };

  if (!options.client && !existsSync(databasePath)) {
    summary.fatalError = `Database not found: ${databasePath}. Run npm run index:build first.`;
    summary.elapsedMs = Date.now() - startMs;
    return summary;
  }

  // Step 1: Touch pass — bump indexed_at for content-fresh rows.
  const touchResult = await runTouchPass(
    options.client ? { client: options.client } : databasePath,
  );
  summary.touched = touchResult.touched;
  summary.contentChanged = touchResult.contentChanged;
  summary.onDiskMissing = touchResult.onDiskMissing;

  // Step 2: Build pass — incremental rebuild for changed or new corpus files.
  if (!options.touchOnly) {
    const buildResult = runBuildPass(databasePath);
    summary.buildPassRan = true;
    summary.buildPassExitCode = buildResult.exitCode;
  }

  summary.elapsedMs = Date.now() - startMs;
  return summary;
}

// ---------------------------------------------------------------------------
// Touch pass
// ---------------------------------------------------------------------------

/**
 * Update `indexed_at` to NOW() for every row whose on-disk content still matches
 * the stored freshness proof. Returns counts for changed and missing files.
 *
 * @param {string|object} databasePathOrOptions - Resolved path to the SQLite database file,
 *   or an options object with a `client` property for Turso/libSQL mode.
 * @returns {Promise<TouchResult>}
 */
export async function runTouchPass(databasePathOrOptions) {
  const result = { touched: 0, contentChanged: 0, onDiskMissing: 0 };

  if (
    typeof databasePathOrOptions === 'object' &&
    databasePathOrOptions !== null &&
    databasePathOrOptions.client
  ) {
    const { client } = databasePathOrOptions;
    const docsResult = await client.execute({
      sql: 'SELECT file_path, mtime_ms, file_size, sha256 FROM documents',
      args: [],
    });
    const allDocuments = docsResult.rows;

    const freshPaths = await collectFreshPaths(allDocuments, result);

    if (freshPaths.length > 0) {
      const nowMs = Date.now();
      for (const filePath of freshPaths) {
        await client.execute({
          sql: 'UPDATE documents SET indexed_at = ? WHERE file_path = ?',
          args: [nowMs, filePath],
        });
      }
      result.touched = freshPaths.length;
    }

    return result;
  }

  const databasePath = databasePathOrOptions;
  const client = createClient({ url: pathToFileURL(databasePath).href });
  const docsResult = await client.execute({
    sql: 'SELECT file_path, mtime_ms, file_size, sha256 FROM documents',
    args: [],
  });
  const allDocuments = docsResult.rows;

  // Step 1: Classify each document row by comparing on-disk freshness proof.
  const freshPaths = await collectFreshPaths(allDocuments, result);

  // Step 2: Batch-update indexed_at for content-fresh rows.
  if (freshPaths.length > 0) {
    const nowMs = Date.now();
    await client.batch(
      freshPaths.map((filePath) => ({
        sql: 'UPDATE documents SET indexed_at = ? WHERE file_path = ?',
        args: [nowMs, filePath],
      })),
      'write',
    );
    result.touched = freshPaths.length;
  }

  await client.close();
  return result;
}

/**
 * Probe each document row against on-disk state and return paths where the
 * content proof still matches (safe to touch without re-chunking).
 * Mutates `result.contentChanged` and `result.onDiskMissing` as a side effect.
 *
 * @param {Array<{file_path: string, mtime_ms: number|string, file_size: number|string, sha256: string}>} documentRows
 * @param {TouchResult} result - Accumulator for mismatch counts.
 * @returns {Promise<string[]>} Repo-relative file paths whose content is still fresh.
 */
async function collectFreshPaths(documentRows, result) {
  const probeResults = await Promise.all(
    documentRows.map((documentRow) => probeDocumentFreshness(documentRow)),
  );

  const freshPaths = [];

  for (const probeResult of probeResults) {
    if (probeResult.status === 'fresh') {
      freshPaths.push(probeResult.filePath);
    } else if (probeResult.status === 'missing') {
      result.onDiskMissing += 1;
    } else {
      result.contentChanged += 1;
    }
  }

  return freshPaths;
}

/**
 * Probe a single document row against its on-disk state.
 *
 * @param {{file_path: string, mtime_ms: number|string, file_size: number|string, sha256: string}} documentRow
 * @returns {Promise<{filePath: string, status: 'fresh'|'changed'|'missing'}>}
 */
async function probeDocumentFreshness(documentRow) {
  const absolutePath = path.join(repoRoot, documentRow.file_path);

  try {
    const freshness = await getFreshnessProof(absolutePath);
    const isFresh =
      Number(documentRow.mtime_ms) === Number(freshness.mtime_ms) &&
      Number(documentRow.file_size) === Number(freshness.size) &&
      documentRow.sha256 === freshness.sha256;

    return {
      filePath: documentRow.file_path,
      status: isFresh ? 'fresh' : 'changed',
    };
  } catch {
    return { filePath: documentRow.file_path, status: 'missing' };
  }
}

// ---------------------------------------------------------------------------
// Build pass
// ---------------------------------------------------------------------------

/**
 * Run `build-index.mjs` as a child process for incremental corpus updates.
 * Content-changed and new corpus files are re-indexed; unchanged files are
 * skipped by the build's own freshness-proof check.
 *
 * @param {string} databasePath - Resolved path to the SQLite database file.
 * @returns {{ exitCode: number }}
 */
export function runBuildPass(databasePath) {
  const spawnResult = spawnSync(
    process.execPath,
    [BUILD_INDEX_PATH, `--database=${databasePath}`],
    { stdio: 'inherit', cwd: repoRoot },
  );

  return { exitCode: spawnResult.status ?? 1 };
}

// ---------------------------------------------------------------------------
// Formatting and help
// ---------------------------------------------------------------------------

/**
 * @param {SessionStartSummary} summary
 * @returns {string}
 */
function formatSummaryText(summary) {
  const lines = [
    `session-start-index: touched=${summary.touched} changed=${summary.contentChanged} missing=${summary.onDiskMissing} buildPass=${summary.buildPassRan} elapsed=${summary.elapsedMs}ms`,
  ];

  if (summary.fatalError) lines.push(`ERROR: ${summary.fatalError}`);
  return lines.join('\n');
}

function printUsage() {
  console.log(
    [
      'session-start-index — lightweight session-start semantic-index refresh',
      '',
      'Usage:',
      '  node scripts/semantic-index/session-start-index.mjs [--json] [--touch-only] [--database=<path>]',
      '  node scripts/semantic-index/session-start-index.mjs --help',
      '',
      'Options:',
      '  --json                 Emit machine-readable JSON summary.',
      '  --touch-only           Run only the touch pass; skip the incremental build pass.',
      '  --database=<path>      Override the SQLite database path.',
      '',
      'Description:',
      '  Prevents the 24-hour indexed_at staleness gate from drifting red for',
      '  content that has not changed. Touches indexed_at for fresh rows (no',
      '  re-chunking), then runs an incremental build for changed or new files.',
      '',
      'Recommended usage:',
      '  npm run index:session-start   # run once at session start',
    ].join('\n'),
  );
}

// ---------------------------------------------------------------------------
// Types (JSDoc only)
// ---------------------------------------------------------------------------

/**
 * @typedef {object} TouchResult
 * @property {number} touched        - Rows where indexed_at was bumped.
 * @property {number} contentChanged - Rows where on-disk content differs from stored hash.
 * @property {number} onDiskMissing  - Rows whose on-disk file no longer exists.
 */

/**
 * @typedef {object} SessionStartSummary
 * @property {string}      databasePath      - Resolved path to the SQLite database.
 * @property {number}      touched           - Rows touched (indexed_at bumped, no re-chunking).
 * @property {number}      contentChanged    - Files whose content changed since last index.
 * @property {number}      onDiskMissing     - Files no longer present on disk.
 * @property {boolean}     buildPassRan      - Whether the incremental build pass ran.
 * @property {number|null} buildPassExitCode - Exit code of the build pass (null if not run).
 * @property {number}      elapsedMs         - Total wall-clock time in milliseconds.
 * @property {string|null} fatalError        - Non-null when the script could not proceed.
 */
