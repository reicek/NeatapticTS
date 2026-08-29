/**
 * @description Lightweight session-start semantic-index refresh. Delegates to
 * `build-index.mjs` (without `--force`) so that content-changed or newly-added
 * corpus files are indexed normally. Unchanged files are skipped by the build's
 * own freshness-proof check.
 *
 * Safe to run at session start every day; the build pass is O(changed files)
 * only.
 *
 * @param {boolean} [--json]             - Emit JSON summary.
 * @param {string}  [--database <path>]  - Path to the SQLite database file (default: `rag-index/data/turso-replica.sqlite`).
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
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseCliArgs, writeJsonOrText } from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const SCRIPT_NAME = 'session-start-index.mjs';
const BUILD_INDEX_PATH = path.join(repoRoot, 'rag-index', 'build-index.mjs');

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
    databasePath: args['database'] ?? undefined,
  })
    .then((summary) => {
      writeJsonOrText(summary, Boolean(args['json']), formatSummaryText);
      process.exit(summary.fatalError ? 1 : 0);
    })
    .catch((error) => {
      let errorMessage;
      /* istanbul ignore next -- defensive: caught errors are always Error instances */
      if (error instanceof Error) errorMessage = error.message;
      else errorMessage = String(error);
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
 * @param {string}  [options.databasePath] - Override the database path.
 * @param {boolean} [options.json]         - Emit JSON output (unused in library mode).
 * @returns {Promise<SessionStartSummary>}
 */
/* istanbul ignore next -- defensive: always called with explicit options */
export async function runSessionStartIndex(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const startMs = Date.now();

  const summary = {
    databasePath,
    buildPassRan: false,
    buildPassExitCode: null,
    elapsedMs: 0,
    fatalError: null,
  };

  if (!existsSync(databasePath)) {
    summary.fatalError = `Database not found: ${databasePath}. Run npm run index:build first.`;
    summary.elapsedMs = Date.now() - startMs;
    return summary;
  }

  // Run incremental build for changed or new corpus files.
  const buildResult = runBuildPass(databasePath);
  summary.buildPassRan = true;
  summary.buildPassExitCode = buildResult.exitCode;

  summary.elapsedMs = Date.now() - startMs;
  return summary;
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

  return { exitCode: /* istanbul ignore next -- defensive: spawnResult.status is always set for synchronous spawns */ spawnResult.status ?? 1 };
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
    `session-start-index: buildPass=${summary.buildPassRan} exitCode=${summary.buildPassExitCode} elapsed=${summary.elapsedMs}ms`,
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
      '  node rag-index/session-start-index.mjs [--json] [--database=<path>]',
      '  node rag-index/session-start-index.mjs --help',
      '',
      'Options:',
      '  --json                 Emit machine-readable JSON summary.',
      '  --database=<path>      Override the SQLite database path.',
      '',
      'Description:',
      '  Runs an incremental build for changed or new corpus files. Unchanged',
      "  files are skipped by the build's own freshness-proof check.",
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
 * @typedef {object} SessionStartSummary
 * @property {string}      databasePath      - Resolved path to the SQLite database.
 * @property {boolean}     buildPassRan      - Whether the incremental build pass ran.
 * @property {number|null} buildPassExitCode - Exit code of the build pass (null if not run).
 * @property {number}      elapsedMs         - Total wall-clock time in milliseconds.
 * @property {string|null} fatalError        - Non-null when the script could not proceed.
 */
