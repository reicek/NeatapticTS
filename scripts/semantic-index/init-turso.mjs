/**
 * @module init-turso
 * @description Turso-aware database initializer that replaces `init-schema.mjs`
 * for the Turso/libSQL RAG stack.
 *
 * This script creates (or connects to) a Turso/libSQL database, applies
 * `schema-turso.sql` (tables, FTS5 virtual tables, triggers, indexes), verifies
 * that FTS5 triggers are functional by inserting and querying a test chunk, and
 * returns a JSON status report with table counts. It is idempotent: the schema
 * uses `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` so re-running
 * is safe.
 *
 * Environment variables:
 *
 * - `TURSO_DATABASE_URL` — Turso database URL. Defaults to
 *   `file:data/turso-replica.sqlite` for local development. Use a `libsql://`
 *   URL for Turso Cloud.
 * - `TURSO_AUTH_TOKEN` — Turso auth token. Not required for local `file:` mode.
 * - `TURSO_SYNC_INTERVAL` — Embedded replica sync interval in **seconds**
 *   (Phase 1 verified the unit is seconds, not milliseconds). Default: `60`.
 *
 * CLI flags:
 *
 * - `--json` — Output only JSON to stdout (for CI gate integration). Without
 *   this flag, a human-readable status report is printed.
 * - `--schema <path>` — Override the schema SQL file path. Defaults to
 *   `scripts/semantic-index/schema-turso.sql`.
 * - `--skip-diskann` — Skip DiskANN vector index creation during init so they
 *   can be created after bulk data load for better build efficiency. The two
 *   `libsql_vector_idx` statements are filtered out of the schema batch.
 *
 * @example
 * ```ts
 * // Local development (no cloud credentials needed)
 * node scripts/semantic-index/init-turso.mjs --json
 *
 * // Turso Cloud
 * TURSO_DATABASE_URL=libsql://my-db.turso.io \
 * TURSO_AUTH_TOKEN=eyJ... \
 * node scripts/semantic-index/init-turso.mjs --json
 *
 * // Programmatic usage
 * import { initTurso } from './init-turso.mjs';
 * const result = await initTurso({ jsonOutput: true });
 * console.log(result);
 * ```
 */

import { createClient } from '@libsql/client';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Default Turso database URL for local development. Uses the embedded libSQL
 * file backend — no cloud credentials required.
 */
const DEFAULT_TURSO_URL = 'file:data/turso-replica.sqlite';

/**
 * Default path to the Turso schema SQL file, relative to repo root.
 */
const DEFAULT_SCHEMA_PATH = 'scripts/semantic-index/schema-turso.sql';

/**
 * Embedded replica sync interval in **seconds** (not milliseconds). Phase 1
 * research verified the libSQL unit is seconds.
 */
const DEFAULT_SYNC_INTERVAL_SECONDS = 60;

/**
 * Tables whose row counts are reported in the JSON status. Ordered to match the
 * schema creation order (dependency order).
 */
const COUNTED_TABLES = [
  'documents',
  'chunks',
  'entities',
  'edges',
  'term_embeddings',
  'feedback_events',
  'feedback_scores',
  '_schema_version',
  '_index_metadata',
];

/**
 * SQL pattern identifying DiskANN `libsql_vector_idx` index statements. When
 * `--skip-diskann` is used, statements matching this pattern are filtered out
 * of the schema batch so vector indexes are not built on empty tables. The
 * pattern requires both `CREATE INDEX` and `libsql_vector_idx` to avoid
 * matching header comments that mention DiskANN.
 */
const DISKANN_INDEX_PATTERN = /CREATE\s+INDEX.*libsql_vector_idx/i;

// ---------------------------------------------------------------------------
// Schema helpers
// ---------------------------------------------------------------------------

/**
 * Split a multi-statement SQL script into individual statements, handling
 * multi-line `CREATE TRIGGER` bodies that contain semicolons inside `BEGIN … END`.
 *
 * @param {string} sql - Raw SQL script text.
 * @returns {string[]} Array of trimmed individual SQL statements.
 */
function splitSqlStatements(sql) {
  const statements = [];
  let buffer = '';
  let inTrigger = false;
  for (const line of sql.split(/\r?\n/)) {
    const trimmedLine = line.trim();
    if (
      /^\s*(CREATE\s+(TEMP\s+|TEMPORARY\s+)?TRIGGER|CREATE\s+TRIGGER)/i.test(
        trimmedLine,
      ) &&
      /BEGIN\s*$/i.test(trimmedLine)
    ) {
      inTrigger = true;
    }
    buffer += line + '\n';
    if (inTrigger) {
      if (/^\s*END\s*;?\s*$/i.test(trimmedLine)) {
        inTrigger = false;
        const stmt = buffer.trim().replace(/;\s*$/, '');
        if (stmt) statements.push(stmt + ';');
        buffer = '';
      }
    } else if (trimmedLine.endsWith(';')) {
      const stmt = buffer.trim().replace(/;\s*$/, '');
      if (stmt) statements.push(stmt + ';');
      buffer = '';
    }
  }
  const tail = buffer.trim();
  if (tail) statements.push(tail);
  return statements;
}

/**
 * Apply a multi-statement SQL schema to a libSQL client via batch execution.
 * Statements are applied in file order (tables before indexes), which matches
 * the dependency order in `schema-turso.sql`.
 *
 * @param {object} client - libSQL client instance.
 * @param {string} schemaSql - Raw SQL schema text.
 * @param {boolean} skipDiskann - When `true`, filter out DiskANN
 *   `libsql_vector_idx` index statements so they can be created after bulk
 *   data load.
 * @returns {Promise<{applied: number, skipped: number}>} Count of statements
 *   applied and skipped.
 */
async function applySchemaToClient(client, schemaSql, skipDiskann) {
  const allStatements = splitSqlStatements(schemaSql);
  const statementsToApply = skipDiskann
    ? allStatements.filter((sql) => !DISKANN_INDEX_PATTERN.test(sql))
    : allStatements;
  const skippedCount = allStatements.length - statementsToApply.length;

  // Apply statements in groups to avoid exceeding batch limits on large schemas.
  const BATCH_GROUP_SIZE = 50;
  for (let i = 0; i < statementsToApply.length; i += BATCH_GROUP_SIZE) {
    const group = statementsToApply
      .slice(i, i + BATCH_GROUP_SIZE)
      .map((sql) => ({ sql, args: [] }));
    await client.batch(group, 'write');
  }

  return { applied: statementsToApply.length, skipped: skippedCount };
}

// ---------------------------------------------------------------------------
// Table count helper
// ---------------------------------------------------------------------------

/**
 * Query the row count for each table in {@link COUNTED_TABLES}.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<Record<string, number>>} Map of table name to row count.
 */
async function getTableCounts(client) {
  const counts = {};
  for (const table of COUNTED_TABLES) {
    const result = await client.execute(`SELECT COUNT(*) AS cnt FROM ${table}`);
    counts[table] = result.rows[0].cnt;
  }
  return counts;
}

// ---------------------------------------------------------------------------
// FTS5 functional verification
// ---------------------------------------------------------------------------

/**
 * Verify that FTS5 triggers are functional by inserting a test document + chunk,
 * querying `chunks_fts` for the test text, and then cleaning up the test data.
 *
 * This confirms the `chunks_ai` (AFTER INSERT) and `chunks_ad` (AFTER DELETE)
 * triggers are wiring the FTS5 virtual table correctly.
 *
 * @param {object} client - libSQL client instance.
 * @returns {Promise<{verified: boolean, matchedRows: number}>} Verification
 *   result.
 */
async function verifyFts5Triggers(client) {
  // Insert a minimal test document.
  const testFilePath = '__init_turso_fts5_test__.ts';
  const testBodyText =
    'init turso fts5 verification test chunk uniquemarkerxyz789';

  // Clean up any stale test data from a prior interrupted run.
  const staleDoc = await client.execute(
    `SELECT doc_id FROM documents WHERE file_path = ?`,
    [testFilePath],
  );
  if (staleDoc.rows.length > 0) {
    await client.execute(`DELETE FROM documents WHERE doc_id = ?`, [
      staleDoc.rows[0].doc_id,
    ]);
  }

  await client.execute({
    sql: `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [testFilePath, '__test__', 0, 0, 'test-sha256', 0],
  });

  const docResult = await client.execute(
    `SELECT doc_id FROM documents WHERE file_path = ?`,
    [testFilePath],
  );
  const docId = docResult.rows[0].doc_id;

  // Insert a test chunk — the chunks_ai trigger should auto-populate chunks_fts.
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [docId, 0, testBodyText, 0, testBodyText.length, 0],
  });

  const chunkResult = await client.execute(
    `SELECT chunk_id FROM chunks WHERE doc_id = ? AND chunk_index = 0`,
    [docId],
  );
  const chunkId = chunkResult.rows[0].chunk_id;

  // Query chunks_fts for the unique marker — should match via the trigger.
  // The FTS5 table uses content_rowid='chunk_id', so chunk_id is the rowid.
  const ftsResult = await client.execute(
    `SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?`,
    ['uniquemarkerxyz789'],
  );
  const matchedRows = ftsResult.rows.length;
  const verified = matchedRows > 0;

  // Clean up: delete the test chunk (chunks_ad trigger removes it from FTS),
  // then delete the test document.
  await client.execute(`DELETE FROM chunks WHERE chunk_id = ?`, [chunkId]);
  await client.execute(`DELETE FROM documents WHERE doc_id = ?`, [docId]);

  return { verified, matchedRows };
}

// ---------------------------------------------------------------------------
// Main initialization function
// ---------------------------------------------------------------------------

/**
 * Initialize a Turso/libSQL database by applying the schema, verifying FTS5
 * triggers, and returning table counts.
 *
 * @param {object} [options] - Initialization options.
 * @param {string} [options.url] - Turso database URL. Defaults to
 *   `process.env.TURSO_DATABASE_URL` or `file:data/turso-replica.sqlite`.
 * @param {string} [options.authToken] - Turso auth token. Defaults to
 *   `process.env.TURSO_AUTH_TOKEN`.
 * @param {string} [options.schemaPath] - Path to schema SQL file.
 * @param {boolean} [options.skipDiskann] - Skip DiskANN index creation.
 * @param {boolean} [options.skipFts5Verify] - Skip the FTS5 functional test.
 * @param {boolean} [options.jsonOutput] - Reserved for CLI; the return value is
 *   always the status object.
 * @returns {Promise<object>} Status object with `success`, `url`, `tableCounts`,
 *   `fts5Verified`, `statementsApplied`, `diskannSkipped`, and `error` fields.
 */
export async function initTurso(options = {}) {
  const url =
    options.url ?? process.env.TURSO_DATABASE_URL ?? DEFAULT_TURSO_URL;
  const authToken =
    options.authToken ?? process.env.TURSO_AUTH_TOKEN ?? undefined;
  const schemaPath = options.schemaPath ?? DEFAULT_SCHEMA_PATH;
  const skipDiskann = options.skipDiskann ?? false;
  const skipFts5Verify = options.skipFts5Verify ?? false;

  // For local file: URLs, ensure the parent directory exists.
  if (url.startsWith('file:')) {
    const filePath = url.slice('file:'.length);
    const dir = path.dirname(path.resolve(filePath));
    await mkdir(dir, { recursive: true });
  }

  const clientConfig = { url };
  if (authToken) {
    clientConfig.authToken = authToken;
  }

  const client = createClient(clientConfig);

  try {
    // Step 1: Read and apply the schema (tables before indexes, in file order).
    const schemaSql = await readFile(schemaPath, 'utf8');
    const { applied, skipped } = await applySchemaToClient(
      client,
      schemaSql,
      skipDiskann,
    );

    // Step 2: Verify FTS5 triggers are functional.
    let fts5Result = { verified: false, matchedRows: 0 };
    if (!skipFts5Verify) {
      fts5Result = await verifyFts5Triggers(client);
    }

    // Step 3: Collect table counts for the status report.
    const tableCounts = await getTableCounts(client);

    return {
      success: true,
      url,
      schemaApplied: true,
      statementsApplied: applied,
      diskannSkipped: skipped,
      fts5Verified: fts5Result.verified,
      fts5MatchedRows: fts5Result.matchedRows,
      tableCounts,
    };
  } catch (err) {
    return {
      success: false,
      url,
      schemaApplied: false,
      statementsApplied: 0,
      diskannSkipped: 0,
      fts5Verified: false,
      fts5MatchedRows: 0,
      tableCounts: {},
      error: err instanceof Error ? err.message : String(err),
    };
  } finally {
    await client.close();
  }
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

/**
 * Run the Turso initializer from the command line.
 *
 * Usage:
 *   node scripts/semantic-index/init-turso.mjs [--json] [--schema <path>] [--skip-diskann]
 */
async function main() {
  const args = process.argv.slice(2);
  const getArg = (name) => {
    const idx = args.indexOf(name);
    return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
  };

  const jsonOutput = args.includes('--json');
  const schemaPath = getArg('--schema') ?? DEFAULT_SCHEMA_PATH;
  const skipDiskann = args.includes('--skip-diskann');

  const result = await initTurso({ schemaPath, skipDiskann });

  if (jsonOutput) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    if (result.success) {
      console.log('Turso database initialized successfully.');
      console.log(`  URL:                  ${result.url}`);
      console.log(
        `  Schema applied:       ${result.statementsApplied} statements`,
      );
      if (result.diskannSkipped > 0) {
        console.log(
          `  DiskANN skipped:      ${result.diskannSkipped} statements (deferred to post-data-load)`,
        );
      }
      console.log(
        `  FTS5 verified:        ${result.fts5Verified} (${result.fts5MatchedRows} match)`,
      );
      console.log('  Table counts:');
      for (const [table, count] of Object.entries(result.tableCounts)) {
        console.log(`    ${table.padEnd(22)} ${count}`);
      }
    } else {
      console.error('Turso initialization FAILED.');
      console.error(`  URL:   ${result.url}`);
      console.error(`  Error: ${result.error}`);
    }
  }

  process.exit(result.success ? 0 : 1);
}

// Run CLI only when executed directly (not when imported).
if (
  process.argv[1] &&
  (process.argv[1].endsWith('init-turso.mjs') ||
    process.argv[1].endsWith('init-turso'))
) {
  main().catch((err) => {
    console.error('init-turso failed:', err);
    process.exit(1);
  });
}
