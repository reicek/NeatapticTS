/**
 * @module turso-test-helpers
 * @description Shared helpers for the semantic-index Turso async-migration red tests.
 *
 * Provides a BEGIN/END-aware SQL splitter, a schema loader that creates the
 * full Turso consolidated schema (`schema-turso.sql`) in an in-memory
 * `@libsql/client` client, and fixture inserters used across the Step 05 red
 * test files.
 *
 * The Turso schema merges `chunk_embeddings` into `chunks.embedding F8_BLOB(384)`
 * and includes DiskANN vector indexes, `_schema_version`, and `_index_metadata`.
 *
 * Pure .mjs helper — not a test file itself (no `.test.mjs` suffix).
 */

import { createClient } from '@libsql/client';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Split a multi-statement SQL script into individual statements.
 *
 * Respects `BEGIN ... END;` trigger bodies so that semicolons inside trigger
 * bodies are not treated as statement terminators. Uses the same line-based
 * approach as `schema-turso.test.mjs` and `init-turso.mjs`.
 *
 * @param {string} sql - Raw SQL script text.
 * @returns {string[]} Array of trimmed individual SQL statements.
 */
export function splitSqlStatements(sql) {
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
          /* istanbul ignore next -- defensive: stmt is always non-empty after trimming */
          if (stmt) statements.push(stmt + ';');
        buffer = '';
      }
    } else if (trimmedLine.endsWith(';')) {
      const stmt = buffer.trim().replace(/;\s*$/, '');
      /* istanbul ignore next -- defensive: stmt is always non-empty after trimming */
      if (stmt) statements.push(stmt + ';');
      buffer = '';
    }
  }
  const tail = buffer.trim();
  /* istanbul ignore next -- defensive: tail is always non-empty when buffer has content */
  if (tail) statements.push(tail);
  return statements;
}

/**
 * Read the Turso consolidated schema SQL from the semantic-index scripts
 * directory.
 *
 * @returns {Promise<string>} Schema SQL text.
 */
export async function readTursoSchema() {
  const schemaPath = path.resolve(__dirname, '..', 'schema-turso.sql');
  return readFile(schemaPath, 'utf8');
}

/**
 * SQL pattern identifying DiskANN `libsql_vector_idx` index statements.
 *
 * These indexes validate vector types on insert, which conflicts with test
 * fixtures that use raw Float32Array buffers. Skipping them (following the
 * `init-turso.mjs --skip-diskann` pattern) avoids the type mismatch while
 * keeping all tables and non-vector indexes intact.
 */
const DISKANN_INDEX_PATTERN = /CREATE\s+INDEX.*libsql_vector_idx/i;

/**
 * Create a `:memory:` libSQL client and load the full Turso consolidated
 * schema into it.
 *
 * The schema includes `documents`, `chunks` (with `embedding F8_BLOB(384)`),
 * `chunks_fts` (FTS5 + triggers), `entities`, `edges`, `feedback_events`,
 * `feedback_scores`, `term_embeddings`, `_schema_version`, and
 * `_index_metadata`. DiskANN `libsql_vector_idx` indexes are skipped so test
 * fixtures can insert raw Float32Array buffers into F8_BLOB columns without
 * vector-index type validation errors. This follows the same pattern as
 * `init-turso.mjs --skip-diskann`.
 *
 * @returns {Promise<import('@libsql/client').Client>} Configured in-memory client.
 */
export async function createSchemaClient() {
  const client = createClient({ url: ':memory:' });
  const schemaSql = await readTursoSchema();
  const allStatements = splitSqlStatements(schemaSql);
  const statementsToApply = allStatements.filter(
    (stmt) => !DISKANN_INDEX_PATTERN.test(stmt),
  );
  await client.batch(
    statementsToApply.map((sql) => ({ sql, args: [] })),
    'write',
  );
  return client;
}

// ---------------------------------------------------------------------------
// Test constants — chosen far outside any realistic repo ID range so values
// can only exist in the injected in-memory test client, never in the real
// on-disk corpus.
// ---------------------------------------------------------------------------

export const TEST_DOC_ID = 600001;
export const TEST_CHUNK_ID = 624242;
export const TEST_PARENT_CHUNK_ID = 624241;
export const TEST_FILE_PATH = 'src/turso-async-script-test.ts';
export const TEST_FAMILY = 'turso-script-test';
export const TEST_SYMBOL = 'TursoAsyncScriptTestSymbol';
export const UNIQUE_QUERY_TERM = 'tursoscriptuniqueword';
export const TEST_MODEL_ID = 'turso-test-model';
export const TEST_MODEL_SHA256 = 'turso-test-sha256-abcdef';
export const TEST_DIMENSION = 384;

// ---------------------------------------------------------------------------
// Fixture inserters
// ---------------------------------------------------------------------------

/**
 * Insert a minimal but complete fixture set into a schema-loaded client.
 *
 * Inserts one document and one chunk whose body contains the
 * {@link UNIQUE_QUERY_TERM} so BM25 queries against the injected client can
 * deterministically find the test chunk. The chunk's `embedding` column is
 * left NULL — use {@link insertEmbeddingFixtures} to populate it.
 *
 * @param {import('@libsql/client').Client} client - Schema-loaded client.
 * @returns {Promise<void>}
 */
export async function insertTestFixtures(client) {
  await client.execute({
    sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      TEST_DOC_ID,
      TEST_FILE_PATH,
      TEST_FAMILY,
      1000,
      500,
      'turso-script-test-sha256',
      1000,
      'network',
    ],
  });

  await client.execute({
    sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth, symbol_name, module_path, arch_layer)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      TEST_CHUNK_ID,
      TEST_DOC_ID,
      0,
      'TursoScriptTestModule',
      `${UNIQUE_QUERY_TERM} body content for semantic index migration test`,
      0,
      60,
      null,
      0,
      TEST_SYMBOL,
      'src/turso-async-script-test.ts',
      'network',
    ],
  });
}

/**
 * Insert a chunk embedding into the `chunks.embedding` column.
 *
 * The Turso schema merges embeddings into the `chunks` table, so the
 * embedding is written directly to the chunk row via UPDATE rather than to
 * a separate `chunk_embeddings` table.
 *
 * @param {import('@libsql/client').Client} client - Schema-loaded client.
 * @param {number} [chunkId=TEST_CHUNK_ID] - Chunk ID to attach the embedding to.
 * @returns {Promise<void>}
 */
export async function insertEmbeddingFixtures(/* istanbul ignore next -- defensive: chunkId default is always provided by callers */ client, chunkId = TEST_CHUNK_ID) {
  // Build a deterministic 384-dimensional Float32Array embedding
  const embedding = new Float32Array(TEST_DIMENSION);
  for (let i = 0; i < TEST_DIMENSION; i += 1) {
    embedding[i] = (i % 7) / 10.0;
  }
  const embeddingBuffer = Buffer.from(
    embedding.buffer,
    embedding.byteOffset,
    embedding.byteLength,
  );

  await client.execute({
    sql: `UPDATE chunks SET embedding = ?, embedding_model = ?, chunk_sha256 = ?, embedded_at = ? WHERE chunk_id = ?`,
    args: [embeddingBuffer, TEST_MODEL_ID, 'turso-chunk-sha256', 1000, chunkId],
  });
}

/**
 * Insert entity/relationship graph fixtures for build-entity-graph tests.
 *
 * Inserts one entity with a unique qualified_name that cannot exist in the
 * real on-disk corpus.
 *
 * @param {import('@libsql/client').Client} client - Schema-loaded client.
 * @returns {Promise<void>}
 */
export async function insertGraphFixtures(client) {
  await client.execute({
    sql: `INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, chunk_id, module_path, file_path)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      950001,
      'function',
      'TursoScriptTestEntityA',
      'src/turso.TursoScriptTestEntityA',
      TEST_DOC_ID,
      TEST_CHUNK_ID,
      'src/turso-async-script-test.ts',
      TEST_FILE_PATH,
    ],
  });
}

// ---------------------------------------------------------------------------
// Env isolation
// ---------------------------------------------------------------------------

/**
 * Env vars managed by the Turso red tests. Each test file saves and restores
 * these around every test so injected-client behavior is not influenced by
 * ambient Turso/CORTEX_DB_PATH configuration.
 */
export const MANAGED_ENV_VARS = [
  'TURSO_DATABASE_URL',
  'TURSO_AUTH_TOKEN',
  'TURSO_SYNC_URL',
  'TURSO_SYNC_INTERVAL',
  'CORTEX_DB_PATH',
];

/**
 * Build a beforeEach/afterEach env isolation pair for the managed env vars.
 *
 * @returns {{ saveEnv: () => void, restoreEnv: () => void }} Save/restore helpers.
 */
export function createEnvIsolation() {
  const savedEnv = {};
  return {
    saveEnv() {
      for (const key of MANAGED_ENV_VARS) {
        savedEnv[key] = process.env[key];
        delete process.env[key];
      }
    },
    restoreEnv() {
      for (const key of MANAGED_ENV_VARS) {
        /* istanbul ignore next -- defensive: savedEnv always has the key set from setupEnv */
        if (savedEnv[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = savedEnv[key];
        }
      }
    },
  };
}
