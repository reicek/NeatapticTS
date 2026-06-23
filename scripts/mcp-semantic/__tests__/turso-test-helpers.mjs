/**
 * @module turso-test-helpers
 * @description Shared helpers for the Turso async-migration test files.
 *
 * Provides a BEGIN/END-aware SQL splitter, a schema loader that creates the
 * production Turso corpus schema in an in-memory `@libsql/client` client, and a
 * small fixture inserter used across the Turso migration test files.
 *
 * Pure .mjs helper — not a test file itself (no `.test.mjs` suffix).
 */

import { createClient } from '@libsql/client';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Split a multi-statement SQL script into individual executable statements.
 *
 * The v2 schema contains `CREATE TRIGGER ... BEGIN ... END;` blocks whose
 * bodies include `;` characters. A naive `split(';')` would break those
 * triggers. This splitter tracks `BEGIN`/`END` keyword boundaries so trigger
 * bodies stay intact, and discards comment-only or empty fragments.
 *
 * @param {string} sql - Raw SQL script text.
 * @returns {string[]} Executable SQL statements.
 */
export function splitSqlStatements(sql) {
  const out = [];
  let buf = '';
  let inBegin = false;
  for (let i = 0; i < sql.length; i++) {
    const ch = sql[i];
    buf += ch;
    const upper = buf.slice(-8).toUpperCase();
    if (!inBegin && /\bBEGIN\b/.test(upper)) {
      inBegin = true;
    }
    if (inBegin && /\bEND\b/.test(buf.slice(-5).toUpperCase())) {
      inBegin = false;
    }
    if (ch === ';' && !inBegin) {
      const stmt = buf.trim();
      if (stmt && stmt !== ';') out.push(stmt);
      buf = '';
    }
  }
  if (buf.trim()) out.push(buf.trim());
  return out;
}

/**
 * Read the production Turso corpus schema SQL from the semantic-index scripts directory.
 *
 * The Turso schema is the authoritative consolidated schema for the migrated
 * RAG system. It adds `embedding F8_BLOB(384)`, `embedding_model`,
 * `chunk_sha256`, `embedded_at`, the `_schema_version` / `_index_metadata`
 * bookkeeping tables, and the DiskANN vector indexes that the local v2 schema
 * does not include. Tests that exercise the Turso migration must use this
 * schema so column references in production queries (for example
 * `c.chunk_sha256` in {@link assembleContext}) resolve correctly.
 *
 * @returns {Promise<string>} Schema SQL text.
 */
export async function readCorpusSchema() {
  const schemaPath = path.join(__dirname, '../../semantic-index/schema-turso.sql');
  return readFile(schemaPath, 'utf8');
}

/**
 * Create a `:memory:` libSQL client and load the full Turso corpus schema into it.
 *
 * The schema includes `documents`, `chunks` (with Turso-specific vector columns
 * and `chunk_sha256`), `chunks_fts` (FTS5 external content table + triggers),
 * `entities`, `edges`, `feedback_events`, `feedback_scores`, `term_embeddings`,
 * plus `_schema_version` and `_index_metadata`. Triggers populate `chunks_fts`
 * automatically when rows are inserted into `chunks`.
 *
 * @returns {Promise<import('@libsql/client').Client>} Configured in-memory client.
 */
export async function createSchemaClient() {
  const client = createClient({ url: ':memory:' });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  return client;
}

/**
 * Create a file-backed libSQL client and load the full Turso corpus schema into it.
 *
 * @param {string} dbPath - Absolute or relative SQLite file path.
 * @returns {Promise<import('@libsql/client').Client>} Configured file-backed client.
 */
export async function createFileSchemaClient(dbPath) {
  const client = createClient({ url: 'file:' + dbPath });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  return client;
}

/**
 * Unique chunk ID used across the Turso red tests.
 *
 * Chosen far outside any realistic repo chunk-id range so the value can only
 * exist in the injected in-memory test client — never in the real on-disk
 * corpus. This makes the "tool ignored the injected client" failure mode
 * deterministic.
 */
export const TEST_CHUNK_ID = 424242;
export const TEST_PARENT_CHUNK_ID = 424241;
export const TEST_DOC_ID = 500001;
export const TEST_FILE_PATH = 'src/turso-async-test.ts';
export const TEST_FAMILY = 'turso-test';
export const TEST_SYMBOL = 'TursoAsyncTestSymbol';
export const UNIQUE_QUERY_TERM = 'tursouniqueword';

/**
 * Insert a minimal but complete fixture set into a schema-loaded client.
 *
 * Inserts one document, one depth-0 parent chunk, one depth-1 sub-chunk, and
 * sets the parent's `symbol_name` to {@link TEST_SYMBOL} so exact-symbol and
 * BM25 paths can both resolve against the injected client.
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
      'turso-test-sha256',
      1000,
      'network',
    ],
  });

  await client.execute({
    sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth, symbol_name, module_path, arch_layer)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      TEST_PARENT_CHUNK_ID,
      TEST_DOC_ID,
      0,
      'TursoAsyncTestModule',
      `${UNIQUE_QUERY_TERM} parent body content`,
      0,
      30,
      null,
      0,
      TEST_SYMBOL,
      'src/turso-async-test.ts',
      'network',
    ],
  });

  await client.execute({
    sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth, symbol_name, module_path, arch_layer)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      TEST_CHUNK_ID,
      TEST_DOC_ID,
      1,
      'TursoAsyncTestModule.TursoAsyncMethod',
      `${UNIQUE_QUERY_TERM} sub body content`,
      30,
      60,
      TEST_PARENT_CHUNK_ID,
      1,
      'TursoAsyncMethod',
      'src/turso-async-test.ts',
      'network',
    ],
  });
}

/**
 * Insert entity/relationship graph fixtures for traverse-graph tests.
 *
 * Inserts two entities linked by a `references` edge with unique qualified
 * names that cannot exist in the real on-disk corpus.
 *
 * @param {import('@libsql/client').Client} client - Schema-loaded client.
 * @returns {Promise<void>}
 */
export async function insertGraphFixtures(client) {
  await client.execute({
    sql: `INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, chunk_id, module_path, file_path)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      900001,
      'function',
      'TursoTestEntityA',
      'src/turso.TursoTestEntityA',
      TEST_DOC_ID,
      TEST_CHUNK_ID,
      'src/turso-async-test.ts',
      TEST_FILE_PATH,
    ],
  });

  await client.execute({
    sql: `INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, chunk_id, module_path, file_path)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      900002,
      'class',
      'TursoTestEntityB',
      'src/turso.TursoTestEntityB',
      TEST_DOC_ID,
      TEST_PARENT_CHUNK_ID,
      'src/turso-async-test.ts',
      TEST_FILE_PATH,
    ],
  });

  await client.execute({
    sql: `INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence)
          VALUES (?, ?, ?, ?)`,
    args: [900001, 900002, 'references', 'high'],
  });
}

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
        if (savedEnv[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = savedEnv[key];
        }
      }
    },
  };
}
