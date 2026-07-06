/**
 * @module migrate-schema
 * @description Idempotent schema migrations for the Turso RAG corpus.
 *
 * Exports small, async, client-aware migration helpers that bring an older
 * semantic-index database up to the current consolidated schema.
 *
 * Each migration accepts either a pre-created `@libsql/client` `Client` via
 * `options.client`, or `options.databasePath` so a client can be opened on
 * demand. They are idempotent: running the same migration twice is a no-op.
 */

import { getTursoClient } from '../scripts/mcp-semantic/tools/cortex-db.mjs';

/**
 * Column addition descriptor.
 *
 * @typedef {Object} ColumnDescriptor
 * @property {string} name - Column name to add if missing.
 * @property {string} type - Full SQLite type clause (e.g. `TEXT`,
 *   `INTEGER NOT NULL DEFAULT 0`).
 */

/** Columns introduced by the V1 → V2 migration on `chunks`. */
const V2_CHUNK_COLUMNS = [
  { name: 'parent_chunk_id', type: 'INTEGER' },
  { name: 'depth', type: 'INTEGER NOT NULL DEFAULT 0' },
  { name: 'context_header', type: 'TEXT' },
  { name: 'symbol_name', type: 'TEXT' },
  { name: 'signature_text', type: 'TEXT' },
  { name: 'jsdoc_text', type: 'TEXT' },
  { name: 'export_type', type: 'TEXT' },
  { name: 'module_path', type: 'TEXT' },
];

/** Indexes introduced by the V1 → V2 migration on `chunks`. */
const V2_CHUNK_INDEXES = [
  'CREATE INDEX IF NOT EXISTS chunks_parent_idx ON chunks(parent_chunk_id)',
  'CREATE INDEX IF NOT EXISTS chunks_depth_idx ON chunks(depth)',
  'CREATE INDEX IF NOT EXISTS chunks_symbol_idx ON chunks(symbol_name)',
  'CREATE INDEX IF NOT EXISTS chunks_module_idx ON chunks(module_path)',
];

/** Columns introduced by the V2 → V3 migration. */
const V3_COLUMNS = [
  { table: 'documents', name: 'arch_layer', type: 'TEXT' },
  { table: 'chunks', name: 'arch_layer', type: 'TEXT' },
  { table: 'chunks', name: 'jsdoc_quality', type: 'TEXT' },
  { table: 'chunks', name: 'test_coverage', type: 'TEXT' },
  { table: 'chunks', name: 'source_path_pattern', type: 'TEXT' },
];

/** Indexes introduced by the V2 → V3 migration. */
const V3_INDEXES = [
  'CREATE INDEX IF NOT EXISTS chunks_arch_layer_idx ON chunks(arch_layer)',
  'CREATE INDEX IF NOT EXISTS chunks_jsdoc_quality_idx ON chunks(jsdoc_quality)',
  'CREATE INDEX IF NOT EXISTS chunks_test_coverage_idx ON chunks(test_coverage)',
  'CREATE INDEX IF NOT EXISTS chunks_export_type_idx ON chunks(export_type)',
  'CREATE INDEX IF NOT EXISTS chunks_source_path_pattern_idx ON chunks(source_path_pattern)',
  'CREATE INDEX IF NOT EXISTS chunks_family_arch_layer_idx ON chunks(doc_id, arch_layer)',
  'CREATE INDEX IF NOT EXISTS chunks_family_export_type_idx ON chunks(doc_id, export_type)',
  'CREATE INDEX IF NOT EXISTS documents_arch_layer_idx ON documents(arch_layer)',
];

/**
 * Resolve a database client from the options.
 *
 * @param {{ client?: import('@libsql/client').Client, databasePath?: string }} options
 * @returns {Promise<import('@libsql/client').Client>}
 */
async function resolveClient(options) {
  if (options?.client) return options.client;
  return getTursoClient(options?.databasePath);
}

/**
 * Check whether a table already has a column.
 *
 * @param {import('@libsql/client').Client} client
 * @param {string} table
 * @param {string} column
 * @returns {Promise<boolean>}
 */
async function tableHasColumn(client, table, column) {
  const result = await client.execute({
    sql: `PRAGMA table_info(${table})`,
    args: [],
  });
  return result.rows.some((row) => row.name === column);
}

/**
 * Add a column to a table if it does not already exist.
 *
 * @param {import('@libsql/client').Client} client
 * @param {string} table
 * @param {ColumnDescriptor} column
 * @returns {Promise<void>}
 */
async function addColumnIfMissing(client, table, column) {
  const exists = await tableHasColumn(client, table, column.name);
  if (exists) return;

  await client.execute({
    sql: `ALTER TABLE ${table} ADD COLUMN ${column.name} ${column.type}`,
    args: [],
  });
}

/**
 * Ensure a list of CREATE INDEX statements have been applied.
 *
 * Uses batch write mode so the whole index creation block is applied
 * atomically.
 *
 * @param {import('@libsql/client').Client} client
 * @param {string[]} statements
 * @returns {Promise<void>}
 */
async function ensureIndexes(client, statements) {
  if (statements.length === 0) return;

  const batch = statements.map((sql) => ({ sql, args: [] }));
  await client.batch(batch, 'write');
}

/**
 * Record the current schema version.
 *
 * `_schema_version` tracks every applied version as a row so migrations can
 * be inspected historically.
 *
 * @param {import('@libsql/client').Client} client
 * @param {number} version
 * @returns {Promise<void>}
 */
async function recordSchemaVersion(client, version) {
  await client.execute({
    sql: `INSERT INTO _schema_version (version, applied_at) VALUES (?, ?)`,
    args: [version, Date.now()],
  });
}

/**
 * Migrate a V1 schema to V2.
 *
 * Adds semantic chunking columns (`parent_chunk_id`, `depth`, `context_header`,
 * `symbol_name`, `signature_text`, `jsdoc_text`, `export_type`, `module_path`)
 * and their supporting indexes to `chunks`.
 *
 * @param {{ client?: import('@libsql/client').Client, databasePath?: string }} [options]
 * @returns {Promise<import('@libsql/client').Client>} The client used for the migration.
 */
export async function migrateSchemaV1ToV2(options = {}) {
  const client = await resolveClient(options);

  for (const column of V2_CHUNK_COLUMNS) {
    await addColumnIfMissing(client, 'chunks', column);
  }

  await ensureIndexes(client, V2_CHUNK_INDEXES);
  await recordSchemaVersion(client, 2);

  return client;
}

/**
 * Migrate a V2 schema to V3.
 *
 * Adds metadata-filter columns (`arch_layer`, `jsdoc_quality`, `test_coverage`,
 * `source_path_pattern`) to `documents` and `chunks`, plus the partial indexes
 * required by the Turso vector query filtering paths.
 *
 * @param {{ client?: import('@libsql/client').Client, databasePath?: string }} [options]
 * @returns {Promise<import('@libsql/client').Client>} The client used for the migration.
 */
export async function migrateSchemaV2ToV3(options = {}) {
  const client = await resolveClient(options);

  for (const column of V3_COLUMNS) {
    await addColumnIfMissing(client, column.table, column);
  }

  await ensureIndexes(client, V3_INDEXES);
  await recordSchemaVersion(client, 3);

  return client;
}
