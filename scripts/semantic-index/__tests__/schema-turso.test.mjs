/**
 * @module schema-turso.test
 * @description Red tests for the Turso/libSQL consolidated schema.
 *
 * These tests load `scripts/semantic-index/schema-turso.sql`, apply it to an
 * in-memory libSQL database via `@libsql/client`, and assert that the schema
 * structure matches the Turso RAG Migration requirements:
 *
 * - All tables from schema-v2.sql are present.
 * - `chunk_embeddings` is merged into `chunks` as `embedding F8_BLOB(384)`.
 * - `term_embeddings.embedding` is `F8_BLOB(384)`.
 * - DiskANN vector indexes (`libsql_vector_idx`) are declared.
 * - `_schema_version` table replaces `PRAGMA user_version`.
 * - `_index_metadata` table replaces `PRAGMA application_id`.
 * - FTS5 virtual table (`chunks_fts`) and its sync triggers are present.
 * - No `PRAGMA user_version` or `VACUUM` statements.
 *
 * Pure .mjs test — runs directly via Jest ESM project (no ts-jest).
 *
 * RED PHASE: schema-turso.sql does not exist yet, so every test fails because
 * the schema file cannot be loaded.
 */

import { createClient } from '@libsql/client';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SCHEMA_PATH = path.resolve(__dirname, '..', 'schema-turso.sql');

/**
 * Read the schema-turso.sql file from disk.
 *
 * @returns {Promise<string>} Raw SQL text.
 * @throws {Error} If the file does not exist or cannot be read.
 */
async function readSchemaSql() {
  return readFile(SCHEMA_PATH, 'utf8');
}

/**
 * Split a multi-statement SQL script into individual statements.
 *
 * Respects `BEGIN ... END;` trigger bodies so that semicolons inside trigger
 * bodies are not treated as statement terminators.
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
 * Apply a multi-statement SQL schema to an in-memory libSQL client.
 *
 * @param {string} schemaSql - Raw SQL schema text.
 * @returns {Promise<object>} A connected libSQL client with the schema applied.
 */
async function applySchema(schemaSql) {
  const client = createClient({ url: ':memory:' });
  const statements = splitSqlStatements(schemaSql);
  await client.batch(
    statements.map((sql) => ({ sql, args: [] })),
    'write',
  );
  return client;
}

/**
 * Query sqlite_master for table names (excluding FTS5 shadow tables).
 *
 * @param {object} client - libSQL client.
 * @returns {Promise<string[]>} Sorted list of table names.
 */
async function getTableNames(client) {
  const result = await client.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
  );
  return result.rows.map((r) => r.name);
}

/**
 * Query sqlite_master for trigger names.
 *
 * @param {object} client - libSQL client.
 * @returns {Promise<string[]>} Sorted list of trigger names.
 */
async function getTriggerNames(client) {
  const result = await client.execute(
    "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name",
  );
  return result.rows.map((r) => r.name);
}

/**
 * Query sqlite_master for index names.
 *
 * @param {object} client - libSQL client.
 * @returns {Promise<string[]>} Sorted list of index names.
 */
async function getIndexNames(client) {
  const result = await client.execute(
    "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name",
  );
  return result.rows.map((r) => r.name);
}

/**
 * Query the full sqlite_master DDL for a specific object.
 *
 * @param {object} client - libSQL client.
 * @param {string} name - Object name.
 * @returns {Promise<string|null>} The original SQL DDL or null.
 */
async function getObjectDdl(client, name) {
  const result = await client.execute({
    sql: 'SELECT sql FROM sqlite_master WHERE name = ?',
    args: [name],
  });
  if (result.rows.length === 0) return null;
  return result.rows[0].sql;
}

/**
 * Return PRAGMA table info for a given table.
 *
 * @param {object} client - libSQL client.
 * @param {string} table - Table name.
 * @returns {Promise<Array<object>>} Column info rows.
 */
async function getTableInfo(client, table) {
  const result = await client.execute({
    sql: `PRAGMA table_info(${table})`,
    args: [],
  });
  return result.rows.map((r) => ({
    cid: r.cid,
    name: r.name,
    type: r.type,
    notnull: r.notnull,
    dflt_value: r.dflt_value,
    pk: r.pk,
  }));
}

// ---------------------------------------------------------------------------
// Red tests — schema-turso.sql structure
// ---------------------------------------------------------------------------

describe('schema-turso.sql', () => {
  describe('file loading', () => {
    it('exists and can be read as UTF-8 text', async () => {
      const sql = await readSchemaSql();
      expect(typeof sql).toBe('string');
      expect(sql.length).toBeGreaterThan(0);
    });
  });

  describe('schema application', () => {
    it('applies cleanly to an in-memory libSQL database', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables.length).toBeGreaterThan(0);
    });
  });

  describe('required tables', () => {
    it('creates the documents table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('documents');
    });

    it('creates the chunks table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('chunks');
    });

    it('does not create a standalone chunk_embeddings table (merged into chunks)', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).not.toContain('chunk_embeddings');
    });

    it('creates the term_embeddings table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('term_embeddings');
    });

    it('creates the entities table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('entities');
    });

    it('creates the edges table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('edges');
    });

    it('creates the feedback_events table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('feedback_events');
    });

    it('creates the feedback_scores table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('feedback_scores');
    });

    it('creates the _schema_version table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('_schema_version');
    });

    it('creates the _index_metadata table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const tables = await getTableNames(client);
      expect(tables).toContain('_index_metadata');
    });
  });

  describe('chunks table vector columns', () => {
    it('has an embedding column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, 'chunks');
      const names = columns.map((c) => c.name);
      expect(names).toContain('embedding');
    });

    it('declares embedding column as F8_BLOB(384)', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, 'chunks');
      const embeddingCol = columns.find((c) => c.name === 'embedding');
      expect(embeddingCol.type.toUpperCase()).toMatch(
        /F8_BLOB\s*\(\s*384\s*\)/,
      );
    });

    it('has an embedding_model TEXT column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, 'chunks');
      const col = columns.find((c) => c.name === 'embedding_model');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('TEXT');
    });

    it('has a chunk_sha256 TEXT column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, 'chunks');
      const col = columns.find((c) => c.name === 'chunk_sha256');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('TEXT');
    });

    it('has an embedded_at INTEGER column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, 'chunks');
      const col = columns.find((c) => c.name === 'embedded_at');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('INTEGER');
    });
  });

  describe('term_embeddings vector column', () => {
    it('declares embedding column as F8_BLOB(384)', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, 'term_embeddings');
      const embeddingCol = columns.find((c) => c.name === 'embedding');
      expect(embeddingCol.type.toUpperCase()).toMatch(
        /F8_BLOB\s*\(\s*384\s*\)/,
      );
    });
  });

  describe('DiskANN vector indexes', () => {
    it('declares a libsql_vector_idx index on chunks.embedding', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const ddl = await getObjectDdl(client, 'chunks_embedding_idx');
      expect(ddl).not.toBeNull();
      expect(ddl).toMatch(/libsql_vector_idx\s*\(\s*embedding\s*\)/i);
    });

    it('declares a libsql_vector_idx index on term_embeddings.embedding', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const ddl = await getObjectDdl(client, 'term_embeddings_embedding_idx');
      expect(ddl).not.toBeNull();
      expect(ddl).toMatch(/libsql_vector_idx\s*\(\s*embedding\s*\)/i);
    });
  });

  describe('_schema_version table', () => {
    it('has a version INTEGER column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, '_schema_version');
      const col = columns.find((c) => c.name === 'version');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('INTEGER');
    });

    it('has an applied_at INTEGER column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, '_schema_version');
      const col = columns.find((c) => c.name === 'applied_at');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('INTEGER');
    });
  });

  describe('_index_metadata table', () => {
    it('has a key TEXT PRIMARY KEY column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, '_index_metadata');
      const col = columns.find((c) => c.name === 'key');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('TEXT');
      expect(col.pk).toBe(1);
    });

    it('has a value TEXT column', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const columns = await getTableInfo(client, '_index_metadata');
      const col = columns.find((c) => c.name === 'value');
      expect(col).toBeDefined();
      expect(col.type.toUpperCase()).toBe('TEXT');
    });
  });

  describe('FTS5 full-text search', () => {
    it('creates the chunks_fts FTS5 virtual table', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const result = await client.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'",
      );
      expect(result.rows.length).toBe(1);
      expect(result.rows[0].sql).toMatch(/fts5/i);
    });

    it('uses porter unicode61 tokenizer', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const result = await client.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'",
      );
      expect(result.rows[0].sql).toMatch(/porter unicode61/i);
    });

    it('creates the chunks_ai insert trigger', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const triggers = await getTriggerNames(client);
      expect(triggers).toContain('chunks_ai');
    });

    it('creates the chunks_ad delete trigger', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const triggers = await getTriggerNames(client);
      expect(triggers).toContain('chunks_ad');
    });

    it('creates the chunks_au update trigger', async () => {
      const sql = await readSchemaSql();
      const client = await applySchema(sql);
      const triggers = await getTriggerNames(client);
      expect(triggers).toContain('chunks_au');
    });
  });

  describe('forbidden statements', () => {
    it('does not contain PRAGMA user_version', async () => {
      const sql = await readSchemaSql();
      expect(sql).not.toMatch(/PRAGMA\s+user_version/i);
    });

    it('does not contain a VACUUM statement', async () => {
      const sql = await readSchemaSql();
      expect(sql).not.toMatch(/(^|[\s;])VACUUM\b/i);
    });
  });
});
