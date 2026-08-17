/**
 * @module init-schema.test
 * @description Branch-coverage tests for initSemanticIndex.
 *
 * Exercises every branch in init-schema.mjs:
 * - Early return when `options.client` is provided
 * - `options.databasePath ?? defaultDatabasePath` (nullish and non-nullish)
 * - splitSqlStatements: trigger-with-BEGIN, non-trigger lines, END lines,
 *   regular statements, comment lines, CREATE TRIGGER without BEGIN on same
 *   line, and trailing statement without semicolon (tail buffer)
 * - applySchemaToClient: batch loop entry and exit
 * - migrateSliceMetadataColumns: columns present (skip) and columns missing
 *   (ALTER TABLE)
 *
 * Uses jest.unstable_mockModule to mock `readFile` from `node:fs/promises`
 * so custom SQL can be injected for edge-case branch coverage while keeping
 * `mkdir` real so directories are created normally.
 */

import { jest } from '@jest/globals';
import { createRequire } from 'node:module';
import { createClient } from '@libsql/client';
import { readFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Obtain real fs/promises before mocking so mkdir/unlink stay functional.
const require = createRequire(import.meta.url);
const realFsPromises = require('fs/promises');

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCHEMA_PATH = path.resolve(__dirname, '..', 'schema-turso.sql');
const realSchemaSql = readFileSync(SCHEMA_PATH, 'utf8');

// Schema with slice-metadata column definitions removed so the migration
// exercises the ALTER TABLE (column-missing) branch.
const schemaWithoutSliceColumns = realSchemaSql
  .replace(/^\s+slice_id TEXT,\s*$/gm, '')
  .replace(/^\s+step_number INTEGER,\s*$/gm, '')
  .replace(/^\s+phase TEXT,\s*$/gm, '')
  .replace(/^\s+status TEXT,\s*$/gm, '');

// Edge-case SQL: a single-line CREATE TRIGGER (BEGIN not at end of line so
// the && short-circuits with left=true/right=false) followed by a trailing
// statement without a semicolon (exercises the tail-buffer branch).
const edgeCaseSql = [
  'CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY, body_text TEXT, heading_path TEXT, slice_id TEXT, step_number INTEGER, phase TEXT, status TEXT);',
  'CREATE TABLE IF NOT EXISTS edge_t (x INTEGER);',
  'CREATE TRIGGER edge_t_ai AFTER INSERT ON edge_t BEGIN SELECT 1; END;',
  'CREATE TABLE IF NOT EXISTS edge_t2 (y INTEGER)',
].join('\n');

// SQL with a standalone-semicolon line to exercise the `if (stmt)` false
// branch in the non-trigger path of splitSqlStatements (line 48). The `;`
// line produces an empty stmt after trim + replace, so it is skipped.
const standaloneSemicolonSql = [
  'CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY, body_text TEXT, heading_path TEXT, slice_id TEXT, step_number INTEGER, phase TEXT, status TEXT);',
  ';',
  'CREATE TABLE IF NOT EXISTS semi_t (z INTEGER);',
].join('\n');

// Mock readFile to control schema SQL; keep real mkdir.
const mockReadFile = jest.fn();
jest.unstable_mockModule('node:fs/promises', () => ({
  mkdir: realFsPromises.mkdir,
  readFile: mockReadFile,
}));

// Import after mock is registered.
const { initSemanticIndex } = await import('../init-schema.mjs');

/**
 * Generate a unique temp SQLite file path.
 *
 * @returns {string} Absolute path inside the OS temp directory.
 */
function tempDbPath() {
  return path.join(
    os.tmpdir(),
    `init-schema-test-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`,
  );
}

describe('initSemanticIndex', () => {
  let clients;
  let tempFiles;

  beforeEach(() => {
    clients = [];
    tempFiles = [];
    mockReadFile.mockReset();
    mockReadFile.mockResolvedValue(realSchemaSql);
  });

  afterEach(async () => {
    for (const c of clients) {
      try {
        await c.close();
      } catch {
        // ignore close errors
      }
    }
    for (const f of tempFiles) {
      try {
        await realFsPromises.unlink(f);
      } catch {
        // ignore unlink errors
      }
    }
  });

  // -------------------------------------------------------------------------
  // Branch: if (options.client) → true (early return)
  // -------------------------------------------------------------------------

  it('returns the provided client directly without touching the filesystem', async () => {
    const client = createClient({ url: ':memory:' });
    const result = await initSemanticIndex({ client });
    expect(result).toBe(client);
    await client.close();
  });

  // -------------------------------------------------------------------------
  // Branch: options.databasePath ?? defaultDatabasePath → false (provided)
  // Full path: mkdir, createClient, applySchemaToClient, migration (columns exist)
  // -------------------------------------------------------------------------

  it('creates a new database with the full schema when no client is provided', async () => {
    const dbPath = tempDbPath();
    tempFiles.push(dbPath);
    const client = await initSemanticIndex({ databasePath: dbPath });
    clients.push(client);

    const tables = await client.execute(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
    );
    const tableNames = tables.rows.map((r) => r.name);
    expect(tableNames).toContain('documents');
    expect(tableNames).toContain('chunks');
    expect(tableNames).toContain('chunks_fts');
  });

  // -------------------------------------------------------------------------
  // Branch: options.databasePath ?? defaultDatabasePath → true (undefined)
  // -------------------------------------------------------------------------

  it('uses the default database path when databasePath is not specified', async () => {
    const client = await initSemanticIndex({});
    clients.push(client);

    const result = await client.execute('SELECT 1 AS ok');
    expect(result.rows[0].ok).toBe(1);
  });

  // -------------------------------------------------------------------------
  // Branch: migrateSliceMetadataColumns → if (!existingColumns.has(name)) true
  // -------------------------------------------------------------------------

  it('migrates missing slice metadata columns via ALTER TABLE', async () => {
    const dbPath = tempDbPath();
    tempFiles.push(dbPath);
    mockReadFile.mockResolvedValue(schemaWithoutSliceColumns);

    const client = await initSemanticIndex({ databasePath: dbPath });
    clients.push(client);

    const info = await client.execute({ sql: 'PRAGMA table_info(chunks)' });
    const columnNames = info.rows.map((r) => r.name);
    expect(columnNames).toContain('slice_id');
    expect(columnNames).toContain('step_number');
    expect(columnNames).toContain('phase');
    expect(columnNames).toContain('status');
  });

  // -------------------------------------------------------------------------
  // Branch: splitSqlStatements — CREATE TRIGGER without BEGIN at end of line
  //         (left=true, right=false for && in trigger detection)
  // Branch: splitSqlStatements — if (tail) true (trailing statement without ;)
  // -------------------------------------------------------------------------

  it('handles single-line triggers and trailing statements without semicolons', async () => {
    const dbPath = tempDbPath();
    tempFiles.push(dbPath);
    mockReadFile.mockResolvedValue(edgeCaseSql);

    const client = await initSemanticIndex({ databasePath: dbPath });
    clients.push(client);

    // The single-line trigger should have been created.
    const triggers = await client.execute(
      "SELECT name FROM sqlite_master WHERE type='trigger' AND name='edge_t_ai'",
    );
    expect(triggers.rows.length).toBe(1);

    // The trailing table (no semicolon) should also have been created.
    const tables = await client.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='edge_t2'",
    );
    expect(tables.rows.length).toBe(1);
  });

  // -------------------------------------------------------------------------
  // Branch: splitSqlStatements — if (stmt) false in non-trigger path (line 48)
  //         A standalone-semicolon line produces an empty stmt that is skipped.
  // -------------------------------------------------------------------------

  it('skips standalone-semicolon lines without producing empty statements', async () => {
    const dbPath = tempDbPath();
    tempFiles.push(dbPath);
    mockReadFile.mockResolvedValue(standaloneSemicolonSql);

    const client = await initSemanticIndex({ databasePath: dbPath });
    clients.push(client);

    // The table after the standalone semicolon should still be created.
    const tables = await client.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='semi_t'",
    );
    expect(tables.rows.length).toBe(1);
  });

  // -------------------------------------------------------------------------
  // Branch: options = {} default parameter (line 106)
  //         Calling initSemanticIndex() with no arguments uses the default.
  // -------------------------------------------------------------------------

  it('uses default parameter when called with no arguments', async () => {
    const client = await initSemanticIndex();
    clients.push(client);

    const result = await client.execute('SELECT 1 AS ok');
    expect(result.rows[0].ok).toBe(1);
  });
});