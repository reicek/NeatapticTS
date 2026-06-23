/**
 * @module cortex-db.turso.test
 * @description Red tests for async Turso client migration of cortex-db.mjs.
 *
 * Phase 3, Step 03, Slice 03-red-cortex-db.
 *
 * These tests verify the expected async API surface after the migration from
 * the synchronous local SQLite driver to @libsql/client:
 *
 * - getTursoClient() returns an async @libsql/client Client (cached per URL)
 * - readChunk() is async and returns normalized chunk data
 * - resolveDatabasePath() respects TURSO_DATABASE_URL env var
 * - Embedded replica configuration from TURSO_SYNC_URL / TURSO_SYNC_INTERVAL
 * - Pure-logic helpers (sanitizeFtsQuery, normalizeLimit, normalizeRepoPath)
 *   remain unchanged (regression guards — these PASS in red phase)
 *
 * RED PHASE: cortex-db.mjs still uses the synchronous local SQLite driver and
 * does not export getTursoClient() or readChunk(). resolveDatabasePath() does not check
 * TURSO_DATABASE_URL. The new API tests fail because the exports do not exist
 * or do not exhibit the expected behavior. The pure-logic tests pass because
 * those functions are unchanged.
 *
 * Pure .mjs test — runs via Jest ESM project (mcp-semantic-mjs).
 */

import { createClient } from '@libsql/client';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { defaultDatabasePath } from '../../semantic-index/init-schema.mjs';

const MODULE_PATH = '../tools/cortex-db.mjs';

/**
 * Env vars managed by these tests. Saved and restored around each test to
 * ensure deterministic isolation.
 */
const MANAGED_ENV_VARS = [
  'TURSO_DATABASE_URL',
  'TURSO_AUTH_TOKEN',
  'TURSO_SYNC_URL',
  'TURSO_SYNC_INTERVAL',
  'CORTEX_DB_PATH',
];

const savedEnv = {};

beforeEach(() => {
  for (const key of MANAGED_ENV_VARS) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of MANAGED_ENV_VARS) {
    if (savedEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = savedEnv[key];
    }
  }
});

/**
 * Dynamically import the module under test.
 *
 * Using dynamic import (rather than top-level) so that importing a
 * non-existent named export yields `undefined` instead of a SyntaxError
 * that would crash the entire test file.
 *
 * @returns {Promise<Record<string, unknown>>} Module namespace object.
 */
async function loadModule() {
  return import(MODULE_PATH);
}

/**
 * Create a `:memory:` libSQL client preloaded with a minimal documents +
 * chunks schema and a single test chunk.
 *
 * @returns {Promise<import('@libsql/client').Client>} Configured in-memory client.
 */
async function setupTestDb() {
  const client = createClient({ url: ':memory:' });

  await client.execute({
    sql: `CREATE TABLE documents (
      doc_id INTEGER PRIMARY KEY,
      file_path TEXT NOT NULL UNIQUE,
      doc_family TEXT NOT NULL,
      mtime_ms INTEGER NOT NULL,
      file_size INTEGER NOT NULL,
      sha256 TEXT NOT NULL,
      indexed_at INTEGER NOT NULL,
      arch_layer TEXT,
      test_coverage TEXT,
      source_path_pattern TEXT
    )`,
  });

  await client.execute({
    sql: `CREATE TABLE chunks (
      chunk_id INTEGER PRIMARY KEY,
      doc_id INTEGER NOT NULL REFERENCES documents(doc_id),
      chunk_index INTEGER NOT NULL,
      heading_path TEXT,
      body_text TEXT NOT NULL,
      char_start INTEGER NOT NULL,
      char_end INTEGER NOT NULL,
      parent_chunk_id INTEGER,
      depth INTEGER NOT NULL DEFAULT 0,
      context_header TEXT,
      symbol_name TEXT,
      signature_text TEXT,
      jsdoc_text TEXT,
      export_type TEXT,
      module_path TEXT,
      arch_layer TEXT,
      jsdoc_quality TEXT,
      jsdoc_word_count INTEGER,
      cyclomatic_complexity INTEGER,
      test_coverage TEXT,
      source_path_pattern TEXT
    )`,
  });

  await client.execute({
    sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
          VALUES (1, 'src/test.ts', 'src', 1000, 500, 'abc123', 1000)`,
  });

  await client.execute({
    sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth)
          VALUES (1, 1, 0, 'TestModule', 'test content here', 0, 17, 0)`,
  });

  return client;
}

// ---------------------------------------------------------------------------
// getTursoClient — async @libsql/client Client factory (NEW API)
// ---------------------------------------------------------------------------

describe('getTursoClient', () => {
  it('is an exported function', async () => {
    const mod = await loadModule();
    expect(typeof mod.getTursoClient).toBe('function');
  });

  it('returns a Promise when called (async)', async () => {
    const { getTursoClient } = await loadModule();
    const result = getTursoClient(':memory:');
    expect(result).toBeInstanceOf(Promise);
  });

  it('resolves to a Client with execute and batch methods', async () => {
    const { getTursoClient } = await loadModule();
    const client = await getTursoClient(':memory:');
    expect(typeof client.execute).toBe('function');
  });

  it('caches client instances per database URL (same URL returns same client)', async () => {
    const { getTursoClient } = await loadModule();
    const clientA = await getTursoClient(':memory:');
    const clientB = await getTursoClient(':memory:');
    expect(clientA).toBe(clientB);
  });

  it('configures embedded replica from TURSO_SYNC_URL and TURSO_SYNC_INTERVAL env vars', async () => {
    const tempDir = await mkdtemp(path.join(tmpdir(), 'turso-cortex-test-'));
    try {
      const dbPath = path.join(tempDir, 'replica.sqlite');
      process.env.TURSO_DATABASE_URL = `file:${dbPath}`;
      process.env.TURSO_SYNC_URL = 'libsql://dummy.turso.io';
      process.env.TURSO_SYNC_INTERVAL = '60';

      const { getTursoClient } = await loadModule();
      const client = await getTursoClient();
      expect(typeof client.execute).toBe('function');
      client.close?.();
    } finally {
      // On Windows, the native SQLite file handle may not release
      // immediately after client.close(). Use retries to handle EBUSY.
      await rm(tempDir, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    }
  });
});

// ---------------------------------------------------------------------------
// readChunk — async chunk reader using client.execute() (NEW API)
// ---------------------------------------------------------------------------

describe('readChunk', () => {
  it('is an exported function', async () => {
    const mod = await loadModule();
    expect(typeof mod.readChunk).toBe('function');
  });

  it('returns a Promise when called (async)', async () => {
    const client = await setupTestDb();
    try {
      const { readChunk } = await loadModule();
      const result = readChunk(client, 1);
      expect(result).toBeInstanceOf(Promise);
    } finally {
      client.close?.();
    }
  });

  it('returns correct chunk data for a given chunk ID', async () => {
    const client = await setupTestDb();
    try {
      const { readChunk } = await loadModule();
      const chunk = await readChunk(client, 1);
      expect(chunk).toMatchObject({
        chunk_id: 1,
        file_path: 'src/test.ts',
        family: 'src',
        chunk_index: 0,
        heading_path: 'TestModule',
        text: 'test content here',
        char_start: 0,
        char_end: 17,
        depth: 0,
      });
    } finally {
      client.close?.();
    }
  });
});

// ---------------------------------------------------------------------------
// resolveDatabasePath — TURSO_DATABASE_URL env var resolution (UPDATED)
// ---------------------------------------------------------------------------

describe('resolveDatabasePath', () => {
  it('uses TURSO_DATABASE_URL when set', async () => {
    process.env.TURSO_DATABASE_URL = 'libsql://test.turso.io';
    const { resolveDatabasePath } = await loadModule();
    expect(resolveDatabasePath()).toBe('libsql://test.turso.io');
  });

  it('falls back to file path when TURSO_DATABASE_URL is unset', async () => {
    const { resolveDatabasePath } = await loadModule();
    expect(resolveDatabasePath()).toBe(path.resolve(defaultDatabasePath));
  });
});

// ---------------------------------------------------------------------------
// Pure-logic helpers — unchanged regression guards (PASS in red phase)
// ---------------------------------------------------------------------------

describe('sanitizeFtsQuery (unchanged pure logic)', () => {
  it('applies prefix wildcards to plain words', async () => {
    const { sanitizeFtsQuery } = await loadModule();
    expect(sanitizeFtsQuery('hello world')).toBe('hello* world*');
  });
});

describe('normalizeLimit (unchanged pure logic)', () => {
  it('clamps values above 50 to 50', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(100)).toBe(50);
  });
});

describe('normalizeRepoPath (unchanged pure logic)', () => {
  it('rejects path traversal attempts', async () => {
    const { normalizeRepoPath } = await loadModule();
    expect(() => normalizeRepoPath('../../etc/passwd')).toThrow();
  });
});
