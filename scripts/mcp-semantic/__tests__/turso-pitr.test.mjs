/**
 * @module turso-pitr.test
 * @description Red tests for Phase 6 Step 05 slice 05-red-pitr — turso_pitr MCP tool.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because:
 * - `scripts/mcp-semantic/tools/turso-pitr.mjs` does not exist yet, so
 *   dynamic `import()` of the module throws and `readFileSync` of the source
 *   file throws ENOENT.
 * - `turso_pitr` is not registered in the MCP tool list yet.
 * - `repo-cortex-mcp.mjs` does not yet import from `turso-pitr.mjs`.
 *
 * Coverage targets (per plan acceptance criteria):
 * - turso_pitr tool creates a database from a timestamp via Platform API.
 * - Timestamp configurable (within PITR retention window).
 * - Tool registered in repo-cortex-mcp.mjs.
 * - 100% coverage on touched files.
 *
 * Pure .mjs test — runs via Jest ESM project `mcp-semantic-mjs`.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { jest } from '@jest/globals';

import { rm } from 'node:fs/promises';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';
import { closeTursoClient } from '../tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TURSO_PITR_PATH = path.resolve(
  __dirname,
  '..',
  'tools',
  'turso-pitr.mjs',
);
const REPO_CORTEX_MCP_PATH = path.resolve(
  __dirname,
  '..',
  'repo-cortex-mcp.mjs',
);

/**
 * Read a source file as UTF-8 text.
 *
 * @param {string} filePath - Absolute path to the source file.
 * @returns {string} File contents.
 */
function readSource(filePath) {
  return readFileSync(filePath, 'utf8');
}

/**
 * Set up an in-memory SQLite database with the v2 corpus schema.
 *
 * Uses `:memory:` so repeated test cycles do not exhaust Windows file handles.
 * The schema/tools-list tests only inspect `tools/list` output, which never
 * opens the database, so an in-memory client is sufficient.
 *
 * @returns {Promise<{ client: object, dbPath: string, tempDir: string }>} Setup state.
 */
async function setupDb() {
  const client = createClient({ url: ':memory:' });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  await client.execute(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES ('src/network.ts', 'ts-source', 0, 100, 'a', 1);
  `);
  const docResult = await client.execute('SELECT doc_id FROM documents');
  const docId = docResult.rows[0].doc_id;
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer)
    VALUES (?, 0, 'NEAT activation function in network', 0, 35, 0, 'network')`,
    args: [docId],
  });
  return { client, dbPath: ':memory:', tempDir: null };
}

/**
 * Tear down a temporary database and its directory.
 *
 * @param {object} client - libSQL client.
 * @param {string} tempDir - Temporary directory path.
 * @param {string} dbPath - SQLite file path.
 * @returns {Promise<void>}
 */
async function teardownDb(client, tempDir, dbPath) {
  await client.close();
  if (dbPath && dbPath !== ':memory:') await closeTursoClient(dbPath);
  if (tempDir) {
    await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
  }
}

// ---------------------------------------------------------------------------
// Group 1: turso-pitr.mjs module exists and exports
// ---------------------------------------------------------------------------

describe('turso-pitr.mjs: module exists and exports', () => {
  it('turso-pitr.mjs exports tursoPitr', async () => {
    const mod = await import(TURSO_PITR_PATH);
    expect(typeof mod.tursoPitr).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Group 2: Platform API usage (source-pattern assertions)
// ---------------------------------------------------------------------------

describe('turso-pitr.mjs: Platform API usage', () => {
  it('turso-pitr.mjs source references the Turso Platform API URL (api.turso.tech)', () => {
    const source = readSource(TURSO_PITR_PATH);
    expect(source).toMatch(/api\.turso\.tech/);
  });

  it('turso-pitr.mjs source uses POST for database creation with a seed', () => {
    const source = readSource(TURSO_PITR_PATH);
    expect(source).toMatch(/POST/);
  });

  it('turso-pitr.mjs source uses a seed block for point-in-time recovery', () => {
    const source = readSource(TURSO_PITR_PATH);
    expect(source).toMatch(/seed/);
  });

  it('turso-pitr.mjs source references a timestamp seed type', () => {
    const source = readSource(TURSO_PITR_PATH);
    expect(source).toMatch(/timestamp/i);
  });
});

// ---------------------------------------------------------------------------
// Group 3: Timestamp configurable (source-pattern)
// ---------------------------------------------------------------------------

describe('turso-pitr.mjs: timestamp configurable', () => {
  it('turso-pitr.mjs source accepts a timestamp parameter', () => {
    const source = readSource(TURSO_PITR_PATH);
    expect(source).toMatch(/timestamp/i);
  });
});

// ---------------------------------------------------------------------------
// Group 4: PITR database creation behavior (functional with mocked fetch)
// ---------------------------------------------------------------------------

describe('turso-pitr.mjs: PITR database creation behavior', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = originalFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('tursoPitr creates a database from a timestamp by POSTing to the Platform API and returns database info', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        name: 'corpus-pitr',
        hostname: 'libsql://corpus-pitr.turso.io',
      }),
    }));
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    const result = await tursoPitr({
      databaseName: 'corpus-pitr',
      organization: 'test-org',
      apiToken: 'test-token',
      sourceDatabaseName: 'corpus',
      timestamp: '2026-01-01T00:00:00Z',
    });
    expect(result.databaseName ?? result.name).toBeDefined();
  });

  it('tursoPitr throws when Platform API returns non-2xx status', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: false,
      status: 400,
      json: async () => ({ error: 'bad request' }),
    }));
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        organization: 'test-org',
        apiToken: 'test-token',
        sourceDatabaseName: 'corpus',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Group 5: turso_pitr MCP tool registered
// ---------------------------------------------------------------------------

describe('turso_pitr MCP tool registration', () => {
  it('registers turso_pitr in the MCP tool list', async () => {
    const { createRepoCortexMcpServer } = await import('../repo-cortex-mcp.mjs');
    const { client, dbPath, tempDir } = await setupDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const listed = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
      });
      expect(listed.tools.map((tool) => tool.name)).toContain('turso_pitr');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('repo-cortex-mcp.mjs imports tursoPitr from turso-pitr.mjs', () => {
    const source = readSource(REPO_CORTEX_MCP_PATH);
    expect(source).toMatch(/turso-pitr\.mjs/);
  });
});

// ---------------------------------------------------------------------------
// Group 6: tool schema completeness
// ---------------------------------------------------------------------------

describe('turso_pitr MCP tool schema completeness', () => {
  it('turso_pitr tool inputSchema includes a timestamp property', async () => {
    const { createRepoCortexMcpServer } = await import('../repo-cortex-mcp.mjs');
    const { client, dbPath, tempDir } = await setupDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const listed = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
      });
      const tool = listed.tools.find((t) => t.name === 'turso_pitr');
      expect(Object.keys(tool.inputSchema.properties)).toContain('timestamp');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('turso_pitr tool inputSchema includes a database_name property for the recovered database', async () => {
    const { createRepoCortexMcpServer } = await import('../repo-cortex-mcp.mjs');
    const { client, dbPath, tempDir } = await setupDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const listed = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
      });
      const tool = listed.tools.find((t) => t.name === 'turso_pitr');
      expect(Object.keys(tool.inputSchema.properties)).toContain('database_name');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });
});

// ---------------------------------------------------------------------------
// Group 7: validation and edge cases (coverage gap closure)
// ---------------------------------------------------------------------------

describe('turso-pitr.mjs: validation and edge cases', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = originalFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('safeReadJson returns null when response body is empty (non-2xx still rejects)', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: false,
      status: 500,
      text: async () => '',
    }));
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        organization: 'test-org',
        apiToken: 'test-token',
        sourceDatabaseName: 'corpus',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toBeDefined();
  });

  it('safeReadJson parses valid JSON text from a non-2xx response (still rejects)', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: false,
      status: 500,
      text: async () => '{"error":"server error"}',
    }));
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        organization: 'test-org',
        apiToken: 'test-token',
        sourceDatabaseName: 'corpus',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toBeDefined();
  });

  it('throws when databaseName is missing', async () => {
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        organization: 'test-org',
        apiToken: 'test-token',
        sourceDatabaseName: 'corpus',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toThrow('tursoPitr requires a databaseName');
  });

  it('throws when organization is missing', async () => {
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        apiToken: 'test-token',
        sourceDatabaseName: 'corpus',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toThrow('tursoPitr requires an organization');
  });

  it('throws when apiToken is missing', async () => {
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        organization: 'test-org',
        sourceDatabaseName: 'corpus',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toThrow('tursoPitr requires an apiToken');
  });

  it('throws when sourceDatabaseName is missing', async () => {
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        organization: 'test-org',
        apiToken: 'test-token',
        timestamp: '2026-01-01T00:00:00Z',
      }),
    ).rejects.toThrow('tursoPitr requires a sourceDatabaseName');
  });

  it('throws when timestamp is missing', async () => {
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(
      tursoPitr({
        databaseName: 'corpus-pitr',
        organization: 'test-org',
        apiToken: 'test-token',
        sourceDatabaseName: 'corpus',
      }),
    ).rejects.toThrow('tursoPitr requires a timestamp');
  });

  it('throws when called with no arguments (default param branch)', async () => {
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    await expect(tursoPitr()).rejects.toThrow('tursoPitr requires a databaseName');
  });

  it('falls back to the passed databaseName when API response omits name', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ hostname: 'libsql://no-name.turso.io' }),
    }));
    const { tursoPitr } = await import(TURSO_PITR_PATH);
    const result = await tursoPitr({
      databaseName: 'fallback-pitr',
      organization: 'test-org',
      apiToken: 'test-token',
      sourceDatabaseName: 'corpus',
      timestamp: '2026-01-01T00:00:00Z',
    });
    expect(result.databaseName).toBe('fallback-pitr');
  });
});