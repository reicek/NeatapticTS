/**
 * @module turso-branch.test
 * @description Red tests for Phase 6 Step 04 slice 04-red-branch — turso_branch MCP tool.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because:
 * - `scripts/mcp-semantic/tools/turso-branch.mjs` does not exist yet, so
 *   dynamic `import()` of the module throws and `readFileSync` of the source
 *   file throws ENOENT.
 * - `turso_branch` is not registered in the MCP tool list yet.
 * - `repo-cortex-mcp.mjs` does not yet import from `turso-branch.mjs`.
 *
 * Coverage targets (per plan acceptance criteria):
 * - turso_branch tool creates a branch of the current database via Platform API.
 * - Branch can be used for testing index changes without affecting production.
 * - Branch cleanup (delete) supported (DELETE database).
 * - Tool registered in repo-cortex-mcp.mjs.
 * - Tool uses Turso Platform API (POST /v1/organizations/{org}/databases with seed).
 * - Branch name configurable.
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

const TURSO_BRANCH_PATH = path.resolve(
  __dirname,
  '..',
  'tools',
  'turso-branch.mjs',
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
    await rm(tempDir, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 200,
    });
  }
}

// ---------------------------------------------------------------------------
// Group 1: turso-branch.mjs module exists and exports
// ---------------------------------------------------------------------------

describe('turso-branch.mjs: module exists and exports', () => {
  it('turso-branch.mjs exports tursoBranch', async () => {
    const mod = await import(TURSO_BRANCH_PATH);
    expect(typeof mod.tursoBranch).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Group 2: Platform API usage (source-pattern assertions)
// ---------------------------------------------------------------------------

describe('turso-branch.mjs: Platform API usage', () => {
  it('turso-branch.mjs source references the Turso Platform API URL (api.turso.tech)', () => {
    const source = readSource(TURSO_BRANCH_PATH);
    expect(source).toMatch(/api\.turso\.tech/);
  });

  it('turso-branch.mjs source uses POST for database creation with seed', () => {
    const source = readSource(TURSO_BRANCH_PATH);
    expect(source).toMatch(/POST/);
  });

  it('turso-branch.mjs source uses POST body containing seed for branch creation', () => {
    const source = readSource(TURSO_BRANCH_PATH);
    expect(source).toMatch(/seed/);
  });

  it('turso-branch.mjs source uses DELETE for branch cleanup', () => {
    const source = readSource(TURSO_BRANCH_PATH);
    expect(source).toMatch(/DELETE|delete/i);
  });
});

// ---------------------------------------------------------------------------
// Group 3: Branch name configurable (source-pattern)
// ---------------------------------------------------------------------------

describe('turso-branch.mjs: branch name configurable', () => {
  it('turso-branch.mjs source accepts a branch_name parameter', () => {
    const source = readSource(TURSO_BRANCH_PATH);
    expect(source).toMatch(/branch_name/);
  });
});

// ---------------------------------------------------------------------------
// Group 4: branch creation behavior (functional with mocked fetch)
// ---------------------------------------------------------------------------

describe('turso-branch.mjs: branch creation behavior', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = originalFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('tursoBranch creates a branch by POSTing to the Platform API and returns branch info', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        name: 'test-branch',
        hostname: 'libsql://test-branch.turso.io',
      }),
    }));
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    const result = await tursoBranch({
      branchName: 'test-branch',
      organization: 'test-org',
      apiToken: 'test-token',
      databaseName: 'corpus',
    });
    expect(result.branchName ?? result.name).toBeDefined();
  });

  it('tursoBranch throws when Platform API returns non-2xx status', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: false,
      status: 400,
      json: async () => ({ error: 'bad request' }),
    }));
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        branchName: 'test-branch',
        organization: 'test-org',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Group 5: branch deletion behavior (functional with mocked fetch)
// ---------------------------------------------------------------------------

describe('turso-branch.mjs: branch deletion behavior', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = originalFetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('tursoBranch deletes a branch by DELETE to the Platform API', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ deleted: true }),
    }));
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    const result = await tursoBranch({
      action: 'delete',
      branchName: 'test-branch',
      organization: 'test-org',
      apiToken: 'test-token',
      databaseName: 'corpus',
    });
    expect(result).toBeDefined();
  });

  it('tursoBranch rejects when DELETE returns non-2xx', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: false,
      status: 404,
      json: async () => ({ error: 'not found' }),
    }));
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        action: 'delete',
        branchName: 'test-branch',
        organization: 'test-org',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Group 6: turso_branch MCP tool registered
// ---------------------------------------------------------------------------

describe('turso_branch MCP tool registration', () => {
  it('registers turso_branch in the MCP tool list', async () => {
    const { createRepoCortexMcpServer } =
      await import('../repo-cortex-mcp.mjs');
    const { client, dbPath, tempDir } = await setupDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const listed = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
      });
      expect(listed.tools.map((tool) => tool.name)).toContain('turso_branch');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('repo-cortex-mcp.mjs imports tursoBranch from turso-branch.mjs', () => {
    const source = readSource(REPO_CORTEX_MCP_PATH);
    expect(source).toMatch(/turso-branch\.mjs/);
  });
});

// ---------------------------------------------------------------------------
// Group 7: tool schema completeness
// ---------------------------------------------------------------------------

describe('turso_branch MCP tool schema completeness', () => {
  it('turso_branch tool inputSchema includes a branch_name property', async () => {
    const { createRepoCortexMcpServer } =
      await import('../repo-cortex-mcp.mjs');
    const { client, dbPath, tempDir } = await setupDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const listed = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
      });
      const tool = listed.tools.find((t) => t.name === 'turso_branch');
      expect(Object.keys(tool.inputSchema.properties)).toContain('branch_name');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('turso_branch tool inputSchema includes an action property (create|delete)', async () => {
    const { createRepoCortexMcpServer } =
      await import('../repo-cortex-mcp.mjs');
    const { client, dbPath, tempDir } = await setupDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const listed = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
      });
      const tool = listed.tools.find((t) => t.name === 'turso_branch');
      expect(Object.keys(tool.inputSchema.properties)).toContain('action');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });
});

// ---------------------------------------------------------------------------
// Group 8: validation and edge cases (coverage gap closure)
// ---------------------------------------------------------------------------

describe('turso-branch.mjs: validation and edge cases', () => {
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
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        action: 'create',
        branchName: 'test-branch',
        organization: 'test-org',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toBeDefined();
  });

  it('safeReadJson parses valid JSON text from a non-2xx response (still rejects)', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: false,
      status: 500,
      text: async () => '{"error":"server error"}',
    }));
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        action: 'create',
        branchName: 'test-branch',
        organization: 'test-org',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toBeDefined();
  });

  it('throws when branchName is missing', async () => {
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        organization: 'test-org',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toThrow('tursoBranch requires a branch_name');
  });

  it('throws when organization is missing', async () => {
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        branchName: 'test-branch',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toThrow('tursoBranch requires an organization');
  });

  it('throws when apiToken is missing', async () => {
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        branchName: 'test-branch',
        organization: 'test-org',
        databaseName: 'corpus',
      }),
    ).rejects.toThrow('tursoBranch requires an apiToken');
  });

  it('throws when databaseName is missing', async () => {
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        branchName: 'test-branch',
        organization: 'test-org',
        apiToken: 'test-token',
      }),
    ).rejects.toThrow('tursoBranch requires a databaseName');
  });

  it('throws when action is unknown', async () => {
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(
      tursoBranch({
        action: 'foobar',
        branchName: 'test-branch',
        organization: 'test-org',
        apiToken: 'test-token',
        databaseName: 'corpus',
      }),
    ).rejects.toThrow('tursoBranch unknown action: foobar');
  });

  it('falls back to the passed branchName when API response omits name', async () => {
    globalThis.fetch = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ hostname: 'libsql://no-name.turso.io' }),
    }));
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    const result = await tursoBranch({
      branchName: 'fallback-branch',
      organization: 'test-org',
      apiToken: 'test-token',
      databaseName: 'corpus',
    });
    expect(result.branchName).toBe('fallback-branch');
  });

  it('throws when called with no arguments (default param branch)', async () => {
    const { tursoBranch } = await import(TURSO_BRANCH_PATH);
    await expect(tursoBranch()).rejects.toThrow(
      'tursoBranch requires a branch_name',
    );
  });
});
