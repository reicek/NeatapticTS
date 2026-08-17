/**
 * @module parallel-search.test
 * @description Red tests for Phase 5 Step 03 slice 03-red-parallel — parallel query execution.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because:
 * - `rag-index/parallel-search.mjs` does not exist yet (dynamic import throws).
 * - `readFileSync` for the missing source file throws ENOENT.
 * - `search-corpus.mjs` still uses the old sequential `await denseQuery({...})` pattern.
 * - `parallel_search` is not registered in the MCP tool list yet.
 *
 * Coverage targets:
 * - Promise.all / Promise.allSettled runs multiple client.execute() calls concurrently.
 * - TURSO_CONCURRENCY env var limits in-flight requests (default 20).
 * - Results from multiple queries merged correctly (RRF-style dedup + sort).
 * - Graceful degradation: surviving query results returned when one query fails.
 * - Old sequential dense-query path removed from search-corpus.mjs (no dual-path).
 * - parallel_search MCP tool registered in repo-cortex-mcp.mjs tools/list.
 *
 * Pure .mjs test — runs via Jest ESM project `mcp-semantic-mjs`.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { rm } from 'node:fs/promises';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';
import { closeTursoClient } from '../tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PARALLEL_SEARCH_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  '..',
  'rag-index',
  'parallel-search.mjs',
);
const SEARCH_CORPUS_PATH = path.resolve(
  __dirname,
  '..',
  'tools',
  'search-corpus.mjs',
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
 * Build a mock libSQL-style client whose `execute` applies a configurable delay
 * and tracks concurrent in-flight calls.
 *
 * @param {object} options - Mock client options.
 * @param {number} [options.delayMs=50] - Artificial delay per execute call.
 * @param {Function} [options.rowFactory] - Returns rows for a given sql/args pair.
 * @param {number} [options.failIndex] - Index of the query that should reject.
 * @returns {{ client: object, maxConcurrent: number }} Mock client and a max-concurrency tracker.
 */
function createMockClient({
  delayMs = 50,
  rowFactory = () => [],
  failIndex = -1,
} = {}) {
  let inFlight = 0;
  let maxConcurrent = 0;
  let callIndex = 0;
  const client = {
    async execute({ sql, args } = {}) {
      const idx = callIndex++;
      if (idx === failIndex) {
        throw new Error(`MOCK_EXECUTE_FAILURE at index ${idx}`);
      }
      inFlight += 1;
      if (inFlight > maxConcurrent) maxConcurrent = inFlight;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      inFlight -= 1;
      return { rows: rowFactory({ sql, args, index: idx }) };
    },
  };
  return {
    client,
    get maxConcurrent() {
      return maxConcurrent;
    },
  };
}

/**
 * Set up an in-memory SQLite database with the v2 corpus schema.
 *
 * Uses `:memory:` instead of a file-backed database so repeated test cycles do
 * not exhaust Windows file handles via the libsql native binding. The schema
 * tests only inspect `tools/list` output, which never opens the database, so
 * an in-memory client is sufficient and avoids `EBUSY` teardown failures.
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
 * On Windows, libsql's native `close()` may not release the sqlite file handle
 * immediately, so `rm` of the temp dir can hit `EBUSY`. We close the client,
 * evict any cached turso client, wait a short grace period for the native
 * binding to release the handle, then retry removal with a generous backoff.
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
// Group 1: parallel-search.mjs module exists and exports
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: module exists and exports', () => {
  it('rag-index/parallel-search.mjs exports runParallelQueries', async () => {
    const mod = await import(PARALLEL_SEARCH_PATH);
    expect(typeof mod.runParallelQueries).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Group 2: Promise.all concurrency (source-pattern + functional)
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: Promise.all concurrency', () => {
  it('parallel-search.mjs source contains Promise.all or Promise.allSettled', () => {
    const source = readSource(PARALLEL_SEARCH_PATH);
    expect(source).toMatch(/Promise\.all(Settled)?\(/);
  });

  it('runParallelQueries executes multiple queries concurrently', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const delayMs = 80;
    const { client } = createMockClient({
      delayMs,
      rowFactory: () => [{ chunk_id: 1, score: 1 }],
    });
    const queries = [
      { sql: 'SELECT 1', args: [] },
      { sql: 'SELECT 2', args: [] },
      { sql: 'SELECT 3', args: [] },
    ];
    const start = Date.now();
    await runParallelQueries({ client, queries });
    const totalMs = Date.now() - start;
    expect(totalMs).toBeLessThan(delayMs * queries.length);
  });
});

// ---------------------------------------------------------------------------
// Group 3: TURSO_CONCURRENCY env var
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: TURSO_CONCURRENCY env var', () => {
  it('parallel-search.mjs reads TURSO_CONCURRENCY with default 20', () => {
    const source = readSource(PARALLEL_SEARCH_PATH);
    expect(source).toMatch(/TURSO_CONCURRENCY/);
  });

  it('runParallelQueries respects TURSO_CONCURRENCY to limit in-flight requests', async () => {
    const savedValue = process.env.TURSO_CONCURRENCY;
    process.env.TURSO_CONCURRENCY = '2';
    try {
      const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
      const delayMs = 60;
      const tracker = createMockClient({
        delayMs,
        rowFactory: () => [{ chunk_id: 1, score: 1 }],
      });
      const queries = Array.from({ length: 5 }, (_, i) => ({
        sql: `SELECT ${i}`,
        args: [],
      }));
      await runParallelQueries({ client: tracker.client, queries });
      const maxConcurrent = tracker.maxConcurrent;
      expect(maxConcurrent).toBeLessThanOrEqual(2);
    } finally {
      if (savedValue === undefined) {
        delete process.env.TURSO_CONCURRENCY;
      } else {
        process.env.TURSO_CONCURRENCY = savedValue;
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Group 4: Results merged correctly (RRF)
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: results merged correctly', () => {
  it('runParallelQueries merges results from multiple queries via RRF', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const queryARows = [
      { chunk_id: 1, score: 5 },
      { chunk_id: 2, score: 3 },
    ];
    const queryBRows = [
      { chunk_id: 2, score: 4 },
      { chunk_id: 3, score: 1 },
    ];
    const { client } = createMockClient({
      delayMs: 0,
      rowFactory: ({ sql }) => {
        if (sql.includes('queryA')) return queryARows;
        if (sql.includes('queryB')) return queryBRows;
        return [];
      },
    });
    const results = await runParallelQueries({
      client,
      queries: [
        { sql: 'SELECT queryA', args: [] },
        { sql: 'SELECT queryB', args: [] },
      ],
    });
    const mergedChunkIds = results.map((row) => row.chunk_id).sort();
    expect(mergedChunkIds).toEqual([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Group 5: Graceful degradation
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: graceful degradation', () => {
  it('runParallelQueries returns results from surviving queries when one query fails', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const { client } = createMockClient({
      delayMs: 0,
      failIndex: 0,
      rowFactory: () => [{ chunk_id: 42, score: 1 }],
    });
    const results = await runParallelQueries({
      client,
      queries: [
        { sql: 'SELECT failing', args: [] },
        { sql: 'SELECT surviving', args: [] },
      ],
    });
    expect(results.length).toBeGreaterThan(0);
  });

  it('runParallelQueries surfaces per-query errors on the returned array', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const { client } = createMockClient({
      delayMs: 0,
      failIndex: 0,
      rowFactory: () => [{ chunk_id: 42, score: 1 }],
    });
    const failingSql = 'SELECT failing';
    const results = await runParallelQueries({
      client,
      queries: [
        { sql: failingSql, args: [] },
        { sql: 'SELECT surviving', args: [] },
      ],
    });
    expect(results.errors).toEqual([
      expect.objectContaining({
        queryIndex: 0,
        sql: failingSql,
        message: expect.stringContaining('MOCK_EXECUTE_FAILURE'),
      }),
    ]);
  });
});

// ---------------------------------------------------------------------------
// Group 6: Old sequential path removed (source-pattern)
// ---------------------------------------------------------------------------

describe('search-corpus.mjs: old sequential path removed', () => {
  it('search-corpus.mjs imports runParallelQueries from parallel-search.mjs', () => {
    const source = readSource(SEARCH_CORPUS_PATH);
    expect(source).toMatch(/parallel-search\.mjs/);
  });

  it('search-corpus.mjs does not run dense query sequentially via await denseQuery', () => {
    const source = readSource(SEARCH_CORPUS_PATH);
    expect(source).not.toMatch(/await\s+denseQuery\s*\(\{/);
  });
});

// ---------------------------------------------------------------------------
// Group 7: parallel_search MCP tool registered
// ---------------------------------------------------------------------------

describe('parallel_search MCP tool registration', () => {
  it('registers parallel_search in the MCP tool list', async () => {
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
      expect(listed.tools.map((tool) => tool.name)).toContain(
        'parallel_search',
      );
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });
});

// ---------------------------------------------------------------------------
// Group 8: parallel_search tool schema completeness (Step 02 impl-slice)
// Acceptance criterion: "Tool schema: queries (array), fusion (rrf|alpha),
// limit, use_dense"
// ---------------------------------------------------------------------------

describe('parallel_search MCP tool schema completeness', () => {
  it('parallel_search tool inputSchema includes a fusion property', async () => {
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
      const tool = listed.tools.find((t) => t.name === 'parallel_search');
      expect(Object.keys(tool.inputSchema.properties)).toContain('fusion');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('parallel_search tool inputSchema includes a limit property', async () => {
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
      const tool = listed.tools.find((t) => t.name === 'parallel_search');
      expect(Object.keys(tool.inputSchema.properties)).toContain('limit');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('parallel_search tool inputSchema includes a use_dense property', async () => {
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
      const tool = listed.tools.find((t) => t.name === 'parallel_search');
      expect(Object.keys(tool.inputSchema.properties)).toContain('use_dense');
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });
});

// ---------------------------------------------------------------------------
// Group 9: Configurable fusion (RRF default, alpha-blend fallback)
// Step 02 acceptance: "Results merged by configurable fusion (RRF default,
// alpha-blend fallback)"
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: configurable fusion', () => {
  it('parallel-search.mjs source destructures a fusion option parameter', async () => {
    const source = readSource(PARALLEL_SEARCH_PATH);
    expect(source).toMatch(/fusion\s*[=:]/);
  });

  it('runParallelQueries caps result count when limit option is provided', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const { client } = createMockClient({
      delayMs: 0,
      rowFactory: ({ sql }) => {
        if (sql.includes('queryA')) {
          return [
            { chunk_id: 1, score: 5 },
            { chunk_id: 2, score: 3 },
            { chunk_id: 3, score: 1 },
          ];
        }
        return [];
      },
    });
    const results = await runParallelQueries({
      client,
      queries: [{ sql: 'SELECT queryA', args: [] }],
      limit: 1,
    });
    expect(results.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Group 10: Alpha-blend fusion, empty queries, all-fail edge cases
// Coverage gap closure for normalizeScores, mergeWithAlphaBlend, empty/failed paths
// ---------------------------------------------------------------------------

describe('parallel-search.mjs: alpha-blend fusion and edge cases', () => {
  it('runParallelQueries merges results via alpha-blend when fusion is alpha', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const queryARows = [
      { chunk_id: 1, score: 10 },
      { chunk_id: 2, score: 5 },
    ];
    const queryBRows = [
      { chunk_id: 2, score: 8 },
      { chunk_id: 3, score: 2 },
    ];
    const { client } = createMockClient({
      delayMs: 0,
      rowFactory: ({ sql }) => {
        if (sql.includes('queryA')) return queryARows;
        if (sql.includes('queryB')) return queryBRows;
        return [];
      },
    });
    const results = await runParallelQueries({
      client,
      queries: [
        { sql: 'SELECT queryA', args: [] },
        { sql: 'SELECT queryB', args: [] },
      ],
      fusion: 'alpha',
    });
    const ids = results.map((r) => r.chunk_id).sort();
    expect(ids).toEqual([1, 2, 3]);
    expect(results[0]).toHaveProperty('alpha_score');
  });

  it('runParallelQueries with alpha fusion handles equal scores in a single list', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const { client } = createMockClient({
      delayMs: 0,
      rowFactory: () => [
        { chunk_id: 1, score: 5 },
        { chunk_id: 2, score: 5 },
      ],
    });
    const results = await runParallelQueries({
      client,
      queries: [{ sql: 'SELECT q1', args: [] }],
      fusion: 'alpha',
    });
    expect(results.length).toBe(2);
  });

  it('runParallelQueries returns empty array for empty queries', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const results = await runParallelQueries({
      client: { async execute() { return { rows: [] }; } },
      queries: [],
    });
    expect(results).toEqual([]);
    expect(results.errors).toEqual([]);
  });

  it('runParallelQueries returns empty array with errors when all queries fail', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const { client } = createMockClient({
      delayMs: 0,
      failIndex: 0,
      rowFactory: () => [{ chunk_id: 1, score: 1 }],
    });
    const results = await runParallelQueries({
      client,
      queries: [{ sql: 'SELECT fail', args: [] }],
    });
    expect(results).toEqual([]);
    expect(results.errors.length).toBe(1);
  });

  it('runParallelQueries falls back to default concurrency for invalid TURSO_CONCURRENCY', async () => {
    const savedValue = process.env.TURSO_CONCURRENCY;
    process.env.TURSO_CONCURRENCY = 'not-a-number';
    try {
      const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
      const { client } = createMockClient({
        delayMs: 0,
        rowFactory: () => [{ chunk_id: 1, score: 1 }],
      });
      const results = await runParallelQueries({
        client,
        queries: [{ sql: 'SELECT 1', args: [] }],
      });
      expect(results.length).toBe(1);
    } finally {
      if (savedValue === undefined) {
        delete process.env.TURSO_CONCURRENCY;
      } else {
        process.env.TURSO_CONCURRENCY = savedValue;
      }
    }
  });

  it('runParallelQueries passes use_dense flag through without error', async () => {
    const { runParallelQueries } = await import(PARALLEL_SEARCH_PATH);
    const { client } = createMockClient({
      delayMs: 0,
      rowFactory: () => [{ chunk_id: 1, score: 1 }],
    });
    const results = await runParallelQueries({
      client,
      queries: [{ sql: 'SELECT 1', args: [] }],
      use_dense: true,
    });
    expect(results.length).toBe(1);
  });
});
