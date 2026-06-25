/**
 * @module multi-hop-search.test
 * @description Red tests for Phase 6 Step 03 slice 03-red-multihop — multi_hop_search MCP tool.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because:
 * - `scripts/mcp-semantic/tools/multi-hop-search.mjs` does not exist yet, so
 *   dynamic `import()` of the module throws and `readFileSync` of the source
 *   file throws ENOENT.
 * - `multi_hop_search` is not registered in the MCP tool list yet.
 *
 * Coverage targets (per plan acceptance criteria):
 * - Hop 1: vector_top_k → JOIN chunks returns seed chunks.
 * - Hop 2: entities JOIN edges finds related entities for seed chunk_ids.
 * - Hop 3: vector_top_k scoped to neighbor chunk_ids (WHERE chunk_id IN (...)).
 * - Combined ranking: vector_distance * graph_proximity_weight, sorted desc.
 * - Tool schema: query, max_hops (1-3), relationship_types, entity_types, limit.
 * - Tool registered in repo-cortex-mcp.mjs tools/list as 'multi_hop_search'.
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

const MULTI_HOP_SEARCH_PATH = path.resolve(
  __dirname,
  '..',
  'tools',
  'multi-hop-search.mjs',
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
 * Build a mock libSQL-style client whose `execute` dispatches to a configurable
 * row factory based on the SQL text. This lets the multi-hop composition be
 * exercised end-to-end without a real DiskANN index or embeddings.
 *
 * @param {object} options - Mock client options.
 * @param {Function} [options.rowFactory] - Returns rows for a given sql/args pair.
 * @returns {{ client: object }} Mock libSQL client.
 */
function createMockClient({ rowFactory = () => [] } = {}) {
  const client = {
    async execute({ sql, args } = {}) {
      return { rows: rowFactory({ sql, args }) };
    },
    async close() {},
  };
  return { client };
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
// Group 1: multi-hop-search.mjs module exists and exports
// ---------------------------------------------------------------------------

describe('multi-hop-search.mjs: module exists and exports', () => {
  it('multi-hop-search.mjs exports multiHopSearch', async () => {
    const mod = await import(MULTI_HOP_SEARCH_PATH);
    expect(typeof mod.multiHopSearch).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Group 2: Tool schema (source-pattern)
// ---------------------------------------------------------------------------

describe('multi-hop-search.mjs: tool schema parameters', () => {
  it('source declares max_hops bounded to 1-3', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/max_hops/);
  });

  it('source declares relationship_types parameter', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/relationship_types/);
  });

  it('source declares entity_types parameter', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/entity_types/);
  });

  it('source declares limit parameter', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/\blimit\b/);
  });
});

// ---------------------------------------------------------------------------
// Group 3: Hop 1 — vector search returns seed chunks
// ---------------------------------------------------------------------------

describe('multi-hop-search.mjs: hop 1 vector search returns seed chunks', () => {
  it('source uses vector_top_k with vector8 for hop 1', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/vector_top_k.*vector8/s);
  });

  it('multiHopSearch hop 1 returns seed chunks from vector search', async () => {
    const { multiHopSearch } = await import(MULTI_HOP_SEARCH_PATH);
    const seedRows = [
      { chunk_id: 1, body_text: 'seed chunk', distance: 0.1 },
      { chunk_id: 2, body_text: 'seed chunk 2', distance: 0.2 },
    ];
    const { client } = createMockClient({
      rowFactory: ({ sql }) => {
        if (/vector_top_k/i.test(sql) && !/chunk_id\s+IN/i.test(sql)) {
          return seedRows;
        }
        return [];
      },
    });
    const result = await multiHopSearch({
      client,
      query: 'network activation',
      max_hops: 1,
      limit: 5,
    });
    expect(result.seed_chunks.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Group 4: Hop 2 — graph traversal finds related entities
// ---------------------------------------------------------------------------

describe('multi-hop-search.mjs: hop 2 graph traversal finds related entities', () => {
  it('source joins entities and edges for hop 2', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/entities.*JOIN.*edges|edges.*JOIN.*entities/is);
  });

  it('multiHopSearch hop 2 traverses graph to find related entities', async () => {
    const { multiHopSearch } = await import(MULTI_HOP_SEARCH_PATH);
    const entityRows = [
      { entity_id: 10, name: 'Network', entity_type: 'class', chunk_id: 1 },
      { entity_id: 11, name: 'activate', entity_type: 'function', chunk_id: 2 },
    ];
    const edgeRows = [
      {
        edge_id: 1,
        src_entity_id: 10,
        dst_entity_id: 11,
        relationship_type: 'owns',
      },
    ];
    const { client } = createMockClient({
      rowFactory: ({ sql }) => {
        if (/entities/i.test(sql) && /edges/i.test(sql)) {
          return [...entityRows, ...edgeRows];
        }
        if (/vector_top_k/i.test(sql) && !/chunk_id\s+IN/i.test(sql)) {
          return [{ chunk_id: 1, body_text: 'seed', distance: 0.1 }];
        }
        return [];
      },
    });
    const result = await multiHopSearch({
      client,
      query: 'network activation',
      max_hops: 2,
      limit: 5,
    });
    expect(result.entities.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Group 5: Hop 3 — vector search on neighbor chunks
// ---------------------------------------------------------------------------

describe('multi-hop-search.mjs: hop 3 vector search on neighbor chunks', () => {
  it('source runs vector_top_k scoped to neighbor chunk_ids for hop 3', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(/vector_top_k.*chunk_id\s+IN/is);
  });

  it('multiHopSearch hop 3 runs vector search scoped to neighbor chunks', async () => {
    const { multiHopSearch } = await import(MULTI_HOP_SEARCH_PATH);
    const neighborRows = [
      { chunk_id: 3, body_text: 'neighbor chunk', distance: 0.3 },
      { chunk_id: 4, body_text: 'neighbor chunk 2', distance: 0.4 },
    ];
    const { client } = createMockClient({
      rowFactory: ({ sql }) => {
        if (/vector_top_k/i.test(sql) && /chunk_id\s+IN/i.test(sql)) {
          return neighborRows;
        }
        if (/vector_top_k/i.test(sql)) {
          return [{ chunk_id: 1, body_text: 'seed', distance: 0.1 }];
        }
        if (/entities/i.test(sql) && /edges/i.test(sql)) {
          return [
            {
              entity_id: 10,
              name: 'Network',
              entity_type: 'class',
              chunk_id: 1,
            },
          ];
        }
        return [];
      },
    });
    const result = await multiHopSearch({
      client,
      query: 'network activation',
      max_hops: 3,
      limit: 5,
    });
    expect(result.results.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Group 6: Combined ranking
// ---------------------------------------------------------------------------

describe('multi-hop-search.mjs: combined ranking', () => {
  it('source computes combined score from vector_distance and graph_proximity', () => {
    const source = readSource(MULTI_HOP_SEARCH_PATH);
    expect(source).toMatch(
      /vector_distance.*graph_proximity|graph_proximity.*vector_distance|combined.*score/is,
    );
  });

  it('multiHopSearch returns results sorted by combined score descending', async () => {
    const { multiHopSearch } = await import(MULTI_HOP_SEARCH_PATH);
    const scoredRows = [
      { chunk_id: 3, body_text: 'low', distance: 0.8, combined_score: 0.3 },
      { chunk_id: 4, body_text: 'high', distance: 0.1, combined_score: 0.9 },
      { chunk_id: 5, body_text: 'mid', distance: 0.4, combined_score: 0.6 },
    ];
    const { client } = createMockClient({
      rowFactory: ({ sql }) => {
        if (/vector_top_k/i.test(sql) && /chunk_id\s+IN/i.test(sql)) {
          return scoredRows;
        }
        if (/vector_top_k/i.test(sql)) {
          return [{ chunk_id: 1, body_text: 'seed', distance: 0.1 }];
        }
        if (/entities/i.test(sql) && /edges/i.test(sql)) {
          return [
            {
              entity_id: 10,
              name: 'Network',
              entity_type: 'class',
              chunk_id: 1,
            },
          ];
        }
        return [];
      },
    });
    const result = await multiHopSearch({
      client,
      query: 'network activation',
      max_hops: 3,
      limit: 5,
    });
    expect(result.results[0].combined_score).toBeGreaterThanOrEqual(
      result.results[1].combined_score,
    );
  });
});

// ---------------------------------------------------------------------------
// Group 7: multi_hop_search MCP tool registered
// ---------------------------------------------------------------------------

describe('multi_hop_search MCP tool registration', () => {
  it('registers multi_hop_search in the MCP tool list', async () => {
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
        'multi_hop_search',
      );
    } finally {
      await teardownDb(client, tempDir, dbPath);
    }
  });

  it('repo-cortex-mcp.mjs imports multiHopSearch from multi-hop-search.mjs', () => {
    const source = readSource(REPO_CORTEX_MCP_PATH);
    expect(source).toMatch(/multi-hop-search\.mjs/);
  });
});
