/**
 * @module search-advanced.red.test
 * @description Red tests for the search_advanced MCP tool.
 *
 * search_advanced is planned in Step 21 of the Repo Cortex Advanced RAG
 * Architecture plan as a full-pipeline orchestration tool:
 * classify -> expand -> retrieve -> rerank -> assemble.
 *
 * These tests express the desired contract and should fail until
 * search-advanced.mjs is implemented and registered in repo-cortex-mcp.mjs.
 */

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';
import { closeTursoClient } from '../tools/cortex-db.mjs';

async function setupDb() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'search-advanced-test-'));
  const dbPath = path.join(tempDir, 'test.sqlite');
  const client = createClient({ url: 'file:' + dbPath });
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
  return { client, dbPath, tempDir };
}

async function teardownDb(client, tempDir, dbPath) {
  await client.close();
  if (dbPath) await closeTursoClient(dbPath);
  await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}

describe('search-advanced', () => {
  describe('server registration', () => {
    it('registers search_advanced in the MCP tool list', async () => {
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
          'search_advanced',
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });
  });

  describe('schema validation', () => {
    it('rejects a missing query parameter with EMPTY_QUERY code', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {},
          },
        });

        expect(result).toEqual(
          expect.objectContaining({
            isError: true,
            structuredContent: expect.objectContaining({
              error: expect.stringContaining('EMPTY_QUERY'),
            }),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });

    it('rejects an invalid budget type', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: { query: 'NEAT', budget: 'big' },
          },
        });

        expect(result).toEqual(
          expect.objectContaining({
            isError: true,
            structuredContent: expect.objectContaining({
              error: expect.any(String),
            }),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });

    it('rejects an invalid limit type', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: { query: 'NEAT', limit: 'many' },
          },
        });

        expect(result).toEqual(
          expect.objectContaining({
            isError: true,
            structuredContent: expect.objectContaining({
              error: expect.any(String),
            }),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });
  });

  describe('classification-aware defaults', () => {
    it('uses BM25-heavy defaults for simple_lookup query class', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              query_class: 'simple_lookup',
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            query_class: 'simple_lookup',
            alpha: expect.closeTo(0.7, 1),
            expand_query: false,
            use_rerank: false,
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });

    it('uses dense expansion defaults for cross_boundary query class', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              query_class: 'cross_boundary',
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            query_class: 'cross_boundary',
            alpha: expect.closeTo(0.4, 1),
            expand_query: true,
            use_rerank: true,
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });
  });

  describe('pipeline orchestration', () => {
    it('returns pipeline metadata: classification, expansion, results', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              query_class: 'simple_lookup',
              expand_query: false,
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            query_class: expect.any(String),
            results: expect.any(Array),
            limit: expect.any(Number),
            use_dense: expect.any(Boolean),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });

    it('assembles context when context_budget is provided', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              query_class: 'simple_lookup',
              context_budget: 512,
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            context: expect.any(String),
            token_count: expect.any(Number),
            tier_counts: expect.objectContaining({
              essential: expect.any(Number),
              supporting: expect.any(Number),
              supplementary: expect.any(Number),
            }),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });
  });

  describe('graceful degradation', () => {
    it('reports dense_state when dense subsystem is cold', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              use_dense: true,
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            dense_state: expect.any(String),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });

    it('reports rerank_state when reranker is cold', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              use_rerank: true,
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            rerank_state: expect.any(String),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });

    it('reports expansion degradation when expansion is cold', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              expand_query: true,
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            expansion: expect.objectContaining({
              degraded: expect.any(Boolean),
            }),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });
  });

  describe('timeout handling', () => {
    it('returns a partial result with CORTEX_TIMEOUT_PARTIAL error code', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const server = createRepoCortexMcpServer({ databasePath: dbPath });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: {
            name: 'search_advanced',
            arguments: {
              query: 'NEAT activation',
              timeout_ms: 1,
            },
          },
        });

        expect(result.structuredContent).toEqual(
          expect.objectContaining({
            error: expect.stringContaining('CORTEX_TIMEOUT_PARTIAL'),
          }),
        );
      } finally {
        await teardownDb(client, tempDir, dbPath);
      }
    });
  });
});
