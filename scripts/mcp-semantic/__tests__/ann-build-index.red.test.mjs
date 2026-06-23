/**
 * @module ann-build-index.red.test
 * @description Red tests for Step 23 ANN MCP tool and response extensions.
 *
 * Verifies that:
 *   - the `ann_build_index` MCP tool is registered and callable,
 *   - `search_corpus` emits a `dense_strategy` field on warm dense responses,
 *   - `index_stats` emits an `ann` section describing strategy/index state.
 */

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';
import { closeTursoClient } from '../tools/cortex-db.mjs';

/**
 * Create a minimal corpus database for search/index_stats fixtures.
 * @returns {Promise<{ dbPath: string, tempDir: string }>}
 */
async function setupCorpusDb() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'ann-build-index-test-'));
  const dbPath = path.join(tempDir, 'corpus.sqlite');
  const client = createClient({ url: 'file:' + dbPath });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  await client.execute(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer)
    VALUES
      ('src/network.ts', 'ts-source', 0, 100, 'a', 1, 'network'),
      ('src/methods.ts', 'ts-source', 0, 100, 'b', 1, 'methods'),
      ('plans/roadmap.md', 'plans', 0, 100, 'c', 1, 'planning');
  `);
  const docsResult = await client.execute(
    'SELECT doc_id, doc_family, arch_layer FROM documents',
  );
  const docs = docsResult.rows;
  for (const doc of docs) {
    await client.execute({
      sql: 'INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer) VALUES (?, 0, ?, 0, 10, 0, ?)',
      args: [
        doc.doc_id,
        `${doc.doc_family} ${doc.arch_layer} content`,
        doc.arch_layer,
      ],
    });
  }
  await client.close();
  return { dbPath, tempDir };
}

/**
 * Remove the temporary fixture directory.
 * @param {string} tempDir
 * @returns {Promise<void>}
 */
async function teardown(tempDir, dbPath) {
  if (dbPath) await closeTursoClient(dbPath);
  await rm(tempDir, {
    recursive: true,
    force: true,
    maxRetries: 10,
    retryDelay: 200,
  });
}

/**
 * Import the ann-index module, returning an empty object if it is missing.
 * @returns {Promise<Record<string, unknown>>}
 */
async function loadAnnIndex() {
  return import('../tools/ann-index.mjs').catch(() => ({}));
}

describe('ann_build_index MCP tool', () => {
  it('is registered in the Repo Cortex MCP server tool list', async () => {
    const { createRepoCortexMcpServer } =
      await import('../repo-cortex-mcp.mjs');
    const server = createRepoCortexMcpServer({
      databasePath: './missing.sqlite',
    });
    const listed = await server.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
    });

    expect(listed.tools.map((tool) => tool.name)).toContain('ann_build_index');
  });

  it('has the expected input schema for force and validate_recall', async () => {
    const { createRepoCortexMcpServer } =
      await import('../repo-cortex-mcp.mjs');
    const server = createRepoCortexMcpServer({
      databasePath: './missing.sqlite',
    });
    const listed = await server.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
    });
    const tool = listed.tools.find(
      (candidate) => candidate.name === 'ann_build_index',
    );

    expect(tool).toEqual(
      expect.objectContaining({
        inputSchema: expect.objectContaining({
          properties: expect.objectContaining({
            force: expect.objectContaining({
              enum: ['diskann'],
            }),
            validate_recall: expect.objectContaining({ type: 'boolean' }),
          }),
        }),
      }),
    );
  });

  it('is callable via tools/call and returns a structured response', async () => {
    const { createRepoCortexMcpServer } =
      await import('../repo-cortex-mcp.mjs');
    const { dbPath, tempDir } = await setupCorpusDb();
    try {
      const server = createRepoCortexMcpServer({ databasePath: dbPath });
      const result = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: { name: 'ann_build_index', arguments: {} },
      });

      expect(result).toEqual(
        expect.objectContaining({
          isError: false,
          structuredContent: expect.objectContaining({
            strategy: expect.any(String),
            build_status: expect.any(String),
          }),
        }),
      );
    } finally {
      await teardown(tempDir, dbPath);
    }
  });

  it('exports a buildAnnIndex handler from ann-index.mjs', async () => {
    const mod = await loadAnnIndex();

    expect(mod.buildAnnIndex).toBeInstanceOf(Function);
  });
});

describe('search_corpus dense_strategy extension', () => {
  it('includes dense_strategy in warm dense responses below threshold', async () => {
    const { searchCorpus } = await import('../tools/search-corpus.mjs');
    const { dbPath, tempDir } = await setupCorpusDb();
    try {
      const result = await searchCorpus({
        databasePath: dbPath,
        query: 'network',
        use_dense: true,
        alpha: 0.5,
        readinessProbe: async () => ({ state: 'warm' }),
        denseQuery: async () => ({
          alpha: 0.5,
          query: 'network',
          limit: 10,
          use_dense: true,
          results: [
            {
              chunk_id: 1,
              file_path: 'src/network.ts',
              family: 'ts-source',
              heading_path: '',
              text: 'network content',
              score: 0.9,
            },
          ],
        }),
      });

      expect(result).toEqual(
        expect.objectContaining({
          dense_strategy: 'diskann',
        }),
      );
    } finally {
      await teardown(tempDir, dbPath);
    }
  });
});

describe('index_stats ann section', () => {
  it('includes an ann section on every response', async () => {
    const { indexStats } = await import('../tools/index-stats.mjs');
    const { dbPath, tempDir } = await setupCorpusDb();
    try {
      const result = await indexStats({ databasePath: dbPath });

      expect(result).toEqual(
        expect.objectContaining({
          ann: expect.objectContaining({
            strategy: expect.any(String),
            threshold: expect.any(Number),
            current_chunk_count: expect.any(Number),
          }),
        }),
      );
    } finally {
      await teardown(tempDir, dbPath);
    }
  });

  it('reports diskann with no index when below threshold', async () => {
    const { indexStats } = await import('../tools/index-stats.mjs');
    const { dbPath, tempDir } = await setupCorpusDb();
    try {
      const result = await indexStats({ databasePath: dbPath });

      expect(result.ann).toEqual(
        expect.objectContaining({
          strategy: 'diskann',
          build_status: 'not_applicable',
          threshold: 50000,
          current_chunk_count: 3,
          index_id: null,
          index_type: null,
        }),
      );
    } finally {
      await teardown(tempDir, dbPath);
    }
  });
});
