/**
 * @module ann-build-index.red.test
 * @description Red tests for Step 23 ANN MCP tool and response extensions.
 *
 * Verifies that:
 *   - the `ann_build_index` MCP tool is registered and callable,
 *   - `search_corpus` emits a `dense_strategy` field on warm dense responses,
 *   - `index_stats` emits an `ann` section describing strategy/index state.
 */

import { createSchemaClient } from './turso-test-helpers.mjs';
import { closeTursoClient, setTursoClient } from '../tools/cortex-db.mjs';

/**
 * Create a minimal corpus database for search/index_stats fixtures.
 * @returns {Promise<{ dbPath: string, tempDir: string, client: object }>}
 */
async function setupCorpusDb() {
  const client = await createSchemaClient();
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
  const dbPath = 'file:./ann-build-index-test.sqlite';
  setTursoClient(dbPath, client);
  return { dbPath, client };
}

/**
 * Release an injected corpus client.
 * @param {object} client
 * @param {string} dbPath
 * @returns {Promise<void>}
 */
async function teardown(client, dbPath) {
  setTursoClient(dbPath, undefined);
  await client.close();
  await closeTursoClient(dbPath);
}

/**
 * Import the ann-index module, returning an empty object if it is missing.
 * @returns {Promise<Record<string, unknown>>}
 */
async function loadAnnIndex() {
  return import('../tools/ann-index.mjs').catch(() => ({}));
}

describe('ann_build_index MCP tool', () => {
  let annDb;

  beforeAll(async () => {
    annDb = await setupCorpusDb();
  });

  afterAll(async () => {
    if (annDb) {
      await teardown(annDb.client, annDb.dbPath);
      annDb = null;
    }
  });

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
    const server = createRepoCortexMcpServer({ databasePath: annDb.dbPath });
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
  });

  it('exports a buildAnnIndex handler from ann-index.mjs', async () => {
    const mod = await loadAnnIndex();

    expect(mod.buildAnnIndex).toBeInstanceOf(Function);
  });
});

describe('search_corpus dense_strategy extension', () => {
  let searchDb;

  beforeAll(async () => {
    searchDb = await setupCorpusDb();
  });

  afterAll(async () => {
    if (searchDb) {
      await teardown(searchDb.client, searchDb.dbPath);
      searchDb = null;
    }
  });

  it('includes dense_strategy in warm dense responses below threshold', async () => {
    const { searchCorpus } = await import('../tools/search-corpus.mjs');
    const result = await searchCorpus({
      databasePath: searchDb.dbPath,
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
  });
});

describe('index_stats ann section', () => {
  let statsDb;

  beforeAll(async () => {
    statsDb = await setupCorpusDb();
  });

  afterAll(async () => {
    if (statsDb) {
      await teardown(statsDb.client, statsDb.dbPath);
      statsDb = null;
    }
  });

  it('includes an ann section on every response', async () => {
    const { indexStats } = await import('../tools/index-stats.mjs');
    const result = await indexStats({ databasePath: statsDb.dbPath });

    expect(result).toEqual(
      expect.objectContaining({
        ann: expect.objectContaining({
          strategy: expect.any(String),
          threshold: expect.any(Number),
          current_chunk_count: expect.any(Number),
        }),
      }),
    );
  });

  it('reports diskann with no index when below threshold', async () => {
    const { indexStats } = await import('../tools/index-stats.mjs');
    const result = await indexStats({ databasePath: statsDb.dbPath });

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
  });
});
