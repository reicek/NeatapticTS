/**
 * @module search-tools.turso.test
 * @description Step 04 red tests — Turso async `client` option for search-corpus,
 * search-advanced, and search-context MCP tools.
 *
 * **Red contract:** Each tool should accept an optional `client` option (a
 * `@libsql/client` Client) and use it for DB access instead of opening the
 * sync local SQLite DB via `openCortexDatabase()`.
 *
 * Tests inject a `:memory:` libSQL client preloaded with the v2 corpus schema
 * and a chunk whose body contains the unique term `tursouniqueword` (chunk_id
 * 424242). The tools currently ignore `client`, open the real on-disk DB, and
 * fail to match the unique term — making every assertion below fail (RED).
 * After migration the tools use the injected client, find the chunk via FTS5
 * BM25, and the assertions pass (GREEN).
 */

import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
  UNIQUE_QUERY_TERM,
} from './turso-test-helpers.mjs';

const SEARCH_CORPUS_PATH = '../tools/search-corpus.mjs';
const SEARCH_ADVANCED_PATH = '../tools/search-advanced.mjs';
const SEARCH_CONTEXT_PATH = '../tools/search-context.mjs';

describe('search-corpus (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(SEARCH_CORPUS_PATH);
  });

  afterEach(async () => {
    restoreEnv();
    if (client && typeof client.close === 'function') {
      try {
        await client.close();
      } catch {
        /* noop */
      }
    }
  });

  describe('searchCorpus', () => {
    it('returns chunk 424242 from the injected libSQL client via BM25', async () => {
      const result = await mod.searchCorpus({
        query: UNIQUE_QUERY_TERM,
        use_dense: false,
        use_rerank: false,
        client,
      });

      expect(result.results.some((r) => r.chunk_id === TEST_CHUNK_ID)).toBe(
        true,
      );
    });
  });
});

describe('search-advanced (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(SEARCH_ADVANCED_PATH);
  });

  afterEach(async () => {
    restoreEnv();
    if (client && typeof client.close === 'function') {
      try {
        await client.close();
      } catch {
        /* noop */
      }
    }
  });

  describe('searchAdvanced', () => {
    it('returns chunk 424242 from the injected libSQL client', async () => {
      const result = await mod.searchAdvanced({
        query: UNIQUE_QUERY_TERM,
        use_dense: false,
        use_rerank: false,
        client,
      });

      expect(result.results.some((r) => r.chunk_id === TEST_CHUNK_ID)).toBe(
        true,
      );
    });
  });
});

describe('search-context (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(SEARCH_CONTEXT_PATH);
  });

  afterEach(async () => {
    restoreEnv();
    if (client && typeof client.close === 'function') {
      try {
        await client.close();
      } catch {
        /* noop */
      }
    }
  });

  describe('searchContext', () => {
    it('assembles context containing the unique term from the injected libSQL client', async () => {
      const result = await mod.searchContext({
        query: UNIQUE_QUERY_TERM,
        use_dense: false,
        client,
      });

      expect(result.chunks_in_context).toBeGreaterThan(0);
    });
  });
});
