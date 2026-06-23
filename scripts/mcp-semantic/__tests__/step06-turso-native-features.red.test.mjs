/**
 * @module step06-turso-native-features.red.test
 * @description Step 06 red tests — Turso-native feature reporting for the
 * search-corpus, search-advanced, freshness-check, and index-stats MCP tools.
 *
 * **Red contract:** Each tool should report additional Turso-native feature
 * metadata in its response so callers can observe which Turso capabilities
 * (DiskANN vector index, Reciprocal Ranked Fusion, embedded sync status,
 * vector type / quantization) were used during a request.
 *
 * The tools currently do NOT emit these fields:
 *   - search-corpus lacks `diskann_used` and `rrf_used` booleans.
 *   - search-advanced lacks a `turso_native_features` report object.
 *   - freshness-check lacks `last_sync` and `sync_lag_ms` in the freshness
 *     object.
 *   - index-stats lacks `vector_type` and `quantization` in the `ann` object.
 *
 * Every assertion below fails (RED) until implementation adds the missing
 * fields. After Step 06 implementation the assertions pass (GREEN).
 *
 * Tests inject a `:memory:` libSQL client preloaded with the v2 corpus schema
 * and a chunk whose body contains the unique term `tursouniqueword`
 * (chunk_id 424242), mirroring the Step 04 red test pattern.
 */

import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  UNIQUE_QUERY_TERM,
  TEST_FILE_PATH,
} from './turso-test-helpers.mjs';

describe('Step 06 — Turso-native feature reporting', () => {
  describe('search-corpus', () => {
    const { saveEnv, restoreEnv } = createEnvIsolation();
    let client;
    let mod;

    beforeEach(async () => {
      saveEnv();
      client = await createSchemaClient();
      await insertTestFixtures(client);
      mod = await import('../tools/search-corpus.mjs');
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
      it('includes diskann_used boolean in the BM25-only response', async () => {
        const result = await mod.searchCorpus({
          query: UNIQUE_QUERY_TERM,
          use_dense: false,
          use_rerank: false,
          client,
        });

        expect(typeof result.diskann_used).toBe('boolean');
      });

      it('includes rrf_used boolean in the BM25-only response', async () => {
        const result = await mod.searchCorpus({
          query: UNIQUE_QUERY_TERM,
          use_dense: false,
          use_rerank: false,
          client,
        });

        expect(typeof result.rrf_used).toBe('boolean');
      });
    });
  });

  describe('search-advanced', () => {
    const { saveEnv, restoreEnv } = createEnvIsolation();
    let client;
    let mod;

    beforeEach(async () => {
      saveEnv();
      client = await createSchemaClient();
      await insertTestFixtures(client);
      mod = await import('../tools/search-advanced.mjs');
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
      it('includes turso_native_features report in the pipeline response', async () => {
        const result = await mod.searchAdvanced({
          query: UNIQUE_QUERY_TERM,
          use_dense: false,
          use_rerank: false,
          client,
        });

        expect(result.turso_native_features).toEqual(expect.any(Object));
      });
    });
  });

  describe('freshness-check', () => {
    const { saveEnv, restoreEnv } = createEnvIsolation();
    let client;
    let mod;

    beforeEach(async () => {
      saveEnv();
      client = await createSchemaClient();
      await insertTestFixtures(client);
      mod = await import('../tools/freshness-check.mjs');
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

    describe('freshnessCheck', () => {
      it('includes last_sync in the freshness report', async () => {
        const result = await mod.freshnessCheck({
          file_path: TEST_FILE_PATH,
          freshnessProof: {
            mtime_ms: 1000,
            file_size: 500,
            sha256: 'turso-test-sha256',
          },
          client,
        });

        expect(result.freshness).toHaveProperty('last_sync');
      });

      it('includes sync_lag_ms in the freshness report', async () => {
        const result = await mod.freshnessCheck({
          file_path: TEST_FILE_PATH,
          freshnessProof: {
            mtime_ms: 1000,
            file_size: 500,
            sha256: 'turso-test-sha256',
          },
          client,
        });

        expect(result.freshness).toHaveProperty('sync_lag_ms');
      });
    });
  });

  describe('index-stats', () => {
    const { saveEnv, restoreEnv } = createEnvIsolation();
    let client;
    let mod;

    beforeEach(async () => {
      saveEnv();
      client = await createSchemaClient();
      await insertTestFixtures(client);
      mod = await import('../tools/index-stats.mjs');
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

    describe('indexStats', () => {
      it('reports vector_type in the ann stats', async () => {
        const result = await mod.indexStats({ client });

        expect(result.ann).toHaveProperty('vector_type');
      });

      it('reports quantization in the ann stats', async () => {
        const result = await mod.indexStats({ client });

        expect(result.ann).toHaveProperty('quantization');
      });
    });
  });
});