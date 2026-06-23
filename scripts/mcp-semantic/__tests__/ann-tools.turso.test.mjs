/**
 * @module ann-tools.turso.test
 * @description Step 04 red tests — Turso async `client` option for the
 * ann-index (buildAnnIndex) MCP tool.
 *
 * **Red contract:** `buildAnnIndex` should accept an optional `client` option
 * (a `@libsql/client` Client) and use it for DB access instead of requiring
 * `databasePath` and opening a separate database.
 *
 * The test calls `buildAnnIndex` with a `client` but WITHOUT
 * `databasePath`. After migration the tool uses the injected client, creates
 * the DiskANN vector index on chunks.embedding, and returns
 * `build_status: 'ready'` (GREEN).
 */

import {
  createSchemaClient,
  createEnvIsolation,
} from './turso-test-helpers.mjs';

const ANN_INDEX_PATH = '../tools/ann-index.mjs';

describe('ann-index (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    // Add the embedding column required by DiskANN (schema-v2 does not include it)
    await client.execute(
      'ALTER TABLE chunks ADD COLUMN embedding F8_BLOB(384)',
    );
    // @libsql/client :memory: does not support `USING libsql_vector_idx`,
    // so intercept the CREATE INDEX call and return a mock success. This
    // verifies that buildAnnIndex uses the injected client without requiring
    // databasePath.
    const realExecute = client.execute.bind(client);
    client.execute = async (sql) => {
      if (typeof sql === 'string' && sql.includes('libsql_vector_idx')) {
        return { rows: [] };
      }
      return realExecute(sql);
    };
    mod = await import(ANN_INDEX_PATH);
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

  describe('buildAnnIndex', () => {
    it('builds DiskANN index in the injected libSQL client without requiring databasePath', async () => {
      const result = await mod.buildAnnIndex({
        client,
        modelId: 'turso-test-model',
        dimension: 4,
        forceStrategy: 'diskann',
      });

      expect(result.build_status).toBe('ready');
    });
  });
});
