/**
 * @module metadata-tools.turso.test
 * @description Step 04 red tests — Turso async `client` option for freshness-check,
 * index-stats, and list-families MCP tools.
 *
 * **Red contract:** Each tool should accept an optional `client` option (a
 * `@libsql/client` Client) and use it for DB access instead of opening the
 * sync local SQLite DB via `openCortexDatabase()`.
 *
 * Tests inject a `:memory:` libSQL client preloaded with the v2 corpus schema
 * and a single document/chunk pair using unique identifiers. The tools
 * currently ignore `client`, open the real on-disk DB, and either throw
 * (file not found) or return real-corpus counts — making every assertion below
 * fail (RED). After migration the tools use the injected client and the
 * assertions pass (GREEN).
 */

import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_FILE_PATH,
  TEST_FAMILY,
} from './turso-test-helpers.mjs';

const FRESHNESS_PATH = '../tools/freshness-check.mjs';
const INDEX_STATS_PATH = '../tools/index-stats.mjs';
const LIST_FAMILIES_PATH = '../tools/list-families.mjs';

describe('freshness-check (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(FRESHNESS_PATH);
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
    it('returns fresh for the test document from the injected libSQL client', async () => {
      const result = await mod.freshnessCheck({
        file_path: TEST_FILE_PATH,
        freshnessProof: {
          mtime_ms: 1000,
          file_size: 500,
          sha256: 'turso-test-sha256',
        },
        client,
      });

      expect(result.fresh).toBe(true);
    });
  });
});

describe('index-stats (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(INDEX_STATS_PATH);
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
    it('returns total_documents 1 from the injected libSQL client', async () => {
      const result = await mod.indexStats({ client });

      expect(result.total_documents).toBe(1);
    });
  });
});

describe('list-families (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(LIST_FAMILIES_PATH);
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

  describe('listFamilies', () => {
    it('includes the turso-test family from the injected libSQL client', async () => {
      const result = await mod.listFamilies({ client });

      expect(result.families.some((f) => f.family === TEST_FAMILY)).toBe(true);
    });
  });
});
