/**
 * @module load-tools.turso.test
 * @description Step 04 red tests — Turso async `client` option for load-chunk,
 * load-document, and load-parent-chunk MCP tools.
 *
 * **Red contract:** Each tool should accept an optional `client` option (a
 * `@libsql/client` Client) and use it for DB access instead of opening the
 * sync local SQLite DB via `openCortexDatabase()` / `readChunkRow()`.
 *
 * Tests inject a `:memory:` libSQL client preloaded with the v2 corpus schema
 * and a single chunk with a unique `chunk_id` (424242) that cannot exist in the
 * real on-disk corpus. The tools currently ignore `client`, open the real DB,
 * and fail to find the test chunk — making every assertion below fail (RED).
 * After migration the tools use the injected client, find the chunk, and the
 * assertions pass (GREEN).
 */

import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
  TEST_PARENT_CHUNK_ID,
} from './turso-test-helpers.mjs';

const LOAD_CHUNK_PATH = '../tools/load-chunk.mjs';
const LOAD_DOC_PATH = '../tools/load-document.mjs';
const LOAD_PARENT_PATH = '../tools/load-parent-chunk.mjs';

describe('load-chunk (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(LOAD_CHUNK_PATH);
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

  describe('loadChunk', () => {
    it('returns chunk 424242 from the injected libSQL client', async () => {
      const result = await mod.loadChunk({
        chunk_id: TEST_CHUNK_ID,
        client,
      });

      expect(result.chunk.chunk_id).toBe(TEST_CHUNK_ID);
    });
  });
});

describe('load-document (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(LOAD_DOC_PATH);
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

  describe('loadDocument', () => {
    it('returns chunks from the injected libSQL client', async () => {
      const result = await mod.loadDocument({
        file_path: 'src/turso-async-test.ts',
        client,
      });

      expect(Array.isArray(result.chunks)).toBe(true);
      expect(result.chunks.length).toBeGreaterThan(0);
    });
  });
});

describe('load-parent-chunk (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(LOAD_PARENT_PATH);
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

  describe('loadParentChunk', () => {
    it('returns parent chunk 424241 for sub-chunk 424242 from the injected libSQL client', async () => {
      const result = await mod.loadParentChunk({
        chunk_id: TEST_CHUNK_ID,
        client,
      });

      expect(result.parent_chunk.chunk_id).toBe(TEST_PARENT_CHUNK_ID);
    });
  });
});
