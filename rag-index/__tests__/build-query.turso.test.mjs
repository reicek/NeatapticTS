/**
 * @module build-query.turso.test
 * @description Red tests for Turso async-migration of build-index, embed-index,
 *              and query-dense scripts.
 *
 * These tests verify that the semantic-index scripts accept an optional
 * `client` parameter (a `@libsql/client` Client) and use it for database
 * operations instead of creating a synchronous native SQLite Database.
 *
 * RED PHASE: the scripts currently use sync native SQLite and ignore the
 * `client` parameter. Every test fails because the function either:
 *  - writes to its own native SQLite DB instead of the injected client, or
 *  - throws because it tries to open a non-existent DB path instead of using
 *    the injected client.
 *
 * Pure .mjs test — runs via Jest ESM project `semantic-index-mjs`.
 */

import {
  createSchemaClient,
  insertTestFixtures,
  insertEmbeddingFixtures,
  createEnvIsolation,
  UNIQUE_QUERY_TERM,
  TEST_CHUNK_ID,
  TEST_DOC_ID,
  TEST_FILE_PATH,
  TEST_FAMILY,
  TEST_MODEL_ID,
  TEST_MODEL_SHA256,
  TEST_DIMENSION,
} from './turso-test-helpers.mjs';
import os from 'node:os';
import path from 'node:path';

const { saveEnv, restoreEnv } = createEnvIsolation();

// ---------------------------------------------------------------------------
// build-index.mjs — buildSemanticIndex
// ---------------------------------------------------------------------------

describe('buildSemanticIndex (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should use the provided libSQL client instead of creating a native SQLite database', async () => {
    const { buildSemanticIndex } = await import('../build-index.mjs');
    const client = await createSchemaClient();
    // Insert a test document so we can verify the function operates on the
    // client. With empty corpusDocuments, the function should purge docs not
    // in the corpus — meaning the test doc should be purged FROM THE CLIENT.
    await client.execute({
      sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [
        TEST_DOC_ID,
        TEST_FILE_PATH,
        TEST_FAMILY,
        1000,
        500,
        'test-sha',
        1000,
      ],
    });
    const tmpPath = path.join(
      os.tmpdir(),
      `turso-test-build-${Date.now()}.sqlite`,
    );
    await buildSemanticIndex({
      client,
      corpusDocuments: [],
      coverageReport: {},
      databasePath: tmpPath,
    });
    // If the function used the client, it would have called
    // deleteMissingDocuments on the client, purging our test doc.
    // Currently the function creates its own native SQLite DB at tmpPath
    // and purges from that — the client is untouched → count is 1 → RED.
    const result = await client.execute(
      'SELECT COUNT(*) AS count FROM documents',
    );
    expect(Number(result.rows[0].count)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// embed-index.mjs — buildEmbeddingIndex
// ---------------------------------------------------------------------------

describe('buildEmbeddingIndex (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should write chunk embeddings to the provided libSQL client', async () => {
    const { buildEmbeddingIndex } = await import('../embed-index.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);

    // Mock embedText that returns a deterministic 384-dim Float32Array
    const mockEmbedText = async () => {
      const vec = new Float32Array(TEST_DIMENSION);
      for (let i = 0; i < TEST_DIMENSION; i += 1) {
        vec[i] = (i % 11) / 10.0;
      }
      return vec;
    };

    // Pass a non-existent corpusDatabasePath so current code throws when it
    // tries to open it with the native driver. After migration, the function
    // should use the client instead.
    const nonExistentPath = path.join(
      os.tmpdir(),
      `turso-nonexistent-${Date.now()}.sqlite`,
    );

    await buildEmbeddingIndex({
      client,
      embedText: mockEmbedText,
      dimension: TEST_DIMENSION,
      modelSha256: TEST_MODEL_SHA256,
      modelId: TEST_MODEL_ID,
      corpusDatabasePath: nonExistentPath,
    });

    // After migration, the embedding should be written to chunks.embedding
    // in the client. Currently the function throws when trying to open the
    // non-existent corpus DB → test fails → RED.
    const result = await client.execute({
      sql: 'SELECT embedding FROM chunks WHERE chunk_id = ?',
      args: [TEST_CHUNK_ID],
    });
    expect(result.rows[0].embedding).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// query-dense.mjs — queryDenseIndex
// ---------------------------------------------------------------------------

describe('queryDenseIndex (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should query BM25 results from the provided libSQL client', async () => {
    const { queryDenseIndex } = await import('../query-dense.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);

    // Search for the unique term that only exists in the client's test fixture.
    // Currently the function opens the real on-disk DB (ignoring client) and
    // cannot find the test chunk → results do not contain TEST_CHUNK_ID → RED.
    const result = await queryDenseIndex({
      client,
      query: UNIQUE_QUERY_TERM,
      limit: 10,
    });

    const chunkIds = result.results.map((r) => r.chunk_id);
    expect(chunkIds).toContain(TEST_CHUNK_ID);
  });
});
