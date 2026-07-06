/**
 * @module validate.turso.test
 * @description Red tests for Turso async-migration of validate-index and
 *              validate-embeddings scripts.
 *
 * These tests verify that the validation scripts accept an optional `client`
 * parameter (a `@libsql/client` Client) and use it for database reads instead
 * of creating a synchronous native SQLite Database.
 *
 * RED PHASE: the scripts currently use sync native SQLite and ignore the
 * `client` parameter. Every test fails because the function either:
 *  - reads from the real on-disk DB instead of the injected client, or
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
  TEST_MODEL_ID,
} from './turso-test-helpers.mjs';
import os from 'node:os';
import path from 'node:path';

const { saveEnv, restoreEnv } = createEnvIsolation();

// ---------------------------------------------------------------------------
// validate-index.mjs — validateDatabase
// ---------------------------------------------------------------------------

describe('validateDatabase (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should read documents and chunks from the provided libSQL client', async () => {
    const { validateDatabase } = await import('../validate-index.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);

    // Pass a non-existent databasePath so current code returns "Database not
    // found". After migration, the function should use the client instead.
    const nonExistentPath = path.join(
      os.tmpdir(),
      `turso-nonexistent-validate-${Date.now()}.sqlite`,
    );

    const result = await validateDatabase({
      client,
      databasePath: nonExistentPath,
    });

    // The client has exactly 1 document and 1 chunk.
    // Currently the function checks existsSync(nonExistentPath) → false →
    // returns { documents: 0, chunks: 0 } → RED.
    expect(result.documents).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// validate-embeddings.mjs — validateEmbeddings
// ---------------------------------------------------------------------------

describe('validateEmbeddings (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should read chunk and embedding counts from the provided libSQL client', async () => {
    const { validateEmbeddings } = await import('../validate-embeddings.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);
    await insertEmbeddingFixtures(client);

    // Pass non-existent paths so current code throws when trying to open
    // them with the native driver. After migration, the function should use
    // the client instead.
    const nonExistentCorpus = path.join(
      os.tmpdir(),
      `turso-nonexistent-ve-c-${Date.now()}.sqlite`,
    );

    const result = await validateEmbeddings({
      client,
      corpusDatabasePath: nonExistentCorpus,
      modelId: TEST_MODEL_ID,
    });

    // The client has 1 chunk and 1 embedding (both in the chunks table).
    // Currently the function throws trying to open nonExistentCorpus → RED.
    expect(result.chunk_count).toBe(1);
  });
});
