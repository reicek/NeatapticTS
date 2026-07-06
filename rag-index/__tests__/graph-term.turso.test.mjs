/**
 * @module graph-term.turso.test
 * @description Red tests for Turso async-migration of build-entity-graph and
 *              build-term-index scripts.
 *
 * These tests verify that the graph and term index scripts accept an optional
 * `client` parameter (a `@libsql/client` Client) and use it for database
 * operations instead of creating a synchronous native SQLite Database.
 *
 * RED PHASE: the scripts currently use sync native SQLite and ignore the
 * `client` parameter. Every test fails because the function either:
 *  - writes to its own native SQLite DB instead of the injected client, or
 *  - calls native SQLite-specific synchronous API (.prepare().all()) on the libSQL client
 *    which does not have that method → TypeError → RED.
 *
 * Pure .mjs test — runs via Jest ESM project `semantic-index-mjs`.
 */

import {
  createSchemaClient,
  insertTestFixtures,
  insertEmbeddingFixtures,
  insertGraphFixtures,
  createEnvIsolation,
  UNIQUE_QUERY_TERM,
  TEST_CHUNK_ID,
  TEST_MODEL_ID,
  TEST_MODEL_SHA256,
  TEST_DIMENSION,
} from './turso-test-helpers.mjs';
import os from 'node:os';
import path from 'node:path';

const { saveEnv, restoreEnv } = createEnvIsolation();

// ---------------------------------------------------------------------------
// build-entity-graph.mjs — buildEntityGraph
// ---------------------------------------------------------------------------

describe('buildEntityGraph (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should use the provided libSQL client for entity graph writes', async () => {
    const { buildEntityGraph } = await import('../build-entity-graph.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);
    await insertGraphFixtures(client);

    const tmpPath = path.join(
      os.tmpdir(),
      `turso-test-graph-${Date.now()}.sqlite`,
    );

    // After migration, the function should use the client for DB writes.
    // It deletes all existing entities before inserting new ones, so the
    // test entity should be purged from the client.
    // Currently the function creates its own native SQLite DB at tmpPath
    // and deletes from that — the client is untouched → test entity remains
    // → count is 1 → RED.
    await buildEntityGraph({
      client,
      databasePath: tmpPath,
    });

    const result = await client.execute({
      sql: 'SELECT COUNT(*) AS count FROM entities WHERE qualified_name = ?',
      args: ['src/turso.TursoScriptTestEntityA'],
    });
    expect(Number(result.rows[0].count)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// build-term-index.mjs — extractQualifyingTerms
// ---------------------------------------------------------------------------

describe('extractQualifyingTerms (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should accept a libSQL client and read qualifying terms from it', async () => {
    const { extractQualifyingTerms } = await import('../build-term-index.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);

    // After migration, the function should accept a libSQL client (which uses
    // client.execute() returning a Promise) instead of a native SQLite
    // Database (which uses db.prepare().all() synchronously).
    // Currently the function calls corpusDatabase.prepare() which does not
    // exist on a libSQL client → TypeError → RED.
    const result = await extractQualifyingTerms(client, {
      minFrequency: 1,
      maxFrequencyRatio: 1.0,
      minTermLength: 3,
    });

    // The test chunk body contains UNIQUE_QUERY_TERM which should appear as
    // a qualifying term.
    expect(result.qualifyingTerms.has(UNIQUE_QUERY_TERM)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// build-term-index.mjs — buildTermEmbeddings
// ---------------------------------------------------------------------------

describe('buildTermEmbeddings (Turso async migration)', () => {
  beforeEach(() => saveEnv());
  afterEach(() => restoreEnv());

  it('should write term embeddings to the provided libSQL client', async () => {
    const { extractQualifyingTerms, buildTermEmbeddings } =
      await import('../build-term-index.mjs');
    const client = await createSchemaClient();
    await insertTestFixtures(client);
    await insertEmbeddingFixtures(client);

    // Extract qualifying terms from the client (will fail in red phase because
    // extractQualifyingTerms is sync and uses native SQLite API).
    // Use a Map with the test term for the buildTermEmbeddings call.
    const qualifyingTerms = new Map();
    qualifyingTerms.set(UNIQUE_QUERY_TERM, {
      frequency: 1,
      doc_family_count: 1,
      chunkIds: [TEST_CHUNK_ID],
    });

    // After migration, buildTermEmbeddings should accept a libSQL client and
    // write term embeddings to the term_embeddings table.
    // Currently the function calls embeddingsDatabase.prepare() which does not
    // exist on a libSQL client → TypeError → RED.
    await buildTermEmbeddings(client, qualifyingTerms, {
      modelId: TEST_MODEL_ID,
      modelSha256: TEST_MODEL_SHA256,
      dimension: TEST_DIMENSION,
    });

    const result = await client.execute({
      sql: 'SELECT COUNT(*) AS count FROM term_embeddings WHERE model_id = ?',
      args: [TEST_MODEL_ID],
    });
    expect(Number(result.rows[0].count)).toBeGreaterThan(0);
  });
});
