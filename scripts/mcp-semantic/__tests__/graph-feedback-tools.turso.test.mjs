/**
 * @module graph-feedback-tools.turso.test
 * @description Step 04 red tests — Turso async `client` option for traverse-graph,
 * submit-feedback, and expand-query MCP tools.
 *
 * **Red contract:** Each tool should accept an optional `client` option (a
 * `@libsql/client` Client) and use it for DB access instead of opening the
 * sync local SQLite DB via `openCortexDatabase()`.
 *
 * Tests inject a `:memory:` libSQL client preloaded with the v2 corpus schema
 * and fixture data using unique identifiers. The tools currently ignore
 * `client`, open the real on-disk DB, and fail to find the test data — making
 * every assertion below fail (RED). After migration the tools use the
 * injected client and the assertions pass (GREEN).
 *
 * The expand-query handler does not touch the DB directly (the DB access is in
 * `semantic-index/expand-query.mjs`, migrated in Step 05). The red contract
 * for expand-query is a handler-level marker: when `client` is provided, the
 * handler should set `expansion.embeddings_source === 'injected-client'` so
 * callers know the injected client will be forwarded. This is achievable in
 * Step 04 without cross-step dependency.
 */

import {
  createSchemaClient,
  insertTestFixtures,
  insertGraphFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
} from './turso-test-helpers.mjs';

const TRAVERSE_GRAPH_PATH = '../tools/traverse-graph.mjs';
const SUBMIT_FEEDBACK_PATH = '../tools/submit-feedback.mjs';
const EXPAND_QUERY_PATH = '../tools/expand-query.mjs';

describe('traverse-graph (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    await insertGraphFixtures(client);
    mod = await import(TRAVERSE_GRAPH_PATH);
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

  describe('traverseGraph', () => {
    it('discovers TursoTestEntityA from the injected libSQL client', async () => {
      const result = await mod.traverseGraph({
        seed_names: ['TursoTestEntityA'],
        client,
      });

      expect(result.entities.some((e) => e.name === 'TursoTestEntityA')).toBe(
        true,
      );
    });
  });
});

describe('submit-feedback (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    mod = await import(SUBMIT_FEEDBACK_PATH);
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

  describe('submitFeedback', () => {
    it('records a positive feedback event for chunk 424242 in the injected libSQL client', async () => {
      await mod.submitFeedback({
        chunk_id: TEST_CHUNK_ID,
        signal_type: 'positive',
        client,
      });

      const rows = await client.execute({
        sql: 'SELECT COUNT(*) AS c FROM feedback_events WHERE chunk_id = ?',
        args: [TEST_CHUNK_ID],
      });

      expect(Number(rows.rows[0].c)).toBeGreaterThan(0);
    });
  });
});

describe('expand-query (Turso async migration)', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;
  let mod;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    mod = await import(EXPAND_QUERY_PATH);
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

  describe('expandQueryHandler', () => {
    it('sets embeddings_source to injected-client when client option is provided', async () => {
      const result = await mod.expandQueryHandler({
        query: 'tursouniqueword',
        expand_query: false,
        client,
      });

      expect(result.expansion.embeddings_source).toBe('injected-client');
    });
  });
});
