/**
 * @module submit-feedback.harden.red.test
 * @description Red tests for hardening submit_feedback to the Step 10/21 design spec.
 *
 * The spec requires irrelevant signal type, signal_strength override,
 * aggregate score updates, and structured error taxonomy.
 */

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';

async function setupDb() {
  const tempDir = await mkdtemp(
    path.join(tmpdir(), 'submit-feedback-harden-test-'),
  );
  const dbPath = path.join(tempDir, 'test.sqlite');
  const client = createClient({ url: pathToFileURL(dbPath).href });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  await client.execute(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES ('src/foo.ts', 'ts-source', 0, 100, 'a', 1);
  `);
  const docResult = await client.execute('SELECT doc_id FROM documents');
  const docId = docResult.rows[0].doc_id;
  await client.execute({
    sql: "INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth) VALUES (?, 0, 'test body', 0, 9, 0)",
    args: [docId],
  });
  return { client, dbPath, tempDir };
}

async function teardown(client, tempDir) {
  await client.close();
  await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}

describe('submit-feedback hardened', () => {
  describe('schema validation', () => {
    it('rejects missing chunk_id with MISSING_CHUNK_ID', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        await expect(
          submitFeedback({ client, signal_type: 'positive' }),
        ).rejects.toThrow(/MISSING_CHUNK_ID/);
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('rejects invalid chunk_id with MISSING_CHUNK_ID', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        await expect(
          submitFeedback({
            client,
            chunk_id: 'not-a-number',
            signal_type: 'positive',
          }),
        ).rejects.toThrow(/MISSING_CHUNK_ID/);
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('rejects invalid signal_type with INVALID_SIGNAL_TYPE', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        await expect(
          submitFeedback({
            client,
            chunk_id: chunkId,
            signal_type: 'like',
          }),
        ).rejects.toThrow(/INVALID_SIGNAL_TYPE/);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('signal types', () => {
    it('accepts the irrelevant signal type', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        const result = await submitFeedback({
          client,
          chunk_id: chunkId,
          signal_type: 'irrelevant',
        });

        expect(result.signal_type).toBe('irrelevant');
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('accepts a signal_strength override and stores it', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        const result = await submitFeedback({
          client,
          chunk_id: chunkId,
          signal_type: 'positive',
          signal_strength: 2.5,
        });

        const eventResult = await client.execute({
          sql: 'SELECT signal_strength FROM feedback_events WHERE chunk_id = ?',
          args: [chunkId],
        });
        const event = eventResult.rows[0];

        expect(result.signal_strength).toBe(2.5);
        expect(event.signal_strength).toBe(2.5);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('aggregate scoring', () => {
    it('records the event in feedback_events', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        await submitFeedback({
          client,
          chunk_id: chunkId,
          signal_type: 'reference',
          query: 'NEAT activation',
          agent_id: 'test-agent',
        });

        const rowResult = await client.execute({
          sql: 'SELECT * FROM feedback_events WHERE chunk_id = ?',
          args: [chunkId],
        });
        const row = rowResult.rows[0];

        expect(row).toEqual(
          expect.objectContaining({
            signal_type: 'reference',
            agent_id: 'test-agent',
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('updates feedback_scores aggregate and returns summary fields', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        const result = await submitFeedback({
          client,
          chunk_id: chunkId,
          signal_type: 'positive',
        });

        expect(result).toEqual(
          expect.objectContaining({
            feedback_score: expect.any(Number),
            total_signals: expect.any(Number),
            feedback_boost: expect.any(Number),
          }),
        );

        const scoreResult = await client.execute({
          sql: 'SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?',
          args: [chunkId],
        });
        const score = scoreResult.rows[0];

        expect(score.feedback_boost).toBeGreaterThan(0);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('optional fields', () => {
    it('truncates context longer than 500 characters', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        await submitFeedback({
          client,
          chunk_id: chunkId,
          signal_type: 'positive',
          context: 'x'.repeat(600),
        });

        const rowResult = await client.execute({
          sql: 'SELECT context FROM feedback_events WHERE chunk_id = ?',
          args: [chunkId],
        });
        const row = rowResult.rows[0];

        expect(row.context.length).toBeLessThanOrEqual(500);
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('accepts optional agent_id and query', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = (await client.execute('SELECT chunk_id FROM chunks'))
          .rows[0].chunk_id;

        const result = await submitFeedback({
          client,
          chunk_id: chunkId,
          signal_type: 'negative',
          query: 'NEAT mutation',
          agent_id: 'agent-42',
        });

        expect(result).toEqual(
          expect.objectContaining({
            chunk_id: chunkId,
            signal_type: 'negative',
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });
  });
});
