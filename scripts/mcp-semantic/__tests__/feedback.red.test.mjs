/**
 * @module feedback.red.test
 * @description Red tests for feedback signal recording, boost computation, time decay,
 * impression decay, and privacy constraints.
 */

import { createHash } from 'node:crypto';
import { rm } from 'node:fs/promises';
import { createClient } from '@libsql/client';
import { splitSqlStatements, readCorpusSchema } from './turso-test-helpers.mjs';

const modulePath = '../tools/feedback-core.mjs';

async function setupDb() {
  // Use an in-memory database to avoid Windows EBUSY file-locking issues
  // during teardown. No test logic depends on file-backed persistence.
  const client = createClient({ url: ':memory:' });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  return { client, tempDir: null };
}

async function teardownDb(client, tempDir) {
  await client.close();
  // In-memory databases need no directory cleanup; tempDir is null.
  if (tempDir !== null) {
    await rm(tempDir, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 200,
    });
  }
}

async function insertDocumentAndChunk(client) {
  await client.execute({
    sql: `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at) VALUES ('test.ts', 'src', 0, 0, 'abc', 0)`,
  });
  const docResult = await client.execute('SELECT doc_id FROM documents');
  const doc = docResult.rows[0];
  await client.execute({
    sql: `INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth) VALUES (?, 0, 'test body', 0, 9, 0)`,
    args: [doc.doc_id],
  });
  const chunkResult = await client.execute('SELECT chunk_id FROM chunks');
  return chunkResult.rows[0].chunk_id;
}

describe('feedback-core', () => {
  describe('schema', () => {
    it('creates feedback_events table with required columns', async () => {
      const { client, tempDir } = await setupDb();
      try {
        const result = await client.execute(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'feedback_events'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ name: 'feedback_events' });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('creates feedback_scores table with required columns', async () => {
      const { client, tempDir } = await setupDb();
      try {
        const result = await client.execute(
          "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'feedback_scores'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ name: 'feedback_scores' });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('cascade deletes feedback_events when referenced chunk is removed', async () => {
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await client.execute({
          sql: "INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength) VALUES ('evt1', ?, 'impression', 0.1)",
          args: [chunkId],
        });
        await client.execute({
          sql: 'DELETE FROM chunks WHERE chunk_id = ?',
          args: [chunkId],
        });
        const countResult = await client.execute(
          'SELECT COUNT(*) as count FROM feedback_events',
        );
        const row = countResult.rows[0];
        expect(row).toEqual({ count: 0 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('cascade deletes feedback_scores when referenced chunk is removed', async () => {
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await client.execute({
          sql: 'INSERT INTO feedback_scores (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, feedback_boost) VALUES (?, 1, 0, 0, 0, 0, 0.1)',
          args: [chunkId],
        });
        await client.execute({
          sql: 'DELETE FROM chunks WHERE chunk_id = ?',
          args: [chunkId],
        });
        const countResult = await client.execute(
          'SELECT COUNT(*) as count FROM feedback_scores',
        );
        const row = countResult.rows[0];
        expect(row).toEqual({ count: 0 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });
  });

  describe('signal recording', () => {
    it('records impression event with pre-computed signal_strength 0.1', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'impression',
          query_hash: 'hash123',
        });
        const result = await client.execute(
          "SELECT signal_strength FROM feedback_events WHERE signal_type = 'impression'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ signal_strength: 0.1 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('records click event with pre-computed signal_strength 0.3', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'click',
          query_hash: 'hash123',
        });
        const result = await client.execute(
          "SELECT signal_strength FROM feedback_events WHERE signal_type = 'click'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ signal_strength: 0.3 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('records reference event with pre-computed signal_strength 0.6', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'reference',
          query_hash: 'hash123',
        });
        const result = await client.execute(
          "SELECT signal_strength FROM feedback_events WHERE signal_type = 'reference'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ signal_strength: 0.6 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('records positive explicit event with pre-computed signal_strength 1.0', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        const result = await client.execute(
          "SELECT signal_strength FROM feedback_events WHERE signal_type = 'positive'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ signal_strength: 1.0 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('records negative explicit event with pre-computed signal_strength -1.0', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'negative',
          query_hash: 'hash123',
        });
        const result = await client.execute(
          "SELECT signal_strength FROM feedback_events WHERE signal_type = 'negative'",
        );
        const row = result.rows[0];
        expect(row).toEqual({ signal_strength: -1.0 });
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('rejects an unknown signal_type', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await expect(
          recordFeedbackEventAsync(client, {
            chunk_id: chunkId,
            signal_type: 'bogus',
            query_hash: 'hash123',
          }),
        ).rejects.toThrow('Unknown signal_type');
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('treats a 64-hex query value as an already-hashed query', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        const preHashed = 'A'.repeat(64);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'impression',
          query_hash: preHashed,
        });
        const result = await client.execute(
          "SELECT query_hash FROM feedback_events WHERE signal_type = 'impression'",
        );
        const row = result.rows[0];
        expect(row.query_hash).toBe(preHashed.toLowerCase());
      } finally {
        await teardownDb(client, tempDir);
      }
    });
  });

  describe('feedback_boost sigmoid dampening', () => {
    it('computes 0.5 * tanh(netFeedback * 2.0) for a moderate positive score', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 1,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(0.5 * Math.tanh(2.0), 6);
    });

    it('computes 0.5 * tanh(netFeedback * 2.0) for a moderate negative score', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 0,
        total_negative: 1,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(-0.5 * Math.tanh(2.0), 6);
    });

    it('treats missing score fields as zero', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const result = computeFeedbackBoost({});
      expect(result).toBe(0);
    });

    it('clamps positive feedback_boost to at most +0.5 for extreme positive scores', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 100,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(0.5, 1);
    });

    it('clamps negative feedback_boost to at least -0.5 for extreme negative scores', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 0,
        total_negative: 100,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(-0.5, 1);
    });

    it('returns zero feedback_boost when positive and negative signals cancel out', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 1,
        total_negative: 1,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(0, 6);
    });
  });

  describe('time decay', () => {
    it('defines FEEDBACK_HALF_LIFE_MS as 7 days in milliseconds', async () => {
      const { FEEDBACK_HALF_LIFE_MS } = await import(modulePath);
      expect(FEEDBACK_HALF_LIFE_MS).toBe(7 * 24 * 60 * 60 * 1000);
    });

    it.skip('applies exponential time decay with 7-day half-life during recompute', async () => {
      const { recordFeedbackEventAsync, recomputeAllFeedbackScores } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        const sevenDaysAgo = new Date(
          Date.now() - 7 * 24 * 60 * 60 * 1000,
        ).toISOString();
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash',
          created_at: sevenDaysAgo,
        });
        await recomputeAllFeedbackScores(client);
        const result = await client.execute({
          sql: 'SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?',
          args: [chunkId],
        });
        const row = result.rows[0];
        // After 7 days decay = 0.5; total_positive_decayed = 0.5; boost = 0.5 * tanh(1.0) ≈ 0.381
        expect(row.feedback_boost).toBeCloseTo(0.5 * Math.tanh(1.0), 2);
      } finally {
        await teardownDb(client, tempDir);
      }
    });
  });

  describe('impression decay', () => {
    it('defines MIN_IMPRESSIONS_FOR_DECAY as 10', async () => {
      const { MIN_IMPRESSIONS_FOR_DECAY } = await import(modulePath);
      expect(MIN_IMPRESSIONS_FOR_DECAY).toBe(10);
    });

    it('defines MIN_CTR_FOR_NEUTRAL as 0.1', async () => {
      const { MIN_CTR_FOR_NEUTRAL } = await import(modulePath);
      expect(MIN_CTR_FOR_NEUTRAL).toBe(0.1);
    });

    it('applies no impression decay when total_impressions is below threshold', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 0,
        total_negative: 0,
        total_impressions: 5,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(0, 6);
    });

    it('applies no impression decay when CTR is above neutral threshold', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 0,
        total_negative: 0,
        total_impressions: 100,
        total_clicks: 20,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeCloseTo(0.5 * Math.tanh(0.12), 6);
    });

    it('applies negative linear impression decay when CTR is below neutral threshold', async () => {
      const { computeFeedbackBoost } = await import(modulePath);
      const scores = {
        total_positive: 0,
        total_negative: 0,
        total_impressions: 100,
        total_clicks: 0,
        total_references: 0,
      };
      const result = computeFeedbackBoost(scores);
      expect(result).toBeLessThan(0);
    });

    it.skip('applies impression decay during full recompute for low-CTR chunks', async () => {
      const { recordFeedbackEventAsync, recomputeAllFeedbackScores } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        for (let i = 0; i < 20; i++) {
          await recordFeedbackEventAsync(client, {
            chunk_id: chunkId,
            signal_type: 'impression',
            query_hash: `hash${i}`,
          });
        }
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'click',
          query_hash: 'hash0',
        });
        await recomputeAllFeedbackScores(client);
        const result = await client.execute({
          sql: 'SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?',
          args: [chunkId],
        });
        const row = result.rows[0];
        // 20 impressions, 1 click => CTR = 0.05 < 0.1 => negative decay applied
        expect(row.feedback_boost).toBeLessThan(0);
      } finally {
        await teardownDb(client, tempDir);
      }
    });
  });

  describe('privacy constraints', () => {
    it('truncates context field to a maximum of 500 characters', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        const longContext = 'x'.repeat(1000);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          context: longContext,
        });
        const result = await client.execute(
          "SELECT context FROM feedback_events WHERE signal_type = 'positive'",
        );
        const row = result.rows[0];
        expect(row.context.length).toBeLessThanOrEqual(500);
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('stores query as SHA-256 hash instead of plaintext', async () => {
      const { recordFeedbackEventAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        const query = 'NEAT crossover';
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'impression',
          query_hash: query,
        });
        const result = await client.execute({
          sql: "SELECT query_hash FROM feedback_events WHERE signal_type = 'impression'",
        });
        const row = result.rows[0];
        const expectedHash = createHash('sha256').update(query).digest('hex');
        expect(row.query_hash).toBe(expectedHash);
      } finally {
        await teardownDb(client, tempDir);
      }
    });
  });

  describe('updateFeedbackScores', () => {
    it('updates scores with negative events', async () => {
      const { recordFeedbackEventAsync, updateFeedbackScoresAsync } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        // Use explicit timestamps to ensure the negative event is more recent
        // than the positive event, producing a deterministic negative boost.
        // Without explicit timestamps, both events may land in the same
        // millisecond, making feedback_boost exactly 0.
        const baseNow = Date.now();
        const positiveCreatedAt = new Date(baseNow - 2000).toISOString();
        const negativeCreatedAt = new Date(baseNow - 1000).toISOString();
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
          created_at: positiveCreatedAt,
        });
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'negative',
          query_hash: 'hash123',
          created_at: negativeCreatedAt,
        });
        const nowMs = baseNow;
        const score = await updateFeedbackScoresAsync(client, chunkId, nowMs);
        expect(score.total_negative).toBeGreaterThan(0);
        expect(score.feedback_boost).toBeLessThan(0);
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('returns null when updating scores for a chunk with no events', async () => {
      const { updateFeedbackScoresAsync } = await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        const score = await updateFeedbackScoresAsync(
          client,
          chunkId,
          Date.now(),
        );
        expect(score).toBeNull();
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('applies time decay from numeric created_at timestamps', async () => {
      const { recordFeedbackEventAsync, updateFeedbackScoresAsync } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        const nowMs = Date.now();
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
          created_at: nowMs - 8 * 24 * 60 * 60 * 1000,
        });
        const score = await updateFeedbackScoresAsync(client, chunkId, nowMs);
        expect(score.total_positive).toBeGreaterThan(0);
        expect(score.total_positive).toBeLessThan(1);
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('falls back to 0 for unparseable created_at strings', async () => {
      const { recordFeedbackEventAsync, updateFeedbackScoresAsync } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
          created_at: 'not-a-date',
        });
        const score = await updateFeedbackScoresAsync(
          client,
          chunkId,
          Date.now(),
        );
        expect(score.total_positive).toBe(0);
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it('uses the current time when now is omitted', async () => {
      const { recordFeedbackEventAsync, updateFeedbackScoresAsync } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        const score = await updateFeedbackScoresAsync(client, chunkId);
        expect(score.total_positive).toBeGreaterThan(0);
      } finally {
        await teardownDb(client, tempDir);
      }
    });

    it.skip('recomputes all scores using the current time when now is omitted', async () => {
      const { recordFeedbackEventAsync, recomputeAllFeedbackScores } =
        await import(modulePath);
      const { client, tempDir } = await setupDb();
      try {
        const chunkId = await insertDocumentAndChunk(client);
        await recordFeedbackEventAsync(client, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        const count = await recomputeAllFeedbackScores(client);
        expect(count).toBe(1);
      } finally {
        await teardownDb(client, tempDir);
      }
    });
  });
});
