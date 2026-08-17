/**
 * @module feedback-core.test
 * @description Coverage tests for feedback-core.mjs — feedback signal recording,
 * boost computation, and score aggregation.
 */
import { createClient } from '@libsql/client';

import {
  FEEDBACK_HALF_LIFE_MS,
  MIN_IMPRESSIONS_FOR_DECAY,
  MIN_CTR_FOR_NEUTRAL,
  VALID_SIGNAL_TYPES,
  computeFeedbackBoost,
  recordFeedbackEventAsync,
  updateFeedbackScoresAsync,
  recomputeAllFeedbackScores,
} from './feedback-core.mjs';

/**
 * Create an in-memory libSQL client with the feedback schema pre-loaded.
 * @returns {Promise<import('@libsql/client').Client>}
 */
async function createFeedbackClient() {
  const client = createClient({ url: ':memory:' });
  await client.execute(`
    CREATE TABLE feedback_events (
      event_id TEXT PRIMARY KEY,
      chunk_id INTEGER NOT NULL,
      signal_type TEXT NOT NULL,
      signal_strength REAL NOT NULL,
      query_hash TEXT,
      agent_id TEXT,
      context TEXT,
      created_at TEXT
    )
  `);
  await client.execute(`
    CREATE TABLE feedback_scores (
      chunk_id INTEGER PRIMARY KEY,
      total_positive REAL DEFAULT 0,
      total_negative REAL DEFAULT 0,
      total_impressions INTEGER DEFAULT 0,
      total_clicks INTEGER DEFAULT 0,
      total_references INTEGER DEFAULT 0,
      last_feedback_at INTEGER,
      feedback_boost REAL DEFAULT 0
    )
  `);
  return client;
}

describe('feedback-core', () => {
  describe('constants', () => {
    it('exports expected constants', () => {
      expect(FEEDBACK_HALF_LIFE_MS).toBe(7 * 24 * 60 * 60 * 1000);
      expect(MIN_IMPRESSIONS_FOR_DECAY).toBe(10);
      expect(MIN_CTR_FOR_NEUTRAL).toBe(0.1);
      expect(VALID_SIGNAL_TYPES).toEqual(
        expect.arrayContaining([
          'click',
          'impression',
          'negative',
          'positive',
          'reference',
          'irrelevant',
        ]),
      );
      expect(Object.isFrozen(VALID_SIGNAL_TYPES)).toBe(true);
    });
  });

  describe('computeFeedbackBoost', () => {
    it('returns 0 for all-zero scores', () => {
      expect(computeFeedbackBoost({})).toBe(0);
    });

    it('returns positive boost for positive signals', () => {
      const boost = computeFeedbackBoost({
        total_positive: 2.0,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      });
      expect(boost).toBeGreaterThan(0);
      expect(boost).toBeLessThanOrEqual(0.5);
    });

    it('returns negative boost for negative signals', () => {
      const boost = computeFeedbackBoost({
        total_positive: 0,
        total_negative: 2.0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      });
      expect(boost).toBeLessThan(0);
      expect(boost).toBeGreaterThanOrEqual(-0.5);
    });

    it('applies click-through-rate boost', () => {
      const boost = computeFeedbackBoost({
        total_positive: 0,
        total_negative: 0,
        total_impressions: 10,
        total_clicks: 5,
        total_references: 0,
      });
      expect(boost).toBeGreaterThan(0);
    });

    it('applies reference bonus (capped at 1.0)', () => {
      const boostFew = computeFeedbackBoost({
        total_positive: 0,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 2,
      });
      const boostMany = computeFeedbackBoost({
        total_positive: 0,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 100,
      });
      expect(boostFew).toBeGreaterThan(0);
      // Many references cap at 1.0 * 0.3 = 0.3 combined positive contribution
      expect(boostMany).toBeGreaterThan(boostFew);
    });

    it('applies impression decay when CTR is low and impressions >= threshold', () => {
      const boost = computeFeedbackBoost({
        total_positive: 0.5,
        total_negative: 0,
        total_impressions: 20,
        total_clicks: 0,
        total_references: 0,
      });
      // With low CTR, impression decay reduces the boost
      expect(boost).toBeLessThan(0.5 * Math.tanh(0.5 * 2.0));
    });

    it('does NOT apply impression decay when CTR >= MIN_CTR_FOR_NEUTRAL', () => {
      const boost = computeFeedbackBoost({
        total_positive: 0.5,
        total_negative: 0,
        total_impressions: 20,
        total_clicks: 5, // CTR = 0.25 > 0.1
        total_references: 0,
      });
      // No decay applied, boost should be positive
      expect(boost).toBeGreaterThan(0);
    });

    it('does NOT apply impression decay when impressions < threshold', () => {
      const boost = computeFeedbackBoost({
        total_positive: 0,
        total_negative: 0,
        total_impressions: 5, // < 10
        total_clicks: 0,
        total_references: 0,
      });
      // CTR = 0/1 = 0 but impressions < 10, no decay
      expect(boost).toBe(0);
    });

    it('clamps result to [-0.5, +0.5]', () => {
      const highPositive = computeFeedbackBoost({
        total_positive: 1000,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      });
      expect(highPositive).toBeCloseTo(0.5, 5);

      const highNegative = computeFeedbackBoost({
        total_positive: 0,
        total_negative: 1000,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      });
      expect(highNegative).toBeCloseTo(-0.5, 5);
    });
  });

  describe('recordFeedbackEventAsync', () => {
    let client;

    beforeEach(async () => {
      client = await createFeedbackClient();
    });

    afterEach(async () => {
      await client.close();
    });

    it('records a click event with default strength', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
      });
      expect(row).toBeDefined();
      expect(Number(row.chunk_id)).toBe(1);
      expect(row.signal_type).toBe('click');
      expect(Number(row.signal_strength)).toBe(0.3);
    });

    it('records a positive event with default strength', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
      });
      expect(row.signal_type).toBe('positive');
      expect(Number(row.signal_strength)).toBe(1.0);
    });

    it('records a negative event with default strength', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'negative',
      });
      expect(row.signal_type).toBe('negative');
      expect(Number(row.signal_strength)).toBe(-1.0);
    });

    it('records an impression event', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'impression',
      });
      expect(Number(row.signal_strength)).toBe(0.1);
    });

    it('records a reference event', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'reference',
      });
      expect(Number(row.signal_strength)).toBe(0.6);
    });

    it('records an irrelevant event', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'irrelevant',
      });
      expect(Number(row.signal_strength)).toBe(-0.5);
    });

    it('throws on unknown signal_type', async () => {
      await expect(
        recordFeedbackEventAsync(client, {
          chunk_id: 1,
          signal_type: 'unknown',
        }),
      ).rejects.toThrow('Unknown signal_type: unknown');
    });

    it('uses explicit signal_strength when provided', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
        signal_strength: 0.5,
      });
      expect(Number(row.signal_strength)).toBe(0.5);
    });

    it('uses explicit signal_strength of 0', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
        signal_strength: 0,
      });
      expect(Number(row.signal_strength)).toBe(0);
    });

    it('hashes plaintext query and stores the hash', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        query: 'my search query',
      });
      expect(row.query_hash).toMatch(/^[0-9a-f]{64}$/);
    });

    it('passes through pre-computed query hash (lowercased)', async () => {
      const hash = 'A'.repeat(64);
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        query_hash: hash,
      });
      expect(row.query_hash).toBe(hash.toLowerCase());
    });

    it('stores null query_hash when no query provided', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
      });
      expect(row.query_hash).toBeNull();
    });

    it('truncates context to 500 characters', async () => {
      const longContext = 'x'.repeat(600);
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        context: longContext,
      });
      expect(row.context.length).toBe(500);
    });

    it('stores null context when not provided', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
      });
      expect(row.context).toBeNull();
    });

    it('stores null context when explicitly null', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        context: null,
      });
      expect(row.context).toBeNull();
    });

    it('stores agent_id when provided', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        agent_id: 'agent-001',
      });
      expect(row.agent_id).toBe('agent-001');
    });

    it('stores null agent_id when not provided', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
      });
      expect(row.agent_id).toBeNull();
    });

    it('uses provided created_at', async () => {
      const row = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        created_at: '2023-01-01T00:00:00.000Z',
      });
      expect(row.created_at).toBe('2023-01-01T00:00:00.000Z');
    });

    it('normalizes duplicate same-session positive signal (returns existing row)', async () => {
      // First positive
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
        agent_id: 'agent-001',
      });

      // Second positive within same session window → should return existing row
      const existingRow = await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
        agent_id: 'agent-001',
      });

      // The returned row should be the existing one, not a new insert
      const allRows = await client.execute(
        'SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = 1 AND signal_type = \'positive\'',
      );
      expect(Number(allRows.rows[0].count)).toBe(1);
      expect(existingRow).toBeDefined();
    });

    it('does NOT normalize when agent_id is null', async () => {
      // First positive with no agent_id
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
      });

      // Second positive with no agent_id → should insert new (not normalized)
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
      });

      const allRows = await client.execute(
        'SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = 1 AND signal_type = \'positive\'',
      );
      expect(Number(allRows.rows[0].count)).toBe(2);
    });

    it('records non-positive signal even with same agent_id', async () => {
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        agent_id: 'agent-001',
      });
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
        agent_id: 'agent-001',
      });

      const allRows = await client.execute(
        'SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = 1 AND signal_type = \'click\'',
      );
      expect(Number(allRows.rows[0].count)).toBe(2);
    });
  });

  describe('updateFeedbackScoresAsync', () => {
    let client;

    beforeEach(async () => {
      client = await createFeedbackClient();
    });

    afterEach(async () => {
      await client.close();
    });

    it('returns null when no events exist for the chunk', async () => {
      const result = await updateFeedbackScoresAsync(client, 999);
      expect(result).toBeNull();
    });

    it('computes and upserts score for a chunk with events', async () => {
      await recordFeedbackEventAsync(client, {
        chunk_id: 5,
        signal_type: 'positive',
      });
      await recordFeedbackEventAsync(client, {
        chunk_id: 5,
        signal_type: 'click',
      });

      const score = await updateFeedbackScoresAsync(client, 5);
      expect(score).not.toBeNull();
      expect(score.chunk_id).toBe(5);
      expect(score.total_positive).toBeGreaterThan(0);
      expect(score.total_clicks).toBe(1);
      expect(typeof score.feedback_boost).toBe('number');

      // Verify it was upserted into feedback_scores
      const stored = await client.execute(
        'SELECT * FROM feedback_scores WHERE chunk_id = 5',
      );
      expect(stored.rows.length).toBe(1);
      expect(Number(stored.rows[0].total_clicks)).toBe(1);
    });

    it('upserts (updates) existing score row on recompute', async () => {
      await recordFeedbackEventAsync(client, {
        chunk_id: 5,
        signal_type: 'positive',
      });
      await updateFeedbackScoresAsync(client, 5);

      // Add another event
      await recordFeedbackEventAsync(client, {
        chunk_id: 5,
        signal_type: 'click',
      });
      await updateFeedbackScoresAsync(client, 5);

      const stored = await client.execute(
        'SELECT * FROM feedback_scores WHERE chunk_id = 5',
      );
      expect(stored.rows.length).toBe(1);
      expect(Number(stored.rows[0].total_clicks)).toBe(1);
    });
  });

  describe('recomputeAllFeedbackScores', () => {
    let client;

    beforeEach(async () => {
      client = await createFeedbackClient();
    });

    afterEach(async () => {
      await client.close();
    });

    it('returns 0 when no events exist', async () => {
      const count = await recomputeAllFeedbackScores(client);
      expect(count).toBe(0);
    });

    it('recomputes scores for all chunks with events', async () => {
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'positive',
      });
      await recordFeedbackEventAsync(client, {
        chunk_id: 2,
        signal_type: 'negative',
      });
      await recordFeedbackEventAsync(client, {
        chunk_id: 3,
        signal_type: 'click',
      });

      const count = await recomputeAllFeedbackScores(client);
      expect(count).toBe(3);

      // Verify scores were upserted
      const scores = await client.execute(
        'SELECT * FROM feedback_scores ORDER BY chunk_id',
      );
      expect(scores.rows.length).toBe(3);
    });

    it('skips chunks with no non-zero aggregate (null from buildChunkAggregate)', async () => {
      // Insert an event that results in event_count > 0
      await recordFeedbackEventAsync(client, {
        chunk_id: 1,
        signal_type: 'click',
      });

      const count = await recomputeAllFeedbackScores(client);
      expect(count).toBe(1);
    });
  });
});