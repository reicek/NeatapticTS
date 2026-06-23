/**
 * @module feedback-turso.test
 * @description Phase 5 Step 07 red tests — SQL time-decay feedback scoring.
 *
 * **Red contract:** These tests assert that the feedback scoring pipeline
 * moves time-decay computation from client-side JavaScript to server-side SQL.
 * Every assertion below fails against the current implementation (RED) because:
 *
 * - `last_feedback_at` is stored as an ISO string (typeof 'text'), not INTEGER.
 * - `runBm25Search` does not JOIN `feedback_scores` and has no `POWER(0.95, days)`.
 * - `recordSearchImpressions` records impressions one-by-one via `execute()`,
 *   not via `client.batch()`.
 * - `buildChunkAggregateAsync` uses `Math.pow(0.5, age / FEEDBACK_HALF_LIFE_MS)`
 *   (JS-side decay) and `SELECT * FROM feedback_events WHERE chunk_id = ?`
 *   (loads all events into JS).
 *
 * After the SQL time-decay migration, all assertions pass (GREEN).
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import {
  createSchemaClient,
  insertTestFixtures,
  TEST_CHUNK_ID,
} from './turso-test-helpers.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FEEDBACK_CORE_PATH = path.join(__dirname, '../tools/feedback-core.mjs');
const SEARCH_CORPUS_PATH = path.join(__dirname, '../tools/search-corpus.mjs');
const FEEDBACK_CORE_MODULE = '../tools/feedback-core.mjs';

/**
 * Extract the source body of a named function from raw source text.
 *
 * Captures from the function signature (`function <name>`) to the start of
 * the next top-level `function` / `async function` / `export function`
 * declaration, or end of file. This allows source-inspection assertions to
 * target a single function without matching comments or unrelated code.
 *
 * @param {string} source - Raw source text.
 * @param {string} funcName - Function name to extract.
 * @returns {string} The function source block, or empty string when not found.
 */
function extractFunctionBody(source, funcName) {
  const marker = `function ${funcName}`;
  const startIdx = source.indexOf(marker);
  if (startIdx === -1) {
    return '';
  }
  const rest = source.slice(startIdx + marker.length);
  const nextFuncMatch = rest.search(
    /\n(?:export\s+)?(?:async\s+)?function\s+\w+/,
  );
  const endIdx =
    nextFuncMatch === -1
      ? source.length
      : startIdx + marker.length + nextFuncMatch;
  return source.slice(startIdx, endIdx);
}

describe('feedback-turso', () => {
  // ---------------------------------------------------------------------------
  // 1. Schema — last_feedback_at stored as INTEGER (Unix epoch)
  // ---------------------------------------------------------------------------
  describe('schema — last_feedback_at as INTEGER', () => {
    it('stores last_feedback_at as INTEGER (Unix epoch) after updateFeedbackScoresAsync', async () => {
      // Arrange: in-memory schema client with a feedback event for TEST_CHUNK_ID
      const client = await createSchemaClient();
      try {
        await insertTestFixtures(client);
        const now = Date.now();
        await client.execute({
          sql: `INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength, created_at)
                VALUES (?, ?, 'positive', 1.0, ?)`,
          args: ['evt-int-1', TEST_CHUNK_ID, new Date(now).toISOString()],
        });

        // Act: recompute scores for the chunk
        const { updateFeedbackScoresAsync } = await import(
          FEEDBACK_CORE_MODULE
        );
        await updateFeedbackScoresAsync(client, TEST_CHUNK_ID, now);

        // Assert: last_feedback_at column type is 'integer', not 'text'
        const result = await client.execute({
          sql: `SELECT typeof(last_feedback_at) AS col_type FROM feedback_scores WHERE chunk_id = ?`,
          args: [TEST_CHUNK_ID],
        });
        expect(result.rows[0].col_type).toBe('integer');
      } finally {
        await client.close();
      }
    });
  });

  // ---------------------------------------------------------------------------
  // 2. SQL time-decay formula in the search query
  // ---------------------------------------------------------------------------
  describe('SQL time-decay formula', () => {
    it('uses POWER(0.95, days) decay in the feedback scoring SQL', async () => {
      // Arrange: read search-corpus.mjs source
      const source = await readFile(SEARCH_CORPUS_PATH, 'utf8');

      // Act + Assert: source contains POWER(0.95, ...) (case-insensitive)
      expect(source.toUpperCase()).toContain('POWER(0.95');
    });

    it('applies feedback boost via SQL JOIN in the search query', async () => {
      // Arrange: read search-corpus.mjs source and extract runBm25Search body
      const source = await readFile(SEARCH_CORPUS_PATH, 'utf8');
      const bm25Body = extractFunctionBody(source, 'runBm25Search');

      // Act + Assert: the BM25 query JOINs feedback_scores (not a separate SELECT)
      expect(bm25Body).toMatch(/JOIN\s+feedback_scores/i);
    });
  });

  // ---------------------------------------------------------------------------
  // 3. Batch impression recording
  // ---------------------------------------------------------------------------
  describe('batch impression recording', () => {
    it('records search impressions via client.batch() not one-by-one execute', async () => {
      // Arrange: read search-corpus.mjs source and extract recordSearchImpressions body
      const source = await readFile(SEARCH_CORPUS_PATH, 'utf8');
      const body = extractFunctionBody(source, 'recordSearchImpressions');

      // Act + Assert: impression recording uses client.batch(), not a for-loop of execute()
      expect(body).toContain('.batch(');
    });
  });

  // ---------------------------------------------------------------------------
  // 4. Client-side time-decay removed
  // ---------------------------------------------------------------------------
  describe('client-side time-decay removed', () => {
    it('does not use Math.pow for time-decay in buildChunkAggregateAsync', async () => {
      // Arrange: read feedback-core.mjs source and extract buildChunkAggregateAsync body
      const source = await readFile(FEEDBACK_CORE_PATH, 'utf8');
      const body = extractFunctionBody(source, 'buildChunkAggregateAsync');

      // Act + Assert: no JS-side Math.pow half-life decay remains
      expect(body.includes('Math.pow')).toBe(false);
    });

    it('does not load all feedback events into JS for scoring', async () => {
      // Arrange: read feedback-core.mjs source and extract buildChunkAggregateAsync body
      const source = await readFile(FEEDBACK_CORE_PATH, 'utf8');
      const body = extractFunctionBody(source, 'buildChunkAggregateAsync');

      // Act + Assert: no "SELECT * FROM feedback_events WHERE chunk_id = ?"
      // pattern that loads all events into JS for client-side aggregation
      expect(
        body.includes('SELECT * FROM feedback_events WHERE chunk_id = ?'),
      ).toBe(false);
    });
  });

  // ---------------------------------------------------------------------------
  // 5. SQL time-decay produces correct results
  // ---------------------------------------------------------------------------
  describe('SQL time-decay produces correct results', () => {
    it('produces boost reflecting POWER(0.95,30) decay not Math.pow(0.5,30/7) for 30-day-old positive event', async () => {
      // Arrange: in-memory schema client with a 30-day-old positive event
      const client = await createSchemaClient();
      try {
        await insertTestFixtures(client);
        const now = Date.now();
        const thirtyDaysMs = 30 * 24 * 60 * 60 * 1000;
        await client.execute({
          sql: `INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength, created_at)
                VALUES (?, ?, 'positive', 1.0, ?)`,
          args: [
            'evt-decay-30d',
            TEST_CHUNK_ID,
            new Date(now - thirtyDaysMs).toISOString(),
          ],
        });

        // Act: recompute scores for the chunk
        const { updateFeedbackScoresAsync } = await import(
          FEEDBACK_CORE_MODULE
        );
        await updateFeedbackScoresAsync(client, TEST_CHUNK_ID, now);

        // Assert: boost reflects SQL POWER(0.95, 30) ≈ 0.2146 → boost ≈ 0.20
        //   NOT JS Math.pow(0.5, 30/7) ≈ 0.055 → boost ≈ 0.05
        //   Threshold 0.15 separates the two formulas with margin.
        const result = await client.execute({
          sql: `SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?`,
          args: [TEST_CHUNK_ID],
        });
        expect(Number(result.rows[0].feedback_boost)).toBeGreaterThan(0.15);
      } finally {
        await client.close();
      }
    });
  });
});