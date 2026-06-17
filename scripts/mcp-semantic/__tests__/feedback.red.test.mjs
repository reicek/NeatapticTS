/**
 * @module feedback.red.test
 * @description Red tests for feedback signal recording, boost computation, time decay,
 * impression decay, and privacy constraints.
 */

import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modulePath = '../tools/feedback-core.mjs';

async function readSchema() {
  const schemaPath = path.join(__dirname, '../../semantic-index/schema-v2.sql');
  return readFile(schemaPath, 'utf8');
}

async function setupDb() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'feedback-test-'));
  const dbPath = path.join(tempDir, 'test.sqlite');
  const db = new Database(dbPath);
  db.exec(await readSchema());
  return { db, tempDir };
}

function teardownDb(db, tempDir) {
  db.close();
  return rm(tempDir, { recursive: true, force: true });
}

function insertDocumentAndChunk(database) {
  database
    .prepare(
      `
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES ('test.ts', 'src', 0, 0, 'abc', 0)
  `,
    )
    .run();
  const doc = database.prepare('SELECT doc_id FROM documents').get();
  database
    .prepare(
      `
    INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth)
    VALUES (?, 0, 'test body', 0, 9, 0)
  `,
    )
    .run(doc.doc_id);
  return database.prepare('SELECT chunk_id FROM chunks').get().chunk_id;
}

describe('feedback-core', () => {
  describe('schema', () => {
    it('creates feedback_events table with required columns', async () => {
      const { db, tempDir } = await setupDb();
      try {
        const row = db
          .prepare(
            `
          SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'feedback_events'
        `,
          )
          .get();
        expect(row).toEqual({ name: 'feedback_events' });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('creates feedback_scores table with required columns', async () => {
      const { db, tempDir } = await setupDb();
      try {
        const row = db
          .prepare(
            `
          SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'feedback_scores'
        `,
          )
          .get();
        expect(row).toEqual({ name: 'feedback_scores' });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('cascade deletes feedback_events when referenced chunk is removed', async () => {
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        db.prepare(
          `
          INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength)
          VALUES ('evt1', ?, 'impression', 0.1)
        `,
        ).run(chunkId);
        db.prepare('DELETE FROM chunks WHERE chunk_id = ?').run(chunkId);
        const row = db
          .prepare('SELECT COUNT(*) as count FROM feedback_events')
          .get();
        expect(row).toEqual({ count: 0 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('cascade deletes feedback_scores when referenced chunk is removed', async () => {
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        db.prepare(
          `
          INSERT INTO feedback_scores (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, feedback_boost)
          VALUES (?, 1, 0, 0, 0, 0, 0.1)
        `,
        ).run(chunkId);
        db.prepare('DELETE FROM chunks WHERE chunk_id = ?').run(chunkId);
        const row = db
          .prepare('SELECT COUNT(*) as count FROM feedback_scores')
          .get();
        expect(row).toEqual({ count: 0 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });

  describe('signal recording', () => {
    it('records impression event with pre-computed signal_strength 0.1', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'impression',
          query_hash: 'hash123',
        });
        const row = db
          .prepare(
            "SELECT signal_strength FROM feedback_events WHERE signal_type = 'impression'",
          )
          .get();
        expect(row).toEqual({ signal_strength: 0.1 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('records click event with pre-computed signal_strength 0.3', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'click',
          query_hash: 'hash123',
        });
        const row = db
          .prepare(
            "SELECT signal_strength FROM feedback_events WHERE signal_type = 'click'",
          )
          .get();
        expect(row).toEqual({ signal_strength: 0.3 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('records reference event with pre-computed signal_strength 0.6', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'reference',
          query_hash: 'hash123',
        });
        const row = db
          .prepare(
            "SELECT signal_strength FROM feedback_events WHERE signal_type = 'reference'",
          )
          .get();
        expect(row).toEqual({ signal_strength: 0.6 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('records positive explicit event with pre-computed signal_strength 1.0', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        const row = db
          .prepare(
            "SELECT signal_strength FROM feedback_events WHERE signal_type = 'positive'",
          )
          .get();
        expect(row).toEqual({ signal_strength: 1.0 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('records negative explicit event with pre-computed signal_strength -1.0', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'negative',
          query_hash: 'hash123',
        });
        const row = db
          .prepare(
            "SELECT signal_strength FROM feedback_events WHERE signal_type = 'negative'",
          )
          .get();
        expect(row).toEqual({ signal_strength: -1.0 });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects an unknown signal_type', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        expect(() =>
          recordFeedbackEvent(db, {
            chunk_id: chunkId,
            signal_type: 'bogus',
            query_hash: 'hash123',
          }),
        ).toThrow('Unknown signal_type');
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('treats a 64-hex query value as an already-hashed query', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        const preHashed = 'A'.repeat(64);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'impression',
          query_hash: preHashed,
        });
        const row = db
          .prepare(
            "SELECT query_hash FROM feedback_events WHERE signal_type = 'impression'",
          )
          .get();
        expect(row.query_hash).toBe(preHashed.toLowerCase());
      } finally {
        await teardownDb(db, tempDir);
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

    it('applies exponential time decay with 7-day half-life during recompute', async () => {
      const { recordFeedbackEvent, recomputeAllFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        const sevenDaysAgo = new Date(
          Date.now() - 7 * 24 * 60 * 60 * 1000,
        ).toISOString();
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash',
          created_at: sevenDaysAgo,
        });
        await recomputeAllFeedbackScores(db);
        const row = db
          .prepare(
            'SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?',
          )
          .get(chunkId);
        // After 7 days decay = 0.5; total_positive_decayed = 0.5; boost = 0.5 * tanh(1.0) ≈ 0.381
        expect(row.feedback_boost).toBeCloseTo(0.5 * Math.tanh(1.0), 2);
      } finally {
        await teardownDb(db, tempDir);
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

    it('applies impression decay during full recompute for low-CTR chunks', async () => {
      const { recordFeedbackEvent, recomputeAllFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        for (let i = 0; i < 20; i++) {
          await recordFeedbackEvent(db, {
            chunk_id: chunkId,
            signal_type: 'impression',
            query_hash: `hash${i}`,
          });
        }
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'click',
          query_hash: 'hash0',
        });
        await recomputeAllFeedbackScores(db);
        const row = db
          .prepare(
            'SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?',
          )
          .get(chunkId);
        // 20 impressions, 1 click => CTR = 0.05 < 0.1 => negative decay applied
        expect(row.feedback_boost).toBeLessThan(0);
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });

  describe('privacy constraints', () => {
    it('truncates context field to a maximum of 500 characters', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        const longContext = 'x'.repeat(1000);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          context: longContext,
        });
        const row = db
          .prepare(
            "SELECT context FROM feedback_events WHERE signal_type = 'positive'",
          )
          .get();
        expect(row.context.length).toBeLessThanOrEqual(500);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('stores query as SHA-256 hash instead of plaintext', async () => {
      const { recordFeedbackEvent } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        const query = 'NEAT crossover';
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'impression',
          query_hash: query,
        });
        const row = db
          .prepare(
            "SELECT query_hash FROM feedback_events WHERE signal_type = 'impression'",
          )
          .get();
        const expectedHash = createHash('sha256').update(query).digest('hex');
        expect(row.query_hash).toBe(expectedHash);
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });

  describe('updateFeedbackScores', () => {
    it('updates scores with negative events', async () => {
      const { recordFeedbackEvent, updateFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'negative',
          query_hash: 'hash123',
        });
        const nowMs = Date.now();
        const score = await updateFeedbackScores(db, chunkId, nowMs);
        expect(score.total_negative).toBeGreaterThan(0);
        expect(score.feedback_boost).toBeLessThan(0);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('returns null when updating scores for a chunk with no events', async () => {
      const { updateFeedbackScores } = await import(modulePath);
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        const score = await updateFeedbackScores(db, chunkId, Date.now());
        expect(score).toBeNull();
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('applies time decay from numeric created_at timestamps', async () => {
      const { recordFeedbackEvent, updateFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        const nowMs = Date.now();
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
          created_at: nowMs - 8 * 24 * 60 * 60 * 1000,
        });
        const score = await updateFeedbackScores(db, chunkId, nowMs);
        expect(score.total_positive).toBeGreaterThan(0);
        expect(score.total_positive).toBeLessThan(1);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('falls back to 0 for unparseable created_at strings', async () => {
      const { recordFeedbackEvent, updateFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
          created_at: 'not-a-date',
        });
        const score = await updateFeedbackScores(db, chunkId, Date.now());
        expect(score.total_positive).toBe(0);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('uses the current time when now is omitted', async () => {
      const { recordFeedbackEvent, updateFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        const score = await updateFeedbackScores(db, chunkId);
        expect(score.total_positive).toBeGreaterThan(0);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('recomputes all scores using the current time when now is omitted', async () => {
      const { recordFeedbackEvent, recomputeAllFeedbackScores } = await import(
        modulePath
      );
      const { db, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db);
        await recordFeedbackEvent(db, {
          chunk_id: chunkId,
          signal_type: 'positive',
          query_hash: 'hash123',
        });
        const count = await recomputeAllFeedbackScores(db);
        expect(count).toBe(1);
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });
});
