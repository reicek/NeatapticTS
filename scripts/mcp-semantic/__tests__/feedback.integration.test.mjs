/**
 * @module feedback.integration.test
 * @description Integration tests for automatic feedback signal collection and
 * the explicit submit_feedback MCP tool.
 */

import { mkdtemp, readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function readSchema() {
  const schemaPath = path.join(__dirname, '../../semantic-index/schema-v2.sql');
  return readFile(schemaPath, 'utf8');
}

async function setupDb() {
  const tempDir = await mkdtemp(
    path.join(process.cwd(), 'tmp-feedback-integration-'),
  );
  const dbPath = path.join(tempDir, 'test.sqlite');
  const db = new Database(dbPath);
  db.exec(await readSchema());
  return { db, dbPath, tempDir };
}

async function teardownDb(db, tempDir) {
  db.close();
  return rm(tempDir, { recursive: true, force: true });
}

function insertDocumentAndChunk(database, bodyText, filePath = 'test.ts') {
  database
    .prepare(
      `
      INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
      VALUES (?, 'src', 0, 0, 'abc', 0)
    `,
    )
    .run(filePath);

  const doc = database
    .prepare('SELECT doc_id FROM documents WHERE file_path = ?')
    .get(filePath);

  database
    .prepare(
      `
      INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth)
      VALUES (?, 0, ?, 0, ?, 0)
    `,
    )
    .run(doc.doc_id, bodyText, bodyText.length);

  return database
    .prepare(
      'SELECT chunk_id FROM chunks WHERE doc_id = ? AND chunk_index = 0',
    )
    .get(doc.doc_id).chunk_id;
}

describe('feedback integration', () => {
  describe('search_corpus impressions', () => {
    it('records impression events for every returned chunk', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'alpha beta gamma',
          'search-test.ts',
        );
        const { searchCorpus } = await import('../tools/search-corpus.mjs');

        const response = await searchCorpus({
          query: 'alpha',
          use_dense: false,
          databasePath: dbPath,
        });

        expect(response.results.length).toBeGreaterThan(0);
        const row = db
          .prepare(
            "SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = ? AND signal_type = 'impression'",
          )
          .get(chunkId);
        expect(row.count).toBe(1);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

  });

  describe('load_chunk clicks', () => {
    it('records a click event with query correlation', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'delta epsilon zeta',
          'load-test.ts',
        );
        const { loadChunk } = await import('../tools/load-chunk.mjs');

        const result = await loadChunk({
          chunk_id: chunkId,
          query: 'delta query',
          databasePath: dbPath,
        });

        expect(result.chunk.chunk_id).toBe(chunkId);
        const row = db
          .prepare(
            "SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = ? AND signal_type = 'click'",
          )
          .get(chunkId);
        expect(row.count).toBe(1);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('deduplicates clicks for the same chunk+query pair', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'eta theta iota',
          'dedupe-test.ts',
        );
        const { loadChunk } = await import('../tools/load-chunk.mjs');

        await loadChunk({
          chunk_id: chunkId,
          query: 'dedupe query',
          databasePath: dbPath,
        });
        await loadChunk({
          chunk_id: chunkId,
          query: 'dedupe query',
          databasePath: dbPath,
        });

        const row = db
          .prepare(
            "SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = ? AND signal_type = 'click'",
          )
          .get(chunkId);
        expect(row.count).toBe(1);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('evicts oldest click cache entry after more than CLICK_CACHE_SIZE distinct pairs', async () => {
      const { createHash } = await import('node:crypto');
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'cache eviction body',
          'cache-test.ts',
        );
        const { loadChunk } = await import('../tools/load-chunk.mjs');
        const originalQuery = 'cache-original';

        await loadChunk({ chunk_id: chunkId, query: originalQuery, databasePath: dbPath });
        // Fill cache with distinct chunk+query pairs for the same chunk.
        for (let i = 1; i <= 60; i++) {
          await loadChunk({ chunk_id: chunkId, query: `cache-filler-${i}`, databasePath: dbPath });
        }
        await new Promise((resolve) => setTimeout(resolve, 50));
        // The original pair should have been evicted, so this reload records a new click.
        await loadChunk({ chunk_id: chunkId, query: originalQuery, databasePath: dbPath });
        await new Promise((resolve) => setTimeout(resolve, 50));

        const originalHash = createHash('sha256').update(originalQuery).digest('hex');
        const row = db
          .prepare(
            "SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = ? AND signal_type = 'click' AND query_hash = ?",
          )
          .get(chunkId, originalHash);
        expect(row.count).toBe(2);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects a non-positive chunk_id', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const { loadChunk } = await import('../tools/load-chunk.mjs');
        await expect(
          loadChunk({ chunk_id: 0, query: 'invalid', databasePath: dbPath }),
        ).rejects.toThrow('chunk_id must be a positive integer');
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects a missing chunk_id when not falling back', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const { loadChunk } = await import('../tools/load-chunk.mjs');
        await expect(
          loadChunk({ chunk_id: 99, databasePath: dbPath }),
        ).rejects.toThrow('Chunk not found');
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects loadChunk without arguments', async () => {
      const { loadChunk } = await import('../tools/load-chunk.mjs');
      await expect(loadChunk()).rejects.toThrow('chunk_id must be a positive integer');
    });

    it('falls back to the lowest-ID chunk when chunk_id is 1 and no exact row exists', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const firstChunkId = insertDocumentAndChunk(db, 'first body', 'fallback-first.ts');
        const secondChunkId = insertDocumentAndChunk(db, 'second body', 'fallback-second.ts');
        db.prepare('DELETE FROM chunks WHERE chunk_id = ?').run(firstChunkId);
        const { loadChunk } = await import('../tools/load-chunk.mjs');

        const result = await loadChunk({ chunk_id: 1, databasePath: dbPath });

        expect(result.chunk.chunk_id).toBe(1);
        expect(result.chunk.text).toBe('second body');
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('records a click event from a pre-hashed query', async () => {
      const { createHash } = await import('node:crypto');
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db, 'hash query body', 'hash-query-test.ts');
        const { loadChunk } = await import('../tools/load-chunk.mjs');
        const queryHash = createHash('sha256').update('pre-hashed-query').digest('hex');

        await loadChunk({ chunk_id: chunkId, query_hash: queryHash, databasePath: dbPath });

        const row = db
          .prepare(
            "SELECT COUNT(*) as count FROM feedback_events WHERE chunk_id = ? AND signal_type = 'click' AND query_hash = ?",
          )
          .get(chunkId, queryHash);
        expect(row.count).toBe(1);
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });

  describe('submit_feedback tool', () => {
    it('returns the expected summary and updates feedback_scores', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'kappa lambda mu',
          'submit-test.ts',
        );
        const { submitFeedback } = await import(
          '../tools/submit-feedback.mjs'
        );

        const summary = await submitFeedback({
          chunk_id: chunkId,
          signal_type: 'reference',
          query: 'test query',
          context: 'useful reference',
          agent_id: 'agent-1',
          databasePath: dbPath,
        });

        expect(summary).toMatchObject({
          chunk_id: chunkId,
          signal_type: 'reference',
          recorded: true,
        });
        expect(typeof summary.feedback_boost_after).toBe('number');

        const score = db
          .prepare('SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?')
          .get(chunkId);
        expect(score.feedback_boost).toBe(summary.feedback_boost_after);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects invalid signal_type values', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(db, 'nu xi omicron', 'invalid-test.ts');
        const { submitFeedback } = await import(
          '../tools/submit-feedback.mjs'
        );

        await expect(
          submitFeedback({
            chunk_id: chunkId,
            signal_type: 'click',
            databasePath: dbPath,
          }),
        ).rejects.toThrow('signal_type');
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects a non-positive chunk_id', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const { submitFeedback } = await import(
          '../tools/submit-feedback.mjs'
        );

        await expect(
          submitFeedback({
            chunk_id: -1,
            signal_type: 'reference',
            databasePath: dbPath,
          }),
        ).rejects.toThrow('chunk_id must be a positive integer');
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('rejects submitFeedback without arguments', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      await expect(submitFeedback()).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });
  });

  describe('index_stats feedback extension', () => {
    it('returns feedback statistics when feedback data exists', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'pi rho sigma',
          'stats-test.ts',
        );
        db.prepare(
          `
          INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength)
          VALUES ('evt1', ?, 'impression', 0.1),
                 ('evt2', ?, 'click', 0.3),
                 ('evt3', ?, 'positive', 1.0)
        `,
        ).run(chunkId, chunkId, chunkId);
        db.prepare(
          `
          INSERT INTO feedback_scores
            (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, last_feedback_at, feedback_boost)
          VALUES (?, 1, 0, 1, 1, 0, '2026-06-14T20:00:00.000Z', 0.25)
        `,
        ).run(chunkId);

        const { indexStats } = await import('../tools/index-stats.mjs');
        const stats = await indexStats({ databasePath: dbPath });

        expect(stats.feedback_stats).toEqual({
          total_events: 3,
          events_by_type: {
            click: 1,
            impression: 1,
            positive: 1,
          },
          chunks_with_feedback: 1,
          average_feedback_boost: 0.25,
          feedback_weight: 1.0,
          feedback_half_life_days: 7,
          last_recomputed_at: '2026-06-14T20:00:00.000Z',
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('returns zero and null feedback statistics when no feedback data exists', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        insertDocumentAndChunk(db, 'tau upsilon phi', 'empty-stats-test.ts');

        const { indexStats } = await import('../tools/index-stats.mjs');
        const stats = await indexStats({ databasePath: dbPath });

        expect(stats.feedback_stats).toEqual({
          total_events: 0,
          events_by_type: {},
          chunks_with_feedback: 0,
          average_feedback_boost: null,
          feedback_weight: 1.0,
          feedback_half_life_days: 7,
          last_recomputed_at: null,
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('formats a positive last indexed timestamp as an ISO string', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        db.prepare(
          `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
           VALUES ('dated.ts', 'src', 0, 0, 'abc', ?)`,
        ).run(Date.now());
        insertDocumentAndChunk(db, 'timestamp body', 'timestamp-chunk.ts');

        const { indexStats } = await import('../tools/index-stats.mjs');
        const stats = await indexStats({ databasePath: dbPath });

        expect(stats.last_build_timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('uses the default database path when called without options', async () => {
      const { indexStats } = await import('../tools/index-stats.mjs');
      const previousEnv = process.env.CORTEX_DB_PATH;
      process.env.CORTEX_DB_PATH = path.join(
        process.cwd(),
        'tmp-missing-cortex.sqlite',
      );
      try {
        await expect(indexStats()).rejects.toThrow('Semantic index not found');
      } finally {
        if (previousEnv === undefined) {
          delete process.env.CORTEX_DB_PATH;
        } else {
          process.env.CORTEX_DB_PATH = previousEnv;
        }
      }
    });
  });

  describe('search_corpus feedback extension', () => {
    it('includes feedback_boost and feedback_signals per result', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'chi psi omega search',
          'search-feedback-test.ts',
        );
        db.prepare(
          `
          INSERT INTO feedback_scores
            (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, last_feedback_at, feedback_boost)
          VALUES (?, 2, 1, 5, 2, 1, '2026-06-14T21:00:00.000Z', 0.18)
        `,
        ).run(chunkId);

        const { searchCorpus } = await import('../tools/search-corpus.mjs');
        const response = await searchCorpus({
          query: 'chi psi omega',
          use_dense: false,
          databasePath: dbPath,
        });

        expect(response.results.length).toBeGreaterThan(0);
        const firstResult = response.results[0];
        expect(firstResult.feedback_boost).toBe(0.18);
        expect(firstResult.feedback_signals).toEqual({
          total_positive: 2,
          total_negative: 1,
          total_impressions: 5,
          total_clicks: 2,
          total_references: 1,
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('uses default feedback_boost and zeroed feedback_signals when no score row exists', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        insertDocumentAndChunk(
          db,
          'default feedback search',
          'search-default-test.ts',
        );

        const { searchCorpus } = await import('../tools/search-corpus.mjs');
        const response = await searchCorpus({
          query: 'default feedback',
          use_dense: false,
          databasePath: dbPath,
        });

        expect(response.results.length).toBeGreaterThan(0);
        const firstResult = response.results[0];
        expect(firstResult.feedback_boost).toBe(0);
        expect(firstResult.feedback_signals).toEqual({
          total_positive: 0,
          total_negative: 0,
          total_impressions: 0,
          total_clicks: 0,
          total_references: 0,
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('handles empty result lists without enrichment', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        insertDocumentAndChunk(db, 'unrelated content', 'empty-search-test.ts');

        const { searchCorpus } = await import('../tools/search-corpus.mjs');
        const response = await searchCorpus({
          query: 'query that matches nothing',
          use_dense: false,
          databasePath: dbPath,
        });

        expect(response.results).toEqual([]);
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('falls back to default signals when the feedback_scores table is missing', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'missing feedback table search',
          'missing-table-test.ts',
        );
        db.exec('DROP TABLE feedback_scores');

        const { searchCorpus } = await import('../tools/search-corpus.mjs');
        const response = await searchCorpus({
          query: 'missing feedback table',
          use_dense: false,
          databasePath: dbPath,
        });

        expect(response.results.length).toBeGreaterThan(0);
        const result = response.results.find((r) => r.chunk_id === chunkId);
        expect(result).toBeDefined();
        expect(result.feedback_boost).toBe(0);
        expect(result.feedback_signals).toEqual({
          total_positive: 0,
          total_negative: 0,
          total_impressions: 0,
          total_clicks: 0,
          total_references: 0,
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });

  describe('attachFeedbackToResults', () => {
    it('leaves an empty result list unchanged', async () => {
      const { attachFeedbackToResults } = await import(
        '../tools/search-corpus.mjs'
      );
      const results = [];

      await attachFeedbackToResults(results);

      expect(results).toEqual([]);
    });

    it('applies default signals to results without a numeric chunk_id', async () => {
      const { attachFeedbackToResults } = await import(
        '../tools/search-corpus.mjs'
      );
      const results = [{ text: 'no chunk id' }];

      await attachFeedbackToResults(results);

      expect(results[0].feedback_boost).toBe(0);
      expect(results[0].feedback_signals).toEqual({
        total_positive: 0,
        total_negative: 0,
        total_impressions: 0,
        total_clicks: 0,
        total_references: 0,
      });
    });

    it('enriches numeric results and defaults non-numeric results in the same list', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'mixed feedback enrichment',
          'mixed-enrichment-test.ts',
        );
        db.prepare(
          `
          INSERT INTO feedback_scores
            (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, last_feedback_at, feedback_boost)
          VALUES (?, 4, 2, 10, 3, 2, '2026-06-14T22:00:00.000Z', 0.42)
        `,
        ).run(chunkId);

        const { attachFeedbackToResults } = await import(
          '../tools/search-corpus.mjs'
        );
        const results = [{ chunk_id: chunkId }, { text: 'plain object' }];
        await attachFeedbackToResults(results, dbPath);

        expect(results[0].feedback_boost).toBe(0.42);
        expect(results[0].feedback_signals).toEqual({
          total_positive: 4,
          total_negative: 2,
          total_impressions: 10,
          total_clicks: 3,
          total_references: 2,
        });
        expect(results[1].feedback_boost).toBe(0);
        expect(results[1].feedback_signals).toEqual({
          total_positive: 0,
          total_negative: 0,
          total_impressions: 0,
          total_clicks: 0,
          total_references: 0,
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });

    it('applies default signals when the feedback_scores table is missing', async () => {
      const { db, dbPath, tempDir } = await setupDb();
      try {
        const chunkId = insertDocumentAndChunk(
          db,
          'enrichment missing table',
          'enrichment-missing-table-test.ts',
        );
        db.exec('DROP TABLE feedback_scores');

        const { attachFeedbackToResults } = await import(
          '../tools/search-corpus.mjs'
        );
        const results = [{ chunk_id: chunkId }];
        await attachFeedbackToResults(results, dbPath);

        expect(results[0].feedback_boost).toBe(0);
        expect(results[0].feedback_signals).toEqual({
          total_positive: 0,
          total_negative: 0,
          total_impressions: 0,
          total_clicks: 0,
          total_references: 0,
        });
      } finally {
        await teardownDb(db, tempDir);
      }
    });
  });

  describe('MCP tool registration', () => {
    it('includes submit_feedback in the Repo Cortex tool list', async () => {
      const { createRepoCortexTools } = await import(
        '../repo-cortex-mcp.mjs'
      );
      const tools = createRepoCortexTools();
      const names = tools.map((tool) => tool.name);

      expect(names).toContain('submit_feedback');
    });
  });
});
