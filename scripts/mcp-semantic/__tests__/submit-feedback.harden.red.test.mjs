/**
 * @module submit-feedback.harden.red.test
 * @description Red tests for hardening submit_feedback to the Step 10/21 design spec.
 *
 * The spec requires irrelevant signal type, signal_strength override,
 * aggregate score updates, and structured error taxonomy.
 */

import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
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
    path.join(tmpdir(), 'submit-feedback-harden-test-'),
  );
  const dbPath = path.join(tempDir, 'test.sqlite');
  const db = new Database(dbPath);
  db.exec(await readSchema());
  db.exec(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES ('src/foo.ts', 'ts-source', 0, 100, 'a', 1);
  `);
  const docId = db.prepare('SELECT doc_id FROM documents').get().doc_id;
  db.prepare(
    `
    INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth)
    VALUES (?, 0, 'test body', 0, 9, 0)
  `,
  ).run(docId);
  db.close();
  return { dbPath, tempDir };
}

function teardown(tempDir) {
  return rm(tempDir, { recursive: true, force: true });
}

describe('submit-feedback hardened', () => {
  describe('schema validation', () => {
    it('rejects missing chunk_id with MISSING_CHUNK_ID', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        await expect(
          submitFeedback({ databasePath: dbPath, signal_type: 'positive' }),
        ).rejects.toThrow(/MISSING_CHUNK_ID/);
      } finally {
        await teardown(tempDir);
      }
    });

    it('rejects invalid chunk_id with MISSING_CHUNK_ID', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        await expect(
          submitFeedback({
            databasePath: dbPath,
            chunk_id: 'not-a-number',
            signal_type: 'positive',
          }),
        ).rejects.toThrow(/MISSING_CHUNK_ID/);
      } finally {
        await teardown(tempDir);
      }
    });

    it('rejects invalid signal_type with INVALID_SIGNAL_TYPE', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        await expect(
          submitFeedback({
            databasePath: dbPath,
            chunk_id: chunkId,
            signal_type: 'like',
          }),
        ).rejects.toThrow(/INVALID_SIGNAL_TYPE/);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('signal types', () => {
    it('accepts the irrelevant signal type', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        const result = await submitFeedback({
          databasePath: dbPath,
          chunk_id: chunkId,
          signal_type: 'irrelevant',
        });

        expect(result.signal_type).toBe('irrelevant');
      } finally {
        await teardown(tempDir);
      }
    });

    it('accepts a signal_strength override and stores it', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        const result = await submitFeedback({
          databasePath: dbPath,
          chunk_id: chunkId,
          signal_type: 'positive',
          signal_strength: 2.5,
        });

        const checkDb = new Database(dbPath);
        const event = checkDb
          .prepare(
            'SELECT signal_strength FROM feedback_events WHERE chunk_id = ?',
          )
          .get(chunkId);
        checkDb.close();

        expect(result.signal_strength).toBe(2.5);
        expect(event.signal_strength).toBe(2.5);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('aggregate scoring', () => {
    it('records the event in feedback_events', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        await submitFeedback({
          databasePath: dbPath,
          chunk_id: chunkId,
          signal_type: 'reference',
          query: 'NEAT activation',
          agent_id: 'test-agent',
        });

        const checkDb = new Database(dbPath);
        const row = checkDb
          .prepare('SELECT * FROM feedback_events WHERE chunk_id = ?')
          .get(chunkId);
        checkDb.close();

        expect(row).toEqual(
          expect.objectContaining({
            signal_type: 'reference',
            agent_id: 'test-agent',
          }),
        );
      } finally {
        await teardown(tempDir);
      }
    });

    it('updates feedback_scores aggregate and returns summary fields', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        const result = await submitFeedback({
          databasePath: dbPath,
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

        const checkDb = new Database(dbPath);
        const score = checkDb
          .prepare(
            'SELECT feedback_boost FROM feedback_scores WHERE chunk_id = ?',
          )
          .get(chunkId);
        checkDb.close();

        expect(score.feedback_boost).toBeGreaterThan(0);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('optional fields', () => {
    it('truncates context longer than 500 characters', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        await submitFeedback({
          databasePath: dbPath,
          chunk_id: chunkId,
          signal_type: 'positive',
          context: 'x'.repeat(600),
        });

        const checkDb = new Database(dbPath);
        const row = checkDb
          .prepare('SELECT context FROM feedback_events WHERE chunk_id = ?')
          .get(chunkId);
        checkDb.close();

        expect(row.context.length).toBeLessThanOrEqual(500);
      } finally {
        await teardown(tempDir);
      }
    });

    it('accepts optional agent_id and query', async () => {
      const { submitFeedback } = await import('../tools/submit-feedback.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const db = new Database(dbPath);
        const chunkId = db
          .prepare('SELECT chunk_id FROM chunks')
          .get().chunk_id;
        db.close();

        const result = await submitFeedback({
          databasePath: dbPath,
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
        await teardown(tempDir);
      }
    });
  });
});
