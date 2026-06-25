/**
 * @module batch-index.test
 * @description Red tests for Phase 5 Step 04 — Batch transactions for bulk indexing.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because `client.batch()` is not yet used by build-index.mjs, embed-index.mjs,
 * or feedback-core.mjs — all of which still use sequential `await client.execute()`
 * inside for-loops.
 *
 * Coverage targets:
 * - build-index.mjs: client.batch() for bulk chunk inserts (write mode)
 * - embed-index.mjs: client.batch() for bulk embedding updates
 * - feedback-core.mjs: client.batch() for feedback event recording + score upsert
 * - submit-feedback.mjs: batch transaction for impression + score update
 * - Batch size configurable (default 1000 statements per batch)
 * - Full rollback on batch failure (atomic)
 * - Old sequential insert code fully removed (no dual-path code)
 *
 * The guard test (Group 5: atomicity) should PASS because it tests the underlying
 * libSQL `client.batch()` API which already works — it confirms the platform
 * supports what we need before the source code is refactored.
 *
 * Pure .mjs test — runs via Jest ESM project `semantic-index-mjs`.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import {
  createSchemaClient,
  insertTestFixtures,
} from './turso-test-helpers.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BUILD_INDEX_PATH = path.resolve(__dirname, '..', 'build-index.mjs');
const EMBED_INDEX_PATH = path.resolve(__dirname, '..', 'embed-index.mjs');
const SUBMIT_FEEDBACK_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'submit-feedback.mjs',
);
const FEEDBACK_CORE_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'feedback-core.mjs',
);

/**
 * Read a source file as UTF-8 text.
 *
 * @param {string} filePath - Absolute path to the source file.
 * @returns {string} File contents.
 */
function readSource(filePath) {
  return readFileSync(filePath, 'utf8');
}

// ---------------------------------------------------------------------------
// Group 1: build-index.mjs uses batch transactions
// ---------------------------------------------------------------------------

describe('Batch transactions: build-index.mjs', () => {
  it('calls client.batch() for bulk chunk inserts', () => {
    const source = readSource(BUILD_INDEX_PATH);
    expect(source).toMatch(/client\.batch\(/);
  });

  it('uses write batch mode for inserts', () => {
    const source = readSource(BUILD_INDEX_PATH);
    expect(source).toMatch(/['"]write['"]/);
  });

  it('does not use sequential await client.execute for INSERT INTO chunks', () => {
    const source = readSource(BUILD_INDEX_PATH);
    expect(source).not.toMatch(
      /await\s+client\.execute\s*\(\s*\{\s*sql:\s*[`'"]INSERT\s+INTO\s+chunks/i,
    );
  });

  it('does not use sequential await client.execute for DELETE FROM chunks in a loop', () => {
    const source = readSource(BUILD_INDEX_PATH);
    expect(source).not.toMatch(
      /await\s+client\.execute\s*\(\s*\{\s*sql:\s*[`'"]DELETE\s+FROM\s+chunks/i,
    );
  });
});

// ---------------------------------------------------------------------------
// Group 2: embed-index.mjs uses batch transactions
// ---------------------------------------------------------------------------

describe('Batch transactions: embed-index.mjs', () => {
  it('calls client.batch() for bulk embedding updates', () => {
    const source = readSource(EMBED_INDEX_PATH);
    expect(source).toMatch(/client\.batch\(/);
  });

  it('does not use sequential await client.execute for UPDATE chunks SET embedding', () => {
    const source = readSource(EMBED_INDEX_PATH);
    expect(source).not.toMatch(
      /await\s+client\.execute\s*\(\s*\{\s*sql:\s*[`'"]UPDATE\s+chunks\s+SET\s+embedding/i,
    );
  });
});

// ---------------------------------------------------------------------------
// Group 3: feedback-core.mjs uses batch transactions
// ---------------------------------------------------------------------------

describe('Batch transactions: feedback-core.mjs', () => {
  it('calls client.batch() for feedback event recording', () => {
    const source = readSource(FEEDBACK_CORE_PATH);
    expect(source).toMatch(/client\.batch\(/);
  });

  it('does not use sequential await client.execute for INSERT INTO feedback_events', () => {
    const source = readSource(FEEDBACK_CORE_PATH);
    expect(source).not.toMatch(
      /await\s+client\.execute\s*\(\s*\{\s*sql:\s*[`'"]INSERT\s+INTO\s+feedback_events/i,
    );
  });

  it('does not use sequential await upsertFeedbackScoreAsync inside a for loop in recomputeAllFeedbackScores', () => {
    const source = readSource(FEEDBACK_CORE_PATH);
    // The recomputeAllFeedbackScores function should use batch, not a for-loop
    // with individual upsertFeedbackScoreAsync calls.
    expect(source).not.toMatch(
      /for\s*\(\s*(?:const|let)\s+\w+\s+of\s+result\.rows[^}]*await\s+upsertFeedbackScoreAsync/s,
    );
  });
});

// ---------------------------------------------------------------------------
// Group 4: submit-feedback.mjs uses batch transactions
// ---------------------------------------------------------------------------

describe('Batch transactions: submit-feedback.mjs', () => {
  it('does not use sequential individual executes for event recording and score update', () => {
    const source = readSource(SUBMIT_FEEDBACK_PATH);
    // After batch implementation, submit-feedback.mjs should use a single
    // batch transaction rather than individual recordFeedbackEventAsync +
    // updateFeedbackScoresAsync + SELECT COUNT calls.
    expect(source).not.toMatch(
      /await\s+recordFeedbackEventAsync\(client,\s*\{/s,
    );
  });
});

// ---------------------------------------------------------------------------
// Group 5: Batch size configuration
// ---------------------------------------------------------------------------

describe('Batch transactions: batch size configuration', () => {
  it('default batch size is 1000 statements in build-index.mjs', () => {
    const source = readSource(BUILD_INDEX_PATH);
    expect(source).toMatch(/(?:BATCH_SIZE|batchSize|batch_size)\s*[=:]\s*1000/);
  });
});

// ---------------------------------------------------------------------------
// Group 6: Batch atomicity (guard test — should PASS)
// ---------------------------------------------------------------------------

describe('Batch transactions: atomicity (guard)', () => {
  it('client.batch rolls back all statements when one fails', async () => {
    const client = await createSchemaClient();
    await insertTestFixtures(client);

    // Build a batch with one valid INSERT and one invalid statement
    // (nonexistent table) to trigger a batch-level failure.
    const statements = [
      {
        sql: `INSERT INTO chunks (doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth, context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path, arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          600001,
          1,
          'BatchTest',
          'batch rollback guard',
          0,
          20,
          null,
          0,
          null,
          null,
          null,
          null,
          null,
          null,
          'network',
          null,
          0,
          0,
          null,
          null,
        ],
      },
      {
        sql: 'INSERT INTO nonexistent_table VALUES (1)',
        args: [],
      },
    ];

    // The batch should fail atomically — neither statement takes effect.
    // Catch the error so the single expect can verify the rollback.
    try {
      await client.batch(statements, 'write');
    } catch {
      // Expected: batch fails atomically.
    }

    // Verify the valid INSERT did NOT take effect (rolled back).
    const result = await client.execute({
      sql: 'SELECT COUNT(*) AS count FROM chunks WHERE heading_path = ?',
      args: ['BatchTest'],
    });
    expect(Number(result.rows[0].count)).toBe(0);
  });
});
