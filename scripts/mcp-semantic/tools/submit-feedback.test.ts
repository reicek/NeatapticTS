/**
 * @module submit-feedback.test
 * @description Red tests for feedback normalization and score bounding.
 *
 * The submit-feedback tool currently accepts arbitrary explicit signal strengths
 * and sums multiple positive signals from the same session without
 * normalization. This lets scores inflate and lets callers bypass the
 * designed signal range. These tests pin down the contract that recorded
 * strengths must be clamped and that repeated same-session positive signals
 * must be normalized rather than accumulated.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';
import Database from 'better-sqlite3';

const REPO_ROOT = path.resolve();

interface FeedbackProbeResult {
  recordedStrength: number;
  score: number;
  singleSessionScore: number;
  multiSessionScore: number;
}

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 */
const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );
  const trimmed = output.trim();
  if (trimmed.length === 0) {
    throw new Error('Module evaluation produced empty output');
  }
  return JSON.parse(trimmed) as Result;
};

/**
 * Build a minimal corpus fixture with one document and one chunk.
 *
 * The temporary database is deleted after the evaluation.
 */
function makeFeedbackFixture(): { databasePath: string; tempDir: string } {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'submit-feedback-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new Database(databasePath);
  try {
    db.exec(fs.readFileSync('./scripts/semantic-index/schema-v2.sql', 'utf8'));
    db.exec(`
      INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
        VALUES (1, 'src/network.ts', 'ts-source', 1, 100, 'sha', 1);
      INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth)
        VALUES (42, 1, 0, 'activate', 'fixture body', 0, 12, 0);
    `);
  } finally {
    db.close();
  }
  return { databasePath, tempDir };
}

describe('submit-feedback.mjs normalization and bounding', () => {
  describe('explicit signal strengths must be clamped to the designed range', () => {
    it('does not record an explicit positive strength beyond the designed upper bound', () => {
      const { databasePath, tempDir } = makeFeedbackFixture();
      try {
        const result = runModuleEvaluation<
          Pick<FeedbackProbeResult, 'recordedStrength' | 'score'>
        >(`
          import { submitFeedback } from './scripts/mcp-semantic/tools/submit-feedback.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await submitFeedback({
            chunk_id: 42,
            signal_type: 'positive',
            signal_strength: 50,
            agent_id: 'agent-red',
            query: 'positive bound test',
            databasePath,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            recordedStrength: response.signal_strength,
            score: response.feedback_score,
          }));
        `);

        expect(result.recordedStrength).toBeLessThanOrEqual(1);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('does not record an explicit negative strength beyond the designed lower bound', () => {
      const { databasePath, tempDir } = makeFeedbackFixture();
      try {
        const result = runModuleEvaluation<
          Pick<FeedbackProbeResult, 'recordedStrength' | 'score'>
        >(`
          import { submitFeedback } from './scripts/mcp-semantic/tools/submit-feedback.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await submitFeedback({
            chunk_id: 42,
            signal_type: 'negative',
            signal_strength: -50,
            agent_id: 'agent-red',
            query: 'negative bound test',
            databasePath,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            recordedStrength: response.signal_strength,
            score: response.feedback_score,
          }));
        `);

        expect(result.recordedStrength).toBeGreaterThanOrEqual(-1);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('repeated same-session positive signals must be normalized', () => {
    it('does not inflate the feedback score when the same session submits many positive signals', () => {
      const { databasePath, tempDir } = makeFeedbackFixture();
      try {
        const result = runModuleEvaluation<
          Pick<FeedbackProbeResult, 'singleSessionScore' | 'multiSessionScore'>
        >(`
          import { submitFeedback } from './scripts/mcp-semantic/tools/submit-feedback.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const shared = {
            chunk_id: 42,
            signal_type: 'positive',
            agent_id: 'agent-red',
            query: 'session normalization test',
            databasePath,
          };

          const single = await submitFeedback({ ...shared, query: 'session normalization test single' });

          for (let i = 0; i < 4; i += 1) {
            await submitFeedback({ ...shared, query: 'session normalization test multi' });
          }
          const multi = await submitFeedback({ ...shared, query: 'session normalization test multi' });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            singleSessionScore: single.feedback_score,
            multiSessionScore: multi.feedback_score,
          }));
        `);

        expect(
          Math.abs(result.multiSessionScore - result.singleSessionScore) <
            0.001,
        ).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });
});
