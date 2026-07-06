/**
 * @module load-chunk.test
 * @description Red tests for the single-chunk loader follow-up navigation contract.
 *
 * The follow-up refs feature in search_advanced needs a reliable way to point
 * at the next sequential chunk in the same document. These tests capture the
 * desired contract before load-chunk.mjs is extended.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';
import { createClient } from '@libsql/client';

const REPO_ROOT = path.resolve();

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
 * Build a minimal corpus SQLite fixture with two sequential chunks in one doc.
 */
async function makeSequentialChunkFixture(): Promise<{
  databasePath: string;
  tempDir: string;
}> {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'load-chunk-red-'));
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = createClient({ url: 'file:' + databasePath });
  try {
    await db.executeMultiple(`
      CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        doc_family TEXT NOT NULL,
        file_path TEXT NOT NULL,
        title TEXT
      );
      CREATE TABLE chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL DEFAULT 0,
        heading_path TEXT,
        body_text TEXT NOT NULL,
        char_start INTEGER NOT NULL DEFAULT 0,
        char_end INTEGER NOT NULL DEFAULT 0,
        parent_chunk_id INTEGER,
        depth INTEGER NOT NULL DEFAULT 0,
        context_header TEXT,
        symbol_name TEXT,
        signature_text TEXT,
        jsdoc_text TEXT,
        export_type TEXT,
        module_path TEXT
      );
      INSERT INTO documents (doc_id, doc_family, file_path, title)
        VALUES (1, 'readme', 'README.md', 'Readme');
      INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_end, context_header)
        VALUES
          (10, 1, 0, 'First sequential chunk body.', 28, 'intro'),
          (11, 1, 1, 'Second sequential chunk body.', 29, 'details');
    `);
  } finally {
    await db.close();
  }
  return { databasePath, tempDir };
}

describe('load-chunk.mjs follow-up navigation', () => {
  describe('next_chunk_id field', () => {
    it('includes the next sequential chunk id for follow-up refs', async () => {
      const { databasePath, tempDir } = await makeSequentialChunkFixture();
      try {
        const result = runModuleEvaluation<{
          nextChunkId: number | null;
        }>(`
          import { loadChunk } from './scripts/mcp-semantic/tools/load-chunk.mjs';
          const databasePath = ${JSON.stringify(databasePath)};
          const response = await loadChunk({ chunk_id: 10, databasePath });
          console.log(JSON.stringify({
            nextChunkId: response.chunk?.next_chunk_id ?? null,
          }));
        `);

        expect(result.nextChunkId).toBe(11);
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
