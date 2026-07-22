/**
 * @module cortex-db.test
 * @description Red test for A1 schema enrichment read-side contract.
 *
 * The schema migration must add `slice_id`, `step_number`, `phase`, and `status`
 * columns to the `chunks` table. `readChunk` must surface those columns as
 * first-class fields on the returned chunk object.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';

const REPO_ROOT = path.resolve();

interface SliceReadReport {
  slice_id: string | null;
  step_number: number | null;
  phase: string | null;
  status: string | null;
}

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      maxBuffer: 8 * 1024 * 1024,
    },
  );
  const trimmed = output.trim();
  if (trimmed.length === 0) {
    throw new Error('Module evaluation produced empty output');
  }
  return JSON.parse(trimmed) as Result;
};

async function makeSliceMetadataFixture(): Promise<{
  databasePath: string;
  tempDir: string;
}> {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'cortex-db-slice-meta-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = createClient({ url: 'file:' + databasePath });
  try {
    await db.executeMultiple(`
      CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        file_path TEXT NOT NULL UNIQUE,
        doc_family TEXT NOT NULL,
        mtime_ms INTEGER NOT NULL,
        file_size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        indexed_at INTEGER NOT NULL
      );
      CREATE TABLE chunks (
        chunk_id INTEGER PRIMARY KEY,
        doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        heading_path TEXT,
        body_text TEXT NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        parent_chunk_id INTEGER,
        depth INTEGER NOT NULL DEFAULT 0,
        context_header TEXT,
        symbol_name TEXT,
        signature_text TEXT,
        jsdoc_text TEXT,
        export_type TEXT,
        module_path TEXT,
        arch_layer TEXT,
        jsdoc_quality TEXT,
        jsdoc_word_count INTEGER,
        cyclomatic_complexity INTEGER,
        test_coverage TEXT,
        source_path_pattern TEXT,
        slice_id TEXT,
        step_number INTEGER,
        phase TEXT,
        status TEXT,
        UNIQUE(doc_id, chunk_index)
      );
      INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
        VALUES (1, 'plans/example.plans.md', 'plan', 1, 100, 'fake', 1);
      INSERT INTO chunks (chunk_id, doc_id, chunk_index, body_text, char_start, char_end, slice_id, step_number, phase, status)
        VALUES (1, 1, 0, 'Slice body text', 0, 17, 'A1-red-tests', 1, 'A', '[PLANNED]');
    `);
  } finally {
    await db.close();
  }
  return { databasePath, tempDir };
}

describe('cortex-db.mjs', () => {
  describe('readChunk', () => {
    it('returns slice_id, step_number, phase, and status when columns exist', async () => {
      const { databasePath, tempDir } = await makeSliceMetadataFixture();
      try {
        const report = runModuleEvaluation<SliceReadReport>(`
          import { createClient } from '@libsql/client';
          import { readChunk } from './scripts/mcp-semantic/tools/cortex-db.mjs';

          const client = createClient({ url: 'file:' + ${JSON.stringify(databasePath)} });
          const chunk = await readChunk(client, 1);
          await client.close();
          console.log(JSON.stringify({
            slice_id: chunk.slice_id ?? null,
            step_number: chunk.step_number ?? null,
            phase: chunk.phase ?? null,
            status: chunk.status ?? null,
          }));
        `);

        expect(report).toEqual({
          slice_id: 'A1-red-tests',
          step_number: 1,
          phase: 'A',
          status: '[PLANNED]',
        });
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
