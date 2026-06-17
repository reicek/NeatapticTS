/**
 * @module search-context.test
 * @description Red tests for conservative default sizes and compact mode in
 * the `search_context` corpus-context assembly pipeline.
 *
 * The `searchContext` tool currently defaults to a 4096-token budget and a
 * 10-result retrieval limit. These tests make the desired conservative
 * defaults (800-1200 token budget, <= 5 retrieved chunks) and compact-mode
 * field filtering/truncation explicit before the implementation changes.
 */
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const REPO_ROOT = path.resolve();

interface SearchContextProbe {
  budget: number;
  totalChunksRetrieved: number;
  resultKeys: string[];
  contextLength: number;
  compactResultKeys: string[];
  compactContextLength: number;
}

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 * Used to import `.mjs` implementation modules that do not yet support the
 * compact option and to run them against real fixtures.
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
 * Build a corpus fixture with six BM25-searchable chunks so default limit
 * caps are observable. Some chunks contain long body text so compact-mode
 * truncation is also observable.
 */
function makeMultiChunkFixture(): { databasePath: string; tempDir: string } {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-context-defaults-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new (require('better-sqlite3'))(databasePath);
  try {
    db.exec(`
      CREATE TABLE documents (
        doc_id INTEGER PRIMARY KEY,
        doc_family TEXT NOT NULL,
        file_path TEXT NOT NULL,
        title TEXT,
        mtime_ms INTEGER,
        file_size INTEGER,
        sha256 TEXT,
        indexed_at INTEGER
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
        module_path TEXT,
        arch_layer TEXT,
        jsdoc_quality TEXT,
        jsdoc_word_count INTEGER,
        cyclomatic_complexity INTEGER,
        test_coverage TEXT,
        source_path_pattern TEXT
      );
      CREATE VIRTUAL TABLE chunks_fts USING fts5(
        body_text,
        content='chunks',
        content_rowid='chunk_id'
      );
      INSERT INTO documents VALUES (1, 'readme', 'README.md', 'Readme', 1, 1, 'sha', 1);
      INSERT INTO chunks (
        chunk_id, doc_id, body_text, char_end, context_header, symbol_name, signature_text, jsdoc_text
      ) VALUES
        (1, 1, 'neural network activation guide part one with a very long body text that should definitely be truncated when compact mode is enabled compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding', 250, 'overview', 'sym1', 'sig1', 'jsdoc one'),
        (2, 1, 'neural network training guide part two with a very long body text that should definitely be truncated when compact mode is enabled compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding', 230, 'training', 'sym2', 'sig2', 'jsdoc two'),
        (3, 1, 'neural network crossover guide part three compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding', 150, 'crossover', 'sym3', 'sig3', 'jsdoc three'),
        (4, 1, 'neural network mutation guide part four compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding', 150, 'mutation', 'sym4', 'sig4', 'jsdoc four'),
        (5, 1, 'neural network selection guide part five compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding', 152, 'selection', 'sym5', 'sig5', 'jsdoc five'),
        (6, 1, 'neural network speciation guide part six compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding', 151, 'speciation', 'sym6', 'sig6', 'jsdoc six');
      INSERT INTO chunks_fts (rowid, body_text) VALUES
        (1, 'neural network activation guide part one with a very long body text that should definitely be truncated when compact mode is enabled compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding'),
        (2, 'neural network training guide part two with a very long body text that should definitely be truncated when compact mode is enabled compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding'),
        (3, 'neural network crossover guide part three compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding'),
        (4, 'neural network mutation guide part four compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding'),
        (5, 'neural network selection guide part five compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding'),
        (6, 'neural network speciation guide part six compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding compact-red-test-padding');
    `);
  } finally {
    db.close();
  }
  return { databasePath, tempDir };
}

describe('search-context conservative defaults and compact mode', () => {
  describe('default response sizes', () => {
    it('defaults to a budget of at most 1200 tokens', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ budget: number }>(`
          import { searchContext } from './scripts/mcp-semantic/tools/search-context.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchContext({
            databasePath,
            query: 'neural network',
            use_dense: false,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
          console.log(JSON.stringify({ budget: response.metadata.budget }));
        `);

        expect(result.budget).toBeLessThanOrEqual(1200);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('retrieves at most five chunks by default', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ totalChunksRetrieved: number }>(`
          import { searchContext } from './scripts/mcp-semantic/tools/search-context.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchContext({
            databasePath,
            query: 'neural network',
            use_dense: false,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
          console.log(JSON.stringify({ totalChunksRetrieved: response.total_chunks_retrieved }));
        `);

        expect(result.totalChunksRetrieved).toBeLessThanOrEqual(5);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('compact mode', () => {
    it('strips non-essential fields from per-chunk results', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ allStripped: boolean }>(`
          import { searchContext } from './scripts/mcp-semantic/tools/search-context.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchContext({
            databasePath,
            query: 'neural network',
            use_dense: false,
            compact: true,
          });

          const nonEssential = ['feedback_boost', 'metadata'];
          const allStripped = response.results.every((result) =>
            nonEssential.every((key) => !(key in result)),
          );

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
          console.log(JSON.stringify({ allStripped }));
        `);

        expect(result.allStripped).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('truncates assembled context text to the compact threshold', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ contextLength: number }>(`
          import { searchContext } from './scripts/mcp-semantic/tools/search-context.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchContext({
            databasePath,
            query: 'neural network',
            use_dense: false,
            compact: true,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
          console.log(JSON.stringify({ contextLength: response.context.length }));
        `);

        expect(result.contextLength).toBeLessThanOrEqual(2000);
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
