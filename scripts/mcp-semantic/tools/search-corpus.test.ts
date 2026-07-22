import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';

// Tests run from the repository root, so cwd is a stable anchor for repo-relative
// paths without needing import.meta.url (which currently breaks ts-jest for the
// mcp-semantic-scripts project).
const REPO_ROOT = path.resolve();

interface ClassificationResult {
  alpha: number;
  family: string | null;
  query_class: string;
  confidence: number;
  classification_fallback: boolean;
}

/**
 * Evaluate a short ESM snippet in a child Node process rooted at the repo root.
 * Used to import `.mjs` implementation modules that do not yet exist in the
 * red phase and to run them against real fixtures.
 *
 * @param source - ESM source string executed as `--input-type=module --eval`.
 * @returns The JSON-parsed value the snippet wrote to stdout.
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

describe('search-corpus.mjs classification-aware routing', () => {
  describe('code identifier detection routes to ts-source', () => {
    it('classifies camelCase identifiers as code_specific with ts-source family', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyForSearchCorpus('findTheNEATSelectionCode')));
      `);
      expect(result).toEqual(
        expect.objectContaining({
          query_class: 'code_specific',
          family: 'ts-source',
        }),
      );
    });

    it('classifies dotted identifiers as code_specific with ts-source family', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyForSearchCorpus('network.activate')));
      `);
      expect(result).toEqual(
        expect.objectContaining({
          query_class: 'code_specific',
          family: 'ts-source',
        }),
      );
    });

    it('classifies snake_case identifiers as code_specific with ts-source family', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyForSearchCorpus('snake_case_function')));
      `);
      expect(result).toEqual(
        expect.objectContaining({
          query_class: 'code_specific',
          family: 'ts-source',
        }),
      );
    });

    it('classifies file-extension hints as code_specific with ts-source family', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyForSearchCorpus('NEAT selection code.ts')));
      `);
      expect(result).toEqual(
        expect.objectContaining({
          query_class: 'code_specific',
          family: 'ts-source',
        }),
      );
    });
  });

  describe('natural language queries avoid code-source bias', () => {
    it('keeps pure natural language queries out of ts-source family', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyForSearchCorpus('how does NEAT work')));
      `);
      expect(result.family).not.toBe('ts-source');
    });

    it('does not regress existing code-keyword classification', () => {
      const result = runModuleEvaluation<ClassificationResult>(`
        import { classifyForSearchCorpus } from './rag-index/classify-query.mjs';
        console.log(JSON.stringify(classifyForSearchCorpus('Network class implementation')));
      `);
      expect(result).toEqual(
        expect.objectContaining({
          query_class: 'code_specific',
          family: 'ts-source',
        }),
      );
    });
  });

  describe('red exact symbol lookup contract', () => {
    it('flags an exact symbol match when the query matches a known symbol_name', async () => {
      const result = runModuleEvaluation<{
        exact_symbol_match?: boolean;
        matchedSymbol?: string | null;
        resultCount: number;
      }>(`
        import { mkdtemp, readFile } from 'node:fs/promises';
        import { tmpdir } from 'node:os';
        import path from 'node:path';
        import { createClient } from '@libsql/client';
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';

        const fixtureDirectory = await mkdtemp(path.join(tmpdir(), 'search-corpus-exact-symbol-'));
        const databasePath = path.join(fixtureDirectory, 'rag-index.sqlite');
        const database = createClient({ url: 'file:' + databasePath });
        const schema = await readFile('./rag-index/schema-turso.sql', 'utf8');
        await database.executeMultiple(schema + \`
          INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
            VALUES (1, 'src/network.ts', 'ts-source', 1, 100, 'fixture-sha', 1);
          INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth, symbol_name)
            VALUES (1, 1, 0, 'activateNetwork', 'Exact symbol lookup fixture body.', 0, 31, 0, 'activateNetwork');
        \`);
        await database.close();

        const response = await searchCorpus({ databasePath, query: 'activateNetwork', use_dense: false });

        console.log(JSON.stringify({
          exact_symbol_match: response.exact_symbol_match,
          matchedSymbol: response.results[0]?.symbol_name ?? null,
          resultCount: response.results.length,
        }));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          exact_symbol_match: true,
          matchedSymbol: 'activateNetwork',
          resultCount: 1,
        }),
      );
    });
  });

  describe('red cold-start latency contract', () => {
    it('reports latency_ms and reduces warm-call latency via caching', () => {
      const result = runModuleEvaluation<{
        denseQueryCalls: number;
        firstLatency?: number;
        latencyPresent: boolean;
        readinessCalls: number;
        secondLatency?: number;
      }>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';

        let denseQueryCalls = 0;
        let readinessCalls = 0;
        const sharedOptions = {
          denseQuery: async () => {
            denseQueryCalls += 1;
            return {
              alpha: 0.5,
              limit: 1,
              query: 'NEAT activation',
              results: [{ chunk_id: 1, score: 0.99, text: 'NEAT activation retrieval contract' }],
              use_dense: true,
            };
          },
          limit: 1,
          query: 'NEAT activation',
          readinessProbe: async () => {
            readinessCalls += 1;
            return { ready: true, state: 'warm', reason: 'All chunks have embeddings.' };
          },
          use_dense: true,
        };

        const first = await searchCorpus(sharedOptions);
        const second = await searchCorpus(sharedOptions);

        console.log(JSON.stringify({
          denseQueryCalls,
          firstLatency: first.latency_ms,
          latencyPresent: typeof first.latency_ms === 'number' && typeof second.latency_ms === 'number',
          readinessCalls,
          secondLatency: second.latency_ms,
        }));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          denseQueryCalls: 2,
          firstLatency: expect.any(Number),
          latencyPresent: true,
          readinessCalls: 1,
          secondLatency: expect.any(Number),
        }),
      );
    });
  });
});

/**
 * Build a corpus fixture with six BM25-searchable chunks so default limit
 * caps are observable. Some chunks contain long body text so compact-mode
 * truncation is also observable.
 */
async function makeMultiChunkFixture(): Promise<{
  databasePath: string;
  tempDir: string;
}> {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-corpus-defaults-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = createClient({ url: 'file:' + databasePath });
  try {
    await db.executeMultiple(`
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
    await db.close();
  }
  return { databasePath, tempDir };
}

describe('search-corpus conservative default response sizes and compact mode', () => {
  describe('default result limit', () => {
    it('defaults to at most five results', async () => {
      const { databasePath, tempDir } = await makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ limit: number }>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
        const databasePath = ${JSON.stringify(databasePath)};
        const response = await searchCorpus({
          databasePath,
          query: 'neural network',
          use_dense: false,
        });
        console.log(JSON.stringify({ limit: response.limit }));
      `);

        expect(result.limit).toBeLessThanOrEqual(5);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('caps the number of returned results to the default limit', async () => {
      const { databasePath, tempDir } = await makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ resultCount: number }>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
        const databasePath = ${JSON.stringify(databasePath)};
        const response = await searchCorpus({
          databasePath,
          query: 'neural network',
          use_dense: false,
        });
        console.log(JSON.stringify({ resultCount: response.results.length }));
      `);

        expect(result.resultCount).toBeLessThanOrEqual(5);
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
    it('strips non-essential metadata fields from results', async () => {
      const { databasePath, tempDir } = await makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ allStripped: boolean }>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
        const databasePath = ${JSON.stringify(databasePath)};
        const response = await searchCorpus({
          databasePath,
          query: 'neural network',
          use_dense: false,
          compact: true,
        });

        const nonEssential = [
          'jsdoc_text', 'signature_text', 'char_start', 'char_end',
          'depth', 'export_type', 'module_path', 'arch_layer',
          'jsdoc_quality', 'jsdoc_word_count', 'cyclomatic_complexity',
          'test_coverage', 'source_path_pattern',
        ];
        const allStripped = response.results.every((result) =>
          nonEssential.every((key) => !(key in result)),
        );
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

    it('truncates result text to the compact threshold', async () => {
      const { databasePath, tempDir } = await makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ maxTextLength: number }>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
        const databasePath = ${JSON.stringify(databasePath)};
        const response = await searchCorpus({
          databasePath,
          query: 'neural network',
          use_dense: false,
          compact: true,
        });

        const maxTextLength = response.results.reduce(
          (max, result) => Math.max(max, result.text?.length ?? 0),
          0,
        );
        console.log(JSON.stringify({ maxTextLength }));
      `);

        expect(result.maxTextLength).toBeLessThanOrEqual(300);
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

/**
 * Red-phase contract for the A2 step_number filter parameter in searchCorpus.
 * The implementation currently ignores the step_number option, so the test
 * observes both matching chunks and fails until the filter is applied.
 */
describe('search-corpus step_number filter parameter', () => {
  it('returns only chunks whose step_number equals the requested step_number', async () => {
    const tempDir = fs.mkdtempSync(
      path.join(os.tmpdir(), 'search-corpus-step-number-red-'),
    );
    const databasePath = path.join(tempDir, 'corpus.sqlite');
    const db = createClient({ url: 'file:' + databasePath });
    try {
      await db.executeMultiple(`
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
          source_path_pattern TEXT,
          slice_id TEXT,
          step_number INTEGER,
          phase TEXT,
          status TEXT
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
          body_text,
          content='chunks',
          content_rowid='chunk_id'
        );
        INSERT INTO documents VALUES (1, 'readme', 'README.md', 'Readme', 1, 1, 'sha', 1);
        INSERT INTO chunks (chunk_id, doc_id, body_text, char_end, slice_id, step_number, phase, status)
          VALUES
            (1, 1, 'step filter fixture body one', 28, 'A1-green', 1, 'A', 'green'),
            (2, 1, 'step filter fixture body two', 27, 'A2-red-tests', 2, 'A', 'red');
        INSERT INTO chunks_fts (rowid, body_text) VALUES
          (1, 'step filter fixture body one'),
          (2, 'step filter fixture body two');
      `);
    } finally {
      await db.close();
    }

    try {
      const result = runModuleEvaluation<{ chunkIds: number[] }>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
        const databasePath = ${JSON.stringify(databasePath)};
        const response = await searchCorpus({
          databasePath,
          query: 'step filter fixture body',
          step_number: 2,
          use_dense: false,
          limit: 10,
        });
        console.log(JSON.stringify({ chunkIds: response.results.map((r) => r.chunk_id) }));
      `);

      expect(result.chunkIds).toEqual([2]);
    } finally {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch {
        // Ignore best-effort cleanup failures.
      }
    }
  });
});
