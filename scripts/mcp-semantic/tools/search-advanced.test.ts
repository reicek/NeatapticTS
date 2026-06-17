/**
 * @module search-advanced.test
 * @description Red tests for the advanced search pipeline reranker integration.
 *
 * The advanced search pipeline is supposed to make `hybrid_rerank` measurably
 * different from `hybrid`. Right now rerank is effectively a no-op because the
 * reranker is never applied when the dense index degrades to BM25-only results.
 * These tests capture the desired contract before the implementation changes.
 *
 * Uses the same `runModuleEvaluation` pattern as `search-corpus.test.ts` so
 * .mjs modules can be exercised from a .ts test file.
 */
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs';

const REPO_ROOT = path.resolve();

interface RerankProbeResult {
  rerankerCalled: boolean;
  rerankState: string;
  useRerank: boolean;
  topIds: number[];
  topScores: number[];
}

interface RankingExplanationResult {
  allHaveExplanations: boolean;
  firstHasRequiredFields: boolean;
  firstReason: string | null;
}

interface GeneratedReadmeFilterResult {
  hasGeneratedReadmeResults: boolean;
  resultCount: number;
}

interface StructuredFallbackResult {
  hasFallbackField: boolean;
  fallbackResultCount: number;
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
 * Build a minimal corpus SQLite fixture with a few BM25-searchable chunks.
 *
 * The fixture is created in a temporary directory and deleted after each
 * evaluation. It contains only the tables/columns required by `runBm25Search`.
 */
function makeCorpusFixture(): { databasePath: string; tempDir: string } {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-advanced-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new (require('better-sqlite3'))(databasePath);
  try {
    db.exec(`
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
      INSERT INTO documents (doc_id, doc_family, file_path, title)
        VALUES (1, 'readme', 'README.md', 'Readme');
      INSERT INTO chunks (
        chunk_id, doc_id, body_text, char_end, context_header
      ) VALUES
        (10, 1, 'neural network activation function for Neataptic', 48, 'activation'),
        (11, 1, 'neural network training guide and crossover details', 51, 'training'),
        (12, 1, 'penguin swimming in cold water facts', 38, 'penguin');
      INSERT INTO chunks_fts (rowid, body_text) VALUES
        (10, 'neural network activation function for Neataptic'),
        (11, 'neural network training guide and crossover details'),
        (12, 'penguin swimming in cold water facts');
    `);
  } finally {
    db.close();
  }
  return { databasePath, tempDir };
}

describe('search-advanced reranker integration', () => {
  describe('reranker should change results when requested', () => {
    it('applies the reranker to BM25 candidates when dense index is degraded', () => {
      const { databasePath, tempDir } = makeCorpusFixture();
      try {
        const result = runModuleEvaluation<RerankProbeResult>(`
          import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          let rerankerCalled = false;

          const rerankerFn = async (_query, candidates) => {
            rerankerCalled = true;
            // Intentionally reorder by descending body_text length so the
            // reranked top-k is observably different from BM25 ordering.
            return candidates
              .toSorted((a, b) => b.body_text.length - a.body_text.length)
              .map((candidate) => ({
                ...candidate,
                rerank_score: 0.99,
              }));
          };

          const response = await searchCorpus({
            query: 'neural network',
            limit: 3,
            use_dense: true,
            use_rerank: true,
            alpha: 0.5,
            databasePath,
            readinessProbe: async () => ({
              state: 'cold',
              ready: false,
              reason: 'forced cold for red test',
            }),
            rerankerFn,
            rerankerReadinessProbe: async () => ({
              state: 'warm',
              ready: true,
              reason: 'forced warm for red test',
              model_id: 'mock-reranker',
              max_sequence_length: 512,
            }),
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            rerankerCalled,
            rerankState: response.rerank_state,
            useRerank: response.use_rerank,
            topIds: response.results?.map((r) => r.chunk_id) ?? [],
            topScores: response.results?.map((r) => r.rerank_score ?? r.score) ?? [],
          }));
        `);

        expect(result.rerankerCalled).toBe(true);
      } finally {
        // Defensive cleanup if the child process failed before removing the dir.
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('reports reranker candidates count and scores when rerank is active', () => {
      const { databasePath, tempDir } = makeCorpusFixture();
      try {
        const result = runModuleEvaluation<{
          hasRerankCandidatesCount: boolean;
          hasRerankScores: boolean;
          useRerank: boolean;
          rerankState: string;
        }>(`
          import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};

          const response = await searchCorpus({
            query: 'neural network',
            limit: 3,
            use_dense: true,
            use_rerank: true,
            alpha: 0.5,
            databasePath,
            readinessProbe: async () => ({
              state: 'cold',
              ready: false,
              reason: 'forced cold for red test',
            }),
            rerankerFn: async (_query, candidates) =>
              candidates.map((c) => ({ ...c, rerank_score: 0.75 })),
            rerankerReadinessProbe: async () => ({
              state: 'warm',
              ready: true,
              reason: 'forced warm for red test',
              model_id: 'mock-reranker',
              max_sequence_length: 512,
            }),
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            hasRerankCandidatesCount: typeof response.rerank_candidates_count === 'number',
            hasRerankScores: response.results?.every((r) => typeof r.rerank_score === 'number'),
            useRerank: response.use_rerank,
            rerankState: response.rerank_state,
          }));
        `);

        expect(result.hasRerankCandidatesCount && result.hasRerankScores).toBe(
          true,
        );
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore.
        }
      }
    });
  });
});

/**
 * Build a mixed-family corpus fixture with a generated README doc and a
 * TypeScript source doc. Used to test README suppression and fallback
 * behavior in the advanced search pipeline.
 */
function makeMixedCorpusFixture(): { databasePath: string; tempDir: string } {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-advanced-mixed-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new (require('better-sqlite3'))(databasePath);
  try {
    db.exec(`
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
    INSERT INTO documents (doc_id, doc_family, file_path, title)
      VALUES
        (1, 'readme', 'README.md', 'Project overview'),
        (2, 'ts-source', 'src/network.ts', 'Network source');
    INSERT INTO chunks (
      chunk_id, doc_id, body_text, char_end, context_header
    ) VALUES
      (10, 1, 'Neataptic neural network overview guide for onboarding', 54, 'overview'),
      (11, 2, 'neural network activation function source code', 46, 'activation'),
      (12, 2, 'neural network training crossover source code', 46, 'training');
    INSERT INTO chunks_fts (rowid, body_text) VALUES
      (10, 'Neataptic neural network overview guide for onboarding'),
      (11, 'neural network activation function source code'),
      (12, 'neural network training crossover source code');
  `);
  } finally {
    db.close();
  }
  return { databasePath, tempDir };
}

describe('search-advanced README suppression', () => {
  describe('include_code_only option suppresses generated README chunks', () => {
    it('excludes readme family results when include_code_only is true', () => {
      const { databasePath, tempDir } = makeMixedCorpusFixture();
      try {
        const result = runModuleEvaluation<{
          hasReadmeResults: boolean;
          resultCount: number;
        }>(`
        import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
        import fs from 'node:fs';

        const databasePath = ${JSON.stringify(databasePath)};

        const response = await searchAdvanced({
          query: 'neural network',
          query_class: 'simple_lookup',
          include_code_only: true,
          use_dense: false,
          limit: 10,
          databasePath,
        });

        fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

        console.log(JSON.stringify({
          hasReadmeResults: response.results?.some((r) => r.family === 'readme'),
          resultCount: response.results?.length ?? 0,
        }));
      `);

        expect(result.hasReadmeResults).toBe(false);
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

describe('search-advanced native fallback', () => {
  describe('auto_fallback option triggers when main results are empty', () => {
    it('sets fallback_triggered when auto_fallback is true and no results match', () => {
      const { databasePath, tempDir } = makeMixedCorpusFixture();
      try {
        const result = runModuleEvaluation<{
          fallbackTriggered: boolean;
          mainResultCount: number;
        }>(`
        import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
        import fs from 'node:fs';

        const databasePath = ${JSON.stringify(databasePath)};

        const response = await searchAdvanced({
          query: 'xyznotfound',
          query_class: 'simple_lookup',
          auto_fallback: true,
          use_dense: false,
          limit: 10,
          databasePath,
        });

        fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

        console.log(JSON.stringify({
          fallbackTriggered: response.fallback_triggered === true,
          mainResultCount: response.results?.length ?? 0,
        }));
      `);

        expect(result.fallbackTriggered).toBe(true);
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
 * Build a corpus fixture where the README family is named `generated-readme`,
 * matching the real corpus convention for auto-generated README documents.
 * Used to verify that `include_code_only` suppresses generated READMEs
 * regardless of the exact family string.
 */
function makeGeneratedReadmeFixture(): {
  databasePath: string;
  tempDir: string;
} {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-advanced-generated-readme-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new (require('better-sqlite3'))(databasePath);
  try {
    db.exec(`
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
      INSERT INTO documents (doc_id, doc_family, file_path, title)
        VALUES
          (1, 'generated-readme', 'README.md', 'Project overview'),
          (2, 'ts-source', 'src/network.ts', 'Network source');
      INSERT INTO chunks (
        chunk_id, doc_id, body_text, char_end, context_header
      ) VALUES
        (10, 1, 'Neataptic neural network overview guide for onboarding', 54, 'overview'),
        (11, 2, 'neural network activation function source code', 46, 'activation'),
        (12, 2, 'neural network training crossover source code', 46, 'training');
      INSERT INTO chunks_fts (rowid, body_text) VALUES
        (10, 'Neataptic neural network overview guide for onboarding'),
        (11, 'neural network activation function source code'),
        (12, 'neural network training crossover source code');
    `);
  } finally {
    db.close();
  }
  return { databasePath, tempDir };
}

describe('search-advanced ranking explanation', () => {
  describe('explain_ranking option', () => {
    it('attaches a ranking_explanation object to every result', () => {
      const { databasePath, tempDir } = makeCorpusFixture();
      try {
        const result = runModuleEvaluation<RankingExplanationResult>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};

          const response = await searchAdvanced({
            query: 'neural network',
            query_class: 'simple_lookup',
            explain_ranking: true,
            use_dense: false,
            limit: 3,
            databasePath,
          });

          const results = response.results ?? [];
          const allHaveExplanations = results.length > 0 && results.every(
            (r) => typeof r.ranking_explanation === 'object' && r.ranking_explanation !== null,
          );

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            allHaveExplanations,
            firstHasRequiredFields: false,
            firstReason: results[0]?.ranking_explanation?.reason ?? null,
          }));
        `);

        expect(result.allHaveExplanations).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('includes bm25_score, dense_score, rerank_score, final_score and a reason string', () => {
      const { databasePath, tempDir } = makeCorpusFixture();
      try {
        const result = runModuleEvaluation<RankingExplanationResult>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};

          const response = await searchAdvanced({
            query: 'neural network',
            query_class: 'simple_lookup',
            explain_ranking: true,
            use_dense: false,
            limit: 3,
            databasePath,
          });

          const ex = response.results?.[0]?.ranking_explanation ?? {};
          const firstHasRequiredFields =
            typeof ex.bm25_score === 'number' &&
            typeof ex.dense_score === 'number' &&
            typeof ex.rerank_score === 'number' &&
            typeof ex.final_score === 'number' &&
            typeof ex.reason === 'string' &&
            ex.reason.trim().length > 0;

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            allHaveExplanations: false,
            firstHasRequiredFields,
            firstReason: ex.reason ?? null,
          }));
        `);

        expect(result.firstHasRequiredFields).toBe(true);
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

describe('search-advanced generated-readme suppression', () => {
  describe('include_code_only option filters generated-readme family', () => {
    it('excludes generated-readme family results when include_code_only is true', () => {
      const { databasePath, tempDir } = makeGeneratedReadmeFixture();
      try {
        const result = runModuleEvaluation<GeneratedReadmeFilterResult>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};

          const response = await searchAdvanced({
            query: 'neural network',
            query_class: 'simple_lookup',
            include_code_only: true,
            use_dense: false,
            limit: 10,
            databasePath,
          });

          const results = response.results ?? [];
          const hasGeneratedReadmeResults = results.some((r) => r.family === 'generated-readme');

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            hasGeneratedReadmeResults,
            resultCount: results.length,
          }));
        `);

        expect(result.hasGeneratedReadmeResults).toBe(false);
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

describe('search-advanced structured fallback', () => {
  describe('auto_fallback option exposes merged fallback results', () => {
    it('returns a fallback object with triggered=true and merged results', () => {
      const { databasePath, tempDir } = makeGeneratedReadmeFixture();
      try {
        const result = runModuleEvaluation<StructuredFallbackResult>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};

          const response = await searchAdvanced({
            query: 'xyznotfound',
            query_class: 'simple_lookup',
            auto_fallback: true,
            use_dense: false,
            limit: 10,
            databasePath,
          });

          const fallback = response.fallback;
          const hasFallbackField =
            typeof fallback === 'object' &&
            fallback !== null &&
            fallback.triggered === true &&
            Array.isArray(fallback.results) &&
            fallback.results.length > 0;

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            hasFallbackField,
            fallbackResultCount: fallback?.results?.length ?? 0,
          }));
        `);

        expect(result.hasFallbackField).toBe(true);
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
 * Build a corpus fixture with six BM25-searchable chunks so default limit
 * caps are observable. Some chunks contain long body text so compact-mode
 * truncation is also observable.
 */
function makeMultiChunkFixture(): { databasePath: string; tempDir: string } {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-advanced-defaults-red-'),
  );
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = new (require('better-sqlite3'))(databasePath);
  try {
    db.exec(`
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
      INSERT INTO documents (doc_id, doc_family, file_path, title)
        VALUES (1, 'readme', 'README.md', 'Readme');
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

describe('search-advanced conservative defaults and compact mode', () => {
  describe('default result limit', () => {
    it('defaults to at most five results', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ limit: number }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network',
            query_class: 'simple_lookup',
            use_dense: false,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
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
  });

  describe('default rerank candidate count', () => {
    it('uses a conservative rerank candidate count when rerank is requested', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{
          rerankCandidatesCount: number | undefined;
        }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network',
            query_class: 'simple_lookup',
            use_dense: false,
            use_rerank: true,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
          console.log(JSON.stringify({
            rerankCandidatesCount: response.rerank_candidates_count,
          }));
        `);

        expect(
          typeof result.rerankCandidatesCount === 'number' &&
            result.rerankCandidatesCount <= 10,
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

  describe('compact mode', () => {
    it('strips non-essential metadata fields from results', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ allStripped: boolean }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network',
            query_class: 'simple_lookup',
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

    it('truncates result text to the compact threshold', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ maxTextLength: number }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network',
            query_class: 'simple_lookup',
            use_dense: false,
            compact: true,
          });

          const maxTextLength = response.results.reduce(
            (max, result) => Math.max(max, result.text?.length ?? 0),
            0,
          );

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });
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

describe('search-advanced single-call search-and-read', () => {
  describe('read_top_result option', () => {
    it('includes a top_result field when read_top_result is requested', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ hasTopResult: boolean }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network activation',
            query_class: 'simple_lookup',
            use_dense: false,
            read_top_result: true,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            hasTopResult: response.top_result != null,
          }));
        `);

        expect(result.hasTopResult).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('returns the full untruncated top result text inline even in compact mode', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{
          topText: string | null;
          expectedText: string;
        }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import Database from 'better-sqlite3';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network activation',
            query_class: 'simple_lookup',
            use_dense: false,
            compact: true,
            read_top_result: true,
          });

          const db = new Database(databasePath);
          const row = db.prepare('SELECT body_text FROM chunks WHERE chunk_id = 1').get();
          db.close();

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            topText: response.top_result?.text ?? null,
            expectedText: row?.body_text ?? null,
          }));
        `);

        expect(result.topText).toBe(result.expectedText);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });
  });

  describe('follow_up_refs option', () => {
    it('emits a follow_up_refs array', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ hasFollowUpRefs: boolean }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network activation',
            query_class: 'simple_lookup',
            use_dense: false,
          });

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({
            hasFollowUpRefs: Array.isArray(response.follow_up_refs),
          }));
        `);

        expect(result.hasFollowUpRefs).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('includes a load_chunk ref for the top result chunk', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ hasTopLoadChunkRef: boolean }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network activation',
            query_class: 'simple_lookup',
            use_dense: false,
          });

          const topChunkId = response.results?.[0]?.chunk_id ?? null;
          const ref = response.follow_up_refs?.find(
            (r) => r.tool === 'load_chunk' && r.args?.chunk_id === topChunkId,
          );

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({ hasTopLoadChunkRef: ref != null }));
        `);

        expect(result.hasTopLoadChunkRef).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('includes a load_chunk ref for the next sequential chunk', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ hasNextLoadChunkRef: boolean }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network activation',
            query_class: 'simple_lookup',
            use_dense: false,
          });

          const nextChunkId = response.results?.[1]?.chunk_id ?? null;
          const ref = response.follow_up_refs?.find(
            (r) => r.tool === 'load_chunk' && r.args?.chunk_id === nextChunkId,
          );

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({ hasNextLoadChunkRef: ref != null }));
        `);

        expect(result.hasNextLoadChunkRef).toBe(true);
      } finally {
        try {
          fs.rmSync(tempDir, { recursive: true, force: true });
        } catch {
          // Ignore best-effort cleanup failures.
        }
      }
    });

    it('includes a search_advanced ref with a suggested follow-up query', () => {
      const { databasePath, tempDir } = makeMultiChunkFixture();
      try {
        const result = runModuleEvaluation<{ hasSearchAdvancedRef: boolean }>(`
          import { searchAdvanced } from './scripts/mcp-semantic/tools/search-advanced.mjs';
          import fs from 'node:fs';

          const databasePath = ${JSON.stringify(databasePath)};
          const response = await searchAdvanced({
            databasePath,
            query: 'neural network activation',
            query_class: 'simple_lookup',
            use_dense: false,
          });

          const ref = response.follow_up_refs?.find(
            (r) => r.tool === 'search_advanced' && typeof r.args?.query === 'string',
          );

          fs.rmSync(${JSON.stringify(tempDir)}, { recursive: true, force: true });

          console.log(JSON.stringify({ hasSearchAdvancedRef: ref != null }));
        `);

        expect(result.hasSearchAdvancedRef).toBe(true);
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
