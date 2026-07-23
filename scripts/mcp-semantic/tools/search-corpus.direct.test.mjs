/**
 * @module search-corpus.direct.test
 * @description Direct-import coverage tests for search-corpus.mjs filter parameters.
 *
 * Runs in the mcp-semantic-mjs project so Jest can instrument the native ESM
 * source file via the V8 coverage provider.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import {
  attachFeedbackToResults,
  buildRankingExplanation,
  buildResponseFreshness,
  compactSearchResult,
  createDegradedBm25Response,
  createEmptyBm25Response,
  estimateResponseTokens,
  getDenseReadiness,
  getRerankerReadiness,
  normalizeAlpha,
  normalizeDenseReason,
  normalizeRerankReason,
  recordSearchImpressions,
  resetReadinessCaches,
  resolveChunkCountForStrategy,
  resolveDenseStrategyForSearch,
  runBm25Search,
  runExactSymbolLookup,
  searchCorpus,
  searchCorpusImpl,
  shouldBypassDenseReadinessCache,
  shouldBypassRerankerReadinessCache,
  tryExactSymbolLookup,
} from './search-corpus.mjs';
import { closeTursoClient } from './cortex-db.mjs';

const CREATED_URLS = new Set();
const TEMP_DIRS = [];

function trackTempDir() {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-corpus-direct-'),
  );
  TEMP_DIRS.push(tempDir);
  return tempDir;
}

async function createFilteredFixture() {
  const tempDir = trackTempDir();
  const databasePath = path.join(tempDir, 'corpus.sqlite');
  const db = createClient({ url: `file:${databasePath}` });
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
      INSERT INTO chunks (chunk_id, doc_id, body_text, char_end, slice_id, step_number, phase, status, symbol_name)
        VALUES
          (1, 1, 'slice filter fixture body alpha', 31, 'A2-red-tests', 2, 'A', 'red', NULL),
          (2, 1, 'slice filter fixture body beta', 30, 'A1-green', 1, 'A', 'green', NULL),
          (3, 1, 'exact symbol fixture gamma', 26, 'A2-red-tests', 2, 'A', 'red', 'fixtureSymbol'),
          (4, 1, '${'corpuslongtoken '.repeat(200)}', 2800, 'A2-red-tests', 2, 'A', 'red', NULL);
      INSERT INTO chunks_fts (rowid, body_text) VALUES
        (1, 'slice filter fixture body alpha'),
        (2, 'slice filter fixture body beta'),
        (3, 'exact symbol fixture gamma'),
        (4, '${'corpuslongtoken '.repeat(200)}');
    `);
  } finally {
    await db.close();
  }
  CREATED_URLS.add(databasePath);
  return databasePath;
}

describe('search-corpus.mjs direct import coverage', () => {
  let databasePath;

  beforeAll(async () => {
    databasePath = await createFilteredFixture();
  });

  beforeEach(() => {
    resetReadinessCaches();
  });

  afterEach(() => {
    process.env.DENSE_FORCE_STATE = 'cold';
    process.env.RERANKER_FORCE_STATE = 'cold';
  });

  afterAll(async () => {
    for (const url of CREATED_URLS) {
      await closeTursoClient(url);
    }
    CREATED_URLS.clear();
    for (const tempDir of TEMP_DIRS) {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch {
        // Ignore Windows file-handle cleanup races for temp directories.
      }
    }
  });

  it('filters BM25 results by slice_id', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A2-red-tests',
      use_dense: false,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([1]);
  });

  it('filters BM25 results by step_number', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      step_number: 1,
      use_dense: false,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([2]);
  });

  it('combines slice_id and step_number filters', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A1-green',
      step_number: 1,
      use_dense: false,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([2]);
  });

  it('combines a metadata.filter with top-level slice_id and step_number', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      step_number: 1,
      use_dense: false,
      limit: 10,
      metadata: {
        filter: { op: 'eq', field: 'slice_id', value: 'A1-green' },
      },
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([2]);
  });

  it('does not filter by an empty or whitespace-only step_number', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      step_number: '  ',
      use_dense: false,
      limit: 10,
    });
    const chunkIds = response.results.map((r) => r.chunk_id);
    expect(chunkIds).toEqual(expect.arrayContaining([1, 2]));
  });

  it('flags exact_symbol_match when the slice_id filter matches', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'fixtureSymbol',
      slice_id: 'A2-red-tests',
      use_dense: false,
      limit: 10,
    });
    expect(response.exact_symbol_match).toBe(true);
  });

  it('returns the exact symbol chunk when the slice_id filter matches', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'fixtureSymbol',
      slice_id: 'A2-red-tests',
      use_dense: false,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([3]);
  });

  it('flags exact_symbol_match when the step_number filter matches', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'fixtureSymbol',
      step_number: 2,
      use_dense: false,
      limit: 10,
    });
    expect(response.exact_symbol_match).toBe(true);
  });

  it('returns the exact symbol chunk when the step_number filter matches', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'fixtureSymbol',
      step_number: 2,
      use_dense: false,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([3]);
  });

  it('marks use_dense false when the dense path is unavailable', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A1-green',
      use_dense: true,
      limit: 10,
    });
    expect(response.use_dense).toBe(false);
  });

  it('still filters BM25 results when the dense path is unavailable', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A1-green',
      use_dense: true,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([2]);
  });

  it('filters BM25 results when dense and reranker are unavailable', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      step_number: 1,
      use_dense: true,
      use_rerank: true,
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([2]);
  });

  it('marks rerank_degraded true when the reranker is unavailable', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      step_number: 1,
      use_dense: true,
      use_rerank: true,
      limit: 10,
    });
    expect(response.rerank_degraded).toBe(true);
  });

  it('rejects a missing query', async () => {
    await expect(
      searchCorpus({ databasePath, query: '', use_dense: false }),
    ).rejects.toThrow('query');
  });

  it('rejects a whitespace-only query', async () => {
    await expect(
      searchCorpus({ databasePath, query: '   ', use_dense: false }),
    ).rejects.toThrow('query');
  });

  it('returns compact summaries with truncated long text', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: false,
      compact: true,
      limit: 10,
    });
    expect(response.results[0].text.length).toBeLessThanOrEqual(300);
  });

  it('marks compact long text with an ellipsis', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: false,
      compact: true,
      limit: 10,
    });
    expect(response.results[0].text.endsWith('\u2026')).toBe(true);
  });

  it('preserves a ranking_explanation field when compact is true', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 1.0,
            ranking_explanation: { reason: 'mock dense' },
          },
        ],
      }),
    });
    expect(response.results[0].ranking_explanation.reason).toBe('mock dense');
  });

  it('keeps the original BM25 results when expandQueryFn throws', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A1-green',
      use_dense: false,
      expand_query: true,
      expandQueryFn: async () => {
        throw new Error('expansion failed');
      },
      limit: 10,
    });
    expect(response.results.map((r) => r.chunk_id)).toEqual([2]);
  });

  it('reports expansion as degraded when expandQueryFn throws', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A1-green',
      use_dense: false,
      expand_query: true,
      expandQueryFn: async () => {
        throw new Error('expansion failed');
      },
      limit: 10,
    });
    expect(response.expansion).toMatchObject({
      applied: false,
      degraded: true,
      reason: 'Query expansion failed',
    });
  });

  it('returns an empty BM25 response for a tokenless query', async () => {
    const response = await searchCorpus({
      databasePath,
      query: '!!!',
      use_dense: false,
      limit: 10,
    });
    expect(response.results).toEqual([]);
  });

  it('returns use_dense false for a tokenless BM25 query', async () => {
    const response = await searchCorpus({
      databasePath,
      query: '!!!',
      use_dense: false,
      limit: 10,
    });
    expect(response.use_dense).toBe(false);
  });

  it('echoes the raw tokenless query in the response', async () => {
    const response = await searchCorpus({
      databasePath,
      query: '!!!',
      use_dense: false,
      limit: 10,
    });
    expect(response.query).toBe('!!!');
  });

  it('uses warm dense results when the index is mocked warm', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 0.99,
          },
        ],
      }),
    });
    expect(response.use_dense).toBe(true);
  });

  it('reports warm dense state when the index is mocked warm', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 0.99,
          },
        ],
      }),
    });
    expect(response.dense_state).toBe('warm');
  });

  it('returns the warm dense result chunk', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 0.99,
          },
        ],
      }),
    });
    expect(response.results[0].chunk_id).toBe(4);
  });

  it('reranks warm dense candidates when reranker is mocked warm', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      use_rerank: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      rerankerReadinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 0.99,
          },
        ],
      }),
      rerankerFn: async (_query, candidates) =>
        candidates.map((candidate) => ({
          ...candidate,
          rerank_score: 1.0,
          score: 1.0,
        })),
    });
    expect(response.use_rerank).toBe(true);
  });

  it('reports warm reranker state when reranker is mocked warm', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      use_rerank: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      rerankerReadinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 0.99,
          },
        ],
      }),
      rerankerFn: async (_query, candidates) =>
        candidates.map((candidate) => ({
          ...candidate,
          rerank_score: 1.0,
          score: 1.0,
        })),
    });
    expect(response.rerank_state).toBe('warm');
  });

  it('returns the reranked warm dense result chunk', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'corpuslongtoken',
      slice_id: 'A2-red-tests',
      use_dense: true,
      use_rerank: true,
      compact: true,
      limit: 10,
      readinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      rerankerReadinessProbe: async () => ({
        state: 'warm',
        reason: 'mock warm',
        hasModel: true,
        hasDatabase: true,
      }),
      denseQuery: async () => ({
        results: [
          {
            chunk_id: 4,
            file_path: 'README.md',
            doc_family: 'readme',
            text: `${'corpuslongtoken '.repeat(200)}`,
            score: 0.99,
          },
        ],
      }),
      rerankerFn: async (_query, candidates) =>
        candidates.map((candidate) => ({
          ...candidate,
          rerank_score: 1.0,
          score: 1.0,
        })),
    });
    expect(response.results[0].chunk_id).toBe(4);
  });

  it('propagates a dense_query failure through the search wrapper', async () => {
    await expect(
      searchCorpus({
        databasePath,
        query: 'slice filter fixture body',
        slice_id: 'A1-green',
        use_dense: true,
        limit: 10,
        readinessProbe: async () => ({
          state: 'warm',
          reason: 'mock warm',
          hasModel: true,
          hasDatabase: true,
        }),
        denseQuery: async () => {
          throw new Error('dense index unavailable');
        },
      }),
    ).rejects.toThrow('dense index unavailable');
  });

  it('propagates a reranker failure through the search wrapper', async () => {
    await expect(
      searchCorpus({
        databasePath,
        query: 'corpuslongtoken',
        slice_id: 'A2-red-tests',
        use_dense: true,
        use_rerank: true,
        compact: true,
        limit: 10,
        readinessProbe: async () => ({
          state: 'warm',
          reason: 'mock warm',
          hasModel: true,
          hasDatabase: true,
        }),
        rerankerReadinessProbe: async () => ({
          state: 'warm',
          reason: 'mock warm',
          hasModel: true,
          hasDatabase: true,
        }),
        denseQuery: async () => ({
          results: [
            {
              chunk_id: 4,
              file_path: 'README.md',
              doc_family: 'readme',
              text: `${'corpuslongtoken '.repeat(200)}`,
              score: 0.99,
            },
          ],
        }),
        rerankerFn: async () => {
          throw new Error('reranker failed');
        },
      }),
    ).rejects.toThrow('reranker failed');
  });

  it('returns empty results for a tokenless query with dense requested', async () => {
    const response = await searchCorpus({
      databasePath,
      query: '!!!',
      use_dense: true,
      limit: 10,
    });
    expect(response.results).toEqual([]);
  });

  it('marks dense_degraded true for a tokenless query with dense requested', async () => {
    const response = await searchCorpus({
      databasePath,
      query: '!!!',
      use_dense: true,
      limit: 10,
    });
    expect(response.dense_degraded).toBe(true);
  });

  it('builds ranking explanation with all score fallbacks', () => {
    expect(buildRankingExplanation({}, true, true)).toEqual({
      bm25_score: 0,
      dense_score: 0,
      rerank_score: 0,
      final_score: 0,
      reason:
        'Final score blends BM25 lexical match, dense cosine similarity, cross-encoder rerank.',
    });
  });

  it('builds ranking explanation from available score fields', () => {
    const result = {
      bm25_score: 0.1,
      cosine_score: 0.2,
      rerank_score: 0.3,
      score: 0.4,
    };
    expect(buildRankingExplanation(result, true, true)).toEqual({
      bm25_score: 0.1,
      dense_score: 0.2,
      rerank_score: 0.3,
      final_score: 0.4,
      reason:
        'Final score blends BM25 lexical match, dense cosine similarity, cross-encoder rerank.',
    });
  });

  it('builds response freshness from a mocked client', async () => {
    const freshness = await buildResponseFreshness(databasePath, {
      execute: async (sql) => {
        if (sql.includes('COUNT')) {
          return {
            rows: [{ count: 5, last_indexed_at: 1234567890 }],
          };
        }
        return {
          rows: [{ mtime_ms: 1000, file_size: 2000, sha256: 'proof-sha' }],
        };
      },
    });
    expect(freshness).toMatchObject({
      stale: false,
      last_update_source: 'corpus_index',
      last_indexed_at: 1234567890,
      freshness_proof: { mtime_ms: 1000, size: 2000, sha256: 'proof-sha' },
    });
  });

  it('returns stale freshness when client query fails', async () => {
    const freshness = await buildResponseFreshness(databasePath, {
      execute: async () => {
        throw new Error('database closed');
      },
    });
    expect(freshness).toEqual({
      timestamp: expect.any(Number),
      stale: true,
      last_update_source: 'unknown',
    });
  });

  // --- normalization helpers ---

  it('normalizes a valid alpha to itself', () => {
    expect(normalizeAlpha(0.75)).toBe(0.75);
  });

  it('normalizes a missing alpha to 0.5', () => {
    expect(normalizeAlpha(undefined)).toBe(0.5);
  });

  it('normalizes a non-finite alpha to 0.5', () => {
    expect(normalizeAlpha(NaN)).toBe(0.5);
  });

  it('returns the provided dense reason unchanged', () => {
    expect(normalizeDenseReason('custom reason', 'cold')).toBe('custom reason');
  });

  it('falls back to a cold dense reason', () => {
    expect(normalizeDenseReason(undefined, 'cold')).toBe(
      'Dense embeddings are unavailable because the model assets are absent.',
    );
  });

  it('falls back to a model-only dense reason', () => {
    expect(normalizeDenseReason(undefined, 'model-only')).toBe(
      'Dense embeddings are unavailable because the embeddings database is missing or incomplete.',
    );
  });

  it('returns the provided reranker reason unchanged', () => {
    expect(normalizeRerankReason('custom reason', 'cold')).toBe(
      'custom reason',
    );
  });

  it('falls back to a cold reranker reason', () => {
    expect(normalizeRerankReason(undefined, 'cold')).toBe(
      'Cross-encoder reranker is unavailable because the model assets are absent.',
    );
  });

  it('falls back to a model-only reranker reason', () => {
    expect(normalizeRerankReason(undefined, 'model-only')).toBe(
      'Cross-encoder reranker is unavailable because the ONNX session could not be created.',
    );
  });

  // --- response builders ---

  it('creates an empty BM25 response without optional fields', () => {
    expect(
      createEmptyBm25Response({ family: null, limit: 5, rawQuery: 'q' }),
    ).toEqual({
      limit: 5,
      query: 'q',
      results: [],
      use_dense: false,
      diskann_used: false,
      rrf_used: false,
    });
  });

  it('creates an empty BM25 response with family and classification metadata', () => {
    const response = createEmptyBm25Response({
      family: 'readme',
      limit: 5,
      rawQuery: 'q',
      classificationMetadata: {
        query_class: 'simple_lookup',
        confidence: 0.9,
        classification_fallback: false,
        family_fallback: true,
      },
    });
    expect(response.family).toBe('readme');
  });

  it('creates a degraded empty BM25 response for an empty query', async () => {
    const response = await createDegradedBm25Response({
      alpha: 0.5,
      family: null,
      limit: 5,
      query: '',
      readinessReport: { state: 'cold', reason: '' },
    });
    expect(response.dense_degraded).toBe(true);
  });

  it('creates a degraded BM25 response by running a real BM25 search', async () => {
    const response = await createDegradedBm25Response({
      alpha: 0.5,
      family: null,
      limit: 5,
      query: 'fixture',
      readinessReport: { state: 'cold', reason: '' },
      databasePath,
    });
    expect(response.use_dense).toBe(false);
  });

  // --- compact & token estimation ---

  it('returns a short result unchanged in compact mode', () => {
    expect(
      compactSearchResult({ chunk_id: 1, text: 'short', score: 0.5 }),
    ).toMatchObject({
      chunk_id: 1,
      text: 'short',
      score: 0.5,
    });
  });

  it('truncates a long result in compact mode', () => {
    const longText = 'a'.repeat(500);
    const result = compactSearchResult({
      chunk_id: 1,
      text: longText,
      score: 0.5,
    });
    expect(result.text.endsWith('…')).toBe(true);
  });

  it('preserves a ranking explanation in compact mode', () => {
    const explanation = { bm25_score: 1, reason: 'test' };
    const result = compactSearchResult({
      chunk_id: 1,
      text: 'short',
      score: 0.5,
      ranking_explanation: explanation,
    });
    expect(result.ranking_explanation).toBe(explanation);
  });

  it('estimates response tokens from result text lengths', () => {
    expect(estimateResponseTokens([{ text: 'abcd' }, { text: 'efghij' }])).toBe(
      3,
    );
  });

  // --- readiness cache helpers ---

  it('bypasses the dense readiness cache when DENSE_FORCE_STATE is cold', () => {
    process.env.DENSE_FORCE_STATE = 'cold';
    const result = shouldBypassDenseReadinessCache();
    delete process.env.DENSE_FORCE_STATE;
    expect(result).toBe(true);
  });

  it('does not bypass the dense readiness cache by default', () => {
    delete process.env.DENSE_FORCE_STATE;
    expect(shouldBypassDenseReadinessCache()).toBe(false);
  });

  it('bypasses the reranker readiness cache when RERANKER_FORCE_STATE is model-only', () => {
    process.env.RERANKER_FORCE_STATE = 'model-only';
    const result = shouldBypassRerankerReadinessCache();
    delete process.env.RERANKER_FORCE_STATE;
    expect(result).toBe(true);
  });

  it('does not bypass the reranker readiness cache by default', () => {
    delete process.env.RERANKER_FORCE_STATE;
    expect(shouldBypassRerankerReadinessCache()).toBe(false);
  });

  it('returns a warm dense readiness report directly from the probe', async () => {
    const report = await getDenseReadiness({
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
    });
    expect(report.state).toBe('warm');
  });

  it('only calls the dense readiness probe once for repeated calls', async () => {
    delete process.env.DENSE_FORCE_STATE;
    let calls = 0;
    const probe = async () => {
      calls += 1;
      return { state: 'warm', reason: 'ok' };
    };
    await getDenseReadiness({ readinessProbe: probe });
    await getDenseReadiness({ readinessProbe: probe });
    expect(calls).toBe(1);
  });

  it('returns a cached warm dense readiness report', async () => {
    delete process.env.DENSE_FORCE_STATE;
    let calls = 0;
    const probe = async () => {
      calls += 1;
      return { state: 'warm', reason: 'ok' };
    };
    await getDenseReadiness({ readinessProbe: probe });
    const report = await getDenseReadiness({ readinessProbe: probe });
    expect(report.state).toBe('warm');
  });

  it('returns a cold dense readiness report directly from the probe', async () => {
    const report = await getDenseReadiness({
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
    });
    expect(report.state).toBe('cold');
  });

  it('returns a warm reranker readiness report from the probe', async () => {
    const report = await getRerankerReadiness({
      rerankerReadinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
    });
    expect(report.state).toBe('warm');
  });

  it('only calls the reranker readiness probe once for repeated calls', async () => {
    delete process.env.RERANKER_FORCE_STATE;
    let calls = 0;
    const probe = async () => {
      calls += 1;
      return { state: 'warm', reason: 'ok' };
    };
    await getRerankerReadiness({ rerankerReadinessProbe: probe });
    await getRerankerReadiness({ rerankerReadinessProbe: probe });
    expect(calls).toBe(1);
  });

  it('returns a cached warm reranker readiness report', async () => {
    delete process.env.RERANKER_FORCE_STATE;
    let calls = 0;
    const probe = async () => {
      calls += 1;
      return { state: 'warm', reason: 'ok' };
    };
    await getRerankerReadiness({ rerankerReadinessProbe: probe });
    const report = await getRerankerReadiness({
      rerankerReadinessProbe: probe,
    });
    expect(report.state).toBe('warm');
  });

  // --- chunk count & strategy ---

  it('resolves chunk count from a warm readiness report', async () => {
    const count = await resolveChunkCountForStrategy(
      databasePath,
      { chunk_count: 42 },
      null,
    );
    expect(count).toBe(42);
  });

  it('resolves chunk count by counting rows when the report omits it', async () => {
    const count = await resolveChunkCountForStrategy(databasePath, {}, null);
    expect(count).toBe(4);
  });

  it('resolves chunk count to zero when the count query fails', async () => {
    const count = await resolveChunkCountForStrategy(
      databasePath,
      {},
      {
        execute: async () => {
          throw new Error('fail');
        },
      },
    );
    expect(count).toBe(0);
  });

  it('resolves the dense strategy using the readiness report status', async () => {
    const strategy = await resolveDenseStrategyForSearch({
      databasePath,
      readinessReport: { state: 'warm', ann_index_status: 'ready' },
    });
    expect(strategy).toBe('diskann');
  });

  it('resolves the dense strategy from state when ann_index_status is absent', async () => {
    const strategy = await resolveDenseStrategyForSearch({
      databasePath,
      readinessReport: { state: 'warm' },
    });
    expect(strategy).toBe('diskann');
  });

  // --- exact symbol helpers ---

  it('returns null from tryExactSymbolLookup when no symbol matches', async () => {
    const result = await tryExactSymbolLookup({
      databasePath,
      rawQuery: 'noSuchSymbol',
      family: null,
      limit: 5,
    });
    expect(result).toBeNull();
  });

  it('returns an exact symbol response when a symbol matches', async () => {
    const result = await tryExactSymbolLookup({
      databasePath,
      rawQuery: 'fixtureSymbol',
      family: null,
      limit: 5,
    });
    expect(result?.exact_symbol_match).toBe(true);
  });

  it('runs an exact symbol lookup with a family filter', async () => {
    const rows = await runExactSymbolLookup({
      databasePath,
      rawQuery: 'fixtureSymbol',
      family: 'readme',
      limit: 5,
    });
    expect(rows.length).toBeGreaterThan(0);
  });

  it('returns an empty array when exact symbol lookup fails', async () => {
    const rows = await runExactSymbolLookup({
      databasePath,
      rawQuery: 'fixtureSymbol',
      family: null,
      limit: 5,
      client: {
        execute: async () => {
          throw new Error('fail');
        },
      },
    });
    expect(rows).toEqual([]);
  });

  // --- BM25 helper branches ---

  it('runs BM25 search successfully when feedback_scores table exists', async () => {
    const tempDir = trackTempDir();
    const dbWithFeedbackPath = path.join(tempDir, 'feedback.sqlite');
    const db = createClient({ url: `file:${dbWithFeedbackPath}` });
    try {
      await db.executeMultiple(`
        CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, doc_family TEXT, file_path TEXT, indexed_at INTEGER);
        CREATE TABLE chunks (
          chunk_id INTEGER PRIMARY KEY,
          doc_id INTEGER,
          chunk_index INTEGER NOT NULL DEFAULT 0,
          heading_path TEXT,
          body_text TEXT,
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
        CREATE VIRTUAL TABLE chunks_fts USING fts5(body_text, content='chunks', content_rowid='chunk_id');
        CREATE TABLE feedback_scores (
          chunk_id INTEGER PRIMARY KEY,
          feedback_boost REAL,
          total_positive INTEGER,
          total_negative INTEGER,
          total_impressions INTEGER,
          total_clicks INTEGER,
          total_references INTEGER
        );
        INSERT INTO documents VALUES (1, 'readme', 'README.md', 1);
        INSERT INTO chunks (chunk_id, doc_id, body_text) VALUES (1, 1, 'feedback score body');
        INSERT INTO chunks_fts (rowid, body_text) VALUES (1, 'feedback score body');
        INSERT INTO feedback_scores VALUES (1, 0.5, 1, 0, 2, 0, 0);
      `);
    } finally {
      await db.close();
    }
    CREATED_URLS.add(dbWithFeedbackPath);

    const result = await runBm25Search({
      databasePath: dbWithFeedbackPath,
      query: 'feedback score',
      family: null,
      limit: 5,
    });
    expect(result.results[0].feedback_signals.total_positive).toBe(1);
  });

  it('falls back to the no-feedback BM25 query when feedback_scores is missing', async () => {
    const result = await runBm25Search({
      databasePath,
      query: 'fixture',
      family: null,
      limit: 5,
    });
    expect(result.results.length).toBeGreaterThan(0);
  });

  // --- feedback & impressions ---

  it('returns early when attaching feedback to an empty result list', async () => {
    const results = [];
    await attachFeedbackToResults(results, databasePath);
    expect(results).toEqual([]);
  });

  it('attaches default feedback signals when no numeric chunk ids are present', async () => {
    const results = [{ chunk_id: 'not-a-number' }];
    await attachFeedbackToResults(results, databasePath);
    expect(results[0].feedback_signals.total_positive).toBe(0);
  });

  it('attaches real feedback scores when rows exist', async () => {
    const tempDir = trackTempDir();
    const feedbackPath = path.join(tempDir, 'feedback.sqlite');
    const db = createClient({ url: `file:${feedbackPath}` });
    try {
      await db.executeMultiple(`
        CREATE TABLE documents (doc_id INTEGER PRIMARY KEY);
        CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY);
        CREATE TABLE feedback_scores (
          chunk_id INTEGER PRIMARY KEY,
          feedback_boost REAL,
          total_positive INTEGER,
          total_negative INTEGER,
          total_impressions INTEGER,
          total_clicks INTEGER,
          total_references INTEGER
        );
        INSERT INTO documents VALUES (1);
        INSERT INTO chunks VALUES (1);
        INSERT INTO feedback_scores VALUES (1, 0.25, 2, 1, 3, 1, 0);
      `);
    } finally {
      await db.close();
    }
    CREATED_URLS.add(feedbackPath);

    const results = [{ chunk_id: 1 }];
    await attachFeedbackToResults(results, feedbackPath);
    expect(results[0].feedback_boost).toBe(0.25);
  });

  it('attaches default feedback signals when a score row is missing', async () => {
    const results = [{ chunk_id: 1 }];
    await attachFeedbackToResults(results, databasePath);
    expect(results[0].feedback_boost).toBe(0);
  });

  it('swallows errors when attaching feedback fails', async () => {
    const results = [{ chunk_id: 1 }];
    await attachFeedbackToResults(results, databasePath, {
      execute: async () => {
        throw new Error('fail');
      },
    });
    expect(results[0].feedback_boost).toBe(0);
  });

  it('returns early from impression recording when results are missing', async () => {
    await expect(
      recordSearchImpressions(
        { query: 'q', results: undefined },
        databasePath,
        {
          batch: async () => {
            throw new Error('should not be called');
          },
        },
      ),
    ).resolves.toBeUndefined();
  });

  it('returns early from impression recording when the query is missing', async () => {
    await expect(
      recordSearchImpressions({ results: [{ chunk_id: 1 }] }, databasePath, {
        batch: async () => {
          throw new Error('should not be called');
        },
      }),
    ).resolves.toBeUndefined();
  });

  it('records impressions for numeric chunk ids', async () => {
    const batched = [];
    await recordSearchImpressions(
      { query: 'q', results: [{ chunk_id: 1 }, { chunk_id: 2 }] },
      databasePath,
      {
        batch: async (statements) => {
          batched.push(...statements);
        },
      },
    );
    expect(batched.length).toBe(2);
  });

  it('skips non-numeric chunk ids when recording impressions', async () => {
    const batched = [];
    await recordSearchImpressions(
      { query: 'q', results: [{ chunk_id: 'bad' }, { chunk_id: 2 }] },
      databasePath,
      {
        batch: async (statements) => {
          batched.push(...statements);
        },
      },
    );
    expect(batched.length).toBe(1);
  });

  it('swallows errors when impression recording fails', async () => {
    await expect(
      recordSearchImpressions(
        { query: 'q', results: [{ chunk_id: 1 }] },
        databasePath,
        {
          batch: async () => {
            throw new Error('fail');
          },
        },
      ),
    ).resolves.toBeUndefined();
  });

  // --- searchCorpusImpl branches ---

  it('uses explicit query_class for classification metadata', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      query_class: 'simple_lookup',
      use_dense: false,
      limit: 5,
    });
    expect(response.query_class).toBe('simple_lookup');
  });

  it('applies classification_hints family override', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      classification_hints: { family: 'readme' },
      use_dense: false,
      limit: 5,
    });
    expect(response.family).toBe('readme');
  });

  it('skips family classification when requested', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      classification_hints: { family: 'nonexistent' },
      skip_family_classification: true,
      use_dense: false,
      limit: 5,
    });
    expect(response.family).toBeUndefined();
  });

  it('expands the query when expand_query is enabled', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true },
        bm25Query: 'expanded fixture',
      }),
      use_dense: false,
      limit: 5,
    });
    expect(response.expansion.applied).toBe(true);
  });

  it('degrades gracefully when query expansion fails', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      expand_query: true,
      expandQueryFn: async () => {
        throw new Error('expansion failed');
      },
      use_dense: false,
      limit: 5,
    });
    expect(response.expansion.degraded).toBe(true);
  });

  it('throws for an invalid metadata filter', async () => {
    await expect(
      searchCorpusImpl({
        databasePath,
        query: 'fixture',
        metadata: { filter: { op: 'unknown', field: 'x', value: 'y' } },
        use_dense: false,
        limit: 5,
      }),
    ).rejects.toThrow();
  });

  it('formats a non-Error metadata-filter compilation error', async () => {
    await expect(
      searchCorpusImpl({
        databasePath,
        query: 'fixture',
        metadata: { filter: { op: 'eq', field: 'slice_id', value: 'x' } },
        use_dense: false,
        limit: 5,
        compileFilterFn: () => {
          throw 'bad filter';
        },
      }),
    ).rejects.toThrow('bad filter');
  });

  it('reports use_dense false for an empty query', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      use_dense: false,
      limit: 5,
    });
    expect(response.use_dense).toBe(false);
  });

  it('reports dense warm and empty results for an empty query with dense warm', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      use_dense: true,
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
    });
    expect(response.dense_state).toBe('warm');
  });

  it('requests rerank on an empty query and degrades to not_requested', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      use_dense: true,
      use_rerank: true,
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
    });
    expect(response.use_rerank).toBe(false);
  });

  it('falls back to BM25 when dense is cold and rerank is not requested', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: false,
      limit: 5,
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
    });
    expect(response.dense_degraded).toBe(true);
  });

  it('applies rerank to degraded BM25 results when reranker is warm', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: true,
      limit: 5,
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
      rerankerReadinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
      rerankerFn: async (_query, candidates) =>
        candidates.map((candidate) => ({ ...candidate, rerank_score: 1.0 })),
    });
    expect(response.use_rerank).toBe(true);
  });

  it('degrades rerank when the reranker is cold on a degraded BM25 path', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: true,
      limit: 5,
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
      rerankerReadinessProbe: async () => ({
        state: 'cold',
        reason: 'no model',
      }),
    });
    expect(response.rerank_state).toBe('cold');
  });

  it('returns warm dense results without rerank', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: false,
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
      denseQuery: async () => ({
        results: [{ chunk_id: 7, text: 'dense result', score: 0.9 }],
      }),
    });
    expect(response.results[0].chunk_id).toBe(7);
  });

  it('degrades rerank when the reranker is cold on a warm dense path', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: true,
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
      denseQuery: async () => ({
        results: [{ chunk_id: 7, text: 'dense result', score: 0.9 }],
      }),
      rerankerReadinessProbe: async () => ({
        state: 'cold',
        reason: 'no model',
      }),
    });
    expect(response.rerank_degraded).toBe(true);
  });

  it('classifies and emits metadata when only alpha is provided', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'fixture',
      use_dense: false,
      alpha: 0.75,
      limit: 5,
    });
    expect(response.query_class).toBeDefined();
  });

  it('maps a missing-table error to a corpus-not-found error', async () => {
    await expect(
      searchCorpus({
        databasePath,
        query: 'fixture',
        use_dense: false,
        limit: 5,
        client: {
          execute: async () => {
            throw new Error('no such table: chunks');
          },
          close: async () => {},
        },
      }),
    ).rejects.toThrow('Corpus database not found');
  });

  it('maps a non-Error thrown by searchCorpusImpl to its string form', async () => {
    await expect(
      searchCorpus({
        databasePath,
        query: 'fixture',
        limit: 5,
        client: {
          execute: async () => {
            throw 'boom';
          },
          close: async () => {},
        },
      }),
    ).rejects.toBe('boom');
  });

  it('defaults use_dense to true and degrades when dense is cold', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      limit: 5,
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
    });
    expect(response.dense_degraded).toBe(true);
  });

  it('resolves chunk count from the table when the report count is null', async () => {
    const count = await resolveChunkCountForStrategy(
      databasePath,
      { chunk_count: null },
      null,
    );
    expect(count).toBe(4);
  });

  it('resolves the dense strategy when ann_index_status is absent for cold state', async () => {
    const strategy = await resolveDenseStrategyForSearch({
      databasePath,
      readinessReport: { state: 'cold' },
    });
    expect(strategy).toBe('diskann');
  });

  it('falls back to the default expandQuery when expandQueryFn is not provided', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'function implementation',
      expand_query: 'domain-only',
      associationsPath: '/no/such/associations.json',
      use_dense: false,
      limit: 5,
    });
    expect(response.expansion.applied).toBe(false);
  });

  it('keeps the original BM25 query when expansion returns no bm25Query', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true },
        bm25Query: null,
      }),
      use_dense: false,
      limit: 5,
    });
    expect(response.expansion.applied).toBe(true);
    expect(response.results.length).toBeGreaterThan(0);
  });

  it('includes expansion metadata in an empty-query BM25 response', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      use_dense: false,
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true, reason: 'empty expansion test' },
        bm25Query: null,
      }),
      limit: 5,
    });
    expect(response.expansion.applied).toBe(true);
  });

  it('includes expansion metadata in an empty-query degraded dense response', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      use_dense: true,
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true, reason: 'empty degraded test' },
        bm25Query: null,
      }),
      limit: 5,
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
    });
    expect(response.expansion.applied).toBe(true);
    expect(response.dense_degraded).toBe(true);
  });

  it('includes expansion metadata in an empty-query warm dense response', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      use_dense: true,
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true, reason: 'empty warm test' },
        bm25Query: null,
      }),
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
    });
    expect(response.expansion.applied).toBe(true);
    expect(response.dense_state).toBe('warm');
  });

  it('builds a ranking explanation with all stages disabled', () => {
    expect(buildRankingExplanation({}, false, false)).toEqual({
      bm25_score: 0,
      dense_score: 0,
      rerank_score: 0,
      final_score: 0,
      reason: 'Final score blends BM25 lexical match.',
    });
  });

  it('falls back to empty text in compact mode when text is absent', () => {
    const result = compactSearchResult({ chunk_id: 1, score: 0.5 });
    expect(result.text).toBe('');
  });

  it('builds a freshness proof with null field values', async () => {
    const freshness = await buildResponseFreshness(databasePath, {
      execute: async (sql) => {
        if (sql.includes('COUNT')) {
          return {
            rows: [{ count: 5, last_indexed_at: 1234567890 }],
          };
        }
        return {
          rows: [{ mtime_ms: null, file_size: null, sha256: null }],
        };
      },
    });
    expect(freshness.freshness_proof).toEqual({
      mtime_ms: null,
      size: null,
      sha256: null,
    });
  });

  it('accepts no options and rejects for a missing query', async () => {
    await expect(searchCorpusImpl()).rejects.toThrow();
  });

  it('skips family classification for an explicit query_class', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      query_class: 'simple_lookup',
      classification_hints: { family: 'ts-source' },
      skip_family_classification: true,
      use_dense: false,
      limit: 5,
    });
    expect(response.family).toBeUndefined();
  });

  it('uses classification alpha in the explicit-alpha branch when hints alpha is null', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      classification_hints: { alpha: null },
      use_dense: false,
      limit: 5,
    });
    expect(response.query_class).toBeDefined();
  });

  it('skips family classification in the explicit-alpha branch', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      alpha: 0.75,
      classification_hints: { family: 'readme' },
      skip_family_classification: true,
      use_dense: false,
      limit: 5,
    });
    expect(response.family).toBeUndefined();
  });

  it('uses the score field as a BM25 fallback in ranking explanation', () => {
    expect(buildRankingExplanation({ score: 0.4 }, false, false)).toMatchObject(
      {
        bm25_score: 0.4,
        final_score: 0.4,
      },
    );
  });

  it('broadens a derived family filter that produced zero results', async () => {
    const tempDir = trackTempDir();
    const broadPath = path.join(tempDir, 'broad.sqlite');
    const db = createClient({ url: `file:${broadPath}` });
    try {
      await db.executeMultiple(`
        CREATE TABLE documents (
          doc_id INTEGER PRIMARY KEY,
          doc_family TEXT NOT NULL,
          file_path TEXT NOT NULL,
          indexed_at INTEGER
        );
        CREATE TABLE chunks (
          chunk_id INTEGER PRIMARY KEY,
          doc_id INTEGER,
          chunk_index INTEGER DEFAULT 0,
          heading_path TEXT,
          body_text TEXT,
          char_start INTEGER DEFAULT 0,
          char_end INTEGER DEFAULT 0,
          parent_chunk_id INTEGER,
          depth INTEGER DEFAULT 0,
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
        CREATE VIRTUAL TABLE chunks_fts USING fts5(body_text, content='chunks', content_rowid='chunk_id');
        INSERT INTO documents VALUES (1, 'ts-source', 'src/a.ts', 1);
        INSERT INTO chunks (chunk_id, doc_id, body_text) VALUES (1, 1, 'unrelated typescript content');
        INSERT INTO chunks_fts (rowid, body_text) VALUES (1, 'unrelated typescript content');
        INSERT INTO documents VALUES (2, 'readme', 'README.md', 1);
        INSERT INTO chunks (chunk_id, doc_id, body_text) VALUES (2, 2, 'fixture implementation details');
        INSERT INTO chunks_fts (rowid, body_text) VALUES (2, 'fixture implementation details');
      `);
    } finally {
      await db.close();
    }
    CREATED_URLS.add(broadPath);

    const response = await searchCorpus({
      databasePath: broadPath,
      query: 'fixture implementation',
      use_dense: false,
      limit: 5,
    });
    expect(response.results.length).toBeGreaterThan(0);
  });

  it('handles an expansion result with an undefined bm25Query', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true },
        bm25Query: undefined,
      }),
      use_dense: false,
      limit: 5,
    });
    expect(response.expansion.applied).toBe(true);
  });

  it('uses the default reranker function when reranker is warm', async () => {
    process.env.RERANKER_FORCE_STATE = 'warm';
    await expect(
      searchCorpus({
        databasePath,
        query: 'fixture',
        use_dense: true,
        use_rerank: true,
      }),
    ).rejects.toThrow();
  });

  it('uses the default options parameter when called without arguments', async () => {
    await expect(searchCorpus()).rejects.toThrow(
      'query must be a non-empty string',
    );
  });

  it('reports the default database path when the default corpus is missing', async () => {
    const tempDir = trackTempDir();
    const emptyDb = path.join(tempDir, 'empty.sqlite');
    const emptyClient = createClient({ url: `file:${emptyDb}` });
    await emptyClient.execute(
      'CREATE TABLE IF NOT EXISTS dummy (id INTEGER PRIMARY KEY);',
    );
    await emptyClient.close();

    const originalUrl = process.env.TURSO_DATABASE_URL;
    process.env.TURSO_DATABASE_URL = `file:${emptyDb}`;
    try {
      await expect(searchCorpus({ query: 'fixture' })).rejects.toThrow(
        'Corpus database not found: default path',
      );
    } finally {
      process.env.TURSO_DATABASE_URL = originalUrl;
    }
  });

  // --- remaining branch-coverage tests ---

  it('resolves chunk count to zero when the count row is missing', async () => {
    const count = await resolveChunkCountForStrategy(
      databasePath,
      {},
      {
        execute: async () => ({ rows: [{}] }),
      },
    );
    expect(count).toBe(0);
  });

  it('honours an explicit family option', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      family: 'readme',
      use_dense: false,
      limit: 5,
    });
    expect(response.family).toBe('readme');
  });

  it('includes family in an empty warm dense response', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: '!!!',
      family: 'readme',
      use_dense: true,
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
    });
    expect(response.family).toBe('readme');
  });

  it('includes expansion metadata in a degraded dense response', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: false,
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true, reason: 'degraded expansion' },
        bm25Query: 'expanded fixture',
      }),
      limit: 5,
      readinessProbe: async () => ({ state: 'cold', reason: 'no model' }),
    });
    expect(response.expansion.applied).toBe(true);
  });

  it('includes expansion metadata in a warm dense response', async () => {
    const response = await searchCorpusImpl({
      databasePath,
      query: 'fixture',
      use_dense: true,
      use_rerank: false,
      expand_query: true,
      expandQueryFn: async () => ({
        expansion: { applied: true, reason: 'warm expansion' },
        bm25Query: 'expanded fixture',
      }),
      limit: 5,
      readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
      denseQuery: async () => ({
        results: [{ chunk_id: 7, text: 'dense result', score: 0.9 }],
      }),
    });
    expect(response.expansion.applied).toBe(true);
  });

  it('uses the default dense query function when denseQuery is not provided', async () => {
    delete process.env.DENSE_FORCE_STATE;
    await expect(
      searchCorpusImpl({
        databasePath,
        query: 'fixture',
        use_dense: true,
        use_rerank: false,
        limit: 5,
        readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
      }),
    ).rejects.toThrow();
  });

  it('uses the default reranker function when rerankerFn is not provided', async () => {
    delete process.env.DENSE_FORCE_STATE;
    delete process.env.RERANKER_FORCE_STATE;
    await expect(
      searchCorpusImpl({
        databasePath,
        query: 'fixture',
        use_dense: true,
        use_rerank: true,
        limit: 5,
        readinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
        denseQuery: async () => ({
          results: [{ chunk_id: 7, text: 'dense result', score: 0.9 }],
        }),
        rerankerReadinessProbe: async () => ({ state: 'warm', reason: 'ok' }),
      }),
    ).rejects.toThrow();
  });

  it('estimates response tokens when a result lacks text', () => {
    expect(
      estimateResponseTokens([{ text: 'abcdefgh' }, { text: undefined }]),
    ).toBe(2);
  });

  it('does not record impressions when all chunk ids are non-numeric', async () => {
    await expect(
      recordSearchImpressions(
        { query: 'q', results: [{ chunk_id: 'bad' }, { chunk_id: null }] },
        databasePath,
        {
          batch: async () => {
            throw new Error('should not be called');
          },
        },
      ),
    ).resolves.toBeUndefined();
  });

  it('builds response freshness for an empty or null-indexed corpus', async () => {
    const freshness = await buildResponseFreshness(databasePath, {
      execute: async (sql) => {
        if (sql.includes('COUNT')) {
          return { rows: [{}] };
        }
        return { rows: [] };
      },
    });
    expect(freshness).toMatchObject({
      stale: true,
      last_update_source: 'empty_corpus_index',
    });
  });

  it('builds response freshness without a proof row', async () => {
    const freshness = await buildResponseFreshness(databasePath, {
      execute: async (sql) => {
        if (sql.includes('COUNT')) {
          return { rows: [{ count: 1, last_indexed_at: Date.now() }] };
        }
        return { rows: [] };
      },
    });
    expect(freshness.last_update_source).toBe('corpus_index');
    expect(freshness.freshness_proof).toBeUndefined();
  });

  it('does not replace results when family broadening still returns empty', async () => {
    const tempDir = trackTempDir();
    const broadPath = path.join(tempDir, 'broad-empty.sqlite');
    const db = createClient({ url: `file:${broadPath}` });
    try {
      await db.executeMultiple(`
        CREATE TABLE documents (
          doc_id INTEGER PRIMARY KEY,
          doc_family TEXT NOT NULL,
          file_path TEXT,
          indexed_at INTEGER
        );
        CREATE TABLE chunks (
          chunk_id INTEGER PRIMARY KEY,
          doc_id INTEGER,
          chunk_index INTEGER DEFAULT 0,
          heading_path TEXT,
          body_text TEXT,
          char_start INTEGER DEFAULT 0,
          char_end INTEGER DEFAULT 0,
          parent_chunk_id INTEGER,
          depth INTEGER DEFAULT 0,
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
        CREATE VIRTUAL TABLE chunks_fts USING fts5(body_text, content='chunks', content_rowid='chunk_id');
        INSERT INTO documents VALUES (1, 'readme', 'README.md', 1);
        INSERT INTO chunks (chunk_id, doc_id, body_text) VALUES (1, 1, 'totally unrelated content');
        INSERT INTO chunks_fts (rowid, body_text) VALUES (1, 'totally unrelated content');
      `);
    } finally {
      await db.close();
    }
    CREATED_URLS.add(broadPath);

    const response = await searchCorpus({
      databasePath: broadPath,
      query: 'xyznotfoundquery',
      use_dense: false,
      limit: 5,
      classification_hints: { family: 'ts-source' },
    });
    expect(response.results).toEqual([]);
  });

  it('does not compact results when compact is false', async () => {
    const response = await searchCorpus({
      databasePath,
      query: 'slice filter fixture body',
      slice_id: 'A1-green',
      use_dense: false,
      compact: false,
      limit: 10,
    });
    expect(response.compact).toBe(false);
  });

  it('does not cache a cold dense readiness report', async () => {
    delete process.env.DENSE_FORCE_STATE;
    let calls = 0;
    const probe = async () => {
      calls += 1;
      return { state: 'cold', reason: 'no model' };
    };
    await getDenseReadiness({ readinessProbe: probe });
    await getDenseReadiness({ readinessProbe: probe });
    expect(calls).toBe(2);
  });

  it('does not cache a cold reranker readiness report', async () => {
    delete process.env.RERANKER_FORCE_STATE;
    let calls = 0;
    const probe = async () => {
      calls += 1;
      return { state: 'cold', reason: 'no model' };
    };
    await getRerankerReadiness({ rerankerReadinessProbe: probe });
    await getRerankerReadiness({ rerankerReadinessProbe: probe });
    expect(calls).toBe(2);
  });

  it('includes family in an exact symbol response', async () => {
    const result = await tryExactSymbolLookup({
      databasePath,
      rawQuery: 'fixtureSymbol',
      family: 'readme',
      limit: 5,
      classificationMetadata: {
        query_class: 'simple_lookup',
        confidence: 0.9,
        classification_fallback: false,
        family_fallback: undefined,
      },
    });
    expect(result?.family).toBe('readme');
  });

  it('creates an empty BM25 response with undefined family_fallback', () => {
    const response = createEmptyBm25Response({
      family: 'readme',
      limit: 5,
      rawQuery: 'q',
      classificationMetadata: {
        query_class: 'simple_lookup',
        confidence: 0.9,
        classification_fallback: false,
      },
    });
    expect(response.family_fallback).toBe(false);
  });

  it('creates a degraded BM25 response with null feedback columns', async () => {
    const response = await createDegradedBm25Response({
      alpha: 0.5,
      family: 'readme',
      limit: 2,
      query: 'x',
      readinessReport: { state: 'cold', reason: '' },
      client: {
        execute: async () => ({
          rows: [
            {
              chunk_id: 1,
              doc_family: 'readme',
              file_path: 'a.ts',
              body_text: 'body',
              score: 0.5,
              fb_total_positive: null,
              fb_total_negative: null,
              fb_total_impressions: null,
              fb_total_clicks: null,
              fb_total_references: null,
            },
          ],
        }),
      },
      classificationMetadata: {
        query_class: 'simple_lookup',
        confidence: 0.9,
        classification_fallback: false,
      },
    });
    expect(response.results[0].feedback_signals).toEqual({
      total_positive: 0,
      total_negative: 0,
      total_impressions: 0,
      total_clicks: 0,
      total_references: 0,
    });
  });
});
