/**
 * @module search-context.direct.test
 * @description Direct-import coverage tests for search-context.mjs filter forwarding.
 *
 * Runs in the mcp-semantic-mjs project so Jest can instrument the native ESM
 * source file via the V8 coverage provider.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import {
  buildFollowUpRefs,
  buildTopResult,
  normalizeSearchResultToChunk,
  searchContext,
  searchContextTool,
  validateBudget,
  validateContextFormat,
} from './search-context.mjs';
import { resetReadinessCaches } from './search-corpus.mjs';
import { closeTursoClient } from './cortex-db.mjs';

const CREATED_URLS = new Set();
const TEMP_DIRS = [];

function trackTempDir() {
  const tempDir = fs.mkdtempSync(
    path.join(os.tmpdir(), 'search-context-direct-'),
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
      INSERT INTO chunks (chunk_id, doc_id, body_text, char_end, slice_id, step_number, phase, status)
        VALUES
          (1, 1, 'context slice filter alpha', 26, 'A2-red-tests', 2, 'A', 'red'),
          (2, 1, 'context slice filter beta', 25, 'A1-green', 1, 'A', 'green'),
          (4, 1, '${'longbodytoken '.repeat(200)}', 2800, 'A2-red-tests', 2, 'A', 'red');
      INSERT INTO chunks_fts (rowid, body_text) VALUES
        (1, 'context slice filter alpha'),
        (2, 'context slice filter beta'),
        (4, '${'longbodytoken '.repeat(200)}');
    `);
  } finally {
    await db.close();
  }
  CREATED_URLS.add(databasePath);
  return databasePath;
}

describe('search-context.mjs direct import coverage', () => {
  let databasePath;

  beforeAll(async () => {
    databasePath = await createFilteredFixture();
  });

  beforeEach(() => {
    resetReadinessCaches();
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

  it('forwards slice_id to corpus search and filters assembled context', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      limit: 10,
    });
    const chunkIds = response.results.map((r) => r.chunk_id);
    expect(chunkIds).toEqual([1]);
  });

  it('forwards step_number to corpus search and filters assembled context', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      step_number: 1,
      use_dense: false,
      limit: 10,
    });
    const chunkIds = response.results.map((r) => r.chunk_id);
    expect(chunkIds).toEqual([2]);
  });

  it('sets compact true when compact is requested', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      compact: true,
      limit: 10,
    });
    expect(response.compact).toBe(true);
  });

  it('returns compact result summaries with chunk_id and truncated fields', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      compact: true,
      limit: 10,
    });
    expect(response.results[0]).toEqual(
      expect.objectContaining({
        chunk_id: expect.any(Number),
        truncated: expect.any(Boolean),
      }),
    );
  });

  it('attaches metadata to results when include_metadata is true', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      include_metadata: true,
      limit: 10,
    });
    expect(response.results[0].metadata).toEqual(
      expect.objectContaining({ file_path: expect.any(String) }),
    );
  });

  it('returns context as a JSON object with chunks when context_format is json', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      context_format: 'json',
      limit: 10,
    });
    expect(response).toEqual(
      expect.objectContaining({
        context_format: 'json',
        context: expect.objectContaining({
          context: expect.any(String),
          chunks: expect.any(Array),
          tokenCount: expect.any(Number),
        }),
      }),
    );
  });

  it('does not include top_result when no results match the filter', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'no-such-slice',
      use_dense: false,
      read_top_result: true,
      limit: 10,
    });
    expect(response).not.toHaveProperty('top_result');
  });

  it('throws when context_format is unsupported', async () => {
    await expect(
      searchContext({
        databasePath,
        query: 'context slice filter',
        context_format: 'xml',
      }),
    ).rejects.toThrow("context_format must be 'markdown' or 'json'.");
  });

  it('throws when budget is not a positive finite number', async () => {
    await expect(
      searchContext({
        databasePath,
        query: 'context slice filter',
        budget: 0,
      }),
    ).rejects.toThrow('budget must be a positive finite number.');
  });

  it('truncates compact context that exceeds the compact threshold', async () => {
    const response = await searchContext({
      databasePath,
      query: 'longbodytoken',
      slice_id: 'A2-red-tests',
      use_dense: false,
      compact: true,
      budget: 4096,
      limit: 10,
    });
    const isTruncated =
      response.compact === true &&
      typeof response.context === 'string' &&
      response.context.length <= 2000 &&
      response.context.endsWith('\u2026');
    expect(isTruncated).toBe(true);
  });

  it('forwards dense_degraded when the corpus falls back to BM25', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: true,
      limit: 10,
    });
    expect(response).toEqual(
      expect.objectContaining({
        results: expect.arrayContaining([
          expect.objectContaining({ chunk_id: expect.any(Number) }),
        ]),
        dense_degraded: true,
      }),
    );
  });

  it('includes a top_result descriptor when read_top_result is true', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      read_top_result: true,
      limit: 10,
    });
    expect(response.top_result).toEqual(
      expect.objectContaining({
        chunk_id: expect.any(Number),
        text: expect.any(String),
      }),
    );
  });

  // --- helper branch coverage ---

  it('validates the default context format as markdown', () => {
    expect(validateContextFormat(undefined)).toBe('markdown');
  });

  it('validates json as a context format', () => {
    expect(validateContextFormat('json')).toBe('json');
  });

  it('normalizes a search result to a chunk using text', () => {
    const chunk = normalizeSearchResultToChunk({ chunk_id: 1, text: 'body' });
    expect(chunk.body_text).toBe('body');
  });

  it('normalizes a search result to a chunk using body_text fallback', () => {
    const chunk = normalizeSearchResultToChunk({
      chunk_id: 1,
      body_text: 'body2',
    });
    expect(chunk.body_text).toBe('body2');
  });

  it('builds a top result descriptor from a raw result', () => {
    const top = buildTopResult({
      chunk_id: 1,
      file_path: 'a.md',
      family: 'readme',
      text: 't',
    });
    expect(top).toEqual({
      chunk_id: 1,
      file_path: 'a.md',
      family: 'readme',
      text: 't',
    });
  });

  it('returns null from buildTopResult when no result is provided', () => {
    expect(buildTopResult(undefined)).toBeNull();
  });

  it('builds follow-up refs for the first result', () => {
    const refs = buildFollowUpRefs([{ chunk_id: 1 }], 'q');
    expect(refs[0]).toEqual(
      expect.objectContaining({
        tool: 'load_chunk',
        args: { chunk_id: 1, query: 'q' },
      }),
    );
  });

  it('builds follow-up refs for the second result', () => {
    const refs = buildFollowUpRefs([{ chunk_id: 1 }, { chunk_id: 2 }], 'q');
    expect(refs[1]).toEqual(
      expect.objectContaining({
        tool: 'load_chunk',
        args: { chunk_id: 2, query: 'q' },
      }),
    );
  });

  it('includes a related search follow-up ref', () => {
    const refs = buildFollowUpRefs([], 'q');
    expect(refs[0]).toEqual(
      expect.objectContaining({
        tool: 'search_context',
        args: { query: 'Related: q' },
      }),
    );
  });

  it('validates the default budget', () => {
    expect(validateBudget(undefined)).toBe(1024);
  });

  it('validates a custom positive budget', () => {
    expect(validateBudget(2048)).toBe(2048);
  });

  it('uses the searchCorpusFn seam', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
        freshness: { timestamp: 1, stale: false, last_update_source: 'test' },
      }),
    });
    expect(response.results[0].chunk_id).toBe(99);
  });

  it('uses the searchCorpusFn freshness fallback', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
      }),
    });
    expect(response.freshness).toEqual(
      expect.objectContaining({
        stale: expect.any(Boolean),
        last_update_source: expect.any(String),
      }),
    );
  });

  it('reports dense_state none when the corpus response lacks it', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
        freshness: { timestamp: 1, stale: false, last_update_source: 'test' },
      }),
    });
    expect(response.dense_state).toBe('none');
  });

  it('reports rerank_state not_requested when the corpus response lacks it', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
        freshness: { timestamp: 1, stale: false, last_update_source: 'test' },
      }),
    });
    expect(response.rerank_state).toBe('not_requested');
  });

  it('uses the non-exact dedup strategy', async () => {
    const response = await searchContext({
      databasePath,
      query: 'context slice filter',
      slice_id: 'A2-red-tests',
      use_dense: false,
      dedup_strategy: 'fuzzy',
      limit: 5,
    });
    expect(response.dedup_strategy).toBe('fuzzy');
  });

  it('exposes searchContextTool as the same function', () => {
    expect(searchContextTool).toBe(searchContext);
  });

  it('defaults missing options to an empty object', async () => {
    await expect(searchContext()).rejects.toThrow();
  });

  it('normalizes a result that only has text', () => {
    const chunk = normalizeSearchResultToChunk({ text: 'only text' });
    expect(chunk.body_text).toBe('only text');
  });

  it('normalizes a result that only has body_text', () => {
    const chunk = normalizeSearchResultToChunk({ body_text: 'only body' });
    expect(chunk.body_text).toBe('only body');
  });

  it('normalizes a result that has neither text nor body_text', () => {
    const chunk = normalizeSearchResultToChunk({ chunk_id: 1 });
    expect(chunk.body_text).toBe('');
  });

  it('builds top_result text from body_text when present', () => {
    const top = buildTopResult({
      chunk_id: 1,
      file_path: 'a.md',
      family: 'f',
      body_text: 'body',
    });
    expect(top.text).toBe('body');
  });

  it('builds top_result text from text when body_text is absent', () => {
    const top = buildTopResult({
      chunk_id: 1,
      file_path: 'a.md',
      family: 'f',
      text: 'txt',
    });
    expect(top.text).toBe('txt');
  });

  it('builds top_result text as empty fallback', () => {
    const top = buildTopResult({ chunk_id: 1, file_path: 'a.md', family: 'f' });
    expect(top.text).toBe('');
  });

  it('handles a corpus response that omits results', async () => {
    const response = await searchContext({
      databasePath,
      query: 'no results',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        freshness: { timestamp: 1, stale: false, last_update_source: 'test' },
      }),
    });
    expect(response.results).toEqual([]);
  });

  it('passes through dense_state and rerank_state strings', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
        dense_state: 'warm',
        rerank_state: 'warm',
      }),
    });
    expect(response.dense_state).toBe('warm');
  });

  it('coalesces missing metadata fields to null', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      include_metadata: true,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
      }),
    });
    expect(response.results[0].metadata).toMatchObject({
      file_path: null,
      family: null,
      chunk_index: null,
      heading_path: null,
      depth: null,
      parent_chunk_id: null,
      context_header: null,
    });
  });

  it('defaults assembleContext missing fields through the seam', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
        freshness: { timestamp: 1, stale: false, last_update_source: 'test' },
      }),
      assembleContextFn: async () => ({
        selectedChunks: undefined,
        tokenCount: undefined,
      }),
    });
    expect(response.results).toEqual([]);
    expect(response.token_count).toBe(0);
    expect(response.tier_counts).toEqual({
      essential: 0,
      supporting: 0,
      supplementary: 0,
    });
  });

  it('defaults dense_state when the corpus response has a non-string value', async () => {
    const response = await searchContext({
      databasePath,
      query: 'seam query',
      use_dense: false,
      limit: 5,
      searchCorpusFn: async () => ({
        results: [{ chunk_id: 99, text: 'seam result' }],
        dense_state: 123,
        rerank_state: null,
      }),
    });
    expect(response.dense_state).toBe('none');
    expect(response.rerank_state).toBe('not_requested');
  });
});
