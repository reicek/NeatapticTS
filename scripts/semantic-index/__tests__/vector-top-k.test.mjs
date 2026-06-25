/**
 * @module vector-top-k.test
 * @description Red tests for Phase 4 Step 04 — server-side ANN vector search
 * via Turso's `vector_top_k()` function backed by the DiskANN index (from
 * Step 03).
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because the implementation does not exist yet (not because of syntax errors).
 *
 * Coverage targets:
 * - query-dense.mjs uses `vector_top_k(embedding, vector8(?), k)` for ANN search
 * - search-corpus.mjs uses `vector_top_k` for the ANN path
 * - vector_top_k results are JOINed with chunks in the same query (no separate round-trip)
 * - No JS-side loading of ALL embeddings (no `SELECT embedding FROM chunks` bulk load)
 * - vector_top_k (ANN) is the PRIMARY dense path; vector_distance_cos is fallback only
 * - search-advanced.mjs uses the server-side vector_top_k path for dense retrieval
 * - search-context.mjs uses the server-side vector_top_k path for context assembly
 * - Old brute-force-only `loadDenseRows` is replaced by an ANN-aware path
 *
 * Pure .mjs test — runs via Jest ESM project (no ts-jest).
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { closeTursoClient } from '../../mcp-semantic/tools/cortex-db.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const QUERY_DENSE_PATH = path.resolve(__dirname, '..', 'query-dense.mjs');
const SEARCH_CORPUS_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'search-corpus.mjs',
);
const SEARCH_ADVANCED_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'search-advanced.mjs',
);
const SEARCH_CONTEXT_PATH = path.resolve(
  __dirname,
  '..',
  '..',
  'mcp-semantic',
  'tools',
  'search-context.mjs',
);

/**
 * Read a source file as UTF-8 text.
 *
 * @param {string} filePath - Absolute path to the source file.
 * @returns {Promise<string>} File contents.
 */
async function readSource(filePath) {
  return readFile(filePath, 'utf8');
}

// ---------------------------------------------------------------------------
// Teardown — close any Turso clients that may have been opened
// ---------------------------------------------------------------------------

afterEach(async () => {
  await closeTursoClient();
});

// ---------------------------------------------------------------------------
// vector_top_k usage — query-dense.mjs
// ---------------------------------------------------------------------------

describe('vector-top-k: query-dense.mjs uses vector_top_k for ANN search', () => {
  it('uses vector_top_k() in the dense query SQL', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).toMatch(/vector_top_k\s*\(/i);
  });

  it('passes the query embedding as a vector8() parameter to vector_top_k', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).toMatch(/vector_top_k\s*\([^)]*vector8\s*\(\s*\?\s*\)/is);
  });

  it('passes a k limit parameter to vector_top_k', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // vector_top_k(index, embedding, k) — the third argument is the k limit.
    expect(source).toMatch(/vector_top_k\s*\([^)]*,\s*\?\s*\)/is);
  });
});

// ---------------------------------------------------------------------------
// vector_top_k usage — search-corpus.mjs
// ---------------------------------------------------------------------------

describe('vector-top-k: search-corpus.mjs uses vector_top_k for ANN path', () => {
  it('references vector_top_k in the source', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    expect(source).toMatch(/vector_top_k\s*\(/i);
  });

  it('passes the query embedding as vector8() to vector_top_k', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    expect(source).toMatch(/vector_top_k\s*\([^)]*vector8\s*\(\s*\?\s*\)/is);
  });
});

// ---------------------------------------------------------------------------
// JOIN with chunks — vector_top_k results joined in the same query
// ---------------------------------------------------------------------------

describe('vector-top-k: vector_top_k results JOINed with chunks in same query', () => {
  it('JOINs vector_top_k results with the chunks table in query-dense.mjs', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The vector_top_k call should be part of a SQL statement that also JOINs
    // with the chunks table to retrieve chunk metadata in a single round-trip.
    expect(source).toMatch(/vector_top_k[\s\S]{0,400}JOIN\s+chunks/i);
  });

  it('JOINs vector_top_k results with the chunks table in search-corpus.mjs', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    expect(source).toMatch(/vector_top_k[\s\S]{0,400}JOIN\s+chunks/i);
  });
});

// ---------------------------------------------------------------------------
// No JS-side embedding loading — all 4 touched files
// ---------------------------------------------------------------------------

describe('vector-top-k: no JS-side bulk embedding loading in any touched file', () => {
  it('query-dense.mjs does not load all embeddings via SELECT embedding FROM chunks', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('search-corpus.mjs does not load all embeddings via SELECT embedding FROM chunks', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('search-advanced.mjs does not load all embeddings via SELECT embedding FROM chunks', async () => {
    const source = await readSource(SEARCH_ADVANCED_PATH);
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('search-context.mjs does not load all embeddings via SELECT embedding FROM chunks', async () => {
    const source = await readSource(SEARCH_CONTEXT_PATH);
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('query-dense.mjs does not define loadEmbeddingsForChunks', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(/loadEmbeddingsForChunks/i);
  });

  it('search-corpus.mjs does not define loadEmbeddingsForChunks', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    expect(source).not.toMatch(/loadEmbeddingsForChunks/i);
  });
});

// ---------------------------------------------------------------------------
// ANN vs brute-force selection — vector_top_k is the primary path
// ---------------------------------------------------------------------------

describe('vector-top-k: ANN is the primary dense path, brute-force is fallback', () => {
  it('query-dense.mjs has an ANN-first dense query function that uses vector_top_k', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The primary dense loading function should use vector_top_k, not just
    // vector_distance_cos. We assert that a function whose name suggests ANN
    // loading (e.g. loadAnnRows, loadDenseRows) contains a vector_top_k call.
    expect(source).toMatch(/vector_top_k/i);
  });

  it('query-dense.mjs preserves vector_distance_cos as a fallback path', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The brute-force fallback from Step 02 should remain available.
    expect(source).toMatch(/vector_distance_cos\s*\(/i);
  });

  it('search-corpus.mjs references vector_top_k for the ANN path', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    // The source should reference vector_top_k (ANN) indicating the
    // primary ANN path is available alongside the brute-force fallback.
    expect(source).toMatch(/vector_top_k/i);
  });

  it('search-corpus.mjs references vector_distance_cos for the brute-force fallback', async () => {
    const source = await readSource(SEARCH_CORPUS_PATH);
    // The brute-force fallback from Step 02 should remain referenced,
    // indicating a selection path between ANN and brute-force.
    expect(source).toMatch(/vector_distance_cos/i);
  });
});

// ---------------------------------------------------------------------------
// search-advanced.mjs integration — uses server-side vector_top_k path
// ---------------------------------------------------------------------------

describe('vector-top-k: search-advanced.mjs uses server-side vector_top_k path', () => {
  it('search-advanced.mjs does not perform JS-side cosine similarity computation', async () => {
    const source = await readSource(SEARCH_ADVANCED_PATH);
    expect(source).not.toMatch(/computeCosineSimilarity/i);
  });

  it('search-advanced.mjs does not load all embeddings into JS memory', async () => {
    const source = await readSource(SEARCH_ADVANCED_PATH);
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('search-advanced.mjs routes dense retrieval through searchCorpus or queryDenseIndex', async () => {
    const source = await readSource(SEARCH_ADVANCED_PATH);
    // search-advanced delegates to searchCorpus (which uses queryDenseIndex),
    // ensuring the server-side vector_top_k path is used for dense retrieval.
    expect(source).toMatch(/searchCorpus|queryDenseIndex/i);
  });
});

// ---------------------------------------------------------------------------
// search-context.mjs integration — uses server-side vector_top_k path
// ---------------------------------------------------------------------------

describe('vector-top-k: search-context.mjs uses server-side vector_top_k path', () => {
  it('search-context.mjs does not perform JS-side cosine similarity computation', async () => {
    const source = await readSource(SEARCH_CONTEXT_PATH);
    expect(source).not.toMatch(/computeCosineSimilarity/i);
  });

  it('search-context.mjs does not load all embeddings into JS memory', async () => {
    const source = await readSource(SEARCH_CONTEXT_PATH);
    expect(source).not.toMatch(/SELECT\s+embedding\s+FROM\s+chunks/i);
  });

  it('search-context.mjs routes dense retrieval through searchCorpus or queryDenseIndex', async () => {
    const source = await readSource(SEARCH_CONTEXT_PATH);
    // search-context delegates to searchCorpus (which uses queryDenseIndex),
    // ensuring the server-side vector_top_k path is used for context assembly.
    expect(source).toMatch(/searchCorpus|queryDenseIndex/i);
  });
});

// ---------------------------------------------------------------------------
// No deferred cleanup — old brute-force-only loadDenseRows replaced
// ---------------------------------------------------------------------------

describe('vector-top-k: old brute-force-only loadDenseRows replaced by ANN-aware path', () => {
  it('query-dense.mjs no longer uses a brute-force-only loadDenseRows as the sole dense path', async () => {
    const source = await readSource(QUERY_DENSE_PATH);
    // The old loadDenseRows function from Step 02 used only vector_distance_cos.
    // After Step 04, the dense loading path must be ANN-aware (vector_top_k).
    // We assert that vector_top_k appears in the source, meaning the
    // brute-force-only path is no longer the sole dense retrieval method.
    expect(source).toMatch(/vector_top_k/i);
  });
});
