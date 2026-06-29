/**
 * @module rrf.test
 * @description Red tests for Phase 5 Step 02 — server-side RRF hybrid search.
 *
 * These tests define the EXPECTED behavior AFTER implementation. They must FAIL
 * because the RRF implementation (`rankRRFResults`) does not exist yet and the
 * alpha-blend code has not yet been removed.
 *
 * Coverage targets:
 * - RRF formula 1/(k+rank) applied correctly (default k=60)
 * - FTS + vector results merged by RRF score
 * - Tie-breaking by original score
 * - Alpha blend removed from BOTH hybrid-rank.mjs and query-dense.mjs
 * - computeCosineSimilarity() preserved for assemble-context.mjs
 * - No deferred cleanup: old alpha blend code deleted in this step
 * - JS-side alpha blend fully removed (no fallback, no dual-path code)
 *
 * Pure .mjs test — runs via Jest ESM project `semantic-index-mjs`.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HYBRID_RANK_PATH = path.resolve(__dirname, '..', 'hybrid-rank.mjs');
const QUERY_DENSE_PATH = path.resolve(__dirname, '..', 'query-dense.mjs');

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
// Group 1: RRF ranking function exists and produces correct scores
// ---------------------------------------------------------------------------

describe('RRF: rankRRFResults exists and produces correct scores', () => {
  it('returns 1/(60+0) for a BM25-only result at rank 0', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 5 }],
      denseResults: [],
    });
    expect(results[0].rrf_score).toBeCloseTo(1 / 60, 6);
  });

  it('returns 1/60 + 1/61 for a result in both lists (BM25 rank 0, dense rank 1)', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 5 }],
      denseResults: [
        { chunk_id: 1, distance: 0.3 },
        { chunk_id: 2, distance: 0.1 },
      ],
    });
    const chunk1 = results.find((result) => result.chunk_id === 1);
    expect(chunk1.rrf_score).toBeCloseTo(1 / 60 + 1 / 61, 6);
  });

  it('uses default k=60 when not specified', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 5 }],
      denseResults: [],
    });
    expect(results[0].rrf_score).toBeCloseTo(1 / (60 + 0), 6);
  });

  it('honors custom k parameter (k=30 produces 1/(30+0) at rank 0)', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 5 }],
      denseResults: [],
      k: 30,
    });
    expect(results[0].rrf_score).toBeCloseTo(1 / (30 + 0), 6);
  });

  it('sorts results descending by rrf_score', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [
        { chunk_id: 1, score: 5 },
        { chunk_id: 2, score: 3 },
      ],
      denseResults: [
        { chunk_id: 2, distance: 0.1 },
        { chunk_id: 3, distance: 0.2 },
      ],
    });
    expect(results[0].chunk_id).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Group 2: FTS + vector results merged by RRF score
// ---------------------------------------------------------------------------

describe('RRF: FTS + vector results merged by RRF score', () => {
  it('gives a dense-only result RRF contribution only from its dense rank', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 5 }],
      denseResults: [
        { chunk_id: 2, distance: 0.1 },
        { chunk_id: 1, distance: 0.3 },
      ],
    });
    const chunk2 = results.find((result) => result.chunk_id === 2);
    expect(chunk2.rrf_score).toBeCloseTo(1 / (60 + 0), 6);
  });

  it('includes all chunk_ids from the union of BM25 and dense results', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    const results = rankRRFResults({
      bm25Results: [{ chunk_id: 1 }, { chunk_id: 2 }],
      denseResults: [{ chunk_id: 2 }, { chunk_id: 3 }],
    });
    expect(results.map((result) => result.chunk_id).sort()).toEqual([1, 2, 3]);
  });
});

// ---------------------------------------------------------------------------
// Group 3: Tie-breaking by original score
// ---------------------------------------------------------------------------

describe('RRF: tie-breaking by original score', () => {
  it('breaks ties by higher original BM25 score when RRF scores are equal', async () => {
    const { rankRRFResults } = await import('../hybrid-rank.mjs');
    // chunk_id=1: BM25 rank 0 (1/60) + dense rank 1 (1/61)
    // chunk_id=2: BM25 rank 1 (1/61) + dense rank 0 (1/60)
    // → identical RRF scores. Tie-break: higher BM25 score wins (10 > 5).
    const results = rankRRFResults({
      bm25Results: [
        { chunk_id: 1, score: 10 },
        { chunk_id: 2, score: 5 },
      ],
      denseResults: [
        { chunk_id: 2, distance: 0.1 },
        { chunk_id: 1, distance: 0.2 },
      ],
    });
    expect(results[0].chunk_id).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Group 4: Alpha-blend code is ABSENT (source-pattern assertions)
// ---------------------------------------------------------------------------

describe('RRF: alpha-blend code removed from hybrid-rank.mjs', () => {
  it('hybrid-rank.mjs does not contain "alpha * bm25Score"', () => {
    const source = readSource(HYBRID_RANK_PATH);
    expect(source).not.toMatch(/alpha\s*\*\s*bm25Score/);
  });

  it('hybrid-rank.mjs does not contain "(1 - alpha) * cosineScore"', () => {
    const source = readSource(HYBRID_RANK_PATH);
    expect(source).not.toMatch(/\(\s*1\s*-\s*alpha\s*\)\s*\*\s*cosineScore/);
  });

  it('hybrid-rank.mjs does not export DEFAULT_HYBRID_ALPHA', () => {
    const source = readSource(HYBRID_RANK_PATH);
    expect(source).not.toMatch(/DEFAULT_HYBRID_ALPHA/);
  });
});

describe('RRF: alpha-blend code removed from query-dense.mjs', () => {
  it('query-dense.mjs does not contain "alpha * bm25Score + (1 - alpha) * cosineScore"', () => {
    const source = readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(
      /alpha\s*\*\s*bm25Score\s*\+\s*\(\s*1\s*-\s*alpha\s*\)\s*\*\s*cosineScore/,
    );
  });

  it('query-dense.mjs does not import DEFAULT_HYBRID_ALPHA from hybrid-rank.mjs', () => {
    const source = readSource(QUERY_DENSE_PATH);
    expect(source).not.toMatch(
      /import\s*\{[^}]*DEFAULT_HYBRID_ALPHA[^}]*\}\s*from\s*['"][^'"]*hybrid-rank\.mjs['"]/,
    );
  });
});

// ---------------------------------------------------------------------------
// Group 5: computeCosineSimilarity is PRESERVED (source-pattern + functional)
// ---------------------------------------------------------------------------

describe('RRF: computeCosineSimilarity is preserved', () => {
  it('hybrid-rank.mjs source exports computeCosineSimilarity', () => {
    const source = readSource(HYBRID_RANK_PATH);
    expect(source).toMatch(/export\s+function\s+computeCosineSimilarity/);
  });

  it('computeCosineSimilarity returns 0 for empty vectors', async () => {
    const { computeCosineSimilarity } = await import('../hybrid-rank.mjs');
    expect(computeCosineSimilarity([], [])).toBe(0);
  });

  it('computeCosineSimilarity returns 1 for identical unit vectors', async () => {
    const { computeCosineSimilarity } = await import('../hybrid-rank.mjs');
    expect(computeCosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1, 6);
  });
});
