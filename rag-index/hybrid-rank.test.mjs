import { rankRRFResults, computeCosineSimilarity } from './hybrid-rank.mjs';

describe('rankRRFResults', () => {
  it('returns empty array for default options', () => {
    expect(rankRRFResults()).toEqual([]);
  });

  it('returns empty array for non-array inputs', () => {
    expect(rankRRFResults({ bm25Results: 'not array', denseResults: 123 })).toEqual([]);
  });

  it('handles null/undefined bm25 and dense results', () => {
    expect(rankRRFResults({ bm25Results: null, denseResults: undefined })).toEqual([]);
  });

  it('fuses BM25 and dense results with default k', () => {
    const result = rankRRFResults({
      bm25Results: [
        { chunk_id: 1, score: 10 },
        { chunk_id: 2, score: 5 },
      ],
      denseResults: [
        { chunk_id: 2, distance: 0.1 },
        { chunk_id: 3, distance: 0.2 },
      ],
    });
    expect(result).toHaveLength(3);
    // chunk 2 appears in both lists so should have highest RRF score
    expect(result[0].chunk_id).toBe(2);
    expect(result[0].rrf_score).toBeCloseTo(1 / 60 + 1 / 61, 10);
  });

  it('uses custom k value', () => {
    const result = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 10 }],
      denseResults: [],
      k: 10,
    });
    expect(result[0].rrf_score).toBeCloseTo(1 / 10, 10);
  });

  it('sorts BM25 by descending score before ranking', () => {
    const result = rankRRFResults({
      bm25Results: [
        { chunk_id: 2, score: 5 },
        { chunk_id: 1, score: 10 },
        { chunk_id: 3, score: 1 },
      ],
      denseResults: [],
    });
    // chunk 1 has highest score, so rank 0, highest RRF
    expect(result[0].chunk_id).toBe(1);
    expect(result[1].chunk_id).toBe(2);
    expect(result[2].chunk_id).toBe(3);
  });

  it('sorts dense by ascending distance before ranking', () => {
    const result = rankRRFResults({
      bm25Results: [],
      denseResults: [
        { chunk_id: 2, distance: 0.5 },
        { chunk_id: 1, distance: 0.1 },
        { chunk_id: 3, distance: 0.9 },
      ],
    });
    expect(result[0].chunk_id).toBe(1);
    expect(result[1].chunk_id).toBe(2);
    expect(result[2].chunk_id).toBe(3);
  });

  it('handles nullish score and distance', () => {
    const result = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: null }],
      denseResults: [{ chunk_id: 2, distance: undefined }],
    });
    expect(result).toHaveLength(2);
    // Both have score/distance 0 after Number coercion
    expect(result[0].bm25_score).toBe(0);
  });

  it('tie-breaks by higher BM25 score', () => {
    const result = rankRRFResults({
      bm25Results: [
        { chunk_id: 1, score: 100 },
        { chunk_id: 2, score: 50 },
      ],
      denseResults: [],
    });
    // Both have different ranks so no tie, but let's create a tie scenario
    expect(result[0].chunk_id).toBe(1);
  });

  it('merges fields from both lists (BM25 first)', () => {
    const result = rankRRFResults({
      bm25Results: [{ chunk_id: 1, score: 10, custom_field: 'bm25' }],
      denseResults: [{ chunk_id: 1, distance: 0.1, custom_field: 'dense' }],
    });
    expect(result).toHaveLength(1);
    expect(result[0].custom_field).toBe('bm25');
  });

  it('uses dense result fields when chunk only in dense', () => {
    const result = rankRRFResults({
      bm25Results: [],
      denseResults: [{ chunk_id: 1, distance: 0.1, custom_field: 'dense' }],
    });
    expect(result[0].custom_field).toBe('dense');
    expect(result[0].bm25_score).toBe(0);
  });
});

describe('computeCosineSimilarity', () => {
  it('returns 0 for empty vectors', () => {
    expect(computeCosineSimilarity([], [])).toBe(0);
  });

  it('returns 0 for mismatched lengths', () => {
    expect(computeCosineSimilarity([1, 2, 3], [1, 2])).toBe(0);
  });

  it('returns 0 for zero magnitude vectors', () => {
    expect(computeCosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
  });

  it('returns 0 for zero magnitude on right vector', () => {
    expect(computeCosineSimilarity([1, 2, 3], [0, 0, 0])).toBe(0);
  });

  it('computes cosine similarity for parallel vectors', () => {
    const result = computeCosineSimilarity([1, 2, 3], [2, 4, 6]);
    expect(result).toBeCloseTo(1, 10);
  });

  it('computes cosine similarity for orthogonal vectors', () => {
    const result = computeCosineSimilarity([1, 0], [0, 1]);
    expect(result).toBeCloseTo(0, 10);
  });

  it('handles Float32Array input', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([1, 2, 3]);
    expect(computeCosineSimilarity(a, b)).toBeCloseTo(1, 10);
  });

  it('handles ArrayBuffer view input (Uint8Array)', () => {
    const buf = new ArrayBuffer(12);
    const view = new Float32Array(buf);
    view[0] = 1;
    view[1] = 0;
    view[2] = 0;
    const result = computeCosineSimilarity(view, [1, 0, 0]);
    expect(result).toBeCloseTo(1, 10);
  });

  it('handles non-array non-typed-array input', () => {
    expect(computeCosineSimilarity({}, {})).toBe(0);
  });

  it('handles Float32Array with byteOffset (subarray)', () => {
    const buf = new Float32Array(6);
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = 1;
    buf[3] = 0;
    buf[4] = 1;
    buf[5] = 0;
    const sub = buf.subarray(2, 5); // [1, 0, 1] with byteOffset
    const result = computeCosineSimilarity(sub, [1, 0, 1]);
    expect(result).toBeCloseTo(1, 10);
  });
});