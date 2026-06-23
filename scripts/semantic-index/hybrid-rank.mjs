/**
 * Default RRF constant (k). Standard value from the original RRF paper
 * (Cormack, Clarke, Buettcher, 2009).
 * @type {number}
 */
const DEFAULT_RRF_K = 60;

/**
 * Merge BM25 and dense search results using Reciprocal Ranked Fusion (RRF).
 *
 * RRF combines two ranked result lists by assigning each result a score
 * equal to the sum of `1/(k + rank_i)` across every list where the result
 * appears. This avoids the score-scale normalization problems of alpha-blend
 * weighting because RRF only depends on rank positions, not raw scores.
 *
 * Results are merged by `chunk_id` (union of both lists), sorted descending
 * by `rrf_score`, with ties broken by higher original BM25 score.
 *
 * @param {object} options - RRF merge options.
 * @param {Array} options.bm25Results - BM25 results sorted by descending score.
 * @param {Array} options.denseResults - Dense results sorted by ascending distance.
 * @param {number} [options.k=60] - RRF constant (standard default: 60).
 * @returns {Array} Merged results sorted by descending `rrf_score`, each
 *   carrying the original result fields plus `rrf_score` and `bm25_score`.
 */
export function rankRRFResults(options = {}) {
  const k = Number(options.k ?? DEFAULT_RRF_K);
  const rawBm25Results = Array.isArray(options.bm25Results)
    ? options.bm25Results
    : [];
  const rawDenseResults = Array.isArray(options.denseResults)
    ? options.denseResults
    : [];

  // Sort BM25 by descending score and dense by ascending distance before
  // assigning rank positions. This ensures rank reflects retrieval quality.
  const bm25Results = rawBm25Results.toSorted(
    (left, right) => Number(right.score ?? 0) - Number(left.score ?? 0),
  );
  const denseResults = rawDenseResults.toSorted(
    (left, right) => Number(left.distance ?? 0) - Number(right.distance ?? 0),
  );

  // Build rank maps: chunk_id → 0-indexed rank position.
  const bm25RankByChunkId = new Map(
    bm25Results.map((result, index) => [result.chunk_id, index]),
  );
  const denseRankByChunkId = new Map(
    denseResults.map((result, index) => [result.chunk_id, index]),
  );

  // Build BM25 score lookup for tie-breaking (uses original score field).
  const bm25ScoreByChunkId = new Map(
    bm25Results.map((result) => [result.chunk_id, Number(result.score ?? 0)]),
  );

  // Union all chunk_ids from both result lists.
  const allChunkIds = new Set([
    ...bm25RankByChunkId.keys(),
    ...denseRankByChunkId.keys(),
  ]);

  // Compute RRF score for each chunk in the union.
  const merged = [...allChunkIds].map((chunkId) => {
    let rrfScore = 0;
    const bm25Rank = bm25RankByChunkId.get(chunkId);
    if (bm25Rank !== undefined) {
      rrfScore += 1 / (k + bm25Rank);
    }
    const denseRank = denseRankByChunkId.get(chunkId);
    if (denseRank !== undefined) {
      rrfScore += 1 / (k + denseRank);
    }

    // Merge fields from whichever list contains this chunk (BM25 first).
    const bm25Result = bm25Results.find((r) => r.chunk_id === chunkId);
    const denseResult = denseResults.find((r) => r.chunk_id === chunkId);
    const base = bm25Result ?? denseResult;

    return {
      ...base,
      chunk_id: chunkId,
      rrf_score: rrfScore,
      bm25_score: bm25ScoreByChunkId.get(chunkId) ?? 0,
    };
  });

  // Sort descending by rrf_score; tie-break by higher original BM25 score.
  return merged.toSorted((left, right) => {
    if (right.rrf_score !== left.rrf_score) {
      return right.rrf_score - left.rrf_score;
    }
    return (right.bm25_score ?? 0) - (left.bm25_score ?? 0);
  });
}

export function computeCosineSimilarity(leftVectorLike, rightVectorLike) {
  const leftVector = toFloat32Array(leftVectorLike);
  const rightVector = toFloat32Array(rightVectorLike);
  if (
    leftVector.length === 0 ||
    rightVector.length === 0 ||
    leftVector.length !== rightVector.length
  ) {
    return 0;
  }

  let dotProduct = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;

  for (let valueIndex = 0; valueIndex < leftVector.length; valueIndex += 1) {
    const leftValue = leftVector[valueIndex];
    const rightValue = rightVector[valueIndex];
    dotProduct += leftValue * rightValue;
    leftMagnitude += leftValue * leftValue;
    rightMagnitude += rightValue * rightValue;
  }

  if (leftMagnitude === 0 || rightMagnitude === 0) return 0;
  return dotProduct / Math.sqrt(leftMagnitude * rightMagnitude);
}

function toFloat32Array(vectorLike) {
  if (vectorLike instanceof Float32Array) return vectorLike;
  if (ArrayBuffer.isView(vectorLike)) {
    return new Float32Array(
      vectorLike.buffer.slice(
        vectorLike.byteOffset,
        vectorLike.byteOffset + vectorLike.byteLength,
      ),
    );
  }
  return Float32Array.from(Array.isArray(vectorLike) ? vectorLike : []);
}
