export const DEFAULT_HYBRID_ALPHA = 0.5;

export function rankHybridResults(options = {}) {
  const alpha = normalizeAlpha(options.alpha);
  const queryEmbedding = toFloat32Array(options.queryEmbedding ?? []);
  const candidates = Array.isArray(options.candidates)
    ? options.candidates
    : [];
  const normalizedBm25Scores = normalizeBm25Scores(candidates);

  return candidates
    .map((candidate, candidateIndex) => {
      const candidateEmbedding = toFloat32Array(candidate.embedding ?? []);
      const cosineScore = computeCosineSimilarity(
        queryEmbedding,
        candidateEmbedding,
      );
      const bm25Score = normalizedBm25Scores[candidateIndex] ?? 0;
      return {
        ...candidate,
        cosine_score: cosineScore,
        normalized_bm25_score: bm25Score,
        score: alpha * bm25Score + (1 - alpha) * cosineScore,
      };
    })
    .toSorted(
      (leftCandidate, rightCandidate) =>
        rightCandidate.score - leftCandidate.score,
    );
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

function normalizeBm25Scores(candidates) {
  if (candidates.length === 0) return [];

  const scores = candidates.map((candidate) =>
    Number(candidate.bm25_score ?? 0),
  );
  const minimumScore = Math.min(...scores);
  const maximumScore = Math.max(...scores);
  if (maximumScore === minimumScore) return scores.map(() => 1);

  return scores.map(
    (score) => (score - minimumScore) / (maximumScore - minimumScore),
  );
}

function normalizeAlpha(alpha) {
  const numericAlpha = Number(alpha ?? DEFAULT_HYBRID_ALPHA);
  if (!Number.isFinite(numericAlpha)) return DEFAULT_HYBRID_ALPHA;
  return Math.min(Math.max(numericAlpha, 0), 1);
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
