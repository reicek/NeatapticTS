/**
 * Normalize one numeric vector with min-max scaling.
 *
 * Degenerate vectors with a single value or zero range return uniform weights so
 * downstream focus math never emits `NaN`.
 *
 * @param values - Raw numeric vector to normalize.
 * @returns A normalized vector in the `[0, 1]` range or uniform degenerate weights.
 */
export function minMaxNormalize(values: readonly number[]): number[] {
  const lowestValue = Math.min(...values);
  const highestValue = Math.max(...values);
  const valueRange = highestValue - lowestValue;

  if (values.length === 1 || valueRange === 0) {
    return values.map(() => 1 / values.length);
  }

  return values.map((value) => (value - lowestValue) / valueRange);
}

/**
 * Return the top-k items after softmax normalization of the `score` field.
 *
 * @param items - Candidate items carrying one scalar score.
 * @param k - Maximum number of items to return.
 * @returns The highest-ranked `k` items with stable tie-breaking by input index.
 */
export function softmaxTopK<T extends { score: number }>(
  items: readonly T[],
  k: number,
): T[] {
  const indexedItems = items.map((item, originalIndex) => ({
    item,
    originalIndex,
  }));
  const highestScore = indexedItems.reduce(
    (currentHighestScore, { item }) =>
      Math.max(currentHighestScore, item.score),
    Number.NEGATIVE_INFINITY,
  );
  const exponentiatedScores = indexedItems.map(({ item }) =>
    Math.exp(item.score - highestScore),
  );
  const totalExponentiatedScore = exponentiatedScores.reduce(
    (currentTotal, exponentiatedScore) => currentTotal + exponentiatedScore,
    0,
  );

  return indexedItems
    .map(({ item, originalIndex }, entryIndex) => ({
      item,
      originalIndex,
      normalizedScore:
        exponentiatedScores[entryIndex] / totalExponentiatedScore,
    }))
    .toSorted((leftEntry, rightEntry) => {
      const scoreDelta = rightEntry.normalizedScore - leftEntry.normalizedScore;

      if (scoreDelta !== 0) {
        return scoreDelta;
      }

      return leftEntry.originalIndex - rightEntry.originalIndex;
    })
    .slice(0, Math.max(0, k))
    .map(({ item }) => item);
}
