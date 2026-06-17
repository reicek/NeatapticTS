/**
 * @module eval-compare
 * @description A/B comparison framework for Repo Cortex RAG evaluation results.
 *
 * Provides side-by-side condition comparison, a Wilcoxon signed-rank test for
 * paired per-query differences, and an alpha sweep helper.
 *
 * @example
 * ```js
 * import { compareResults, alphaSweep } from './eval-compare.mjs';
 *
 * const comparison = compareResults(resultA, resultB);
 * const sweep = await alphaSweep({ queries, alphas: [0, 0.5, 1] });
 * ```
 */

import { runEval } from './eval-runner.mjs';

/**
 * Compute the standard normal CDF using a rational approximation.
 *
 * @param {number} x
 * @returns {number}
 */
export function normalCdf(x) {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x >= 0 ? 1 : -1;
  const absX = Math.abs(x) / Math.sqrt(2);
  const t = 1 / (1 + p * absX);
  const y =
    1 -
    ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);
  return 0.5 * (1 + sign * y);
}

/**
 * Compute a two-tailed p-value from a normal z-score.
 *
 * @param {number} z
 * @returns {number}
 */
function twoTailedPValue(z) {
  return 2 * (1 - normalCdf(Math.abs(z)));
}

/**
 * Wilcoxon signed-rank test for paired samples.
 *
 * Uses the normal approximation with a continuity correction. Ties (zero
 * differences) are excluded from ranking, and the remaining ranks are assigned
 * using average ranks for tied absolute differences.
 *
 * @param {number[]} seriesA
 * @param {number[]} seriesB
 * @returns {{p_value: number, significant: boolean, statistic: number}}
 */
export function wilcoxonSignedRankTest(seriesA, seriesB) {
  if (!Array.isArray(seriesA) || !Array.isArray(seriesB)) {
    throw new Error('wilcoxonSignedRankTest requires two numeric arrays.');
  }
  if (seriesA.length !== seriesB.length) {
    throw new Error('Wilcoxon test requires paired samples of equal length.');
  }
  if (seriesA.length === 0) {
    return { p_value: 1, significant: false, statistic: 0 };
  }

  const pairs = seriesA
    .map((value, index) => ({
      a: value,
      b: seriesB[index],
      diff: value - seriesB[index],
    }))
    .filter((pair) => pair.diff !== 0);

  if (pairs.length === 0) {
    return { p_value: 1, significant: false, statistic: 0 };
  }

  const ranked = pairs
    .map((pair) => ({ ...pair, abs: Math.abs(pair.diff) }))
    .toSorted((a, b) => a.abs - b.abs);

  // Average-rank assignment for ties.
  const rankedWithRank = [];
  let index = 0;
  while (index < ranked.length) {
    let tieEnd = index;
    while (
      tieEnd < ranked.length - 1 &&
      ranked[tieEnd].abs === ranked[tieEnd + 1].abs
    ) {
      tieEnd += 1;
    }
    const averageRank = (index + 1 + tieEnd + 1) / 2;
    for (let tieIndex = index; tieIndex <= tieEnd; tieIndex += 1) {
      rankedWithRank.push({ ...ranked[tieIndex], rank: averageRank });
    }
    index = tieEnd + 1;
  }

  let positiveRankSum = 0;
  let negativeRankSum = 0;
  for (const pair of rankedWithRank) {
    if (pair.diff > 0) positiveRankSum += pair.rank;
    else negativeRankSum += pair.rank;
  }

  const statistic = Math.min(positiveRankSum, negativeRankSum);
  const n = pairs.length;
  const mean = (n * (n + 1)) / 4;
  const variance = (n * (n + 1) * (2 * n + 1)) / 24;
  const stdDev = Math.sqrt(variance);
  const z = (statistic - mean + 0.5 * Math.sign(statistic - mean)) / stdDev;
  const pValue = twoTailedPValue(z);

  return {
    p_value: Math.max(0, Math.min(1, pValue)),
    significant: pValue < 0.05,
    statistic,
  };
}

/**
 * Format a delta as a signed fixed string.
 *
 * @param {number} delta
 * @returns {string}
 */
function formatDelta(delta) {
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${delta.toFixed(3)}`;
}

/**
 * Compare per-query results to count wins for each condition.
 *
 * @param {object[]} perQueryA
 * @param {object[]} perQueryB
 * @param {string} metricName
 * @returns {{a_wins: number, b_wins: number, ties: number}}
 */
function countWins(perQueryA, perQueryB, metricName) {
  let aWins = 0;
  let bWins = 0;
  let ties = 0;
  const length = Math.min(perQueryA.length, perQueryB.length);
  for (let index = 0; index < length; index += 1) {
    const aValue = perQueryA[index]?.[metricName] ?? 0;
    const bValue = perQueryB[index]?.[metricName] ?? 0;
    if (aValue > bValue) aWins += 1;
    else if (bValue > aValue) bWins += 1;
    else ties += 1;
  }
  return { a_wins: aWins, b_wins: bWins, ties };
}

/**
 * Build per-class delta summary between two conditions.
 *
 * @param {Record<string, object>} perClassA
 * @param {Record<string, object>} perClassB
 * @returns {Record<string, object>}
 */
function buildPerClassDelta(perClassA, perClassB) {
  const classes = new Set([
    ...Object.keys(perClassA ?? {}),
    ...Object.keys(perClassB ?? {}),
  ]);
  const delta = {};
  for (const className of [...classes].toSorted()) {
    const aValue = perClassA?.[className]?.mrr_at_5 ?? 0;
    const bValue = perClassB?.[className]?.mrr_at_5 ?? 0;
    delta[className] = {
      mrr_at_5_delta: formatDelta(bValue - aValue),
    };
  }
  return delta;
}

/**
 * Compare two eval result sets side-by-side.
 *
 * @param {object} resultA
 * @param {object} resultB
 * @returns {object} Comparison with per-metric deltas and significance.
 * @throws {Error} When inputs are invalid.
 */
export function compareResults(resultA, resultB) {
  if (!resultA || typeof resultA !== 'object') {
    throw new Error('resultA is required.');
  }
  if (!resultB || typeof resultB !== 'object') {
    throw new Error('resultB is required.');
  }

  const comparisonId = `comparison-${new Date().toISOString()}`;
  const perQueryA = Array.isArray(resultA.per_query) ? resultA.per_query : [];
  const perQueryB = Array.isArray(resultB.per_query) ? resultB.per_query : [];

  const metrics = {};
  const metricNames = ['mrr_at_5', 'ndcg_at_5', 'recall_at_5'];
  for (const metricName of metricNames) {
    const aValue = Number(resultA.metrics?.[metricName] ?? 0);
    const bValue = Number(resultB.metrics?.[metricName] ?? 0);
    const seriesA = perQueryA.map((row) => Number(row?.[metricName] ?? 0));
    const seriesB = perQueryB.map((row) => Number(row?.[metricName] ?? 0));
    const test = wilcoxonSignedRankTest(seriesA, seriesB);
    metrics[metricName] = {
      a: aValue,
      b: bValue,
      delta: formatDelta(bValue - aValue),
      significant: test.significant,
      p_value: test.p_value,
    };
  }

  const perQueryWins = countWins(perQueryA, perQueryB, 'mrr_at_5');

  return {
    comparison_id: comparisonId,
    condition_a: resultA.condition,
    condition_b: resultB.condition,
    query_count: Math.min(perQueryA.length, perQueryB.length),
    metrics,
    per_query_wins: perQueryWins,
    per_class_delta: buildPerClassDelta(resultA.per_class, resultB.per_class),
  };
}

/**
 * Run the hybrid condition across a set of alpha values and return the
 * resulting metrics for each alpha.
 *
 * @param {object} options
 * @param {object[]} options.queries - Eval query specifications.
 * @param {number[]} options.alphas - Alpha values to sweep.
 * @param {string} [options.condition='hybrid']
 * @returns {Promise<Array>}
 */
export async function alphaSweep(options = {}) {
  const queries = Array.isArray(options.queries) ? options.queries : [];
  const alphas = Array.isArray(options.alphas) ? options.alphas : [];
  const condition = options.condition ?? 'hybrid';

  const results = [];
  for (const alpha of alphas) {
    const run = await runEval({ queries, condition, alpha });
    results.push({
      alpha,
      mrr_at_5: run.metrics.mrr_at_5,
      ndcg_at_5: run.metrics.ndcg_at_5,
      recall_at_5: run.metrics.recall_at_5,
      latency_ms: run.metrics.latency_ms,
    });
  }
  return results;
}
