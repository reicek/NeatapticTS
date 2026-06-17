/**
 * @module eval-metrics
 * @description RAG evaluation metrics for the Repo Cortex advanced eval suite.
 *
 * Implements MRR@k, nDCG@k, Recall@k, context relevance, and latency helpers
 * used by {@link ../semantic-index/eval-runner.mjs}.
 *
 * @example
 * ```js
 * import { computeMrr, computeNdcg, computeRecall } from './eval-metrics.mjs';
 *
 * const query = {
 *   expected_doc_families: ['ts-source'],
 *   expected_heading_contains: 'activate',
 * };
 * const results = [
 *   { chunk_id: 1, family: 'readme', heading_path: 'Readme' },
 *   { chunk_id: 2, family: 'ts-source', heading_path: 'Network activate' },
 * ];
 * computeMrr(results, query, 5); // 0.5
 * ```
 */

/** Valid k values for MRR@k. */
const MRR_K_VALUES = Object.freeze([1, 3, 5, 10]);

/** Valid k values for nDCG@k. */
const NDCG_K_VALUES = Object.freeze([5, 10]);

/** Valid k values for Recall@k. */
const RECALL_K_VALUES = Object.freeze([5, 10, 20]);

/**
 * Ensure k is a positive safe integer within an allowed set.
 *
 * @param {unknown} k
 * @param {readonly number[]} allowed
 * @param {string} name
 * @returns {number}
 * @throws {Error} When k is not an allowed positive integer.
 */
function validateK(k, allowed, name) {
  if (!Number.isInteger(k) || k <= 0 || !allowed.includes(k)) {
    throw new Error(
      `${name} must be one of ${allowed.join(', ')}; received ${String(k)}.`,
    );
  }
  return k;
}

/**
 * Normalize a needle value to a lowercase string for substring matching.
 *
 * @param {unknown} value
 * @returns {string}
 */
function normalizeNeedle(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase();
}

/**
 * Determine whether a result chunk matches a query expectation.
 *
 * A result is relevant when its family is in `expected_doc_families` (or the
 * list is empty/absent) AND any required heading/symbol substring appears in
 * the chunk heading path. Matching follows the same rules as the existing
 * `eval-embeddings.mjs` evaluator so metric definitions stay consistent.
 *
 * @param {object} result - Corpus result row.
 * @param {object} querySpec - Eval query specification.
 * @returns {boolean}
 */
export function matchesQueryExpectation(result, querySpec) {
  const expectedFamilies = Array.isArray(querySpec?.expected_doc_families)
    ? querySpec.expected_doc_families
    : [];
  const family = result.family ?? result.doc_family ?? null;
  if (expectedFamilies.length > 0 && !expectedFamilies.includes(family)) {
    return false;
  }

  const headingNeedle = normalizeNeedle(querySpec?.expected_heading_contains);
  const symbolNeedle = normalizeNeedle(querySpec?.expected_symbol_contains);
  const headingHaystack = normalizeNeedle(
    result.heading_path ?? result.symbol_name ?? '',
  );

  if (headingNeedle && !headingHaystack.includes(headingNeedle)) return false;
  if (symbolNeedle && !headingHaystack.includes(symbolNeedle)) return false;
  return true;
}

/**
 * Compute the graded relevance of a result for nDCG@k.
 *
 * When `relevance_grades` are provided, the first matching rule determines the
 * grade. When no rules match, the result falls back to binary relevance (1 if
 * it satisfies the query expectation, 0 otherwise).
 *
 * @param {object} result - Corpus result row.
 * @param {object} querySpec - Eval query specification.
 * @returns {number} Grade in the range [0, 3].
 */
function gradeResult(result, querySpec) {
  const grades = Array.isArray(querySpec?.relevance_grades)
    ? querySpec.relevance_grades
    : [];
  const headingPath = String(result.heading_path ?? '').toLowerCase();
  const family = result.family ?? result.doc_family ?? '';

  for (const rule of grades) {
    const ruleFamily = String(rule.family ?? '').toLowerCase();
    const ruleHeading = normalizeNeedle(rule.heading_path_contains);
    const familyMatches =
      ruleFamily === '' || family.toLowerCase() === ruleFamily;
    const headingMatches =
      ruleHeading === '' || headingPath.includes(ruleHeading);
    if (familyMatches && headingMatches) {
      const grade = Number(rule.grade ?? 0);
      return Number.isFinite(grade)
        ? Math.max(0, Math.min(3, Math.round(grade)))
        : 0;
    }
  }

  return matchesQueryExpectation(result, querySpec) ? 1 : 0;
}

/**
 * Compute MRR@k (Mean Reciprocal Rank at k).
 *
 * @param {object[]} results - Ranked result chunks.
 * @param {object} querySpec - Eval query specification.
 * @param {number} k - Cut-off rank (1, 3, 5, or 10).
 * @returns {number} Reciprocal rank of the first relevant result, or 0.
 * @throws {Error} When inputs are invalid.
 */
export function computeMrr(results, querySpec, k) {
  if (!Array.isArray(results)) {
    throw new Error('results must be an array.');
  }
  if (querySpec === undefined || querySpec === null) {
    throw new Error('querySpec is required.');
  }
  const cutoff = validateK(k, MRR_K_VALUES, 'MRR k');

  for (let index = 0; index < Math.min(results.length, cutoff); index += 1) {
    if (matchesQueryExpectation(results[index], querySpec)) {
      return 1 / (index + 1);
    }
  }
  return 0;
}

/**
 * Compute DCG@k from a ranked list of relevance grades.
 *
 * @param {number[]} grades - Relevance grades at ranks 1..k.
 * @param {number} k
 * @returns {number}
 */
function computeDcg(grades, k) {
  let dcg = 0;
  for (let index = 0; index < k; index += 1) {
    const grade = grades[index] ?? 0;
    if (grade > 0) {
      dcg += (2 ** grade - 1) / Math.log2(index + 2);
    }
  }
  return dcg;
}

/**
 * Compute nDCG@k (Normalized Discounted Cumulative Gain at k).
 *
 * Uses graded relevance annotations when present; otherwise falls back to
 * binary relevance (1 for a hit, 0 otherwise).
 *
 * @param {object[]} results - Ranked result chunks.
 * @param {object} querySpec - Eval query specification.
 * @param {number} k - Cut-off rank (5 or 10).
 * @returns {number} nDCG in the range [0, 1].
 * @throws {Error} When inputs are invalid.
 */
export function computeNdcg(results, querySpec, k) {
  if (!Array.isArray(results)) {
    throw new Error('results must be an array.');
  }
  if (querySpec === undefined || querySpec === null) {
    throw new Error('querySpec is required.');
  }
  const cutoff = validateK(k, NDCG_K_VALUES, 'nDCG k');

  const actualGrades = results
    .slice(0, cutoff)
    .map((result) => gradeResult(result, querySpec));
  if (actualGrades.every((grade) => grade === 0)) return 0;

  const idealGrades = actualGrades.toSorted((a, b) => b - a);
  const idealDcg = computeDcg(idealGrades, cutoff);

  return computeDcg(actualGrades, cutoff) / idealDcg;
}

/**
 * Compute Recall@k.
 *
 * When `expected_chunk_ids` is provided, recall is the fraction of those IDs
 * that appear in the top-k results. Otherwise recall uses family+heading matches
 * against the full result set as the relevant pool.
 *
 * @param {object[]} results - Ranked result chunks.
 * @param {object} querySpec - Eval query specification.
 * @param {number} k - Cut-off rank (5, 10, or 20).
 * @returns {number} Recall in the range [0, 1].
 * @throws {Error} When inputs are invalid.
 */
export function computeRecall(results, querySpec, k) {
  if (!Array.isArray(results)) {
    throw new Error('results must be an array.');
  }
  if (querySpec === undefined || querySpec === null) {
    throw new Error('querySpec is required.');
  }
  const cutoff = validateK(k, RECALL_K_VALUES, 'Recall k');

  const topK = results.slice(0, cutoff);
  const expectedChunkIds = Array.isArray(querySpec?.expected_chunk_ids)
    ? querySpec.expected_chunk_ids
    : [];

  if (expectedChunkIds.length > 0) {
    const foundIds = new Set(topK.map((result) => result.chunk_id));
    const foundCount = expectedChunkIds.filter((id) => foundIds.has(id)).length;
    return Math.min(1, foundCount / expectedChunkIds.length);
  }

  const relevantInTopK = topK.filter((result) =>
    matchesQueryExpectation(result, querySpec),
  ).length;
  const relevantPool = results.filter((result) =>
    matchesQueryExpectation(result, querySpec),
  ).length;
  if (relevantPool === 0) return 0;
  return Math.min(1, relevantInTopK / relevantPool);
}

/**
 * Compute context relevance: the fraction of assembled context chunks that
 * match the query expectation.
 *
 * @param {object[]} assembledChunks - Chunks returned by context assembly.
 * @param {object} querySpec - Eval query specification.
 * @returns {number} Fraction in the range [0, 1].
 * @throws {Error} When inputs are invalid.
 */
export function computeContextRelevance(assembledChunks, querySpec) {
  if (!Array.isArray(assembledChunks)) {
    throw new Error('assembledChunks must be an array.');
  }
  if (querySpec === undefined || querySpec === null) {
    throw new Error('querySpec is required.');
  }
  if (assembledChunks.length === 0) return 0;

  const relevantCount = assembledChunks.filter((chunk) =>
    matchesQueryExpectation(chunk, querySpec),
  ).length;
  return relevantCount / assembledChunks.length;
}

/**
 * Measure the latency of an async operation in milliseconds.
 *
 * @param {Function} asyncFn - Function that returns a Promise.
 * @returns {Promise<{result: unknown, latency_ms: number}>}
 * @throws {Error} When asyncFn is not a function.
 */
export async function measureLatency(asyncFn) {
  if (typeof asyncFn !== 'function') {
    throw new Error('measureLatency requires a function.');
  }
  const startMs = performance.now();
  const result = await asyncFn();
  const endMs = performance.now();
  return { result, latency_ms: endMs - startMs };
}

/**
 * Compute latency from explicit start and end timestamps in milliseconds.
 *
 * @param {number} startMs
 * @param {number} endMs
 * @returns {number}
 */
export function computeLatency(startMs, endMs) {
  return Math.max(0, Number(endMs ?? 0) - Number(startMs ?? 0));
}

/**
 * Compute the p-th percentile of a numeric sample using linear interpolation.
 *
 * @param {number[]} values
 * @param {number} p - Percentile in the range [0, 100].
 * @returns {number}
 */
function percentile(values, p) {
  if (values.length === 0) return 0;
  const sorted = values.toSorted((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Aggregate latency samples into p50, p95, and max statistics.
 *
 * @param {number[]} latencies
 * @returns {{p50: number, p95: number, max: number}}
 */
export function aggregateLatency(latencies) {
  const values = Array.isArray(latencies) ? latencies : [];
  return {
    p50: percentile(values, 50),
    p95: percentile(values, 95),
    max: values.length > 0 ? Math.max(...values) : 0,
  };
}

/**
 * Aggregate per-query metric rows into condition-level summary metrics.
 *
 * @param {object[]} perQueryResults
 * @returns {object} Summary with MRR@k, nDCG@k, Recall@k, and latency.
 */
export function aggregateMetrics(perQueryResults) {
  const rows = Array.isArray(perQueryResults) ? perQueryResults : [];
  const mean = (numbers) =>
    numbers.length > 0
      ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length
      : 0;

  const mrrKeys = ['mrr_at_1', 'mrr_at_3', 'mrr_at_5', 'mrr_at_10'];
  const ndcgKeys = ['ndcg_at_5', 'ndcg_at_10'];
  const recallKeys = ['recall_at_5', 'recall_at_10', 'recall_at_20'];

  const metrics = {
    latency_ms: aggregateLatency(rows.map((row) => row.latency_ms ?? 0)),
    zero_hit_queries: rows.filter((row) => (row.mrr_at_5 ?? 0) === 0).length,
  };

  for (const key of mrrKeys) {
    metrics[key] = mean(rows.map((row) => row[key] ?? 0));
  }
  for (const key of ndcgKeys) {
    metrics[key] = mean(rows.map((row) => row[key] ?? 0));
  }
  for (const key of recallKeys) {
    metrics[key] = mean(rows.map((row) => row[key] ?? 0));
  }

  return metrics;
}

/**
 * Group per-query results by taxonomy class and aggregate metrics within each.
 *
 * @param {object[]} perQueryResults
 * @returns {Record<string, object>}
 */
export function aggregateByClass(perQueryResults) {
  const rows = Array.isArray(perQueryResults) ? perQueryResults : [];
  const groups = Object.create(null);
  for (const row of rows) {
    const className = row.class ?? 'unknown';
    groups[className] = groups[className] ?? [];
    groups[className].push(row);
  }

  const perClass = {};
  for (const className of Object.keys(groups).toSorted()) {
    perClass[className] = aggregateMetrics(groups[className]);
  }
  return perClass;
}
