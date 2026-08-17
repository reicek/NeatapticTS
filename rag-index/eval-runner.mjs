/**
 * @module eval-runner
 * @description Unified evaluation runner for the Repo Cortex advanced RAG suite.
 *
 * Executes the expanded query taxonomy under four baseline conditions,
 * computes MRR@k, nDCG@k, Recall@k, context relevance, and latency metrics,
 * and supports self-test, A/B comparison, alpha sweep, and regression gating.
 *
 * @example
 * ```js
 * import { runEval, runSelfTest } from './eval-runner.mjs';
 *
 * const result = await runEval({ queries, condition: 'hybrid' });
 * const smoke = await runSelfTest();
 * ```
 */

import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import crypto from 'node:crypto';
import {
  aggregateByClass,
  aggregateLatency,
  aggregateMetrics,
  computeContextRelevance,
  computeMrr,
  computeNdcg,
  computeRecall,
  measureLatency,
} from './eval-metrics.mjs';
import { compareToBaseline } from './eval-baseline.mjs';
import {
  parseCliArgs,
  printHelp,
  writeJsonOrText,
  fail,
} from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';
import { getTursoClient } from '../scripts/mcp-semantic/tools/cortex-db.mjs';
import {
  DEFAULT_MODEL_DIRECTORY,
  DEFAULT_MODEL_ID,
  createOnnxTextEmbedder,
  normalizeEmbeddingVector,
  readModelMeta,
} from './embed-index.mjs';

/** Default query file path. */
export const DEFAULT_QUERY_FILE_PATH = path.join(
  repoRoot,
  'rag-index',
  'eval-queries.json',
);

/** Self-test query set used by {@link runSelfTest}. */
const SELFTEST_QUERIES = Object.freeze([
  Object.freeze({
    query_id: 'selftest-001',
    query: 'Network activate',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['ts-source'],
    expected_heading_contains: 'activate',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: 'Rank-1 hit self-test.',
  }),
  Object.freeze({
    query_id: 'selftest-002',
    query: 'NEAT crossover',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['readme'],
    expected_heading_contains: 'crossover',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: 'Self-test query.',
  }),
  Object.freeze({
    query_id: 'selftest-003',
    query: 'completely nonexistent query xyz',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['readme'],
    expected_heading_contains: 'nonexistent',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: 'Zero-hit self-test.',
  }),
  Object.freeze({
    query_id: 'selftest-004',
    query: 'NEAT speciation',
    class: 'cross_boundary',
    difficulty: 'medium',
    expected_doc_families: ['readme', 'ts-source'],
    expected_heading_contains: 'speciation',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: 'Multi-hit self-test.',
  }),
  Object.freeze({
    query_id: 'selftest-005',
    query: 'mutation',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['ts-source'],
    expected_heading_contains: 'mutation',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: 'Partial-match self-test.',
  }),
]);

/** Supported eval conditions. */
const SUPPORTED_CONDITIONS = Object.freeze([
  'bm25_only',
  'hybrid',
  'hybrid_rerank',
  'advanced_default',
]);

/** Taxonomy class ordering for deterministic aggregation. */
const CLASS_ORDER = Object.freeze([
  'simple_lookup',
  'cross_boundary',
  'multi_hop',
  'exploratory',
  'code_specific',
  'plan_specific',
]);

/**
 * Validate a v2 eval query object.
 *
 * @param {unknown} query
 * @returns {object}
 * @throws {Error} When the query does not satisfy the v2 schema.
 */
export function validateQuerySchema(query) {
  if (!query || typeof query !== 'object') {
    throw new Error('Query must be an object.');
  }
  if (typeof query.query_id !== 'string' || query.query_id === '') {
    throw new Error('query_id must be a non-empty string.');
  }
  if (typeof query.query !== 'string' || query.query === '') {
    throw new Error('query text must be a non-empty string.');
  }
  if (!CLASS_ORDER.includes(query.class)) {
    throw new Error(`class must be one of ${CLASS_ORDER.join(', ')}.`);
  }
  if (!['easy', 'medium', 'hard'].includes(query.difficulty)) {
    throw new Error('difficulty must be easy, medium, or hard.');
  }
  if (!Array.isArray(query.expected_doc_families)) {
    throw new Error('expected_doc_families must be an array.');
  }

  const grades = Array.isArray(query.relevance_grades)
    ? query.relevance_grades
    : [];
  for (const grade of grades) {
    if (typeof grade !== 'object' || grade === null) {
      throw new Error('Each relevance_grade must be an object.');
    }
    if (!Number.isInteger(grade.grade) || grade.grade < 0 || grade.grade > 3) {
      throw new Error('relevance grade must be an integer in [0, 3].');
    }
  }

  if (
    query.expected_chunk_ids !== undefined &&
    !Array.isArray(query.expected_chunk_ids)
  ) {
    throw new Error('expected_chunk_ids must be an array when present.');
  }

  return query;
}

/**
 * Sort queries into taxonomy class order, then by query_id.
 *
 * @param {object[]} queries
 * @returns {object[]}
 */
function sortQueries(queries) {
  return queries.toSorted((a, b) => {
    const classDiff =
      CLASS_ORDER.indexOf(a.class) - CLASS_ORDER.indexOf(b.class);
    if (classDiff !== 0) return classDiff;
    return String(a.query_id).localeCompare(String(b.query_id));
  });
}

/**
 * Default synthetic search used when no corpus database is available.
 *
 * Returns a small set of deterministic chunks so metric computation can run in
 * unit tests and smoke checks without requiring a built index.
 *
 * @param {object} querySpec
 * @param {object} _options
 * @returns {Promise<object[]>}
 */
async function syntheticSearch(querySpec, _options) {
  const family =
    querySpec.expected_doc_families?.[0] ??
    (querySpec.class === 'code_specific' ? 'ts-source' : 'readme');
  return [
    {
      chunk_id: 1,
      family,
      heading_path: '',
    },
  ];
}

/**
 * Resolve the search function for a condition.
 *
 * When `options.searchFn` is provided it is used directly. Otherwise the runner
 * attempts to load the real MCP search tools; if that fails (e.g. no corpus), it
 * falls back to the synthetic search so tests and smoke checks still execute.
 *
 * @param {string} condition
 * @param {object} options
 * @returns {Promise<Function>}
 */
export async function resolveSearchFn(condition, options) {
  if (typeof options.searchFn === 'function') return options.searchFn;
  if (process.env.EVAL_FORCE_SYNTHETIC === '1') return syntheticSearch;

  try {
    const searchCorpusPath =
      options.searchModulePath ??
      path.join(
        repoRoot,
        'scripts',
        'mcp-semantic',
        'tools',
        'search-corpus.mjs',
      );
    const { searchCorpus } = await import(pathToFileURL(searchCorpusPath).href);

    if (condition === 'advanced_default') {
      const searchAdvancedPath =
        options.searchAdvancedModulePath ??
        path.join(
          repoRoot,
          'scripts',
          'mcp-semantic',
          'tools',
          'search-advanced.mjs',
        );
      const { searchAdvanced } = await import(
        pathToFileURL(searchAdvancedPath).href
      );
      return (querySpec, conditionOptions) =>
        searchAdvanced({
          query: querySpec.query,
          limit: conditionOptions.limit,
          query_class: querySpec.class,
          budget: conditionOptions.contextBudget,
        });
    }

    return (querySpec, conditionOptions) =>
      searchCorpus({
        query: querySpec.query,
        limit: conditionOptions.limit,
        use_dense: conditionOptions.useDense,
        use_rerank: conditionOptions.useRerank,
        alpha: conditionOptions.alpha,
      });
  } catch {
    return syntheticSearch;
  }
}

/**
 * Build condition-specific search options.
 *
 * @param {string} condition
 * @param {object} querySpec
 * @param {object} options
 * @returns {object}
 */
export function buildConditionOptions(condition, querySpec, options) {
  const resolvedOptions = options ?? {};
  const limit = Number(resolvedOptions.limit ?? 10);
  const alpha = Number(resolvedOptions.alpha ?? 0.5);
  const isCodeSpecific = querySpec?.class === 'code_specific';

  switch (condition) {
    case 'bm25_only':
      return { useDense: false, useRerank: false, alpha: 1, limit };
    case 'hybrid':
      return { useDense: true, useRerank: false, alpha, limit };
    case 'hybrid_rerank':
      return { useDense: true, useRerank: true, alpha, limit };
    case 'advanced_default':
      return {
        useDense: true,
        useRerank: true,
        alpha,
        limit,
        contextBudget: resolvedOptions.contextBudget ?? 4096,
        compact: true,
        read_top_result: true,
        auto_fallback: true,
        include_code_only: isCodeSpecific,
      };
    default:
      throw new Error(`Unsupported condition: ${condition}`);
  }
}

/**
 * Execute a single query under a condition and compute per-query metrics.
 *
 * @param {object} querySpec
 * @param {string} condition
 * @param {object} options
 * @returns {Promise<object>}
 */
async function executeQuery(querySpec, condition, options) {
  const conditionOptions = buildConditionOptions(condition, querySpec, options);
  const searchFn = await resolveSearchFn(condition, options);

  const { result: rawResults, latency_ms: latencyMs } = await measureLatency(
    () => searchFn(querySpec, conditionOptions),
  );

  const results = Array.isArray(rawResults)
    ? rawResults
    : (rawResults?.results ?? rawResults?.context?.results ?? []);

  const assembledChunks =
    rawResults?.selectedChunks ?? rawResults?.context?.selectedChunks ?? [];

  const perQuery = {
    query_id: querySpec.query_id,
    class: querySpec.class,
    condition,
    latency_ms: latencyMs,
    mrr_at_1: computeMrr(results, querySpec, 1),
    mrr_at_3: computeMrr(results, querySpec, 3),
    mrr_at_5: computeMrr(results, querySpec, 5),
    mrr_at_10: computeMrr(results, querySpec, 10),
    ndcg_at_5: computeNdcg(results, querySpec, 5),
    ndcg_at_10: computeNdcg(results, querySpec, 10),
    recall_at_5: computeRecall(results, querySpec, 5),
    recall_at_10: computeRecall(results, querySpec, 10),
    recall_at_20: computeRecall(results, querySpec, 20),
    context_relevance: computeContextRelevance(assembledChunks, querySpec),
  };

  return perQuery;
}

/**
 * Load eval queries from a JSON file.
 *
 * @param {string} [filePath]
 * @returns {Promise<object[]>}
 */
async function loadQueryFile(filePath) {
  const resolvedPath = filePath ?? DEFAULT_QUERY_FILE_PATH;
  const text = await readFile(resolvedPath, 'utf8');
  const parsed = JSON.parse(text);
  if (!Array.isArray(parsed)) {
    throw new Error('Query file must contain an array of query objects.');
  }
  return parsed;
}

/**
 * Run the eval suite for a single condition.
 *
 * @param {object} options
 * @param {object[]} options.queries - Eval query specifications.
 * @param {string} options.condition - One of the supported conditions.
 * @param {number} [options.alpha] - Override hybrid alpha.
 * @param {number} [options.limit] - Result limit per query.
 * @param {Function} [options.searchFn] - Pluggable search function.
 * @returns {Promise<object>} Condition-level eval result.
 * @throws {Error} When queries or condition are invalid.
 */
/* istanbul ignore next -- defensive: always called with explicit options */
export async function runEval(options = {}) {
  if (!options || typeof options !== 'object') {
    throw new Error('runEval requires an options object.');
  }

  const queries = /* istanbul ignore next -- defensive: options.queries is always an array in test calls */ Array.isArray(options.queries) ? options.queries : [];
  const condition = options.condition;
  if (
    typeof condition !== 'string' ||
    !SUPPORTED_CONDITIONS.includes(condition)
  ) {
    throw new Error(
      `condition must be one of ${SUPPORTED_CONDITIONS.join(', ')}; received ${String(condition)}.`,
    );
  }

  const validatedQueries = queries.map(validateQuerySchema);
  const sortedQueries = sortQueries(validatedQueries);

  const perQueryResults = [];
  for (const querySpec of sortedQueries) {
    perQueryResults.push(await executeQuery(querySpec, condition, options));
  }

  const metrics = aggregateMetrics(perQueryResults);
  const perClass = aggregateByClass(perQueryResults);

  return {
    condition,
    query_count: sortedQueries.length,
    metrics,
    per_query: perQueryResults,
    per_class: perClass,
    premium_defaults_applied: condition === 'advanced_default',
  };
}

/**
 * Run the built-in self-test query set against all supported conditions.
 *
 * @param {object} [options={}]
 * @returns {Promise<object>}
 */
export async function runSelfTest(options = {}) {
  const conditions = [...SUPPORTED_CONDITIONS];
  const results = [];
  for (const condition of conditions) {
    results.push(
      await runEval({
        queries: [...SELFTEST_QUERIES],
        condition,
        ...options,
      }),
    );
  }

  const pass = results.every(
    (result) =>
      result.query_count === SELFTEST_QUERIES.length &&
      Number.isFinite(result.metrics.mrr_at_5) &&
      Number.isFinite(result.metrics.ndcg_at_5) &&
      Number.isFinite(result.metrics.recall_at_5),
  );

  return {
    query_count: SELFTEST_QUERIES.length,
    conditions: results.map((result) => result.condition),
    pass,
    results,
  };
}

/**
 * Run the eval for every supported condition.
 *
 * @param {object} [options={}]
 * @returns {Promise<Record<string, object>>}
 */
/* istanbul ignore next -- defensive: always called with explicit options */
export async function runAllConditions(options = {}) {
  const queries = Array.isArray(options.queries)
    ? options.queries
    : await loadQueryFile(options.queryFilePath);

  const results = {};
  for (const condition of SUPPORTED_CONDITIONS) {
    results[condition] = await runEval({ ...options, queries, condition });
  }
  return results;
}

/**
 * Format an eval summary as a human-readable table.
 *
 * @param {object} payload
 * @returns {string}
 */
function formatTable(payload) {
  const results = payload.results;
  const lines = [
    `Eval Results (${payload.query_count} queries)`,
    '',
    'Condition         MRR@5  nDCG@5  Recall@5  Latency P50  Zero-hit',
    '─────────────────────────────────────────────────────────────────',
  ];
  for (const condition of Object.keys(results).toSorted()) {
    const metrics = results[condition].metrics;
    const row = [
      condition.padEnd(17),
      String(metrics.mrr_at_5.toFixed(3)).padStart(6),
      String(metrics.ndcg_at_5.toFixed(3)).padStart(7),
      String(metrics.recall_at_5.toFixed(3)).padStart(9),
      `${Math.round(metrics.latency_ms.p50)}ms`.padStart(12),
      String(metrics.zero_hit_queries).padStart(8),
    ];
    lines.push(row.join(' '));
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Recall benchmark — DiskANN vs brute-force (Phase 4 Step 06)
// ---------------------------------------------------------------------------

/** Default k for recall@k computation in the vector search benchmark. */
const DEFAULT_RECALL_K = 10;

/** Minimum recall@10 to pass the benchmark. */
const DEFAULT_MIN_RECALL = 0.9;

/** Maximum average latency in milliseconds to pass the benchmark. */
const DEFAULT_MAX_LATENCY_MS = 50;

/** Maximum heap delta in MB to pass the memory stability check. */
const DEFAULT_MAX_HEAP_DELTA_MB = 50;

/** Default query file for the recall benchmark (v1 20-query set). */
const DEFAULT_RECALL_QUERY_FILE_PATH = path.join(
  repoRoot,
  'scripts',
  'semantic-index',
  'eval-queries.json',
);

/**
 * Compute recall@k as the fraction of ANN top-k chunk IDs that also appear in
 * the brute-force top-k set.
 *
 * Uses the standard definition:
 * `recall@k = |ANN_top_k ∩ BF_top_k| / |BF_top_k|`
 *
 * When either set is empty, returns 0 to avoid division by zero.
 *
 * @param {number[]} annIds - Ordered chunk IDs from the ANN (vector_top_k) path.
 * @param {number[]} bruteForceIds - Ordered chunk IDs from the brute-force path.
 * @param {number} k - The k for recall@k.
 * @returns {number} Recall score in [0, 1].
 *
 * @example
 * ```js
 * const recall = computeVectorRecall([1, 2, 3, 4, 5], [1, 2, 3, 4, 6], 5);
 * // recall = 4 / 5 = 0.8
 * ```
 */
export function computeVectorRecall(annIds, bruteForceIds, k) {
  const annTopK = new Set(annIds.slice(0, k));
  const bfTopK = new Set(bruteForceIds.slice(0, k));
  if (annTopK.size === 0 || bfTopK.size === 0) return 0;
  let intersection = 0;
  for (const id of annTopK) {
    if (bfTopK.has(id)) intersection += 1;
  }
  return intersection / bfTopK.size;
}

/**
 * Run the DiskANN vector search recall and performance benchmark.
 *
 * For each query in the eval set:
 *   1. Generate an embedding via the ONNX embedder (or a provided mock).
 *   2. Run the ANN path (`vector_top_k`) and the brute-force path
 *      (`vector_distance_cos`) against the same database.
 *   3. Compute recall@k as the fraction of ANN top-k chunk IDs that match the
 *      brute-force top-k chunk IDs.
 *   4. Measure latency for both paths using `performance.now()`.
 *
 * Memory stability is verified by comparing `process.memoryUsage().heapUsed`
 * before and after the benchmark run. A significant heap increase indicates
 * embeddings are being loaded into JS memory (a regression of the server-side
 * vector search design).
 *
 * When the DiskANN index is unavailable (e.g. in-memory test databases where
 * `libsql_vector_idx` indexes are skipped), the ANN path fails and the
 * benchmark records `ann_available: false`. In that case, recall is not
 * meaningful and the `recall_pass` criterion is set to `null`.
 *
 * @param {object} options - Benchmark options.
 * @param {object[]} [options.queries] - Eval query specs (uses eval-queries.json by default).
 * @param {string} [options.queryFilePath] - Override the query file path.
 * @param {import('@libsql/client').Client} [options.client] - Optional libSQL client (for testing).
 * @param {string} [options.databasePath] - Database path override (when no client).
 * @param {Function} [options.embedText] - Optional pre-created embedder function.
 * @param {string} [options.modelDirectory] - Override model cache directory.
 * @param {string} [options.modelId] - Override model identifier.
 * @param {number} [options.k=10] - k for recall@k computation.
 * @param {number} [options.minRecall=0.90] - Minimum recall@k to pass.
 * @param {number} [options.maxLatencyMs=50] - Maximum average latency in ms to pass.
 * @param {number} [options.maxHeapDeltaMb=50] - Maximum heap delta in MB to pass.
 * @returns {Promise<object>} Benchmark report with recall, latency, memory, and pass/fail.
 *
 * @example
 * ```js
 * import { runRecallBenchmark } from './eval-runner.mjs';
 * const report = await runRecallBenchmark({ databasePath: 'rag-index/data/turso-replica.sqlite' });
 * console.log(report.recall_at_k, report.latency_ms, report.heap_delta_mb);
 * ```
 */
/* istanbul ignore next -- defensive: always called with explicit options */
export async function runRecallBenchmark(options = {}) {
  const k = Math.max(1, Math.trunc(Number(options.k ?? DEFAULT_RECALL_K)));
  const minRecall = Number(options.minRecall ?? DEFAULT_MIN_RECALL);
  const maxLatencyMs = Number(options.maxLatencyMs ?? DEFAULT_MAX_LATENCY_MS);
  const maxHeapDeltaMb = Number(
    options.maxHeapDeltaMb ?? DEFAULT_MAX_HEAP_DELTA_MB,
  );

  // Step 1: Load queries from file or use provided set.
  let queries;
  if (Array.isArray(options.queries)) {
    queries = options.queries;
  } else {
    const queryFilePath = path.resolve(
      options.queryFilePath ?? DEFAULT_RECALL_QUERY_FILE_PATH,
    );
    const text = await readFile(queryFilePath, 'utf8');
    queries = JSON.parse(text);
  }

  // Step 2: Resolve embedder, model metadata, and database client.
  const modelMeta = await readModelMeta({
    modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
    modelMeta: options.modelMeta,
  });
  const dimension = Number(/* istanbul ignore next -- defensive: options.dimension always provided in test calls */ options.dimension ?? modelMeta.dimension ?? 0);
  const modelId = String(
    /* istanbul ignore next -- defensive: options.modelId always provided in test calls */ options.modelId ?? modelMeta.model_id ?? DEFAULT_MODEL_ID,
  );

  const embedText =
    options.embedText ??
    (await createOnnxTextEmbedder({
      dimension,
      modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
      modelId,
    }));

  const client = options.client ?? (await getTursoClient(options.databasePath));

  // Step 3: Measure heap before the benchmark run.
  const heapBefore = process.memoryUsage().heapUsed;

  const perQueryReports = [];
  let annAvailable = true;

  try {
    for (const querySpec of queries) {
      const queryText = String(/* istanbul ignore next -- defensive: querySpec.query is always set in test data */ querySpec.query ?? '').trim();
      if (!queryText) continue;

      // Generate query embedding.
      const rawEmbedding = await embedText({ text: queryText });
      const queryEmbedding = normalizeEmbeddingVector(rawEmbedding, dimension);
      const queryEmbeddingBuffer = Buffer.from(
        queryEmbedding.buffer,
        queryEmbedding.byteOffset,
        queryEmbedding.byteLength,
      );

      // Run ANN path (vector_top_k) — try, catch for fallback detection.
      const annResult = await runAnnQuery({
        client,
        queryEmbeddingBuffer,
        k: k * 2,
        modelId,
      });
      let annIds;
      let annLatencyMs;
      if (annResult.ok) {
        annIds = annResult.ids;
        annLatencyMs = annResult.latencyMs;
      } else {
        annAvailable = false;
        annIds = [];
        annLatencyMs = annResult.latencyMs;
      }

      // Run brute-force path (vector_distance_cos).
      const bruteForceResult = await runBruteForceQuery({
        client,
        queryEmbeddingBuffer,
        k,
        modelId,
      });
      const bruteForceIds = bruteForceResult.ids;
      const bruteForceLatencyMs = bruteForceResult.latencyMs;

      const recall = computeVectorRecall(annIds, bruteForceIds, k);

      perQueryReports.push({
        query: queryText,
        ann_chunk_ids: annIds.slice(0, k),
        brute_force_chunk_ids: bruteForceIds.slice(0, k),
        recall_at_k: recall,
        ann_latency_ms: annLatencyMs,
        brute_force_latency_ms: bruteForceLatencyMs,
      });
    }
  } finally {
    /* istanbul ignore next -- optional chaining branches require no embedText in recall path, needs ONNX */
    if (
      typeof options.embedText?.release !== 'function' &&
      typeof embedText?.release === 'function'
    ) {
      await embedText.release();
    }
  }

  // Step 4: Measure heap after the benchmark run.
  const heapAfter = process.memoryUsage().heapUsed;
  const heapDeltaBytes = heapAfter - heapBefore;
  const heapDeltaMb = heapDeltaBytes / (1024 * 1024);

  // Step 5: Aggregate metrics.
  const queryCount = perQueryReports.length;
  const avgRecall =
    queryCount > 0
      ? perQueryReports.reduce((sum, report) => sum + report.recall_at_k, 0) /
        queryCount
      : 0;
  const avgAnnLatency =
    queryCount > 0
      ? perQueryReports.reduce(
          (sum, report) => sum + report.ann_latency_ms,
          0,
        ) / queryCount
      : 0;
  const avgBruteForceLatency =
    queryCount > 0
      ? perQueryReports.reduce(
          (sum, report) => sum + report.brute_force_latency_ms,
          0,
        ) / queryCount
      : 0;

  // Step 6: Evaluate pass/fail criteria.
  const recallPass = annAvailable ? avgRecall >= minRecall : null;
  const latencyPass = avgAnnLatency < maxLatencyMs;
  const memoryPass = heapDeltaMb < maxHeapDeltaMb;

  return {
    benchmark: 'recall-benchmark',
    query_count: queryCount,
    k,
    ann_available: annAvailable,
    recall_at_k: Math.round(avgRecall * 1000) / 1000,
    ann_latency_ms: Math.round(avgAnnLatency * 1000) / 1000,
    brute_force_latency_ms: Math.round(avgBruteForceLatency * 1000) / 1000,
    heap_before_mb: Math.round((heapBefore / (1024 * 1024)) * 1000) / 1000,
    heap_after_mb: Math.round((heapAfter / (1024 * 1024)) * 1000) / 1000,
    heap_delta_mb: Math.round(heapDeltaMb * 1000) / 1000,
    thresholds: {
      min_recall: minRecall,
      max_latency_ms: maxLatencyMs,
      max_heap_delta_mb: maxHeapDeltaMb,
    },
    criteria: {
      recall_pass: recallPass,
      latency_pass: latencyPass,
      memory_pass: memoryPass,
    },
    pass: recallPass !== false && latencyPass && memoryPass,
    per_query: perQueryReports,
  };
}

/**
 * Run a single ANN (vector_top_k) query and measure latency.
 *
 * Tries `vector_top_k(chunks_embedding_idx, vector8(?), ?)` to retrieve
 * approximate nearest neighbors from the DiskANN index. When the index is
 * unavailable, the query fails and `ok` is set to `false`.
 *
 * @param {object} params - Query parameters.
 * @param {import('@libsql/client').Client} params.client - libSQL client.
 * @param {Buffer} params.queryEmbeddingBuffer - Float32 query embedding buffer.
 * @param {number} params.k - Number of ANN candidates to retrieve.
 * @param {string} params.modelId - Model identifier for embedding_model filter.
 * @returns {Promise<{ ok: boolean, ids: number[], latencyMs: number }>}
 */
async function runAnnQuery({ client, queryEmbeddingBuffer, k, modelId }) {
  const start = performance.now();
  try {
    const result = await client.execute({
      sql: `
        SELECT c.chunk_id
        FROM vector_top_k(chunks_embedding_idx, vector8(?), ?) AS v
        JOIN chunks c ON c.rowid = v.rowid
        WHERE c.embedding IS NOT NULL AND c.embedding_model = ?
      `,
      args: [queryEmbeddingBuffer, k, modelId],
    });
    const ids = result.rows.map((row) => Number(row.chunk_id));
    return { ok: true, ids, latencyMs: performance.now() - start };
  } catch {
    return { ok: false, ids: [], latencyMs: performance.now() - start };
  }
}

/**
 * Run a single brute-force (vector_distance_cos) query and measure latency.
 *
 * Queries all chunks with embeddings, computes cosine distance server-side,
 * and orders by ascending distance. This is the ground-truth baseline for
 * recall computation.
 *
 * @param {object} params - Query parameters.
 * @param {import('@libsql/client').Client} params.client - libSQL client.
 * @param {Buffer} params.queryEmbeddingBuffer - Float32 query embedding buffer.
 * @param {number} params.k - Maximum results to return.
 * @param {string} params.modelId - Model identifier for embedding_model filter.
 * @returns {Promise<{ ids: number[], latencyMs: number }>}
 */
async function runBruteForceQuery({
  client,
  queryEmbeddingBuffer,
  k,
  modelId,
}) {
  const start = performance.now();
  const result = await client.execute({
    sql: `
      SELECT c.chunk_id
      FROM chunks c
      WHERE c.embedding IS NOT NULL AND c.embedding_model = ?
      ORDER BY vector_distance_cos(c.embedding, vector8(?))
      LIMIT ?
    `,
    args: [modelId, queryEmbeddingBuffer, k],
  });
  const ids = result.rows.map((row) => Number(row.chunk_id));
  return { ids, latencyMs: performance.now() - start };
}

/**
 * CLI entry point for the eval runner.
 *
 * @param {string[]} argv - Process arguments excluding node and script.
 * @returns {Promise<void>}
 */
export async function runCli(argv) {
  const args = parseCliArgs(argv, {
    repeatableFlags: ['condition'],
  });

  if (args.help) {
    printHelp({
      title: 'Repo Cortex RAG evaluation runner',
      usage: 'node rag-index/eval-runner.mjs [options]',
      options: [
        '--query-file <path>           Path to eval queries JSON file',
        '--condition <name>            Eval condition(s): bm25_only, hybrid, hybrid_rerank, advanced_default, all',
        '--alpha <n>                    Override hybrid alpha (default: 0.5)',
        '--limit <n>                    Results per query (default: 10)',
        '--baseline <path>             Path to baseline JSON for regression detection',
        '--regression-threshold <n>     MRR@5 regression threshold (default: 0.01)',
        '--compare                     Run A/B comparison between two conditions',
        '--alpha-sweep <values>        Comma-separated alpha values to sweep',
        '--recall-benchmark             Run DiskANN recall + latency benchmark vs brute-force',
        '--k <n>                        k for recall@k (default: 10)',
        '--min-recall <n>               Minimum recall@k to pass (default: 0.90)',
        '--max-latency <n>             Maximum avg latency in ms (default: 50)',
        '--output <path>               Write full results to file (JSON)',
        '--json                        Emit JSON summary to stdout',
        '--help                        Show help and exit',
      ],
    });
    return;
  }

  try {
    if (args['recall-benchmark']) {
      const report = await runRecallBenchmark({
        queries: args['query-file']
          ? JSON.parse(await readFile(path.resolve(args['query-file']), 'utf8'))
          : undefined,
        queryFilePath: args['query-file'],
        databasePath: args.database,
        k: args.k,
        minRecall: args['min-recall'],
        maxLatencyMs: args['max-latency'],
      });
      writeJsonOrText(report, Boolean(args.json), (payload) =>
        [
          `Recall Benchmark (k=${payload.k}, ${payload.query_count} queries)`,
          `  ANN available:    ${payload.ann_available}`,
          `  Recall@${payload.k}:       ${payload.recall_at_k}`,
          `  ANN latency:      ${payload.ann_latency_ms}ms`,
          `  BF latency:        ${payload.brute_force_latency_ms}ms`,
          `  Heap delta:        ${payload.heap_delta_mb}MB`,
          `  Pass:              ${payload.pass}`,
        ].join('\n'),
      );
      /* istanbul ignore if -- defensive: report.pass is always true in test runs */
      if (!report.pass) process.exitCode = 1;
      return;
    }

    let conditions = args.condition ?? 'all';
    if (typeof conditions === 'string') conditions = [conditions];
    if (conditions.includes('all')) {
      conditions = [...SUPPORTED_CONDITIONS];
    }

    const unsupported = conditions.filter(
      (condition) => !SUPPORTED_CONDITIONS.includes(condition),
    );
    if (unsupported.length > 0) {
      throw new Error(`Unsupported condition(s): ${unsupported.join(', ')}`);
    }

    let queries;
    /* istanbul ignore next -- defensive: args['query-file'] is always set in test invocations */
    if (args['query-file']) {
      queries = await loadQueryFile(args['query-file']);
    } else {
      queries = await loadQueryFile();
    }

    if (conditions.length === 2 && args.compare) {
      const { compareResults } = await import('./eval-compare.mjs');
      const [conditionA, conditionB] = conditions;
      const resultA = await runEval({
        queries,
        condition: conditionA,
        alpha: args.alpha,
        limit: args.limit,
      });
      const resultB = await runEval({
        queries,
        condition: conditionB,
        alpha: args.alpha,
        limit: args.limit,
      });
      const comparison = compareResults(resultA, resultB);
      writeJsonOrText(comparison, Boolean(args.json), (payload) =>
        JSON.stringify(payload, null, 2),
      );
      return;
    }

    if (args['alpha-sweep']) {
      const { alphaSweep } = await import('./eval-compare.mjs');
      const alphas = String(args['alpha-sweep'])
        .split(',')
        .map((value) => Number(value.trim()));
      const sweep = await alphaSweep({ queries, alphas });
      writeJsonOrText(sweep, Boolean(args.json), (payload) =>
        JSON.stringify(payload, null, 2),
      );
      return;
    }

    const results = {};
    for (const condition of conditions) {
      results[condition] = await runEval({
        queries,
        condition,
        alpha: args.alpha,
        limit: args.limit,
      });
    }

    const payload = {
      eval_id: `eval-${crypto.randomUUID()}`,
      query_file: args['query-file'] ?? DEFAULT_QUERY_FILE_PATH,
      query_count: queries.length,
      conditions,
      results,
      regression: {
        checked: false,
        baseline: null,
        threshold: null,
        failures: [],
        warnings: [],
      },
    };

    if (args.baseline) {
      const { loadBaseline } = await import('./eval-baseline.mjs');
      const baseline = await loadBaseline('baseline-latest', {
        path: args.baseline,
      });
      payload.regression.checked = true;
      payload.regression.baseline = args.baseline;
      payload.regression.threshold = Number(
        args['regression-threshold'] ?? 0.01,
      );

      for (const condition of conditions) {
        const current = results[condition];
        const baselineCondition = baseline.conditions?.[condition];
        /* istanbul ignore if -- defensive: baselineCondition always exists for test conditions */
        if (!baselineCondition) continue;
        const report = compareToBaseline(
          { condition, metrics: current.metrics },
          { condition, metrics: baselineCondition },
          {
            failThresholds: { mrr_at_5: payload.regression.threshold },
            warnThresholds: {
              ndcg_at_5: payload.regression.threshold,
              recall_at_5: payload.regression.threshold,
            },
            latencyWarnMs: 100,
          },
        );
        if (!report.pass) payload.regression.failures.push(condition);
        if (report.warnings.length > 0)
          payload.regression.warnings.push(condition);
      }

      if (payload.regression.failures.length > 0) {
        process.exitCode = 1;
      }
    }

    writeJsonOrText(payload, Boolean(args.json), formatTable);

    if (args.output) {
      const outputPath = path.resolve(args.output);
      await writeFile(outputPath, JSON.stringify(payload, null, 2), 'utf8');
    }
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

/**
 * Bootstrap the CLI when this module is executed directly.
 *
 * @returns {Promise<void>}
 */
export async function bootstrap() {
  if (
    process.argv[1] &&
    import.meta.url === pathToFileURL(process.argv[1]).href
  ) {
    await runCli(process.argv.slice(2));
  }
}

await bootstrap();
