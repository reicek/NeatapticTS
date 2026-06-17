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

/** Default query file path. */
export const DEFAULT_QUERY_FILE_PATH = path.join(
  repoRoot,
  'scripts',
  'semantic-index',
  'eval-queries-v2.json',
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
  const limit = Number(options.limit ?? 10);
  const alpha = Number(options.alpha ?? 0.5);
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
        contextBudget: options.contextBudget ?? 4096,
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
export async function runEval(options = {}) {
  if (!options || typeof options !== 'object') {
    throw new Error('runEval requires an options object.');
  }

  const queries = Array.isArray(options.queries) ? options.queries : [];
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
      usage: 'node scripts/semantic-index/eval-runner.mjs [options]',
      options: [
        '--query-file <path>           Path to eval queries JSON file',
        '--condition <name>            Eval condition(s): bm25_only, hybrid, hybrid_rerank, advanced_default, all',
        '--alpha <n>                    Override hybrid alpha (default: 0.5)',
        '--limit <n>                    Results per query (default: 10)',
        '--baseline <path>             Path to baseline JSON for regression detection',
        '--regression-threshold <n>     MRR@5 regression threshold (default: 0.01)',
        '--compare                     Run A/B comparison between two conditions',
        '--alpha-sweep <values>        Comma-separated alpha values to sweep',
        '--output <path>               Write full results to file (JSON)',
        '--json                        Emit JSON summary to stdout',
        '--help                        Show help and exit',
      ],
    });
    return;
  }

  try {
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

    const queries = args['query-file']
      ? await loadQueryFile(args['query-file'])
      : await loadQueryFile();

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
      eval_id: `eval-${new Date().toISOString()}`,
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
