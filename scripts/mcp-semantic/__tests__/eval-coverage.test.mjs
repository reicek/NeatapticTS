/**
 * @module eval-coverage.test
 * @description Coverage tests for the RAG evaluation modules.
 */

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { jest } from '@jest/globals';
import {
  computeMrr,
  computeNdcg,
  computeRecall,
  computeContextRelevance,
  measureLatency,
  computeLatency,
  matchesQueryExpectation,
  aggregateMetrics,
  aggregateByClass,
} from '../../semantic-index/eval-metrics.mjs';
import {
  storeBaseline,
  loadBaseline,
  compareToBaseline,
} from '../../semantic-index/eval-baseline.mjs';
import {
  wilcoxonSignedRankTest,
  compareResults,
  alphaSweep,
  normalCdf,
} from '../../semantic-index/eval-compare.mjs';
import {
  validateQuerySchema,
  runEval,
  runSelfTest,
  runCli,
  bootstrap,
  buildConditionOptions,
  resolveSearchFn,
  runAllConditions,
  DEFAULT_QUERY_FILE_PATH,
} from '../../semantic-index/eval-runner.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const runnerPath = path.resolve(
  __dirname,
  '../../semantic-index/eval-runner.mjs',
);

afterEach(() => {
  process.exitCode = 0;
});

function makeQuery(overrides = {}) {
  return {
    query_id: 'sl-001',
    query: 'how does NEAT crossover work',
    class: 'simple_lookup',
    difficulty: 'easy',
    expected_doc_families: ['readme'],
    expected_heading_contains: 'crossover',
    expected_symbol_contains: null,
    expected_chunk_ids: [],
    relevance_grades: [],
    notes: '',
    ...overrides,
  };
}

function makeFullQuery(overrides = {}) {
  return {
    ...makeQuery(),
    expected_chunk_ids: [1],
    ...overrides,
  };
}

const dummyQuery = {
  expected_doc_families: ['readme'],
  expected_heading_contains: 'crossover',
};

describe('eval-metrics coverage', () => {
  it('throws when computeMrr receives an invalid k value', () => {
    expect(() => computeMrr([], dummyQuery, 2)).toThrow();
  });

  it('throws when computeNdcg receives invalid inputs', () => {
    expect(() => computeNdcg(null, dummyQuery, 5)).toThrow();
    expect(() => computeNdcg([], null, 5)).toThrow();
    expect(() => computeNdcg([], dummyQuery, 2)).toThrow();
  });

  it('throws when computeRecall receives invalid inputs', () => {
    expect(() => computeRecall(null, dummyQuery, 5)).toThrow();
    expect(() => computeRecall([], null, 5)).toThrow();
    expect(() => computeRecall([], dummyQuery, 2)).toThrow();
  });

  it('throws when computeContextRelevance receives invalid inputs', () => {
    expect(() => computeContextRelevance(null, dummyQuery)).toThrow();
    expect(() => computeContextRelevance([], null)).toThrow();
  });

  it('throws when measureLatency receives a non-function input', async () => {
    await expect(measureLatency('not-a-function')).rejects.toThrow(
      'measureLatency requires a function.',
    );
  });

  it('computes MRR when no results are present', () => {
    expect(computeMrr([], dummyQuery, 5)).toBe(0);
  });

  it('computes nDCG using graded relevance rules', () => {
    const query = {
      expected_doc_families: [],
      expected_heading_contains: null,
      relevance_grades: [
        { family: 'readme', heading_path_contains: 'network', grade: 2 },
      ],
    };
    const results = [
      { family: 'readme', heading_path: 'network activate' },
      { family: 'readme', heading_path: 'other' },
    ];
    expect(computeNdcg(results, query, 5)).toBeGreaterThan(0);
  });

  it('clamps relevance grades to the 0-3 range', () => {
    const query = {
      expected_doc_families: [],
      expected_heading_contains: null,
      relevance_grades: [{ family: '', heading_path_contains: '', grade: 10 }],
    };
    const results = [{ family: 'readme', heading_path: 'anything' }];
    expect(computeNdcg(results, query, 5)).toBe(1);
  });
});

describe('eval-baseline coverage', () => {
  let tempDir;
  let baselineFile;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(process.cwd(), '.test-baseline-'));
    baselineFile = path.join(tempDir, 'baseline.json');
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
  });

  it('throws when storeBaseline is missing a baseline id', async () => {
    await expect(storeBaseline({ metrics: {} })).rejects.toThrow();
  });

  it('throws when loadBaseline is missing a baseline id', async () => {
    await expect(loadBaseline('')).rejects.toThrow();
  });

  it('throws when compareToBaseline is missing inputs', () => {
    expect(() => compareToBaseline(null, { metrics: {} })).toThrow();
    expect(() => compareToBaseline({ metrics: {} }, null)).toThrow();
  });

  it('throws when compareToBaseline conditions mismatch', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'a', metrics: { mrr_at_5: 0.5 } },
        { condition: 'b', metrics: { mrr_at_5: 0.6 } },
      ),
    ).toThrow();
  });

  it('throws when compareToBaseline metrics are missing', () => {
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid' },
        { condition: 'hybrid', metrics: {} },
      ),
    ).toThrow();
    expect(() =>
      compareToBaseline(
        { condition: 'hybrid', metrics: {} },
        { condition: 'hybrid' },
      ),
    ).toThrow();
  });

  it('stores and loads a baseline from a custom path', async () => {
    const baseline = {
      baseline_id: 'test-baseline',
      condition: 'hybrid',
      metrics: { mrr_at_5: 0.5 },
    };
    await storeBaseline(baseline, { path: baselineFile });
    const loaded = await loadBaseline('test-baseline', { path: baselineFile });
    expect(loaded.metrics.mrr_at_5).toBe(0.5);
  });

  it('stores and loads a baseline from a relative custom path', async () => {
    const relativePath = path.relative(process.cwd(), baselineFile);
    const baseline = {
      baseline_id: 'rel-baseline',
      metrics: { mrr_at_5: 0.7 },
    };
    await storeBaseline(baseline, { path: relativePath });
    const loaded = await loadBaseline('rel-baseline', { path: relativePath });
    expect(loaded.metrics.mrr_at_5).toBe(0.7);
  });

  it('detects a latency regression warning', () => {
    const current = {
      condition: 'hybrid',
      metrics: { mrr_at_5: 0.5, latency_ms: { p95: 250 } },
    };
    const baseline = {
      condition: 'hybrid',
      metrics: { mrr_at_5: 0.5, latency_ms: { p95: 1 } },
    };
    const report = compareToBaseline(current, baseline, { latencyWarnMs: 100 });
    expect(report.warnings).toContain('latency_p95');
  });

  it('detects a metric regression failure', () => {
    const current = {
      condition: 'hybrid',
      metrics: { mrr_at_5: 0.1, ndcg_at_5: 0.1, recall_at_5: 0.1 },
    };
    const baseline = {
      condition: 'hybrid',
      metrics: { mrr_at_5: 1, ndcg_at_5: 1, recall_at_5: 1 },
    };
    const report = compareToBaseline(current, baseline, {
      failThresholds: { mrr_at_5: 0.01 },
    });
    expect(report.regressions).toContain('mrr_at_5');
    expect(report.pass).toBe(false);
  });
});

describe('eval-compare coverage', () => {
  it('throws when Wilcoxon inputs are invalid', () => {
    expect(() => wilcoxonSignedRankTest(null, [1, 2])).toThrow();
    expect(() => wilcoxonSignedRankTest([1, 2], 'not-array')).toThrow();
    expect(() => wilcoxonSignedRankTest([1], [1, 2])).toThrow();
  });

  it('returns a non-significant result for empty series', () => {
    const stats = wilcoxonSignedRankTest([], []);
    expect(stats.significant).toBe(false);
  });

  it('handles series where all differences are zero', () => {
    const stats = wilcoxonSignedRankTest([0.5, 0.5], [0.5, 0.5]);
    expect(stats.p_value).toBe(1);
  });

  it('handles ties in compareResults as non-significant', () => {
    const resultA = {
      condition: 'a',
      query_count: 2,
      metrics: { mrr_at_5: 0.5 },
      per_query: [
        { query_id: 'q1', mrr_at_5: 0.5 },
        { query_id: 'q2', mrr_at_5: 0.5 },
      ],
    };
    const resultB = {
      condition: 'b',
      query_count: 2,
      metrics: { mrr_at_5: 0.5 },
      per_query: [
        { query_id: 'q1', mrr_at_5: 0.5 },
        { query_id: 'q2', mrr_at_5: 0.5 },
      ],
    };
    const report = compareResults(resultA, resultB);
    expect(report.metrics.mrr_at_5.significant).toBe(false);
    expect(report.per_query_wins.ties).toBe(2);
  });

  it('throws when compareResults receives invalid inputs', () => {
    expect(() => compareResults(null, { condition: 'b' })).toThrow();
    expect(() => compareResults({ condition: 'a' }, null)).toThrow();
  });

  it('returns an empty alpha sweep for empty inputs', async () => {
    const sweep = await alphaSweep({});
    expect(sweep).toEqual([]);
  });

  it('ignores non-array alphas in alphaSweep', async () => {
    const sweep = await alphaSweep({ queries: [], alphas: '1,2' });
    expect(sweep).toEqual([]);
  });

  it('computes per-class deltas in compareResults', () => {
    const resultA = {
      condition: 'a',
      query_count: 1,
      metrics: { mrr_at_5: 0.4 },
      per_query: [{ query_id: 'q1', class: 'simple_lookup', mrr_at_5: 0.4 }],
      per_class: { simple_lookup: { mrr_at_5: 0.4 } },
    };
    const resultB = {
      condition: 'b',
      query_count: 1,
      metrics: { mrr_at_5: 0.6 },
      per_query: [{ query_id: 'q1', class: 'simple_lookup', mrr_at_5: 0.6 }],
      per_class: { simple_lookup: { mrr_at_5: 0.6 } },
    };
    const report = compareResults(resultA, resultB);
    expect(report.per_class_delta).toHaveProperty('simple_lookup');
  });
});

describe('eval-runner coverage', () => {
  let consoleLogSpy;
  let consoleErrorSpy;
  let tempDir;

  beforeEach(async () => {
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    process.exitCode = undefined;
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    tempDir = await mkdtemp(path.join(process.cwd(), '.test-runner-'));
  });

  afterEach(async () => {
    consoleLogSpy.mockRestore();
    consoleErrorSpy.mockRestore();
    process.exitCode = undefined;
    delete process.env.EVAL_FORCE_SYNTHETIC;
    await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
  });

  it('validates a well-formed query', () => {
    expect(() => validateQuerySchema(makeFullQuery())).not.toThrow();
  });

  it('throws when query is not an object', () => {
    expect(() => validateQuerySchema(null)).toThrow();
  });

  it('throws when query_id is missing', () => {
    expect(() =>
      validateQuerySchema({ query: 'test', class: 'simple_lookup' }),
    ).toThrow();
  });

  it('throws when query text is missing', () => {
    expect(() =>
      validateQuerySchema({ query_id: 'x', query: '', class: 'simple_lookup' }),
    ).toThrow();
  });

  it('throws for an unsupported query class', () => {
    expect(() =>
      validateQuerySchema(makeQuery({ class: 'unknown_class' })),
    ).toThrow();
  });

  it('throws for an invalid difficulty value', () => {
    expect(() =>
      validateQuerySchema(makeQuery({ difficulty: 'very_hard' })),
    ).toThrow();
  });

  it('throws when expected_doc_families is not an array', () => {
    expect(() =>
      validateQuerySchema(makeQuery({ expected_doc_families: undefined })),
    ).toThrow();
  });

  it('throws when a relevance grade entry is not an object', () => {
    expect(() =>
      validateQuerySchema(makeQuery({ relevance_grades: ['bad'] })),
    ).toThrow();
  });

  it('throws when a relevance grade is out of range', () => {
    expect(() =>
      validateQuerySchema(makeQuery({ relevance_grades: [{ grade: 5 }] })),
    ).toThrow();
  });

  it('throws when expected_chunk_ids is not an array', () => {
    expect(() =>
      validateQuerySchema(makeQuery({ expected_chunk_ids: 'bad' })),
    ).toThrow();
  });

  it('throws when runEval is called without options', async () => {
    await expect(runEval()).rejects.toThrow();
  });

  it('throws for an unsupported condition in runEval', async () => {
    await expect(
      runEval({ queries: [makeFullQuery()], condition: 'no_such_condition' }),
    ).rejects.toThrow();
  });

  it('falls back to synthetic search when no search function is provided', async () => {
    const result = await runEval({
      queries: [makeFullQuery()],
      condition: 'bm25_only',
    });
    expect(result.metrics).toHaveProperty('mrr_at_5');
  });

  it('runs a self-test successfully', async () => {
    const result = await runSelfTest();
    expect(result.pass).toBe(true);
  });

  it('prints help from the CLI', async () => {
    await runCli(['--help']);
    expect(consoleLogSpy).toHaveBeenCalled();
  });

  it('runs all conditions from the CLI', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli(['--query-file', queryFile]);
    expect(consoleLogSpy).toHaveBeenCalled();
  });

  it('compares two conditions from the CLI', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli([
      '--compare',
      '--condition',
      'bm25_only',
      '--condition',
      'hybrid',
      '--query-file',
      queryFile,
    ]);
    expect(consoleLogSpy).toHaveBeenCalled();
  });

  it('runs an alpha sweep from the CLI', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli(['--alpha-sweep', '0,1', '--query-file', queryFile]);
    expect(consoleLogSpy).toHaveBeenCalled();
  });

  it('detects a regression from the CLI baseline mode', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(
      queryFile,
      JSON.stringify([
        makeFullQuery({
          expected_chunk_ids: ['missing-chunk'],
          relevance_grades: [],
        }),
      ]),
    );
    const baselinePath = path.join(tempDir, 'baseline-latest.json');
    await writeFile(
      baselinePath,
      JSON.stringify({
        conditions: {
          hybrid: {
            mrr_at_5: 1,
            ndcg_at_5: 1,
            recall_at_5: 1,
            latency_ms: { p95: 1 },
          },
        },
      }),
    );
    await runCli([
      '--condition',
      'hybrid',
      '--query-file',
      queryFile,
      '--baseline',
      baselinePath,
      '--regression-threshold',
      '0.01',
    ]);
    expect(process.exitCode).toBe(1);
  });

  it('fails gracefully for an unsupported CLI condition', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli(['--condition', 'bad_condition', '--query-file', queryFile]);
    expect(process.exitCode).toBe(1);
  });

  it('fails gracefully for a missing query file', async () => {
    await runCli(['--query-file', path.join(tempDir, 'missing.json')]);
    expect(process.exitCode).toBe(1);
  });

  it('emits JSON output from the CLI', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli([
      '--query-file',
      queryFile,
      '--condition',
      'bm25_only',
      '--json',
    ]);
    const output = consoleLogSpy.mock.calls[0]?.[0];
    expect(typeof output).toBe('string');
  });

  it('executes bootstrap when the module is the main entrypoint', async () => {
    const originalArgv = process.argv.slice();
    process.argv = [process.argv[0], runnerPath, '--help'];
    await bootstrap();
    process.argv = originalArgv;
    expect(consoleLogSpy).toHaveBeenCalled();
  });
});

describe('eval-metrics remaining coverage', () => {
  it('throws when computeMrr results is not an array', () => {
    expect(() => computeMrr(null, { expected_doc_families: [] }, 5)).toThrow(
      'results must be an array.',
    );
  });

  it('throws when computeMrr querySpec is missing', () => {
    expect(() => computeMrr([], null, 5)).toThrow('querySpec is required.');
  });

  it('computes context relevance for non-empty chunks', () => {
    const query = { expected_doc_families: ['ts-source'] };
    const chunks = [
      { family: 'ts-source', heading_path: 'network' },
      { family: 'readme', heading_path: 'readme' },
    ];
    expect(computeContextRelevance(chunks, query)).toBe(0.5);
  });

  it('returns zero context relevance when no chunks match', () => {
    const query = { expected_doc_families: ['ts-source'] };
    const chunks = [{ family: 'readme', heading_path: 'readme' }];
    expect(computeContextRelevance(chunks, query)).toBe(0);
  });

  it('clamps computeLatency to non-negative values', () => {
    expect(computeLatency(10, 5)).toBe(0);
    expect(computeLatency(undefined, undefined)).toBe(0);
  });
});

describe('eval-compare remaining coverage', () => {
  it('computes Wilcoxon for distinct non-zero differences', () => {
    const test = wilcoxonSignedRankTest([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]);
    expect(test.p_value).toBeLessThan(0.05);
    expect(test.significant).toBe(true);
  });

  it('computes Wilcoxon with tied absolute differences', () => {
    const test = wilcoxonSignedRankTest([2, 4, 6], [0, 2, 4]);
    expect(test.significant).toBe(false);
    expect(Number.isFinite(test.p_value)).toBe(true);
  });

  it('compares results with per-query series and different classes', () => {
    const resultA = {
      condition: 'bm25_only',
      metrics: { mrr_at_5: 0.1, ndcg_at_5: 0.2, recall_at_5: 0.3 },
      per_query: [{ mrr_at_5: 0.1, ndcg_at_5: 0.2, recall_at_5: 0.3 }],
      per_class: { simple_lookup: { mrr_at_5: 0.1 } },
    };
    const resultB = {
      condition: 'hybrid',
      metrics: { mrr_at_5: 0.5, ndcg_at_5: 0.6, recall_at_5: 0.7 },
      per_query: [{ mrr_at_5: 0.5, ndcg_at_5: 0.6, recall_at_5: 0.7 }],
      per_class: { cross_boundary: { mrr_at_5: 0.5 } },
    };
    const report = compareResults(resultA, resultB);
    expect(report.per_query_wins.b_wins).toBe(1);
    expect(report.per_class_delta).toHaveProperty('cross_boundary');
    expect(report.per_class_delta).toHaveProperty('simple_lookup');
  });
});

describe('eval-baseline remaining coverage', () => {
  it('stores and loads a baseline at the default path', async () => {
    const baseline = {
      baseline_id: 'default-path-test',
      timestamp: new Date().toISOString(),
      conditions: { hybrid: { mrr_at_5: 0.5 } },
    };
    const stored = await storeBaseline(baseline);
    const loaded = await loadBaseline('default-path-test');
    expect(loaded.baseline_id).toBe('default-path-test');
    expect(loaded.conditions.hybrid.mrr_at_5).toBe(0.5);
    expect(stored.timestamp).toBe(baseline.timestamp);
  });
});

describe('eval-runner remaining coverage', () => {
  let tempDir;

  beforeEach(async () => {
    tempDir = await mkdtemp(path.join(process.cwd(), '.test-runner-extra-'));
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('rejects when runEval receives null options', async () => {
    await expect(runEval(null)).rejects.toThrow(
      'runEval requires an options object.',
    );
  });

  it('builds condition options for all supported conditions', () => {
    expect(buildConditionOptions('bm25_only', {})).toEqual({
      useDense: false,
      useRerank: false,
      alpha: 1,
      limit: 10,
    });
    expect(buildConditionOptions('hybrid', {})).toEqual({
      useDense: true,
      useRerank: false,
      alpha: 0.5,
      limit: 10,
    });
    expect(buildConditionOptions('hybrid_rerank', {})).toEqual({
      useDense: true,
      useRerank: true,
      alpha: 0.5,
      limit: 10,
    });
    expect(buildConditionOptions('advanced_default', {})).toEqual({
      useDense: true,
      useRerank: true,
      alpha: 0.5,
      limit: 10,
      contextBudget: 4096,
    });
  });

  it('throws for an unsupported condition in buildConditionOptions', () => {
    expect(() => buildConditionOptions('unknown', {})).toThrow(
      'Unsupported condition: unknown',
    );
  });

  it('uses the provided search function', async () => {
    const searchFn = () => [
      { chunk_id: 1, family: 'readme', heading_path: '' },
    ];
    const fn = await resolveSearchFn('hybrid', { searchFn });
    expect(fn).toBe(searchFn);
  });

  it('falls back to synthetic search when forced by env', async () => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    const fn = await resolveSearchFn('hybrid', {});
    const result = await fn({ expected_doc_families: ['readme'] }, {});
    expect(Array.isArray(result)).toBe(true);
  });

  it('falls back to synthetic search when the search module is missing', async () => {
    const fn = await resolveSearchFn('hybrid', {
      searchModulePath: path.join(tempDir, 'missing-search.mjs'),
    });
    const result = await fn({ expected_doc_families: ['readme'] }, {});
    expect(Array.isArray(result)).toBe(true);
  });

  it('runs eval with empty expected_doc_families using the synthetic family fallback', async () => {
    const codeQuery = makeQuery({
      query_id: 'code-empty',
      class: 'code_specific',
      expected_doc_families: [],
    });
    const readmeQuery = makeQuery({
      query_id: 'readme-empty',
      class: 'simple_lookup',
      expected_doc_families: [],
    });
    const result = await runEval({
      queries: [codeQuery, readmeQuery],
      condition: 'hybrid',
      searchModulePath: path.join(tempDir, 'missing-search.mjs'),
    });
    expect(result.query_count).toBe(2);
    expect(Number.isFinite(result.metrics.mrr_at_5)).toBe(true);
  });

  it('runs all supported conditions through runAllConditions', async () => {
    const queryFile = path.join(tempDir, 'queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    const results = await runAllConditions({
      queryFilePath: queryFile,
      searchModulePath: path.join(tempDir, 'missing-search.mjs'),
    });
    expect(Object.keys(results).sort()).toEqual([
      'advanced_default',
      'bm25_only',
      'hybrid',
      'hybrid_rerank',
    ]);
  });

  it('fails when the query file does not contain an array', async () => {
    const queryFile = path.join(tempDir, 'bad-queries.json');
    await writeFile(queryFile, JSON.stringify({ not: 'array' }));
    await runCli(['--query-file', queryFile]);
    expect(process.exitCode).toBe(1);
  });
});

describe('eval-coverage extra', () => {
  let tempDir;

  beforeAll(async () => {
    tempDir = await mkdtemp(path.join(process.cwd(), '.test-extra-'));
  });

  beforeEach(() => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
    tempDir = await mkdtemp(path.join(process.cwd(), '.test-extra-'));
    process.exitCode = 0;
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('stores baseline at an absolute custom path', async () => {
    const abs = path.join(tempDir, 'absolute-baseline.json');
    await storeBaseline({ baseline_id: 'abs', conditions: {} }, { path: abs });
    const loaded = await loadBaseline('abs', { path: abs });
    expect(loaded.baseline_id).toBe('abs');
  });

  it('loads baseline latency p95 from an object', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { latency_ms: { p95: 12 } } },
      { condition: 'hybrid', metrics: { latency_ms: { p95: 5 } } },
      { latencyWarnMs: 5 },
    );
    expect(report.warnings).toContain('latency_p95');
  });

  it('reports a metric warning without a fail threshold', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { ndcg_at_5: 0.2 } },
      { condition: 'hybrid', metrics: { ndcg_at_5: 0.3 } },
      { warnThresholds: { ndcg_at_5: 0.05 } },
    );
    expect(report.pass).toBe(true);
    expect(report.warnings).toContain('ndcg_at_5');
    expect(report.metrics.ndcg_at_5.status).toBe('WARN');
  });

  it('returns PASS with no thresholds', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: {} },
      { condition: 'hybrid', metrics: {} },
      {},
    );
    expect(report.pass).toBe(true);
  });

  it('uses symbol_name fallback in matchesQueryExpectation', () => {
    const query = {
      expected_doc_families: [],
      expected_symbol_contains: 'foo',
    };
    const result = { chunk_id: 1, family: 'readme', symbol_name: 'foo bar' };
    expect(matchesQueryExpectation(result, query)).toBe(true);
  });

  it('matches heading needle via symbol_name', () => {
    const query = {
      expected_doc_families: [],
      expected_heading_contains: 'bar',
    };
    const result = { chunk_id: 1, family: 'readme', symbol_name: 'bar baz' };
    expect(matchesQueryExpectation(result, query)).toBe(true);
  });

  it('returns false when family does not match', () => {
    const query = { expected_doc_families: ['ts-source'] };
    const result = { chunk_id: 1, family: 'readme', heading_path: '' };
    expect(matchesQueryExpectation(result, query)).toBe(false);
  });

  it('returns true with no family restrictions', () => {
    const query = { expected_doc_families: [] };
    const result = { chunk_id: 1, family: 'readme', heading_path: '' };
    expect(matchesQueryExpectation(result, query)).toBe(true);
  });

  it('grades a result with empty family and heading rule', () => {
    const query = { relevance_grades: [{ grade: 3 }] };
    const result = { family: 'any', heading_path: 'any' };
    expect(computeNdcg([result], query, 5)).toBe(1);
  });

  it('treats a non-finite grade as zero', () => {
    const query = {
      relevance_grades: [
        { family: 'any', heading_path_contains: 'any', grade: Number.NaN },
      ],
    };
    const result = { family: 'any', heading_path: 'any' };
    expect(computeNdcg([result], query, 5)).toBe(0);
  });

  it('computes recall using expected chunk ids', () => {
    const query = { expected_chunk_ids: [10, 20] };
    const results = [
      { chunk_id: 20, family: 'readme', heading_path: '' },
      { chunk_id: 30, family: 'readme', heading_path: '' },
    ];
    expect(computeRecall(results, query, 5)).toBe(0.5);
  });

  it('computes recall using relevant pool fallback', () => {
    const query = {
      expected_doc_families: ['readme'],
      expected_chunk_ids: [],
    };
    const results = [
      { chunk_id: 1, family: 'readme', heading_path: '' },
      { chunk_id: 2, family: 'ts-source', heading_path: '' },
    ];
    expect(computeRecall(results, query, 5)).toBe(1);
  });

  it('returns zero recall when no relevant pool exists', () => {
    const query = { expected_doc_families: ['readme'] };
    const results = [{ chunk_id: 1, family: 'ts-source', heading_path: '' }];
    expect(computeRecall(results, query, 5)).toBe(0);
  });

  it('aggregates latency and metrics', () => {
    const perQuery = [
      {
        latency_ms: 10,
        mrr_at_5: 1,
        ndcg_at_5: 0.8,
        recall_at_5: 0.5,
        class: 'a',
      },
      {
        latency_ms: 20,
        mrr_at_5: 0,
        ndcg_at_5: 0.4,
        recall_at_5: 0.2,
        class: 'a',
      },
    ];
    const metrics = aggregateMetrics(perQuery);
    expect(metrics.mrr_at_5).toBe(0.5);
    expect(metrics.latency_ms.p50).toBe(15);
    const byClass = aggregateByClass(perQuery);
    expect(byClass.a.mrr_at_5).toBe(0.5);
  });

  it('aggregates with empty rows', () => {
    expect(aggregateMetrics([]).mrr_at_5).toBe(0);
    expect(aggregateByClass([])).toEqual({});
  });

  it('validates a query with valid relevance grades', () => {
    const q = makeFullQuery({
      relevance_grades: [
        { family: 'readme', heading_path_contains: 'x', grade: 2 },
      ],
    });
    expect(validateQuerySchema(q).query_id).toBe(q.query_id);
  });

  it('throws for a non-integer relevance grade', () => {
    const q = makeFullQuery({
      relevance_grades: [{ grade: 1.5 }],
    });
    expect(() => validateQuerySchema(q)).toThrow(
      'relevance grade must be an integer in [0, 3].',
    );
  });

  it('executes query with searchFn returning a results wrapper', async () => {
    const searchFn = async () => ({
      results: [{ chunk_id: 1, family: 'readme', heading_path: 'foo' }],
    });
    const res = await runEval({
      queries: [makeFullQuery({ expected_doc_families: ['readme'] })],
      condition: 'hybrid',
      searchFn,
    });
    expect(res.query_count).toBe(1);
  });

  it('runs default query file via runAllConditions', async () => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    const results = await runAllConditions();
    expect(Object.keys(results)).toContain('hybrid');
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('runs CLI with a custom query file', async () => {
    const queryFile = path.join(tempDir, 'cli-queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli([
      '--query-file',
      queryFile,
      '--condition',
      'hybrid',
      '--json',
    ]);
    expect(process.exitCode).toBe(0);
  });

  it('skips missing baseline conditions during regression', async () => {
    const baselineFile = path.join(tempDir, 'partial-baseline.json');
    await writeFile(
      baselineFile,
      JSON.stringify({
        baseline_id: 'partial',
        conditions: { hybrid: { mrr_at_5: 1.02 } },
      }),
    );
    const queryFile = path.join(tempDir, 'reg-queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    await runCli([
      '--query-file',
      queryFile,
      '--baseline',
      baselineFile,
      '--condition',
      'all',
      '--json',
    ]);
    expect(process.exitCode).toBe(1);
  });

  it('counts ties in per-query wins', () => {
    const a = {
      condition: 'a',
      metrics: { mrr_at_5: 0.5 },
      per_query: [{ mrr_at_5: 0.5 }, { mrr_at_5: 0.5 }],
    };
    const b = {
      condition: 'b',
      metrics: { mrr_at_5: 0.5 },
      per_query: [{ mrr_at_5: 0.5 }, { mrr_at_5: 0.5 }],
    };
    const comp = compareResults(a, b);
    expect(comp.per_query_wins.ties).toBe(2);
  });

  it('computes Wilcoxon with negative z-score', () => {
    const test = wilcoxonSignedRankTest([0, 0, 0], [1, 2, 3]);
    expect(test.significant).toBe(false);
    expect(Number.isFinite(test.p_value)).toBe(true);
  });

  it('runs alpha sweep with default condition and empty queries', async () => {
    const sweep = await alphaSweep();
    expect(sweep).toEqual([]);
  });

  it('matches expectations when optional fields are absent', () => {
    expect(matchesQueryExpectation({}, {})).toBe(true);
    expect(
      matchesQueryExpectation(
        { family: 'readme' },
        { expected_heading_contains: 'missing' },
      ),
    ).toBe(false);
    expect(
      matchesQueryExpectation(
        { family: 'readme', heading_path: 'foo' },
        { expected_symbol_contains: 'missing' },
      ),
    ).toBe(false);
    expect(
      matchesQueryExpectation(
        { doc_family: 'ts-source' },
        { expected_doc_families: ['ts-source'] },
      ),
    ).toBe(true);
    expect(
      matchesQueryExpectation(
        { symbol_name: 'bar' },
        { expected_heading_contains: 'bar' },
      ),
    ).toBe(true);
  });

  it('computes nDCG with relevance grade rules', () => {
    const query = {
      expected_doc_families: ['readme'],
      relevance_grades: [
        { family: 'readme', heading_path_contains: 'foo', grade: 3 },
        { family: 'other', heading_path_contains: 'foo', grade: 3 },
        { heading_path_contains: 'bar', grade: 2 },
        { grade: 1 },
      ],
    };
    const results = [
      { family: 'readme', heading_path: 'foo' },
      { family: 'readme', heading_path: 'foo' },
      { family: 'readme', heading_path: 'baz' },
    ];
    expect(computeNdcg(results, query, 5)).toBeGreaterThan(0);
  });

  it('falls back to binary relevance when grade rules do not match', () => {
    const query = {
      expected_doc_families: ['readme'],
      relevance_grades: [
        { family: 'other', heading_path_contains: 'foo', grade: 3 },
      ],
    };
    const results = [{ family: 'readme', heading_path: 'foo' }];
    expect(computeNdcg(results, query, 5)).toBe(1);
  });

  it('uses a missing grade default of 0 in relevance rules', () => {
    const query = {
      expected_doc_families: ['readme'],
      relevance_grades: [{ heading_path_contains: 'foo' }],
    };
    const results = [{ family: 'readme', heading_path: 'foo' }];
    expect(computeNdcg(results, query, 5)).toBe(0);
  });

  it('normalCdf handles negative inputs', () => {
    expect(normalCdf(-1.5)).toBeLessThan(0.5);
    expect(normalCdf(1.5)).toBeGreaterThan(0.5);
  });

  it('compareResults tolerates missing or sparse per_query arrays', () => {
    const a = { metrics: { mrr_at_5: 1 } };
    const b = { metrics: { mrr_at_5: 0.5 } };
    const comp = compareResults(a, b);
    expect(comp.metrics.mrr_at_5.a).toBe(1);
    expect(comp.metrics.mrr_at_5.b).toBe(0.5);
  });

  it('countWins handles null rows in per_query', () => {
    const a = {
      condition: 'a',
      metrics: {},
      per_query: [{ mrr_at_5: 1 }, null, undefined],
    };
    const b = {
      condition: 'b',
      metrics: {},
      per_query: [{ mrr_at_5: 0 }, { mrr_at_5: 0 }, { mrr_at_5: 0 }],
    };
    const comp = compareResults(a, b);
    expect(comp.per_query_wins.a_wins).toBe(1);
    expect(comp.per_query_wins.ties).toBe(2);
  });

  it('compareToBaseline uses default p95 of 0 when latency object lacks p95', () => {
    const report = compareToBaseline(
      {
        condition: 'hybrid',
        metrics: {
          latency_ms: { p95: 200 },
          mrr_at_5: 0.5,
          ndcg_at_5: 0.5,
          recall_at_5: 0.5,
        },
      },
      {
        condition: 'hybrid',
        metrics: {
          latency_ms: { p95: null },
          mrr_at_5: 0.6,
          ndcg_at_5: 0.6,
          recall_at_5: 0.6,
        },
      },
      { failThresholds: { mrr_at_5: 0.01 }, latencyWarnMs: 5 },
    );
    expect(report.pass).toBe(false);
    expect(report.warnings).toContain('latency_p95');
  });

  it('compareToBaseline warns on a metric-only threshold', () => {
    const report = compareToBaseline(
      {
        condition: 'hybrid',
        metrics: { mrr_at_5: 0.5, custom_metric: 0.5 },
      },
      {
        condition: 'hybrid',
        metrics: { mrr_at_5: 0.6, custom_metric: 0.6 },
      },
      { warnThresholds: { custom_metric: 0.05 } },
    );
    expect(report.warnings).toContain('custom_metric');
  });

  it('stores and loads a baseline with an absolute path', async () => {
    const absolutePath = path.join(tempDir, 'abs-baseline.json');
    await storeBaseline(
      { baseline_id: 'abs-test', conditions: { hybrid: { mrr_at_5: 0.9 } } },
      { path: absolutePath },
    );
    const loaded = await loadBaseline('abs-test', { path: absolutePath });
    expect(loaded.conditions.hybrid.mrr_at_5).toBe(0.9);
  });

  it('runs CLI with default query file path', async () => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    await runCli(['--condition', 'hybrid', '--json']);
    expect(process.exitCode).toBe(0);
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('runs CLI with a baseline that causes a regression failure', async () => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    const baselineFile = path.join(tempDir, 'bad-baseline.json');
    await writeFile(
      baselineFile,
      JSON.stringify({
        baseline_id: 'bad',
        conditions: {
          hybrid: {
            mrr_at_5: 1,
            ndcg_at_5: 1,
            recall_at_5: 1,
            latency_ms: { p95: 1 },
          },
        },
      }),
    );
    await runCli([
      '--baseline',
      baselineFile,
      '--condition',
      'hybrid',
      '--json',
    ]);
    expect(process.exitCode).toBe(1);
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('executes query with selectedChunks wrapper', async () => {
    const searchFn = async () => ({
      results: [{ chunk_id: 1, family: 'readme', heading_path: 'foo' }],
      selectedChunks: [{ chunk_id: 1, family: 'readme', heading_path: 'foo' }],
    });
    const res = await runEval({
      queries: [makeFullQuery({ expected_doc_families: ['readme'] })],
      condition: 'hybrid',
      searchFn,
    });
    expect(res.query_count).toBe(1);
  });

  it('runAllConditions uses a provided queryFilePath', async () => {
    const queryFile = path.join(tempDir, 'all-queries.json');
    await writeFile(queryFile, JSON.stringify([makeFullQuery()]));
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    const results = await runAllConditions({ queryFilePath: queryFile });
    expect(results.hybrid.query_count).toBe(1);
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('validates a query with no relevance grades', () => {
    const q = makeFullQuery();
    delete q.relevance_grades;
    expect(validateQuerySchema(q).query_id).toBe(q.query_id);
  });

  it('executes query with context.results wrapper', async () => {
    const searchFn = async () => ({
      context: {
        results: [{ chunk_id: 1, family: 'readme', heading_path: 'foo' }],
        selectedChunks: [{ chunk_id: 1 }],
      },
    });
    const res = await runEval({
      queries: [makeFullQuery({ expected_doc_families: ['readme'] })],
      condition: 'hybrid',
      searchFn,
    });
    expect(res.query_count).toBe(1);
  });

  it('executes query with opaque search result', async () => {
    const searchFn = async () => ({ foo: 'bar' });
    const res = await runEval({
      queries: [makeFullQuery()],
      condition: 'hybrid',
      searchFn,
    });
    expect(res.query_count).toBe(1);
  });

  it('grades result using doc_family fallback', () => {
    const query = {
      expected_doc_families: ['readme'],
      relevance_grades: [
        { family: 'readme', heading_path_contains: 'foo', grade: 3 },
      ],
    };
    const result = [{ doc_family: 'readme', heading_path: 'foo' }];
    expect(computeNdcg(result, query, 5)).toBe(1);
  });

  it('aggregates zero-hit metrics', () => {
    const metrics = aggregateMetrics([
      {
        mrr_at_5: 0,
        ndcg_at_5: 0,
        recall_at_5: 0,
        latency_ms: 0,
        class: 'a',
        query_id: 'q1',
      },
    ]);
    expect(metrics.zero_hit_queries).toBe(1);
    expect(metrics.latency_ms.max).toBe(0);
  });

  it('compareResults tolerates missing metrics objects', () => {
    const comp = compareResults({ per_query: [] }, { per_query: [] });
    expect(comp.metrics.mrr_at_5.a).toBe(0);
    expect(comp.metrics.mrr_at_5.b).toBe(0);
  });

  it('countWins handles null rows on both sides', () => {
    const a = {
      condition: 'a',
      metrics: {},
      per_query: [{ mrr_at_5: 1 }, null],
    };
    const b = {
      condition: 'b',
      metrics: {},
      per_query: [null, { mrr_at_5: 1 }],
    };
    const comp = compareResults(a, b);
    expect(comp.per_query_wins.a_wins).toBe(1);
    expect(comp.per_query_wins.b_wins).toBe(1);
  });

  it('compareToBaseline warns when current metric is missing', () => {
    const report = compareToBaseline(
      { condition: 'hybrid', metrics: { mrr_at_5: 0.5 } },
      { condition: 'hybrid', metrics: { mrr_at_5: 0.6, custom_metric: 0.6 } },
      { warnThresholds: { custom_metric: 0.05 } },
    );
    expect(report.warnings).toContain('custom_metric');
  });

  it('runAllConditions accepts an inline query array', async () => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    const results = await runAllConditions({ queries: [makeFullQuery()] });
    expect(results.hybrid.query_count).toBe(1);
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });

  it('runs CLI with a baseline that passes regression', async () => {
    process.env.EVAL_FORCE_SYNTHETIC = '1';
    const baselineFile = path.join(tempDir, 'good-baseline.json');
    await writeFile(
      baselineFile,
      JSON.stringify({
        baseline_id: 'good',
        conditions: {
          hybrid: {
            mrr_at_5: 0,
            ndcg_at_5: 0,
            recall_at_5: 0,
            latency_ms: { p95: 99999 },
          },
        },
      }),
    );
    await runCli([
      '--baseline',
      baselineFile,
      '--condition',
      'hybrid',
      '--json',
    ]);
    expect(process.exitCode).toBe(0);
    delete process.env.EVAL_FORCE_SYNTHETIC;
  });
});
