import { jest } from '@jest/globals';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

// Capture real contract module before mocking
const realContract =
  await import('../../../rag-index/docs-quality/docs-quality.contract.mjs');

// Mutable mock state
let scanImpl = async () => scanResult;
let validationOverride = null;

let scanResult = {
  pass: true,
  evidence: [],
  issueDimensions: [],
  fixHint: null,
  owner: 'docs',
};
let cliArgs = { help: false, json: false };
let printHelpCalls = [];
let writeJsonCalls = [];
let failCalls = [];

// Mock code-quality-scanner.mjs
jest.unstable_mockModule('../../../rag-index/code-quality-scanner.mjs', () => ({
  scanCodeQuality: async (opts) => scanImpl(opts),
}));

// Mock cli-utils.mjs
jest.unstable_mockModule('../../../rag-index/cli-utils.mjs', () => ({
  parseCliArgs: (argv) => ({
    ...cliArgs,
    _: argv.filter((a) => !a.startsWith('--')),
  }),
  printHelp: (config) => {
    printHelpCalls.push(config);
  },
  writeJsonOrText: (payload, asJson, textFormatter) => {
    writeJsonCalls.push({ payload, asJson, textFormatter });
    if (!asJson && textFormatter) textFormatter(payload);
  },
  fail: (message, asJson) => {
    failCalls.push({ message, asJson });
    process.exitCode = 1;
  },
  toRepoRelative: (filePath) => filePath.replaceAll('\\', '/'),
}));

// Mock docs-quality.contract.mjs — override validateDocsQualityManifestV1 for invalid manifest test
jest.unstable_mockModule(
  '../../../rag-index/docs-quality/docs-quality.contract.mjs',
  () => ({
    ...realContract,
    validateDocsQualityManifestV1: (manifest) =>
      validationOverride ??
      realContract.validateDocsQualityManifestV1(manifest),
  }),
);

// Import metrics module AFTER all mocks
const { runDocsQualityMetrics } =
  await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');

const PROJECT_ROOT = process.cwd();
const METRICS_MODULE_PATH = path.resolve(
  PROJECT_ROOT,
  'rag-index',
  'docs-quality',
  'docs-quality.metrics.mjs',
);
const TMP_BASE = path.resolve(PROJECT_ROOT, '__test_tmp_metrics_cov');

function createTempDir() {
  rmSync(TMP_BASE, { recursive: true, force: true });
  mkdirSync(TMP_BASE, { recursive: true });
  return TMP_BASE;
}

function cleanupTempDir() {
  try {
    rmSync(TMP_BASE, { recursive: true, force: true });
  } catch {
    // ignore
  }
}

function writeCoverageFiles(dir, opts = {}) {
  const covDir = path.join(dir, 'coverage');
  mkdirSync(covDir, { recursive: true });

  if (opts.lcovContent !== undefined) {
    writeFileSync(path.join(covDir, 'lcov.info'), opts.lcovContent, 'utf8');
  }

  if (opts.coverageSummary !== undefined) {
    writeFileSync(
      path.join(covDir, 'coverage-summary.json'),
      typeof opts.coverageSummary === 'string'
        ? opts.coverageSummary
        : JSON.stringify(opts.coverageSummary),
      'utf8',
    );
  }
}

function lcovRecord(sf, opts = {}) {
  const lines = [`SF:${sf}`];
  const lh = opts.lineHits ?? 0;
  const lf = opts.lineFound ?? 0;
  const brh = opts.branchHits ?? 0;
  const brf = opts.branchFound ?? 0;
  const fnh = opts.functionHits ?? 0;
  const fnf = opts.functionFound ?? 0;

  lines.push(`LF:${lf}`);
  lines.push(`LH:${lh}`);
  lines.push(`BRF:${brf}`);
  lines.push(`BRH:${brh}`);
  lines.push(`FNF:${fnf}`);
  lines.push(`FNH:${fnh}`);

  if (opts.daLines) {
    for (const [lineNum, hitCount] of opts.daLines) {
      lines.push(`DA:${lineNum},${hitCount}`);
    }
  }

  if (opts.brdaLines) {
    for (const [lineNum, blockNum, branchNum, taken] of opts.brdaLines) {
      lines.push(`BRDA:${lineNum},${blockNum},${branchNum},${taken}`);
    }
  }

  if (opts.fndaLines) {
    for (const [hitCount, funcName] of opts.fndaLines) {
      lines.push(`FNDA:${hitCount},${funcName}`);
    }
  }

  lines.push('end_of_record');
  return lines.join('\n');
}

describe('docs-quality.metrics.mjs coverage', () => {
  let originalCwd;
  let tempDir;
  let originalGitCommit;

  beforeEach(() => {
    originalCwd = process.cwd();
    tempDir = createTempDir();
    process.chdir(tempDir);
    originalGitCommit = process.env.GIT_COMMIT;
    delete process.env.GIT_COMMIT;

    scanResult = {
      pass: true,
      evidence: [],
      issueDimensions: [],
      fixHint: null,
      owner: 'docs',
    };
    scanImpl = async () => scanResult;
    cliArgs = { help: false, json: false };
    printHelpCalls = [];
    writeJsonCalls = [];
    failCalls = [];
    validationOverride = null;
    process.exitCode = 0;
  });

  afterEach(() => {
    if (originalGitCommit !== undefined) {
      process.env.GIT_COMMIT = originalGitCommit;
    } else {
      delete process.env.GIT_COMMIT;
    }
    process.chdir(originalCwd);
    cleanupTempDir();
  });

  // -------------------------------------------------------------------------
  // Basic paths
  // -------------------------------------------------------------------------
  describe('runDocsQualityMetrics — basic paths', () => {
    it('runs with default options and no coverage files', async () => {
      const result = await runDocsQualityMetrics();
      expect(result.pass).toBe(true);
      expect(result.evidence).toEqual([]);
      expect(result.summary.coverage.available).toBe(false);
    });

    it('runs with custom thresholds', async () => {
      const result = await runDocsQualityMetrics({
        complexityThreshold: 15,
        minJsdocWords: 5,
      });
      expect(result.manifest.thresholdConfig.complexityThreshold).toBe(15);
      expect(result.manifest.thresholdConfig.minJsdocWords).toBe(5);
    });

    it('runs with scope=paths', async () => {
      const result = await runDocsQualityMetrics({
        scope: 'paths',
        sourcePaths: ['src/foo.ts'],
      });
      expect(result.manifest.scopeConfig.scopeType).toBe('paths');
    });

    it('auto-detects paths scope when sourcePaths provided', async () => {
      const result = await runDocsQualityMetrics({
        sourcePaths: ['src/bar.ts'],
      });
      expect(result.manifest.scopeConfig.scopeType).toBe('paths');
    });

    it('uses custom runId', async () => {
      const result = await runDocsQualityMetrics({ runId: 'cov-test-run' });
      expect(result.artifacts.runDirectory).toContain('cov-test-run');
    });

    it('treats non-paths scope as src', async () => {
      const result = await runDocsQualityMetrics({ scope: 'custom' });
      expect(result.manifest.scopeConfig.scopeType).toBe('src');
    });
  });

  // -------------------------------------------------------------------------
  // resolveThresholds errors
  // -------------------------------------------------------------------------
  describe('resolveThresholds error branches', () => {
    it('throws on non-finite complexityThreshold', async () => {
      await expect(
        runDocsQualityMetrics({ complexityThreshold: 'abc' }),
      ).rejects.toThrow('complexityThreshold must be a non-negative number.');
    });

    it('throws on negative complexityThreshold', async () => {
      await expect(
        runDocsQualityMetrics({ complexityThreshold: -1 }),
      ).rejects.toThrow('complexityThreshold must be a non-negative number.');
    });

    it('throws on non-finite minJsdocWords', async () => {
      await expect(
        runDocsQualityMetrics({ minJsdocWords: 'xyz' }),
      ).rejects.toThrow('minJsdocWords must be a non-negative number.');
    });

    it('throws on negative minJsdocWords', async () => {
      await expect(
        runDocsQualityMetrics({ minJsdocWords: -5 }),
      ).rejects.toThrow('minJsdocWords must be a non-negative number.');
    });
  });

  // -------------------------------------------------------------------------
  // resolveScopeConfig error
  // -------------------------------------------------------------------------
  describe('resolveScopeConfig error branch', () => {
    it('throws when scope=paths with no sourcePaths', async () => {
      await expect(runDocsQualityMetrics({ scope: 'paths' })).rejects.toThrow(
        'scope=paths requires at least one source path.',
      );
    });

    it('throws when scope=paths with empty sourcePaths', async () => {
      await expect(
        runDocsQualityMetrics({ scope: 'paths', sourcePaths: [] }),
      ).rejects.toThrow('scope=paths requires at least one source path.');
    });
  });

  // -------------------------------------------------------------------------
  // summarizeIssueBreakdown
  // -------------------------------------------------------------------------
  describe('summarizeIssueBreakdown', () => {
    it('counts all four issue types plus unknown', async () => {
      scanResult = {
        pass: false,
        evidence: [
          {
            file: 'src/a.ts',
            issue: 'missing JSDoc',
            numericValue: 0,
            symbol: 'a',
          },
          {
            file: 'src/b.ts',
            issue: 'weak JSDoc',
            numericValue: 3,
            symbol: 'b',
          },
          {
            file: 'src/c.ts',
            issue: 'high complexity',
            numericValue: 15,
            symbol: 'c',
          },
          {
            file: 'src/d.ts',
            issue: 'incomplete JSDoc tags',
            numericValue: 2,
            symbol: 'd',
          },
          {
            file: 'src/e.ts',
            issue: 'missing JSDoc',
            numericValue: 0,
            symbol: 'e',
          },
          {
            file: 'src/f.ts',
            issue: 'unknown issue',
            numericValue: 0,
            symbol: 'f',
          },
        ],
        issueDimensions: [],
        fixHint: 'fix hint',
        owner: 'docs',
      };

      const result = await runDocsQualityMetrics();
      expect(result.pass).toBe(false);
      expect(result.summary.missingJsdoc).toBe(2);
      expect(result.summary.weakJsdoc).toBe(1);
      expect(result.summary.highComplexity).toBe(1);
      expect(result.summary.incompleteJsdocTags).toBe(1);
      expect(result.summary.weakCount).toBe(1);
    });
  });

  // -------------------------------------------------------------------------
  // parseLcovSummary
  // -------------------------------------------------------------------------
  describe('parseLcovSummary', () => {
    it('returns available:false when lcov.info does not exist', async () => {
      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.available).toBe(false);
    });

    it('parses lcov.info without coverage-summary.json (line fallback)', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 8,
            branchFound: 5,
            branchHits: 4,
            functionFound: 3,
            functionHits: 2,
            daLines: [
              [1, 1],
              [2, 0],
              [3, 1],
            ],
            brdaLines: [
              [1, 0, 0, '-'],
              [2, 0, 1, 0],
              [3, 0, 2, 1],
            ],
            fndaLines: [
              [0, 'uncoveredFn'],
              [1, 'coveredFn'],
              [0, ''],
            ],
          }),
          lcovRecord('src/file2.ts', {
            lineFound: 5,
            lineHits: 5,
            branchFound: 3,
            branchHits: 3,
            functionFound: 2,
            functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.available).toBe(true);
      expect(cov.totalFiles).toBe(2);
      expect(cov.filesBelow100).toBe(1);
      expect(cov.filesBelow100Detail[0].statementCoverageSource).toBe(
        'lcov-line-fallback',
      );
      expect(cov.filesBelow100Detail[0].uncoveredLines).toContain(2);
      expect(cov.filesBelow100Detail[0].uncoveredBranches).toHaveLength(2);
      expect(cov.filesBelow100Detail[0].uncoveredBranches[0].taken).toBe(null);
      expect(cov.filesBelow100Detail[0].uncoveredBranches[1].taken).toBe(0);
      expect(cov.filesBelow100Detail[0].uncoveredFunctions).toEqual([
        'uncoveredFn',
      ]);
    });

    it('parses lcov.info with coverage-summary.json', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: {
            statements: { pct: 85 },
            branches: { pct: 60 },
            functions: { pct: 80 },
            lines: { pct: 70 },
          },
          'src/file1.ts': {
            statements: { pct: 75 },
            branches: { pct: 60 },
            functions: { pct: 80 },
            lines: { pct: 70 },
          },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.filesBelow100Detail[0].statements).toBe(75);
      expect(cov.filesBelow100Detail[0].statementCoverageSource).toBe(
        'coverage-summary',
      );
      expect(cov.isPartial).toBe(true);
      expect(cov.coveragePass).toBe(false);
    });

    it('skips records without SF: line', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          'TN:test\nLF:5\nLH:5\nend_of_record\n',
          lcovRecord('src/file1.ts', {
            lineFound: 5,
            lineHits: 5,
            branchFound: 3,
            branchHits: 3,
            functionFound: 2,
            functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.totalFiles).toBe(1);
    });

    it('skips records out of scope', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('lib/file1.ts', {
            lineFound: 10,
            lineHits: 5,
            branchFound: 5,
            branchHits: 2,
            functionFound: 3,
            functionHits: 1,
          }),
          lcovRecord('src/file2.ts', {
            lineFound: 5,
            lineHits: 5,
            branchFound: 3,
            branchHits: 3,
            functionFound: 2,
            functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics({ scope: 'src' });
      expect(result.summary.coverage.totalFiles).toBe(1);
    });

    it('handles 100% coverage', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 10,
            branchFound: 5,
            branchHits: 5,
            functionFound: 3,
            functionHits: 3,
          }),
        ].join('\n'),
        coverageSummary: {
          total: {
            statements: { pct: 100 },
            branches: { pct: 100 },
            functions: { pct: 100 },
            lines: { pct: 100 },
          },
          'src/file1.ts': {
            statements: { pct: 100 },
            branches: { pct: 100 },
            functions: { pct: 100 },
            lines: { pct: 100 },
          },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.filesBelow100).toBe(0);
      expect(cov.coveragePass).toBe(true);
      expect(cov.isPartial).toBeUndefined();
    });

    it('sorts filesBelow100Detail by worst coverage then file name', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/zzz.ts', {
            lineFound: 10,
            lineHits: 9,
            branchFound: 5,
            branchHits: 5,
            functionFound: 3,
            functionHits: 3,
          }),
          lcovRecord('src/aaa.ts', {
            lineFound: 10,
            lineHits: 5,
            branchFound: 5,
            branchHits: 5,
            functionFound: 3,
            functionHits: 3,
          }),
          lcovRecord('src/bbb.ts', {
            lineFound: 10,
            lineHits: 5,
            branchFound: 5,
            branchHits: 5,
            functionFound: 3,
            functionHits: 3,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const detail = result.summary.coverage.filesBelow100Detail;
      expect(detail[0].file).toBe('src/aaa.ts');
      expect(detail[1].file).toBe('src/bbb.ts');
      expect(detail[2].file).toBe('src/zzz.ts');
    });

    it('handles absolute SF: paths within cwd', async () => {
      const absPath = path.join(tempDir, 'src', 'file1.ts');
      mkdirSync(path.join(tempDir, 'src'), { recursive: true });
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord(absPath, {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.totalFiles).toBe(1);
      expect(result.summary.coverage.filesBelow100Detail[0].file).toContain(
        'src/file1.ts',
      );
    });

    it('handles absolute SF: paths outside cwd', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('/other/dir/src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics({ scope: 'src' });
      expect(result.summary.coverage.totalFiles).toBe(0);
    });

    it('handles SF:unknown path', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('unknown', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics({ scope: 'src' });
      expect(result.summary.coverage.totalFiles).toBe(0);
    });

    it('filters coverage records by explicit sourcePaths', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 5,
            branchFound: 5,
            branchHits: 2,
            functionFound: 3,
            functionHits: 1,
          }),
          lcovRecord('lib/file2.ts', {
            lineFound: 10,
            lineHits: 5,
            branchFound: 5,
            branchHits: 2,
            functionFound: 3,
            functionHits: 1,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics({
        scope: 'paths',
        sourcePaths: ['src/file1.ts'],
      });
      expect(result.summary.coverage.totalFiles).toBe(1);
    });
  });

  // -------------------------------------------------------------------------
  // readStatementCoverageByFile edge cases
  // -------------------------------------------------------------------------
  describe('readStatementCoverageByFile edge cases', () => {
    it('filters non-plain-object entries and non-finite pct', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: { pct: 70 } },
          'src/file1.ts': { statements: { pct: 80 } },
          'src/bad-entry.ts': 'not-an-object',
          'src/bad-statements.ts': { statements: 'not-an-object' },
          'src/bad-pct.ts': { statements: { pct: 'abc' } },
        },
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.filesBelow100Detail[0].statements).toBe(
        80,
      );
      expect(
        result.summary.coverage.filesBelow100Detail[0].statementCoverageSource,
      ).toBe('coverage-summary');
    });

    it('handles invalid JSON in coverage-summary.json', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: '{ invalid json {{{',
      });

      const result = await runDocsQualityMetrics();
      expect(
        result.summary.coverage.filesBelow100Detail[0].statementCoverageSource,
      ).toBe('lcov-line-fallback');
      expect(result.summary.coverage.isPartial).toBeUndefined();
    });
  });

  // -------------------------------------------------------------------------
  // readTotalStatementCoverage edge cases
  // -------------------------------------------------------------------------
  describe('readTotalStatementCoverage edge cases', () => {
    it('handles total not being a plain object', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: 'not-an-object',
          'src/file1.ts': { statements: { pct: 80 } },
        },
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.isPartial).toBeUndefined();
    });

    it('handles total.statements not being a plain object', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: 'not-an-object' },
          'src/file1.ts': { statements: { pct: 80 } },
        },
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.isPartial).toBeUndefined();
    });

    it('handles non-finite total pct', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10,
            lineHits: 7,
            branchFound: 5,
            branchHits: 3,
            functionFound: 3,
            functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: { pct: 'abc' } },
          'src/file1.ts': { statements: { pct: 80 } },
        },
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.isPartial).toBeUndefined();
    });
  });

  // -------------------------------------------------------------------------
  // resolveGitCommit
  // -------------------------------------------------------------------------
  describe('resolveGitCommit', () => {
    it('returns trimmed GIT_COMMIT when set', async () => {
      process.env.GIT_COMMIT = '  abc123  ';
      const result = await runDocsQualityMetrics();
      expect(result.manifest.gitCommit).toBe('abc123');
    });

    it('returns unknown when GIT_COMMIT is empty', async () => {
      process.env.GIT_COMMIT = '';
      const result = await runDocsQualityMetrics();
      expect(result.manifest.gitCommit).toBe('unknown');
    });

    it('returns unknown when GIT_COMMIT is whitespace only', async () => {
      process.env.GIT_COMMIT = '   ';
      const result = await runDocsQualityMetrics();
      expect(result.manifest.gitCommit).toBe('unknown');
    });

    it('returns unknown when GIT_COMMIT not set', async () => {
      const result = await runDocsQualityMetrics();
      expect(result.manifest.gitCommit).toBe('unknown');
    });
  });

  // -------------------------------------------------------------------------
  // toCoveragePercent / readLcovCounter edge cases
  // -------------------------------------------------------------------------
  describe('toCoveragePercent and readLcovCounter', () => {
    it('returns 100 when totalCount is 0', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/empty.ts', {
            lineFound: 0,
            lineHits: 0,
            branchFound: 0,
            branchHits: 0,
            functionFound: 0,
            functionHits: 0,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.filesBelow100).toBe(0);
      expect(cov.overallLines).toBe(100);
      expect(cov.overallBranches).toBe(100);
      expect(cov.overallFunctions).toBe(100);
    });

    it('handles missing counters (null match)', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: ['SF:src/nocounters.ts\nend_of_record\n'].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.totalFiles).toBe(1);
      expect(result.summary.coverage.filesBelow100).toBe(0);
    });
  });

  // -------------------------------------------------------------------------
  // Manifest validation
  // -------------------------------------------------------------------------
  describe('manifest validation', () => {
    it('produces valid manifest with evidence', async () => {
      scanResult = {
        pass: false,
        evidence: [
          {
            file: 'src/a.ts',
            issue: 'missing JSDoc',
            numericValue: 0,
            symbol: 'a',
          },
        ],
        issueDimensions: [],
        fixHint: 'fix',
        owner: 'docs',
      };

      const result = await runDocsQualityMetrics();
      expect(result.manifest.metricVersion).toBeDefined();
      expect(result.manifest.pass).toBe(false);
      expect(result.manifest.issueBreakdown.missingJsdoc).toBe(1);
    });

    it('throws when manifest validation fails', async () => {
      validationOverride = {
        valid: false,
        errors: [{ field: 'test', message: 'test error' }],
      };

      await expect(runDocsQualityMetrics()).rejects.toThrow(
        'Invalid docs-quality manifest: test',
      );
    });
  });

  // -------------------------------------------------------------------------
  // CLI main()
  // -------------------------------------------------------------------------
  describe('CLI main()', () => {
    let originalArgv;
    let originalExitCode;

    beforeEach(() => {
      originalArgv = process.argv;
      originalExitCode = process.exitCode;
    });

    afterEach(() => {
      process.argv = originalArgv;
      process.exitCode = originalExitCode;
    });

    it('prints help when --help is provided', async () => {
      cliArgs = { help: true, json: false };
      process.argv = ['node', METRICS_MODULE_PATH, '--help'];

      await jest.isolateModulesAsync(async () => {
        await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');
      });
      await new Promise((r) => setTimeout(r, 100));

      expect(printHelpCalls.length).toBeGreaterThanOrEqual(1);
      expect(printHelpCalls[0].title).toBe('Docs quality metrics runner');
    });

    it('runs with --json output on success', async () => {
      cliArgs = { help: false, json: true, _: [] };
      process.argv = ['node', METRICS_MODULE_PATH, '--json'];

      await jest.isolateModulesAsync(async () => {
        await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');
      });
      await new Promise((r) => setTimeout(r, 100));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].asJson).toBe(true);
    });

    it('runs with text output on success', async () => {
      cliArgs = { help: false, json: false, _: [] };
      process.argv = ['node', METRICS_MODULE_PATH];

      await jest.isolateModulesAsync(async () => {
        await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');
      });
      await new Promise((r) => setTimeout(r, 100));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].asJson).toBe(false);
    });

    it('calls fail on Error (json mode)', async () => {
      cliArgs = { help: false, json: true, _: [] };
      scanImpl = async () => {
        throw new Error('scanner failed');
      };
      process.argv = ['node', METRICS_MODULE_PATH, '--json'];

      await jest.isolateModulesAsync(async () => {
        await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');
      });
      await new Promise((r) => setTimeout(r, 100));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
      expect(failCalls[0].message).toBe('scanner failed');
      expect(failCalls[0].asJson).toBe(true);
    });

    it('calls fail on non-Error throw (text mode)', async () => {
      cliArgs = { help: false, json: false, _: [] };
      scanImpl = async () => {
        throw 'string error';
      };
      process.argv = ['node', METRICS_MODULE_PATH];

      await jest.isolateModulesAsync(async () => {
        await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');
      });
      await new Promise((r) => setTimeout(r, 100));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
      expect(failCalls[0].message).toBe('string error');
      expect(failCalls[0].asJson).toBe(false);
    });

    it('passes --source args as sourcePaths', async () => {
      cliArgs = { help: false, json: true, _: ['src/custom.ts'] };
      process.argv = ['node', METRICS_MODULE_PATH, '--json', 'src/custom.ts'];

      await jest.isolateModulesAsync(async () => {
        await import('../../../rag-index/docs-quality/docs-quality.metrics.mjs');
      });
      await new Promise((r) => setTimeout(r, 100));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
    });
  });
});
