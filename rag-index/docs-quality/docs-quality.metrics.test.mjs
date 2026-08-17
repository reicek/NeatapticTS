import { jest } from '@jest/globals';
import {
  existsSync,
  mkdirSync,
  rmSync,
  writeFileSync,
  readFileSync,
} from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Mock state for scanCodeQuality
// ---------------------------------------------------------------------------
let scanResult = {
  pass: true,
  evidence: [],
  issueDimensions: [],
  fixHint: null,
  owner: 'docs',
};
// When non-null, the mock scanCodeQuality throws this value instead of returning.
let scanThrowValue = null;
// Captures the opts passed to scanCodeQuality on the most recent invocation.
let receivedScanOpts = null;

// ---------------------------------------------------------------------------
// Mock state for cli-utils
// ---------------------------------------------------------------------------
let cliArgs = { help: false, json: false };
let printHelpCalls = [];
let writeJsonCalls = [];
let failCalls = [];

// ---------------------------------------------------------------------------
// Mock code-quality-scanner.mjs
// ---------------------------------------------------------------------------
jest.unstable_mockModule('../code-quality-scanner.mjs', () => ({
  scanCodeQuality: async (opts) => {
    receivedScanOpts = opts;
    if (scanThrowValue !== null) throw scanThrowValue;
    return scanResult;
  },
}));

// ---------------------------------------------------------------------------
// Mock cli-utils.mjs — pass through real implementations except controlled parts
// ---------------------------------------------------------------------------
const realCliUtils = await import('../cli-utils.mjs');
jest.unstable_mockModule('../cli-utils.mjs', () => ({
  ...realCliUtils,
  parseCliArgs: (argv) => ({ ...cliArgs, _: cliArgs._ !== undefined ? cliArgs._ : argv.filter((a) => !a.startsWith('--')) }),
  printHelp: (config) => {
    printHelpCalls.push(config);
  },
  writeJsonOrText: (payload, asJson, textFormatter) => {
    writeJsonCalls.push({ payload, asJson, textFormatter });
    if (asJson) {
      process.stdout.write(JSON.stringify(payload, null, 2));
    } else {
      process.stdout.write(textFormatter(payload));
    }
  },
  fail: (message, asJson) => {
    failCalls.push({ message, asJson });
    process.exitCode = 1;
    if (asJson) {
      process.stdout.write(JSON.stringify({ error: message }));
    } else {
      process.stderr.write(message);
    }
  },
}));

// Restore real cli-utils.mjs after all tests so the mock does not leak
// into other test files in the same Jest process.
afterAll(() => {
  jest.unstable_mockModule('../cli-utils.mjs', () => ({ ...realCliUtils }));
});

// ---------------------------------------------------------------------------
// Import module under test AFTER mocks
// ---------------------------------------------------------------------------
const { runDocsQualityMetrics } = await import('./docs-quality.metrics.mjs');

// ---------------------------------------------------------------------------
// Temp directory helper
// ---------------------------------------------------------------------------
function createTempDir() {
  const dir = path.resolve(__dirname, '__test_tmp_metrics__');
  try {
    rmSync(dir, { recursive: true, force: true });
  } catch {
    // ignore EPERM on Windows — file handles may still be open
  }
  mkdirSync(dir, { recursive: true });
  return dir;
}

function cleanupTempDir(dir) {
  try {
    process.chdir(__dirname);
    rmSync(dir, { recursive: true, force: true });
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

// Build a single LCOV record
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

  // DA lines (line coverage data)
  if (opts.daLines) {
    for (const [lineNum, hitCount] of opts.daLines) {
      lines.push(`DA:${lineNum},${hitCount}`);
    }
  }

  // BRDA lines (branch coverage data)
  if (opts.brdaLines) {
    for (const [lineNum, blockNum, branchNum, taken] of opts.brdaLines) {
      lines.push(`BRDA:${lineNum},${blockNum},${branchNum},${taken}`);
    }
  }

  // FNDA lines (function coverage data)
  if ( opts.fndaLines) {
    for (const [hitCount, funcName] of opts.fndaLines) {
      lines.push(`FNDA:${hitCount},${funcName}`);
    }
  }

  lines.push('end_of_record');
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('docs-quality.metrics.mjs', () => {
  let originalCwd;
  let tempDir;
  let originalGitCommit;

  beforeEach(() => {
    originalCwd = process.cwd();
    tempDir = createTempDir();
    process.chdir(tempDir);
    originalGitCommit = process.env.GIT_COMMIT;
    delete process.env.GIT_COMMIT;

    // Reset mock state
    scanResult = {
      pass: true,
      evidence: [],
      issueDimensions: [],
      fixHint: null,
      owner: 'docs',
    };
    scanThrowValue = null;
    receivedScanOpts = null;
    cliArgs = { help: false, json: false };
    printHelpCalls = [];
    writeJsonCalls = [];
    failCalls = [];
    process.exitCode = 0;
  });

  afterEach(() => {
    if (originalGitCommit !== undefined) {
      process.env.GIT_COMMIT = originalGitCommit;
    } else {
      delete process.env.GIT_COMMIT;
    }
    cleanupTempDir(tempDir);
    process.chdir(originalCwd);
  });

  // =========================================================================
  // runDocsQualityMetrics — basic paths
  // =========================================================================
  describe('runDocsQualityMetrics — basic paths', () => {
    it('runs with default options and no coverage files', async () => {
      const result = await runDocsQualityMetrics();
      expect(result.pass).toBe(true);
      expect(result.evidence).toEqual([]);
      expect(result.summary.pass).toBe(true);
      expect(result.summary.evidenceCount).toBe(0);
      expect(result.summary.coverage.available).toBe(false);
      expect(result.manifest).toBeDefined();
      expect(result.artifacts).toBeDefined();
    });

    it('runs with custom thresholds and scope=src', async () => {
      const result = await runDocsQualityMetrics({
        complexityThreshold: 15,
        minJsdocWords: 5,
        scope: 'src',
      });
      expect(result.pass).toBe(true);
      expect(result.manifest.thresholdConfig.complexityThreshold).toBe(15);
      expect(result.manifest.thresholdConfig.minJsdocWords).toBe(5);
      expect(result.manifest.scopeConfig.scopeType).toBe('src');
    });

    it('runs with scope=paths and sourcePaths', async () => {
      const result = await runDocsQualityMetrics({
        scope: 'paths',
        sourcePaths: ['src/foo.ts'],
      });
      expect(result.manifest.scopeConfig.scopeType).toBe('paths');
      expect(result.manifest.scopeConfig.scopeValue).toEqual(['src/foo.ts']);
    });

    it('auto-detects paths scope when sourcePaths provided without scope', async () => {
      const result = await runDocsQualityMetrics({
        sourcePaths: ['src/bar.ts'],
      });
      expect(result.manifest.scopeConfig.scopeType).toBe('paths');
    });

    it('uses custom runId', async () => {
      const result = await runDocsQualityMetrics({ runId: 'my-run' });
      expect(result.artifacts.runDirectory).toContain('my-run');
    });

    it('passes thresholds to scanCodeQuality', async () => {
      await runDocsQualityMetrics({
        complexityThreshold: 20,
        minJsdocWords: 3,
        scope: 'src',
      });
      expect(receivedScanOpts.complexityThreshold).toBe(20);
      expect(receivedScanOpts.minJsdocWords).toBe(3);
    });
  });

  // =========================================================================
  // resolveThresholds — error branches
  // =========================================================================
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

  // =========================================================================
  // resolveScopeConfig — error branches
  // =========================================================================
  describe('resolveScopeConfig error branches', () => {
    it('throws when scope=paths but no sourcePaths provided', async () => {
      await expect(
        runDocsQualityMetrics({ scope: 'paths' }),
      ).rejects.toThrow('scope=paths requires at least one source path.');
    });

    it('throws when scope=paths and sourcePaths is empty array', async () => {
      await expect(
        runDocsQualityMetrics({ scope: 'paths', sourcePaths: [] }),
      ).rejects.toThrow('scope=paths requires at least one source path.');
    });

    it('treats non-paths scope as src', async () => {
      const result = await runDocsQualityMetrics({ scope: 'custom' });
      expect(result.manifest.scopeConfig.scopeType).toBe('src');
    });
  });

  // =========================================================================
  // summarizeIssueBreakdown — all issue types
  // =========================================================================
  describe('summarizeIssueBreakdown', () => {
    it('counts all four issue types', async () => {
      scanResult = {
        pass: false,
        evidence: [
          { file: 'src/a.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'a' },
          { file: 'src/b.ts', issue: 'weak JSDoc', numericValue: 3, symbol: 'b' },
          { file: 'src/c.ts', issue: 'high complexity', numericValue: 15, symbol: 'c' },
          { file: 'src/d.ts', issue: 'incomplete JSDoc tags', numericValue: 2, symbol: 'd' },
          { file: 'src/e.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'e' },
          { file: 'src/f.ts', issue: 'unknown issue', numericValue: 0, symbol: 'f' },
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

  // =========================================================================
  // parseLcovSummary — coverage parsing
  // =========================================================================
  describe('parseLcovSummary', () => {
    it('returns available:false when lcov.info does not exist', async () => {
      const result = await runDocsQualityMetrics();
      expect(result.summary.coverage.available).toBe(false);
    });

    it('parses lcov.info without coverage-summary.json (line fallback)', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 8,
            branchFound: 5, branchHits: 4,
            functionFound: 3, functionHits: 2,
            daLines: [[1, 1], [2, 0], [3, 1]],
            brdaLines: [[1, 0, 0, '-'], [2, 0, 1, 0], [3, 0, 2, 1]],
            fndaLines: [[0, 'uncoveredFn'], [1, 'coveredFn'], [0, '']],
          }),
          lcovRecord('src/file2.ts', {
            lineFound: 5, lineHits: 5,
            branchFound: 3, branchHits: 3,
            functionFound: 2, functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.available).toBe(true);
      expect(cov.totalFiles).toBe(2);
      expect(cov.filesBelow100).toBe(1);
      expect(cov.filesBelow100Detail).toHaveLength(1);
      expect(cov.filesBelow100Detail[0].file).toBe('src/file1.ts');
      // No coverage-summary.json → line fallback
      expect(cov.filesBelow100Detail[0].statementCoverageSource).toBe('lcov-line-fallback');
      // Uncovered lines: line 2 (DA:2,0)
      expect(cov.filesBelow100Detail[0].uncoveredLines).toContain(2);
      // Uncovered branches: BRDA with '-' and 0
      expect(cov.filesBelow100Detail[0].uncoveredBranches).toHaveLength(2);
      expect(cov.filesBelow100Detail[0].uncoveredBranches[0].taken).toBe(null); // '-' case
      expect(cov.filesBelow100Detail[0].uncoveredBranches[1].taken).toBe(0);
      // Uncovered functions: FNDA:0,uncoveredFn (empty name filtered)
      expect(cov.filesBelow100Detail[0].uncoveredFunctions).toEqual(['uncoveredFn']);
    });

    it('handles malformed DA/BRDA/FNDA lines with missing fields', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          'SF:src/malformed.ts',
          'LF:5',
          'LH:3',
          'BRF:2',
          'BRH:1',
          'FNF:2',
          'FNH:1',
          'DA:5',
          'BRDA:5',
          'BRDA:5,0',
          'BRDA:5,0,0',
          'FNDA:0',
          'end_of_record',
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.available).toBe(true);
      expect(cov.totalFiles).toBe(1);
      // Malformed DA:5 (no hit count) → hitCount undefined → ?? '0' → parseInt 0 → uncovered
      expect(cov.filesBelow100Detail[0].uncoveredLines).toContain(5);
      // Malformed BRDA lines with missing fields → takenCount undefined → ?? '0' → 0 → uncovered
      expect(cov.filesBelow100Detail[0].uncoveredBranches.length).toBeGreaterThanOrEqual(1);
      // Malformed FNDA:0 (no function name) → functionName undefined → ?? 'unknown'
      expect(cov.filesBelow100Detail[0].uncoveredFunctions).toContain('unknown');
    });

    it('parses lcov.info with coverage-summary.json (coverage-summary source)', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: { pct: 85 }, branches: { pct: 60 }, functions: { pct: 80 }, lines: { pct: 70 } },
          'src/file1.ts': { statements: { pct: 75 }, branches: { pct: 60 }, functions: { pct: 80 }, lines: { pct: 70 } },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.available).toBe(true);
      expect(cov.filesBelow100Detail[0].statements).toBe(75);
      expect(cov.filesBelow100Detail[0].statementCoverageSource).toBe('coverage-summary');
      // isPartial: totalStatementCoverage = 85 < 99
      expect(cov.isPartial).toBe(true);
      // coveragePass: overallLines = round(7/10*100) = 70 < 99
      expect(cov.coveragePass).toBe(false);
    });

    it('skips records without SF: line', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          'TN:test\nLF:5\nLH:5\nend_of_record\n',
          lcovRecord('src/file1.ts', {
            lineFound: 5, lineHits: 5,
            branchFound: 3, branchHits: 3,
            functionFound: 2, functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.totalFiles).toBe(1);
    });

    it('skips records out of scope', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('lib/file1.ts', {
            lineFound: 10, lineHits: 5,
            branchFound: 5, branchHits: 2,
            functionFound: 3, functionHits: 1,
          }),
          lcovRecord('src/file2.ts', {
            lineFound: 5, lineHits: 5,
            branchFound: 3, branchHits: 3,
            functionFound: 2, functionHits: 2,
          }),
        ].join('\n'),
      });

      // scope=src should include only src/ files
      const result = await runDocsQualityMetrics({ scope: 'src' });
      const cov = result.summary.coverage;
      expect(cov.totalFiles).toBe(1);
    });

    it('handles 100% coverage (no filesBelow100)', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 10,
            branchFound: 5, branchHits: 5,
            functionFound: 3, functionHits: 3,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: { pct: 100 }, branches: { pct: 100 }, functions: { pct: 100 }, lines: { pct: 100 } },
          'src/file1.ts': { statements: { pct: 100 }, branches: { pct: 100 }, functions: { pct: 100 }, lines: { pct: 100 } },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.filesBelow100).toBe(0);
      expect(cov.filesBelow100Detail).toEqual([]);
      expect(cov.coveragePass).toBe(true);
      // isPartial: totalStatementCoverage = 100 >= 99 → no isPartial
      expect(cov.isPartial).toBeUndefined();
    });

    it('sorts filesBelow100Detail by worst coverage then file name', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/zzz.ts', {
            lineFound: 10, lineHits: 9,
            branchFound: 5, branchHits: 5,
            functionFound: 3, functionHits: 3,
          }),
          lcovRecord('src/aaa.ts', {
            lineFound: 10, lineHits: 5,
            branchFound: 5, branchHits: 5,
            functionFound: 3, functionHits: 3,
          }),
          lcovRecord('src/bbb.ts', {
            lineFound: 10, lineHits: 5,
            branchFound: 5, branchHits: 5,
            functionFound: 3, functionHits: 3,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const detail = result.summary.coverage.filesBelow100Detail;
      // aaa.ts and bbb.ts both have 50% line coverage (worst=50)
      // zzz.ts has 90% line coverage (worst=90)
      // Sort: lowest worst first → aaa/bbb (50) before zzz (90)
      // Same worst → sort by file name → aaa before bbb
      expect(detail[0].file).toBe('src/aaa.ts');
      expect(detail[1].file).toBe('src/bbb.ts');
      expect(detail[2].file).toBe('src/zzz.ts');
    });

    it('handles absolute SF: paths within cwd', async () => {
      const absPath = path.join(tempDir, 'src', 'file1.ts');
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord(absPath, {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.totalFiles).toBe(1);
      // Absolute path should be normalized to repo-relative
      expect(cov.filesBelow100Detail[0].file).toContain('src/file1.ts');
    });

    it('handles absolute SF: paths outside cwd', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('/other/dir/src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
      });

      // scope=src, path is /other/dir/... → starts with '..' relative to cwd
      // → normalizeCoverageFilePath returns toRepoRelativePath(coverageFilePath)
      // → isPathInScope checks if it starts with 'src/' → no → skipped
      const result = await runDocsQualityMetrics({ scope: 'src' });
      const cov = result.summary.coverage;
      expect(cov.totalFiles).toBe(0);
    });

    it('handles SF:unknown path', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('unknown', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics({ scope: 'src' });
      const cov = result.summary.coverage;
      // 'unknown' path → isPathInScope returns false → skipped
      expect(cov.totalFiles).toBe(0);
    });
  });

  // =========================================================================
  // readStatementCoverageByFile — edge cases via coverage-summary.json
  // =========================================================================
  describe('readStatementCoverageByFile edge cases', () => {
    it('filters non-plain-object entries and non-finite pct values', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
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
      const cov = result.summary.coverage;
      // file1.ts should use coverage-summary pct=80
      expect(cov.filesBelow100Detail[0].statements).toBe(80);
      expect(cov.filesBelow100Detail[0].statementCoverageSource).toBe('coverage-summary');
    });

    it('handles invalid JSON in coverage-summary.json (catch block)', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: '{ invalid json {{{',
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      // Invalid JSON → readStatementCoverageByFile returns empty Map → line fallback
      expect(cov.filesBelow100Detail[0].statementCoverageSource).toBe('lcov-line-fallback');
      // readTotalStatementCoverage → null → isPartialCoverage false
      expect(cov.isPartial).toBeUndefined();
    });
  });

  // =========================================================================
  // readTotalStatementCoverage — edge cases
  // =========================================================================
  describe('readTotalStatementCoverage edge cases', () => {
    it('handles total not being a plain object', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: 'not-an-object',
          'src/file1.ts': { statements: { pct: 80 } },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      // total not plain object → readTotalStatementCoverage returns null → no isPartial
      expect(cov.isPartial).toBeUndefined();
    });

    it('handles total.statements not being a plain object', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: 'not-an-object' },
          'src/file1.ts': { statements: { pct: 80 } },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.isPartial).toBeUndefined();
    });

    it('handles non-finite total pct', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 7,
            branchFound: 5, branchHits: 3,
            functionFound: 3, functionHits: 2,
          }),
        ].join('\n'),
        coverageSummary: {
          total: { statements: { pct: 'abc' } },
          'src/file1.ts': { statements: { pct: 80 } },
        },
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      expect(cov.isPartial).toBeUndefined();
    });
  });

  // =========================================================================
  // resolveGitCommit
  // =========================================================================
  describe('resolveGitCommit', () => {
    it('returns trimmed GIT_COMMIT when set', async () => {
      process.env.GIT_COMMIT = '  abc123  ';
      const result = await runDocsQualityMetrics();
      expect(result.manifest.gitCommit).toBe('abc123');
    });

    it('returns unknown when GIT_COMMIT is empty string', async () => {
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

  // =========================================================================
  // toRepoRelativePath — backslash conversion
  // =========================================================================
  describe('toRepoRelativePath', () => {
    it('converts backslashes to forward slashes in manifest paths', async () => {
      // On Windows, path.join uses backslashes, so manifest paths will have them
      // toRepoRelativePath converts them to forward slashes
      const result = await runDocsQualityMetrics({ runId: 'test-run' });
      expect(result.manifest.manifestPath).toContain('/');
      expect(result.manifest.manifestPath).not.toContain('\\');
    });
  });

  // =========================================================================
  // toCoveragePercent — edge cases
  // =========================================================================
  describe('toCoveragePercent edge cases', () => {
    it('returns 100 when totalCount is 0', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          // File with 0 lines, 0 branches, 0 functions → all toCoveragePercent(0, 0) = 100
          lcovRecord('src/empty.ts', {
            lineFound: 0, lineHits: 0,
            branchFound: 0, branchHits: 0,
            functionFound: 0, functionHits: 0,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      // 0/0 → 100% → not below 100
      expect(cov.filesBelow100).toBe(0);
      expect(cov.overallLines).toBe(100);
      expect(cov.overallBranches).toBe(100);
      expect(cov.overallFunctions).toBe(100);
    });
  });

  // =========================================================================
  // readLcovCounter — missing counter (match is null)
  // =========================================================================
  describe('readLcovCounter missing counter', () => {
    it('handles missing LH/LF/BRH/BRF/FNH/FNF counters', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          // Record with SF but no counters at all
          'SF:src/nocounters.ts\nend_of_record\n',
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics();
      const cov = result.summary.coverage;
      // All counters default to 0 → toCoveragePercent(0, 0) = 100 → not below 100
      expect(cov.totalFiles).toBe(1);
      expect(cov.filesBelow100).toBe(0);
    });
  });

  // =========================================================================
  // Coverage with scope=paths filtering
  // =========================================================================
  describe('scope=paths coverage filtering', () => {
    it('filters coverage records by explicit sourcePaths', async () => {
      writeCoverageFiles(tempDir, {
        lcovContent: [
          lcovRecord('src/file1.ts', {
            lineFound: 10, lineHits: 5,
            branchFound: 5, branchHits: 2,
            functionFound: 3, functionHits: 1,
          }),
          lcovRecord('lib/file2.ts', {
            lineFound: 10, lineHits: 5,
            branchFound: 5, branchHits: 2,
            functionFound: 3, functionHits: 1,
          }),
        ].join('\n'),
      });

      const result = await runDocsQualityMetrics({
        scope: 'paths',
        sourcePaths: ['src/file1.ts'],
      });
      const cov = result.summary.coverage;
      // Only src/file1.ts should be in scope
      expect(cov.totalFiles).toBe(1);
    });
  });

  // =========================================================================
  // CLI main()
  // =========================================================================
  describe('CLI main()', () => {
    let writeSpy;
    let errorSpy;
    let originalExit;
    let scriptPath;

    beforeEach(() => {
      writeSpy = jest
        .spyOn(process.stdout, 'write')
        .mockImplementation(() => true);
      errorSpy = jest
        .spyOn(process.stderr, 'write')
        .mockImplementation(() => true);
      originalExit = process.exit;
      process.exit = () => {};
      scriptPath = path.resolve(
        originalCwd,
        'rag-index',
        'docs-quality',
        'docs-quality.metrics.mjs',
      );
    });

    afterEach(() => {
      writeSpy.mockRestore();
      errorSpy.mockRestore();
      process.exit = originalExit;
    });

    it('prints help when --help is provided', async () => {
      cliArgs = { help: true, json: false };
      process.argv = ['node', scriptPath, '--help'];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now());
      await new Promise((r) => setTimeout(r, 300));

      expect(printHelpCalls.length).toBeGreaterThanOrEqual(1);
      expect(printHelpCalls[0].title).toBe('Docs quality metrics runner');
    });

    it('runs with --json output on success', async () => {
      cliArgs = { help: false, json: true, _: [] };
      process.argv = ['node', scriptPath, '--json'];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now() + '2');
      await new Promise((r) => setTimeout(r, 300));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].asJson).toBe(true);
    });

    it('runs with text output on success', async () => {
      cliArgs = { help: false, json: false, _: [] };
      process.argv = ['node', scriptPath];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now() + '3');
      await new Promise((r) => setTimeout(r, 300));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].asJson).toBe(false);
    });

    it('calls fail on error (json mode)', async () => {
      cliArgs = { help: false, json: true, _: [] };
      scanThrowValue = new Error('scanner failed');

      process.argv = ['node', scriptPath, '--json'];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now() + '4');
      await new Promise((r) => setTimeout(r, 300));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
      expect(failCalls[0].message).toBe('scanner failed');
      expect(failCalls[0].asJson).toBe(true);
    });

    it('calls fail on error (text mode) with non-Error throw', async () => {
      cliArgs = { help: false, json: false, _: [] };
      scanThrowValue = 'string error';

      process.argv = ['node', scriptPath];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now() + '5');
      await new Promise((r) => setTimeout(r, 300));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
      expect(failCalls[0].message).toBe('string error');
      expect(failCalls[0].asJson).toBe(false);
    });

    it('passes --source args as sourcePaths', async () => {
      cliArgs = { help: false, json: true, _: ['src/custom.ts'] };
      process.argv = ['node', scriptPath, '--json', 'src/custom.ts'];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now() + '6');
      await new Promise((r) => setTimeout(r, 300));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
    });

    it('handles non-array args._ from CLI parser', async () => {
      cliArgs = { help: false, json: false, _: null };
      process.argv = ['node', scriptPath];

      await import('./docs-quality.metrics.mjs?cli-test=' + Date.now() + '7');
      await new Promise((r) => setTimeout(r, 300));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
    });
  });

  // =========================================================================
  // Manifest validation
  // =========================================================================
  describe('manifest validation', () => {
    it('produces a valid manifest that passes validation', async () => {
      scanResult = {
        pass: false,
        evidence: [
          { file: 'src/a.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'a' },
        ],
        issueDimensions: [],
        fixHint: 'fix',
        owner: 'docs',
      };

      const result = await runDocsQualityMetrics();
      expect(result.manifest.metricVersion).toBeDefined();
      expect(result.manifest.scannerVersion).toBeDefined();
      expect(result.manifest.pass).toBe(false);
      expect(result.manifest.issueBreakdown.missingJsdoc).toBe(1);
    });
  });
});