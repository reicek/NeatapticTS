/**
 * @module merge-coverage-summaries.createcoverage-throw.test
 * @description Covers the createCoverageMap catch block (line 150) in
 *   loadCoverageFinalFileEntries by mocking istanbul-lib-coverage to throw.
 *   Separate file so the mock does not affect the main coverage tests.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const actualLibCoverage = await import('istanbul-lib-coverage');

jest.unstable_mockModule('istanbul-lib-coverage', () => ({
  default: {
    ...actualLibCoverage.default,
    createCoverageMap: () => {
      throw new Error('mocked createCoverageMap failure');
    },
  },
}));

const { mergeCoverageSummaries } = await import(
  './merge-coverage-summaries.mjs'
);

const REPO_ROOT = path.resolve();

describe('merge-coverage-summaries createCoverageMap throw coverage', () => {
  let tempDir;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'merge-cov-throw-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('skips coverage-final.json entries when createCoverageMap throws', async () => {
    // Create a coverage directory with a coverage-final.json
    const coverageDir = path.join(tempDir, 'coverage');
    mkdirSync(coverageDir, { recursive: true });
    writeFileSync(
      path.join(coverageDir, 'coverage-final.json'),
      JSON.stringify({
        '/repo/src/file.ts': {
          statementMap: {},
          fnMap: {},
          branchMap: {},
          s: {},
          f: {},
          b: {},
          path: '/repo/src/file.ts',
        },
      }),
      'utf8',
    );

    // Also need a coverage-summary.json so mergedFiles is not empty
    const summaryPath = path.join(coverageDir, 'coverage-summary.json');
    writeFileSync(
      summaryPath,
      JSON.stringify({
        total: {
          lines: { total: 0, covered: 0, skipped: 0, pct: 100 },
          statements: { total: 0, covered: 0, skipped: 0, pct: 100 },
          functions: { total: 0, covered: 0, skipped: 0, pct: 100 },
          branches: { total: 0, covered: 0, skipped: 0, pct: 100 },
        },
      }),
      'utf8',
    );

    const result = await mergeCoverageSummaries({
      coverageDir,
      summaryPath,
    });

    // The coverage-final.json should be in mergedFiles (the path is collected),
    // but the file entries from it should be skipped because createCoverageMap threw.
    assert.ok(result.mergedFiles.some((f) => f.endsWith('coverage-final.json')));
  });
});