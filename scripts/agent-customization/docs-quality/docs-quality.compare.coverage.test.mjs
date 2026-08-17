import { jest } from '@jest/globals';
import { writeFileSync, mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const PROJECT_ROOT = process.cwd();
const COMPARE_MODULE = path.resolve(
  PROJECT_ROOT,
  'rag-index',
  'docs-quality',
  'docs-quality.compare.mjs',
);
const TMP_DIR = path.resolve(PROJECT_ROOT, '__test_tmp_compare_cov');
const MANIFEST_LEFT = path.join(TMP_DIR, 'left-manifest.json');
const MANIFEST_RIGHT = path.join(TMP_DIR, 'right-manifest.json');
const SUMMARY_LEFT = path.join(TMP_DIR, 'left-summary.json');
const SUMMARY_RIGHT = path.join(TMP_DIR, 'right-summary.json');

// Mutable mock state
let cliArgs = { help: false, json: false };
let printHelpCalls = [];
let writeJsonCalls = [];
let failCalls = [];
let writeJsonThrowNonError = false;

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
    if (writeJsonThrowNonError) throw 'non-error string from writeJsonOrText';
    writeJsonCalls.push({ payload, asJson, textFormatter });
    if (!asJson && textFormatter) textFormatter(payload);
  },
  fail: (message, asJson) => {
    failCalls.push({ message, asJson });
    process.exitCode = 1;
  },
  toRepoRelative: (filePath) => filePath.replaceAll('\\', '/'),
}));

// --- Initial import (covers FALSE branch of CLI guard — process.argv is jest) ---
const { compareDocsQualityRuns, REASON_CODES } = await import(
  '../../../rag-index/docs-quality/docs-quality.compare.mjs'
);
// --- End initial import ---

function writeJson(filePath, data) {
  writeFileSync(filePath, JSON.stringify(data), 'utf8');
}

function makeManifest(opts = {}) {
  return {
    metricVersion: opts.metricVersion ?? 2,
    scannerVersion: opts.scannerVersion ?? '2.0.0',
    thresholdConfig: opts.thresholdConfig ?? { minJsdocWords: 10, complexityThreshold: 10 },
    scopeConfig: opts.scopeConfig ?? { scopeType: 'src', scopeDigest: 'digest-abc' },
    summaryPath: opts.summaryPath ?? SUMMARY_LEFT,
  };
}

function makeSummary(opts = {}) {
  return {
    missingJsdoc: opts.missingJsdoc ?? 5,
    weakJsdoc: opts.weakJsdoc ?? 3,
    highComplexity: opts.highComplexity ?? 2,
    evidenceCount: opts.evidenceCount ?? 10,
    issueBreakdown: opts.issueBreakdown ?? {
      missingJsdoc: 5,
      weakJsdoc: 3,
      highComplexity: 2,
    },
  };
}

describe('docs-quality.compare.mjs coverage', () => {
  describe('compareDocsQualityRuns — mismatch reason codes', () => {
    it('rejects with METRIC_VERSION_MISMATCH', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ metricVersion: 2 }),
        rightManifest: makeManifest({ metricVersion: 3 }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.METRIC_VERSION_MISMATCH);
    });

    it('rejects with SCOPE_TYPE_MISMATCH', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } }),
        rightManifest: makeManifest({ scopeConfig: { scopeType: 'paths', scopeDigest: 'abc' } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.SCOPE_TYPE_MISMATCH);
    });

    it('rejects with SCOPE_DIGEST_MISMATCH', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } }),
        rightManifest: makeManifest({ scopeConfig: { scopeType: 'src', scopeDigest: 'xyz' } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.SCOPE_DIGEST_MISMATCH);
    });

    it('rejects with SCANNER_VERSION_MISMATCH', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scannerVersion: '2.0.0' }),
        rightManifest: makeManifest({ scannerVersion: '2.1.0' }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.SCANNER_VERSION_MISMATCH);
    });

    it('rejects with THRESHOLD_MISMATCH (minJsdocWords)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 } }),
        rightManifest: makeManifest({ thresholdConfig: { minJsdocWords: 15, complexityThreshold: 10 } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.THRESHOLD_MISMATCH);
    });

    it('accepts with delta when all dimensions match', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: makeSummary({ missingJsdoc: 5, weakJsdoc: 3, highComplexity: 2, evidenceCount: 10 }),
        rightSummary: makeSummary({ missingJsdoc: 3, weakJsdoc: 1, highComplexity: 1, evidenceCount: 5 }),
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.missingJsdoc).toBe(-2);
    });

    it('handles missing thresholdConfig (defaults to empty object)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { metricVersion: 2, scannerVersion: '2.0.0', scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } },
        rightManifest: { metricVersion: 2, scannerVersion: '2.0.0', scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } },
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('handles null scopeConfig', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scopeConfig: null }),
        rightManifest: makeManifest({ scopeConfig: null }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('handles array scopeConfig (not a plain object)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scopeConfig: ['src'] }),
        rightManifest: makeManifest({ scopeConfig: ['src'] }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('normalizes summary with issueBreakdown fallback', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: { issueBreakdown: { missingJsdoc: 1, weakJsdoc: 2, highComplexity: 3 } },
        rightSummary: { issueBreakdown: { missingJsdoc: 4, weakJsdoc: 5, highComplexity: 6 } },
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.missingJsdoc).toBe(3);
    });

    it('covers ?? fallback branches in normalizeManifest (missing fields)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } },
        rightManifest: { scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } },
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
      // metricVersion defaults to 0 for both, scannerVersion defaults to ''
    });

    it('covers thresholdConfig fallback to empty object and threshold ?? 0', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { metricVersion: 2, scannerVersion: '2.0.0', scopeConfig: { scopeType: 'src', scopeDigest: 'abc' }, thresholdConfig: {} },
        rightManifest: { metricVersion: 2, scannerVersion: '2.0.0', scopeConfig: { scopeType: 'src', scopeDigest: 'abc' }, thresholdConfig: {} },
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('covers scopeConfig fallback to empty object and scopeType/scopeDigest ?? empty', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { metricVersion: 2, scannerVersion: '2.0.0', thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 }, scopeConfig: {} },
        rightManifest: { metricVersion: 2, scannerVersion: '2.0.0', thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 }, scopeConfig: {} },
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('covers normalizeSummary with empty summary (all ?? 0 fallbacks)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: {},
        rightSummary: {},
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.missingJsdoc).toBe(0);
      expect(result.delta.weakJsdoc).toBe(0);
      expect(result.delta.highComplexity).toBe(0);
      expect(result.delta.evidenceCount).toBe(0);
    });

    it('covers normalizeSummary with partial issueBreakdown (some ?? 0)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: { missingJsdoc: 5, issueBreakdown: { weakJsdoc: 3 } },
        rightSummary: { missingJsdoc: 3, issueBreakdown: { weakJsdoc: 1 } },
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.missingJsdoc).toBe(-2);
      expect(result.delta.weakJsdoc).toBe(-2);
      expect(result.delta.highComplexity).toBe(0);
    });
  });

  describe('CLI main()', () => {
    let originalArgv;
    let originalExitCode;

    beforeEach(() => {
      originalArgv = process.argv;
      originalExitCode = process.exitCode;
      mkdirSync(TMP_DIR, { recursive: true });

      cliArgs = { help: false, json: false };
      printHelpCalls = [];
      writeJsonCalls = [];
      failCalls = [];
      writeJsonThrowNonError = false;
      process.exitCode = 0;
    });

    afterEach(() => {
      process.argv = originalArgv;
      process.exitCode = originalExitCode;
      rmSync(TMP_DIR, { recursive: true, force: true });
    });

    it('prints help when --help is passed', async () => {
      cliArgs = { help: true, json: false };
      process.argv = ['node', COMPARE_MODULE, '--help'];

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(printHelpCalls.length).toBeGreaterThanOrEqual(1);
      expect(printHelpCalls[0].title).toBe('Docs quality compare');
    });

    it('fails when --left and --right are not provided', async () => {
      cliArgs = { help: false, json: false };
      process.argv = ['node', COMPARE_MODULE];

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
      expect(failCalls[0].message).toContain('required');
    });

    it('compares two runs successfully (text output)', async () => {
      cliArgs = { help: false, json: false, left: MANIFEST_LEFT, right: MANIFEST_RIGHT };
      writeJson(MANIFEST_LEFT, makeManifest({ summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary({ evidenceCount: 10 }));
      writeJson(SUMMARY_RIGHT, makeSummary({ evidenceCount: 5 }));

      process.argv = ['node', COMPARE_MODULE, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`];

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].asJson).toBe(false);
      expect(writeJsonCalls[0].payload.accepted).toBe(true);
    });

    it('outputs json when --json flag is passed', async () => {
      cliArgs = { help: false, json: true, left: MANIFEST_LEFT, right: MANIFEST_RIGHT };
      writeJson(MANIFEST_LEFT, makeManifest({ summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary({ evidenceCount: 10 }));
      writeJson(SUMMARY_RIGHT, makeSummary({ evidenceCount: 5 }));

      process.argv = ['node', COMPARE_MODULE, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`, '--json'];

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].asJson).toBe(true);
    });

    it('sets exitCode to 1 when comparison is rejected', async () => {
      cliArgs = { help: false, json: false, left: MANIFEST_LEFT, right: MANIFEST_RIGHT };
      writeJson(MANIFEST_LEFT, makeManifest({ metricVersion: 2, summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ metricVersion: 3, summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary());
      writeJson(SUMMARY_RIGHT, makeSummary());

      process.argv = ['node', COMPARE_MODULE, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`];
      process.exitCode = 0;

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(writeJsonCalls.length).toBeGreaterThanOrEqual(1);
      expect(writeJsonCalls[0].payload.accepted).toBe(false);
      expect(process.exitCode).toBe(1);
    });

    it('does not run main when not invoked as CLI', async () => {
      cliArgs = { help: false, json: false };
      process.argv = ['node', 'some-other-script.mjs'];

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(failCalls.length).toBe(0);
      expect(printHelpCalls.length).toBe(0);
      expect(writeJsonCalls.length).toBe(0);
    });

    it('handles read error with Error object (nonexistent files)', async () => {
      cliArgs = {
        help: false, json: false,
        left: path.join(TMP_DIR, 'nonexistent.json'),
        right: path.join(TMP_DIR, 'also.json'),
      };
      process.argv = ['node', COMPARE_MODULE, `--left=${path.join(TMP_DIR, 'nonexistent.json')}`, `--right=${path.join(TMP_DIR, 'also.json')}`];

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
    });

    it('handles non-Error throw from writeJsonOrText (covers String(error) branch)', async () => {
      cliArgs = { help: false, json: false, left: MANIFEST_LEFT, right: MANIFEST_RIGHT };
      writeJson(MANIFEST_LEFT, makeManifest({ summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary({ evidenceCount: 10 }));
      writeJson(SUMMARY_RIGHT, makeSummary({ evidenceCount: 5 }));

      process.argv = ['node', COMPARE_MODULE, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`];
      writeJsonThrowNonError = true;

      jest.resetModules();
      await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');
      await new Promise((r) => setTimeout(r, 100));

      expect(failCalls.length).toBeGreaterThanOrEqual(1);
      expect(failCalls[0].message).toBe('non-error string from writeJsonOrText');
    });

  });
});