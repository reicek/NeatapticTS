import { jest } from '@jest/globals';
import { writeFileSync, mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { compareDocsQualityRuns, REASON_CODES } from './docs-quality.compare.mjs';

const TMP_DIR = path.resolve(process.cwd(), 'rag-index', '__test_tmp_compare');
const MANIFEST_LEFT = path.join(TMP_DIR, 'left-manifest.json');
const MANIFEST_RIGHT = path.join(TMP_DIR, 'right-manifest.json');
const SUMMARY_LEFT = path.join(TMP_DIR, 'left-summary.json');
const SUMMARY_RIGHT = path.join(TMP_DIR, 'right-summary.json');

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

describe('docs-quality.compare.mjs', () => {
  describe('REASON_CODES', () => {
    it('exports expected reason codes', () => {
      expect(REASON_CODES.METRIC_VERSION_MISMATCH).toBe('METRIC_VERSION_MISMATCH');
      expect(REASON_CODES.THRESHOLD_MISMATCH).toBe('THRESHOLD_MISMATCH');
      expect(REASON_CODES.SCOPE_TYPE_MISMATCH).toBe('SCOPE_TYPE_MISMATCH');
      expect(REASON_CODES.SCOPE_DIGEST_MISMATCH).toBe('SCOPE_DIGEST_MISMATCH');
      expect(REASON_CODES.SCANNER_VERSION_MISMATCH).toBe('SCANNER_VERSION_MISMATCH');
    });
  });

  describe('compareDocsQualityRuns', () => {
    it('returns accepted with delta when all dimensions match', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: makeSummary({ missingJsdoc: 5, weakJsdoc: 3, highComplexity: 2, evidenceCount: 10 }),
        rightSummary: makeSummary({ missingJsdoc: 3, weakJsdoc: 1, highComplexity: 1, evidenceCount: 5 }),
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.missingJsdoc).toBe(-2);
      expect(result.delta.weakJsdoc).toBe(-2);
      expect(result.delta.highComplexity).toBe(-1);
      expect(result.delta.evidenceCount).toBe(-5);
    });

    it('rejects when metricVersion mismatch', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ metricVersion: 2 }),
        rightManifest: makeManifest({ metricVersion: 3 }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.METRIC_VERSION_MISMATCH);
    });

    it('rejects when threshold minJsdocWords mismatch', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 } }),
        rightManifest: makeManifest({ thresholdConfig: { minJsdocWords: 15, complexityThreshold: 10 } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.THRESHOLD_MISMATCH);
    });

    it('rejects when threshold complexityThreshold mismatch', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 } }),
        rightManifest: makeManifest({ thresholdConfig: { minJsdocWords: 10, complexityThreshold: 15 } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.THRESHOLD_MISMATCH);
    });

    it('rejects when scopeType mismatch', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } }),
        rightManifest: makeManifest({ scopeConfig: { scopeType: 'paths', scopeDigest: 'abc' } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.SCOPE_TYPE_MISMATCH);
    });

    it('rejects when scopeDigest mismatch', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scopeConfig: { scopeType: 'src', scopeDigest: 'abc' } }),
        rightManifest: makeManifest({ scopeConfig: { scopeType: 'src', scopeDigest: 'xyz' } }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.SCOPE_DIGEST_MISMATCH);
    });

    it('rejects when scannerVersion mismatch', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ scannerVersion: '2.0.0' }),
        rightManifest: makeManifest({ scannerVersion: '2.1.0' }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(false);
      expect(result.reasonCode).toBe(REASON_CODES.SCANNER_VERSION_MISMATCH);
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

    it('handles null thresholdConfig', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ thresholdConfig: null }),
        rightManifest: makeManifest({ thresholdConfig: null }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('handles array thresholdConfig', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest({ thresholdConfig: [1, 2] }),
        rightManifest: makeManifest({ thresholdConfig: [1, 2] }),
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('handles missing scopeConfig (defaults to empty object)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { metricVersion: 2, scannerVersion: '2.0.0', thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 } },
        rightManifest: { metricVersion: 2, scannerVersion: '2.0.0', thresholdConfig: { minJsdocWords: 10, complexityThreshold: 10 } },
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

    it('handles missing metricVersion (defaults to 0)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { scannerVersion: '2.0.0', thresholdConfig: {}, scopeConfig: {} },
        rightManifest: { scannerVersion: '2.0.0', thresholdConfig: {}, scopeConfig: {} },
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('handles missing scannerVersion (defaults to empty string)', () => {
      const result = compareDocsQualityRuns({
        leftManifest: { metricVersion: 2, thresholdConfig: {}, scopeConfig: {} },
        rightManifest: { metricVersion: 2, thresholdConfig: {}, scopeConfig: {} },
        leftSummary: makeSummary(),
        rightSummary: makeSummary(),
      });
      expect(result.accepted).toBe(true);
    });

    it('normalizes summary with issueBreakdown fallback when top-level fields missing', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: { issueBreakdown: { missingJsdoc: 1, weakJsdoc: 2, highComplexity: 3 } },
        rightSummary: { issueBreakdown: { missingJsdoc: 4, weakJsdoc: 5, highComplexity: 6 } },
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.missingJsdoc).toBe(3);
      expect(result.delta.weakJsdoc).toBe(3);
      expect(result.delta.highComplexity).toBe(3);
    });

    it('handles non-object issueBreakdown in summary', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: { issueBreakdown: null, evidenceCount: 5 },
        rightSummary: { issueBreakdown: 'not-an-object', evidenceCount: 10 },
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.evidenceCount).toBe(5);
      expect(result.delta.missingJsdoc).toBe(0);
    });

    it('handles missing evidenceCount in summary', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: {},
        rightSummary: {},
      });
      expect(result.accepted).toBe(true);
      expect(result.delta.evidenceCount).toBe(0);
      expect(result.delta.missingJsdoc).toBe(0);
    });

    it('handles array issueBreakdown in summary', () => {
      const result = compareDocsQualityRuns({
        leftManifest: makeManifest(),
        rightManifest: makeManifest(),
        leftSummary: { issueBreakdown: [1, 2] },
        rightSummary: { issueBreakdown: [3, 4] },
      });
      expect(result.accepted).toBe(true);
    });
  });

  describe('main (CLI entry point)', () => {
    let originalArgv;
    let originalExitCode;

    beforeEach(() => {
      originalArgv = process.argv;
      originalExitCode = process.exitCode;
      mkdirSync(TMP_DIR, { recursive: true });
    });

    afterEach(() => {
      process.argv = originalArgv;
      process.exitCode = originalExitCode;
      rmSync(TMP_DIR, { recursive: true, force: true });
    });

    it('prints help when --help is passed', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'docs-quality',
        'docs-quality.compare.mjs',
      );
      process.argv = ['node', scriptPath, '--help'];

      const writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      // Use cache-busting import to trigger CLI guard
      await import(`./docs-quality.compare.mjs?cli-test=${Date.now()}`);

      const output = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(output).toContain('Docs quality compare');
      writeSpy.mockRestore();
      consoleLogSpy.mockRestore();
    });

    it('fails when --left and --right are not provided', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'docs-quality',
        'docs-quality.compare.mjs',
      );
      process.argv = ['node', scriptPath];

      const writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      const errorSpy = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
      const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => true);
      await import(`./docs-quality.compare.mjs?cli-test=${Date.now()}-2`);

      const output = [...writeSpy.mock.calls, ...errorSpy.mock.calls, ...consoleLogSpy.mock.calls, ...consoleErrorSpy.mock.calls].map((c) => c[0]).join('');
      expect(output).toContain('required');
      writeSpy.mockRestore();
      errorSpy.mockRestore();
      consoleLogSpy.mockRestore();
      consoleErrorSpy.mockRestore();
    });

    it('compares two runs successfully', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'docs-quality',
        'docs-quality.compare.mjs',
      );

      writeJson(MANIFEST_LEFT, makeManifest({ summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary({ evidenceCount: 10 }));
      writeJson(SUMMARY_RIGHT, makeSummary({ evidenceCount: 5 }));

      process.argv = ['node', scriptPath, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`];

      const writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      await import(`./docs-quality.compare.mjs?cli-test=${Date.now()}-3`);

      const output = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(output).toContain('accepted');
      writeSpy.mockRestore();
      consoleLogSpy.mockRestore();
    });

    it('outputs json when --json flag is passed', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'docs-quality',
        'docs-quality.compare.mjs',
      );

      writeJson(MANIFEST_LEFT, makeManifest({ summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary({ evidenceCount: 10 }));
      writeJson(SUMMARY_RIGHT, makeSummary({ evidenceCount: 5 }));

      process.argv = ['node', scriptPath, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`, '--json'];

      const writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      await import(`./docs-quality.compare.mjs?cli-test=${Date.now()}-4`);

      const output = [...writeSpy.mock.calls, ...consoleLogSpy.mock.calls].map((c) => c[0]).join('');
      expect(output).toContain('"accepted"');
      writeSpy.mockRestore();
      consoleLogSpy.mockRestore();
    });

    it('handles read error gracefully', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'docs-quality',
        'docs-quality.compare.mjs',
      );

      process.argv = ['node', scriptPath, `--left=${path.join(TMP_DIR, 'nonexistent.json')}`, `--right=${path.join(TMP_DIR, 'also-nonexistent.json')}`];

      const writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      const errorSpy = jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
      const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => true);
      await import(`./docs-quality.compare.mjs?cli-test=${Date.now()}-5`);

      const output = [...writeSpy.mock.calls, ...errorSpy.mock.calls, ...consoleLogSpy.mock.calls, ...consoleErrorSpy.mock.calls].map((c) => c[0]).join('');
      expect(output).toBeTruthy();
      writeSpy.mockRestore();
      errorSpy.mockRestore();
      consoleLogSpy.mockRestore();
      consoleErrorSpy.mockRestore();
    });

    it('sets exitCode to 1 when comparison is rejected', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'docs-quality',
        'docs-quality.compare.mjs',
      );

      writeJson(MANIFEST_LEFT, makeManifest({ metricVersion: 2, summaryPath: SUMMARY_LEFT }));
      writeJson(MANIFEST_RIGHT, makeManifest({ metricVersion: 3, summaryPath: SUMMARY_RIGHT }));
      writeJson(SUMMARY_LEFT, makeSummary());
      writeJson(SUMMARY_RIGHT, makeSummary());

      process.argv = ['node', scriptPath, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`];

      const writeSpy = jest.spyOn(process.stdout, 'write').mockImplementation(() => true);
      const consoleLogSpy = jest.spyOn(console, 'log').mockImplementation(() => true);
      process.exitCode = undefined;
      await import(`./docs-quality.compare.mjs?cli-test=${Date.now()}-6`);
      expect(process.exitCode).toBe(1);
      writeSpy.mockRestore();
      consoleLogSpy.mockRestore();
    });
  });
});