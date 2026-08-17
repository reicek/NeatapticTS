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
const TMP_DIR = path.resolve(PROJECT_ROOT, '__test_tmp_compare_errthrow');
const MANIFEST_LEFT = path.join(TMP_DIR, 'left-manifest.json');
const MANIFEST_RIGHT = path.join(TMP_DIR, 'right-manifest.json');
const SUMMARY_LEFT = path.join(TMP_DIR, 'left-summary.json');
const SUMMARY_RIGHT = path.join(TMP_DIR, 'right-summary.json');

let failCalls = [];

jest.unstable_mockModule('../../../rag-index/cli-utils.mjs', () => ({
  parseCliArgs: (argv) => ({
    help: false,
    json: false,
    left: MANIFEST_LEFT,
    right: MANIFEST_RIGHT,
    _: argv.filter((a) => !a.startsWith('--')),
  }),
  printHelp: () => {},
  writeJsonOrText: (payload, asJson, textFormatter) => {
    if (!asJson && textFormatter) textFormatter(payload);
  },
  fail: (message, asJson) => {
    failCalls.push({ message, asJson });
    process.exitCode = 1;
  },
  toRepoRelative: (filePath) => filePath.replaceAll('\\', '/'),
}));

// Create valid temp files so readFileSync succeeds, but JSON.parse throws an Error.
// This exercises the `error.message` branch at line 165 of docs-quality.compare.mjs
// (error instanceof Error is true, so error.message is evaluated).
mkdirSync(TMP_DIR, { recursive: true });
const manifestData = {
  metricVersion: 2,
  scannerVersion: '2.0.0',
  scopeConfig: { scopeType: 'src', scopeDigest: 'abc' },
  summaryPath: SUMMARY_LEFT,
};
writeFileSync(MANIFEST_LEFT, JSON.stringify(manifestData), 'utf8');
writeFileSync(MANIFEST_RIGHT, JSON.stringify({ ...manifestData, summaryPath: SUMMARY_RIGHT }), 'utf8');
writeFileSync(SUMMARY_LEFT, JSON.stringify({ evidenceCount: 10 }), 'utf8');
writeFileSync(SUMMARY_RIGHT, JSON.stringify({ evidenceCount: 5 }), 'utf8');

// Override JSON.parse to throw an Error (not a string) for manifest content.
// Since Error instances pass the `error instanceof Error` check, this covers
// the `error.message` branch of the ternary at line 165.
const originalParse = JSON.parse;
JSON.parse = function (...args) {
  if (typeof args[0] === 'string' && args[0].includes('metricVersion')) {
    throw new Error('deliberate parse error for coverage');
  }
  return originalParse.apply(JSON, args);
};

// Set process.argv so the CLI guard fires during the initial import.
const originalArgv = process.argv;
const originalExitCode = process.exitCode;
process.argv = ['node', COMPARE_MODULE, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`];
process.exitCode = 0;

await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');

JSON.parse = originalParse;
process.argv = originalArgv;
process.exitCode = originalExitCode;

rmSync(TMP_DIR, { recursive: true, force: true });

describe('docs-quality.compare.mjs error.message branch (initial import)', () => {
  it('covers error.message branch via initial import with Error throw', () => {
    expect(failCalls.length).toBeGreaterThanOrEqual(1);
    expect(failCalls[0].message).toBe('deliberate parse error for coverage');
  });
});