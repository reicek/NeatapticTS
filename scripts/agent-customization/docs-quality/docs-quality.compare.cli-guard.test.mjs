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
const TMP_DIR = path.resolve(PROJECT_ROOT, '__test_tmp_compare_cliguard');
const MANIFEST_LEFT = path.join(TMP_DIR, 'left-manifest.json');
const MANIFEST_RIGHT = path.join(TMP_DIR, 'right-manifest.json');
const SUMMARY_LEFT = path.join(TMP_DIR, 'left-summary.json');
const SUMMARY_RIGHT = path.join(TMP_DIR, 'right-summary.json');

let failCalls = [];

jest.unstable_mockModule('../../../rag-index/cli-utils.mjs', () => ({
  parseCliArgs: (argv) => ({
    help: false,
    json: true,
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

// Create temp files before import so readFileSync can read them
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

// Override JSON.parse to throw a non-Error (string) for manifest content
// This exercises the `String(error)` branch at line 165 of docs-quality.compare.mjs
const originalParse = JSON.parse;
JSON.parse = function (...args) {
  if (typeof args[0] === 'string' && args[0].includes('metricVersion')) {
    throw 'string error';
  }
  return originalParse.apply(JSON, args);
};

// Set process.argv so the CLI guard fires during the initial import
// (import.meta.url === pathToFileURL(process.argv[1]).href)
const originalArgv = process.argv;
const originalExitCode = process.exitCode;
process.argv = ['node', COMPARE_MODULE, `--left=${MANIFEST_LEFT}`, `--right=${MANIFEST_RIGHT}`, '--json'];
process.exitCode = 0;

// Initial import — NOT inside isolateModulesAsync — so branch coverage is tracked normally.
// The CLI guard fires, main() runs, JSON.parse throws a string, and the catch block
// evaluates `error instanceof Error ? error.message : String(error)` — covering the
// String(error) branch that isolateModulesAsync fails to merge.
await import('../../../rag-index/docs-quality/docs-quality.compare.mjs');

// Restore global state
JSON.parse = originalParse;
process.argv = originalArgv;
process.exitCode = originalExitCode;

// Clean up temp directory
rmSync(TMP_DIR, { recursive: true, force: true });

describe('docs-quality.compare.mjs CLI guard (initial import)', () => {
  it('covers String(error) branch via initial import (no isolateModulesAsync)', () => {
    expect(failCalls.length).toBeGreaterThanOrEqual(1);
    expect(failCalls[0].message).toBe('string error');
    expect(failCalls[0].asJson).toBe(true);
  });
});