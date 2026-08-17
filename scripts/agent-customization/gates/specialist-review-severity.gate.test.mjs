/**
 * @module specialist-review-severity.gate.test
 * @description Coverage tests for specialist-review-severity.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/specialist-review-severity.gate.mjs',
);

let mockOptions;
let mockReadFileSyncResult;

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: () => mockOptions,
}));

jest.unstable_mockModule('node:fs', () => ({
  readFileSync: () => mockReadFileSyncResult,
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGateMain(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
      await import('./specialist-review-severity.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('specialist-review-severity gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockOptions = { json: false, help: false, input: undefined };
    mockReadFileSyncResult = '';
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('classifySeverity', () => {
    it('returns TRIVIAL for empty array', async () => {
      const { classifySeverity } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./specialist-review-severity.gate.mjs'),
      );
      const result = classifySeverity([]);
      assert.equal(result.severity, 'TRIVIAL');
      assert.equal(result.specialistCount, 0);
      assert.deepEqual(result.trivialFiles, []);
      assert.deepEqual(result.nonTrivialFiles, []);
    });

    it('returns TRIVIAL when all files are trivial (test, spec, md, formatting, lockfile)', async () => {
      const { classifySeverity } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./specialist-review-severity.gate.mjs'),
      );
      const result = classifySeverity([
        'foo.test.ts',
        'bar.spec.js',
        'baz.test.mjs',
        'qux.spec.cjs',
        'README.md',
        '.prettierrc',
        '.prettierignore',
        '.editorconfig',
        '.gitattributes',
        '.eslintignore',
        '.gitignore',
        '.npmrc',
        '.nvmrc',
        'package-lock.json',
        'yarn.lock',
        'pnpm-lock.yaml',
      ]);
      assert.equal(result.severity, 'TRIVIAL');
      assert.equal(result.specialistCount, 0);
      assert.equal(result.trivialFiles.length, 16);
      assert.equal(result.nonTrivialFiles.length, 0);
    });

    it('returns FULL when mixed trivial and non-trivial', async () => {
      const { classifySeverity } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./specialist-review-severity.gate.mjs'),
      );
      const result = classifySeverity([
        'foo.test.ts',
        'src/bar.ts',
        'examples/baz.js',
      ]);
      assert.equal(result.severity, 'FULL');
      assert.equal(result.specialistCount, 1);
      assert.deepEqual(result.trivialFiles, ['foo.test.ts']);
      assert.deepEqual(result.nonTrivialFiles, ['src/bar.ts', 'examples/baz.js']);
    });

    it('throws TypeError for non-array input', async () => {
      const { classifySeverity } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./specialist-review-severity.gate.mjs'),
      );
      assert.throws(() => classifySeverity('not-an-array'), {
        name: 'TypeError',
        message: 'classifySeverity expects an array of file paths',
      });
    });

    it('throws TypeError for non-string element', async () => {
      const { classifySeverity } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./specialist-review-severity.gate.mjs'),
      );
      assert.throws(() => classifySeverity([42]), {
        name: 'TypeError',
        message: 'classifySeverity expects file paths to be strings',
      });
    });
  });

  describe('CLI via import.meta.url guard', () => {
    it('prints usage with --help', async () => {
      mockOptions = { json: false, help: true, input: undefined };
      const logs = await importGateMain(['--help']);
      assert.ok(logs.some((l) => l.includes('specialist-review-severity gate')));
      assert.equal(process.exitCode, 0);
    });

    it('emits JSON with --json and non-trivial input', async () => {
      mockOptions = { json: true, help: false, input: 'src/foo.ts' };
      const logs = await importGateMain(['--json', '--input=src/foo.ts']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.classification.severity, 'FULL');
      assert.equal(
        parsed.evidence.reason,
        'runtime logic changed under src/, examples/, or benchmarks/',
      );
    });

    it('emits JSON with --json and trivial input', async () => {
      mockOptions = { json: true, help: false, input: 'foo.test.ts' };
      const logs = await importGateMain(['--json', '--input=foo.test.ts']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.classification.severity, 'TRIVIAL');
      assert.equal(
        parsed.evidence.reason,
        'all changes are test, doc, formatting, or lock files',
      );
    });

    it('emits JSON with --json and no input', async () => {
      mockOptions = { json: true, help: false, input: undefined };
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.classification.severity, 'TRIVIAL');
      assert.equal(parsed.evidence.reason, 'no changed files supplied');
    });

    it('emits text without --json for FULL severity with non-trivial files listing', async () => {
      mockOptions = { json: false, help: false, input: 'src/foo.ts,foo.test.ts' };
      const logs = await importGateMain(['--input=src/foo.ts,foo.test.ts']);
      assert.ok(
        logs.some((l) => l.includes('FULL specialist-review-severity gate')),
      );
      assert.ok(logs.some((l) => l.includes('non-trivial files:')));
      assert.ok(logs.some((l) => l.includes('src/foo.ts')));
    });

    it('emits text without --json for TRIVIAL severity with files', async () => {
      mockOptions = { json: false, help: false, input: 'foo.test.ts' };
      const logs = await importGateMain(['--input=foo.test.ts']);
      assert.ok(
        logs.some((l) => l.includes('TRIVIAL specialist-review-severity gate')),
      );
    });

    it('reads file list from @file syntax', async () => {
      mockOptions = { json: true, help: false, input: '@file-list.txt' };
      mockReadFileSyncResult = 'src/foo.ts\n# comment\nsrc/bar.js\n';
      const logs = await importGateMain(['--json', '--input=@file-list.txt']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.evidence.classification.severity, 'FULL');
      assert.equal(parsed.evidence.classification.nonTrivialFiles.length, 2);
    });

    it('handles comma-separated input with empty entries (filter Boolean)', async () => {
      mockOptions = { json: true, help: false, input: 'src/foo.ts,,foo.test.ts' };
      const logs = await importGateMain([
        '--json',
        '--input=src/foo.ts,,foo.test.ts',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.evidence.classification.severity, 'FULL');
      assert.equal(parsed.evidence.classification.nonTrivialFiles.length, 1);
      assert.equal(parsed.evidence.classification.trivialFiles.length, 1);
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run CLI when argv[1] does not match', async () => {
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./specialist-review-severity.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});