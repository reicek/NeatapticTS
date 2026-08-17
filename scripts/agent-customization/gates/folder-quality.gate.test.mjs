/**
 * @module folder-quality.gate.test
 * @description Coverage tests for folder-quality.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/folder-quality.gate.mjs',
);

let mockReport;

jest.unstable_mockModule('../../folder-quality-metrics.mjs', () => ({
  runFolderQualityMetrics: async () => mockReport,
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
  const stderrChunks = [];
  const originalLog = console.log;
  const originalStderrWrite = process.stderr.write.bind(process.stderr);
  console.log = (...args) => logs.push(args.map(String).join(' '));
  process.stderr.write = (chunk) => { stderrChunks.push(String(chunk)); return true; };
  try {
    jest.resetModules();
    await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
      await import('./folder-quality.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
    process.stderr.write = originalStderrWrite;
  }
  return { logs, stderr: stderrChunks.join('') };
}

describe('folder-quality gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockReport = {
      pass: true,
      evidence: { summary: 'ok' },
      folderChecked: 'src',
      smells: [],
    };
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runFolderQualityGate', () => {
    it('returns pass=true with no smells', async () => {
      const { runFolderQualityGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./folder-quality.gate.mjs'),
      );
      const result = await runFolderQualityGate({ folderPath: 'src' });
      assert.equal(result.pass, true);
      assert.equal(result.fixHint, null);
      assert.equal(result.schema_version, 1);
    });

    it('returns pass=false with smells and fixHint', async () => {
      mockReport = {
        pass: false,
        evidence: { summary: 'bad' },
        folderChecked: 'src',
        smells: [{ type: 'large-file', file: 'a.ts' }],
      };
      const { runFolderQualityGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./folder-quality.gate.mjs'),
      );
      const result = await runFolderQualityGate({ folderPath: 'src' });
      assert.equal(result.pass, false);
      assert.ok(result.fixHint.includes('quality:folder'));
      assert.equal(result.evidence.smellCount, 1);
    });
  });

  describe('main via import.meta.url guard', () => {
    it('prints usage and exits with --help', async () => {
      const { logs } = await importGateMain(['--help']);
      assert.ok(logs.some((l) => l.includes('Folder-quality gate')));
    });

    it('prints usage and exits with -h', async () => {
      const { logs } = await importGateMain(['-h']);
      assert.ok(logs.some((l) => l.includes('Folder-quality gate')));
    });

    it('errors when --folder is missing', async () => {
      const { stderr, logs } = await importGateMain([]);
      assert.ok(stderr.includes('--folder=<path> is required'));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=true', async () => {
      const { logs } = await importGateMain(['--folder=src', '--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });

    it('emits PASS text without --json when pass=true', async () => {
      const { logs } = await importGateMain(['--folder=src']);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with fixHint when pass=false', async () => {
      mockReport = {
        pass: false,
        evidence: {},
        folderChecked: 'src',
        smells: [{ type: 'big' }],
      };
      const { logs } = await importGateMain(['--folder=src']);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockReport = {
        pass: false,
        evidence: {},
        folderChecked: 'src',
        smells: [],
      };
      const { logs } = await importGateMain(['--folder=src', '--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });

    it('handles unknown argument error (Error instance)', async () => {
      const { stderr } = await importGateMain(['--bogus=1']);
      assert.ok(stderr.includes('Unknown argument: --bogus=1'));
      assert.equal(process.exitCode, 1);
    });

    it('handles non-Error throw in main catch block', async () => {
      jest.resetModules();
      const { logs, stderr } = {};
      const originalStderrWrite = process.stderr.write.bind(process.stderr);
      const stderrChunks = [];
      process.stderr.write = (chunk) => { stderrChunks.push(String(chunk)); return true; };
      const originalLog = console.log;
      const logChunks = [];
      console.log = (...args) => logChunks.push(args.map(String).join(' '));
      // Force runFolderQualityMetrics to throw a non-Error
      jest.unstable_mockModule('../../folder-quality-metrics.mjs', () => ({
        runFolderQualityMetrics: async () => { throw 'string error'; },
      }));
      try {
        await withArgv([process.execPath, GATE_PATH, '--folder=src'], async () => {
          await import('./folder-quality.gate.mjs');
          await new Promise((r) => setTimeout(r, 200));
        });
      } finally {
        console.log = originalLog;
        process.stderr.write = originalStderrWrite;
        // Restore original mock
        jest.unstable_mockModule('../../folder-quality-metrics.mjs', () => ({
          runFolderQualityMetrics: async () => mockReport,
        }));
      }
      assert.ok(stderrChunks.join('').includes('string error'));
      assert.equal(process.exitCode, 1);
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run main when argv[1] does not match', async () => {
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./folder-quality.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});