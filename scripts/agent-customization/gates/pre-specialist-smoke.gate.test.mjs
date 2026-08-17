/**
 * @module pre-specialist-smoke.gate.test
 * @description Coverage tests for pre-specialist-smoke.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const REPO_ROOT = process.cwd();

let mockDerivedTestFiles;
let mockSpawnResult;

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => ({
    json: argv.includes('--json'),
    help: argv.includes('--help') || argv.includes('-h'),
  }),
  repoRoot: REPO_ROOT,
}));

jest.unstable_mockModule('./gate-test-utils.mjs', () => ({
  deriveTestFiles: () => mockDerivedTestFiles,
}));

jest.unstable_mockModule('node:child_process', () => ({
  spawnSync: () => mockSpawnResult,
}));

describe('pre-specialist-smoke gate', () => {
  let originalExit;
  let originalExitCode;
  let originalLog;
  let logs;

  beforeEach(() => {
    jest.resetModules();
    mockDerivedTestFiles = [];
    mockSpawnResult = { status: 0, stdout: '', stderr: '' };
    originalExit = process.exit;
    originalExitCode = process.exitCode;
    originalLog = console.log;
    logs = [];
    console.log = (...args) => logs.push(args.map(String).join(' '));
  });

  afterEach(() => {
    process.exit = originalExit;
    process.exitCode = originalExitCode ?? 0;
    console.log = originalLog;
  });

  describe('runPreSpecialistSmokeGate', () => {
    it('returns standalone descriptor when no changed files', async () => {
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({ changedFiles: [] });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
      assert.equal(result.fixHint, 'n/a');
    });

    it('returns standalone descriptor when called with no arguments', async () => {
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });

    it('returns standalone descriptor when changedFiles is undefined', async () => {
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({});
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });

    it('passes when changed files map to no test files', async () => {
      mockDerivedTestFiles = [];
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.md'],
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.testFiles.length, 0);
      assert.equal(result.fixHint, null);
    });

    it('passes when runner returns status 0', async () => {
      mockDerivedTestFiles = ['src/foo.test.ts'];
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.ts'],
        runner: async () => ({ status: 0, stdout: 'ok', stderr: '' }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.exitStatus, 0);
    });

    it('fails when runner returns non-zero status', async () => {
      mockDerivedTestFiles = ['src/foo.test.ts'];
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.ts'],
        runner: async () => ({ status: 1, stdout: 'fail', stderr: 'err' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.exitStatus, 1);
      assert.ok(result.fixHint.includes('04-implementing'));
    });
  });

  describe('defaultJestRunner', () => {
    it('parses valid JSON output from spawned child', async () => {
      mockDerivedTestFiles = ['src/foo.test.mjs'];
      mockSpawnResult = {
        status: 0,
        stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
        stderr: '',
      };
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.ts'],
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.exitStatus, 0);
    });

    it('falls back to raw spawn result when JSON parse fails', async () => {
      mockDerivedTestFiles = ['src/foo.test.mjs'];
      mockSpawnResult = {
        status: 2,
        stdout: 'not-json',
        stderr: 'some error',
      };
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.ts'],
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.exitStatus, 2);
    });

    it('uses status=1 when spawn status is null and no signal', async () => {
      mockDerivedTestFiles = ['src/foo.test.mjs'];
      mockSpawnResult = {
        status: null,
        stdout: 'garbage',
        stderr: '',
      };
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.ts'],
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.exitStatus, 1);
    });

    it('handles null stdout in catch fallback', async () => {
      mockDerivedTestFiles = ['src/foo.test.mjs'];
      mockSpawnResult = {
        status: null,
        stdout: null,
        stderr: null,
      };
      const { runPreSpecialistSmokeGate } = await import(
        './pre-specialist-smoke.gate.mjs'
      );
      const result = await runPreSpecialistSmokeGate({
        changedFiles: ['src/foo.ts'],
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.exitStatus, 1);
      assert.equal(result.evidence.stdout, '');
      assert.equal(result.evidence.stderr, '');
    });
  });

  describe('main', () => {
    it('prints usage and exits 0 on --help', async () => {
      process.exit = (code) => {
        throw new Error(`EXIT:${code}`);
      };
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      await assert.rejects(() => main(['--help']), /EXIT:0/);
      assert.ok(logs.some((l) => l.includes('pre-specialist-smoke gate')));
    });

    it('emits standalone descriptor with --json and no --changed-files', async () => {
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main(['--json']);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
      assert.equal(JSON.parse(logs[0]).pass, true);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text with no --json and no --changed-files', async () => {
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main([]);
      assert.equal(result.pass, true);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('uses process.argv.slice(2) when called with no arguments', async () => {
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });

    it('runs the gate with --changed-files=value', async () => {
      mockDerivedTestFiles = ['src/foo.test.ts'];
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main(['--json', '--changed-files=src/foo.ts']);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.exitStatus, 0);
    });

    it('runs the gate with space-separated --changed-files value', async () => {
      mockDerivedTestFiles = ['src/foo.test.ts'];
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main(['--changed-files', 'src/foo.ts']);
      assert.equal(result.pass, true);
      assert.ok(logs.some((l) => l.includes('PASS')));
    });

    it('splits --changed-files on commas and newlines', async () => {
      mockDerivedTestFiles = ['src/a.test.ts', 'src/b.test.ts'];
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main([
        '--json',
        '--changed-files=src/a.ts,src/b.ts\r\nsrc/c.ts',
      ]);
      assert.equal(result.pass, true);
    });

    it('emits FAIL text with fixHint when runner fails', async () => {
      mockDerivedTestFiles = ['src/foo.test.ts'];
      mockSpawnResult = { status: 1, stdout: 'fail', stderr: 'err' };
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main([
        '--changed-files=src/foo.ts',
      ]);
      assert.equal(result.pass, false);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('returns [] when --changed-files flag has no following value', async () => {
      const { main } = await import('./pre-specialist-smoke.gate.mjs');
      const result = await main(['--json', '--changed-files']);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });
  });
});