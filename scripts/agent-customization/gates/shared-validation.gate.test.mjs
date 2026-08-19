/**
 * @module shared-validation.gate.test
 * @description Coverage tests for shared-validation.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = process.cwd();

let mockDerivedTestFiles;
let mockSpawnResult;
let mockExistsSync;
let writtenArtifacts;

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
  spawnSync: (cmd, args) =>
    typeof mockSpawnResult === 'function'
      ? mockSpawnResult(cmd, args)
      : mockSpawnResult,
}));

jest.unstable_mockModule('node:fs', () => ({
  writeFileSync: (target, content) => {
    writtenArtifacts.push({ target, content });
  },
  existsSync: (p) =>
    typeof mockExistsSync === 'function' ? mockExistsSync(p) : mockExistsSync,
  mkdirSync: () => {},
}));

describe('shared-validation gate', () => {
  let originalExit;
  let originalExitCode;
  let originalLog;
  let logs;

  beforeEach(() => {
    jest.resetModules();
    mockDerivedTestFiles = [];
    mockSpawnResult = { status: 0, stdout: '', stderr: '' };
    mockExistsSync = true;
    writtenArtifacts = [];
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

  describe('runSharedValidationGate', () => {
    it('returns standalone descriptor when no changed files', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({ changedFiles: [] });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
      assert.equal(result.fixHint, 'n/a');
    });

    it('returns standalone descriptor when changedFiles undefined', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({});
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });

    it('passes and writes artifact when all DI runners pass', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: 't', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: 'b', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: 'l', stderr: '' }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.fixHint, null);
      assert.equal(writtenArtifacts.length, 1);
      assert.ok(writtenArtifacts[0].target.includes('shared-validation.json'));
    });

    it('fails when test runner fails (fixHint mentions tests)', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 1, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.ok(result.fixHint.includes('tests'));
    });

    it('fails when build runner fails (fixHint mentions build)', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 2, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.ok(result.fixHint.includes('build'));
    });

    it('fails when lint runner fails (fixHint mentions lint)', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 3, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.ok(result.fixHint.includes('lint'));
    });

    it('reports multiple failed runners in fixHint', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 1, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 2, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.ok(result.fixHint.includes('tests'));
      assert.ok(result.fixHint.includes('build'));
    });

    it('writes artifact to custom relative artifactPath and mkdirs when dir missing', async () => {
      mockExistsSync = false;
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        artifactPath: 'coverage/custom/art.json',
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
      assert.equal(writtenArtifacts.length, 1);
      assert.ok(writtenArtifacts[0].target.includes('custom'));
    });

    it('accepts an absolute artifactPath', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const abs = path.join(REPO_ROOT, 'coverage', 'abs-artifact.json');
      await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        artifactPath: abs,
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(writtenArtifacts[0].target, abs);
    });
  });

  describe('defaultTestRunner', () => {
    it('returns status 0 when no test files selected', async () => {
      mockDerivedTestFiles = [];
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.testResult.status, 0);
    });

    it('returns status 0 when no existing test files on disk', async () => {
      mockDerivedTestFiles = ['src/missing.test.ts'];
      mockExistsSync = () => false;
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.testResult.status, 0);
    });

    it('groups mjs and other files and aggregates group results (pass)', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs', 'src/b.test.ts'];
      mockExistsSync = () => true;
      mockSpawnResult = (cmd, args) => {
        // jest runner spawns via --eval with 'jest' in script
        return {
          status: 0,
          stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
          stderr: '',
        };
      };
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.testResult.status, 0);
    });

    it('propagates non-zero group status (fail)', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs'];
      mockExistsSync = () => true;
      mockSpawnResult = () => ({
        status: 0,
        stdout: JSON.stringify({ status: 7, stdout: 'fail', stderr: '' }),
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.testResult.status, 7);
    });

    it('handles parsed.error from jest group (status fallback to 1)', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs'];
      mockExistsSync = () => true;
      mockSpawnResult = () => ({
        status: 0,
        stdout: JSON.stringify({
          status: null,
          stdout: '',
          stderr: '',
          error: 'spawn ENOENT',
        }),
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.testResult.status, 1);
    });

    it('falls back to raw spawn result when jest JSON is invalid', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs'];
      mockExistsSync = () => true;
      mockSpawnResult = () => ({
        status: 3,
        stdout: 'not-json',
        stderr: 'err',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.testResult.status, 3);
    });

    it('uses status=1 when jest spawn status is null and no error', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs'];
      mockExistsSync = () => true;
      mockSpawnResult = () => ({
        status: null,
        stdout: 'garbage',
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.testResult.status, 1);
    });
  });

  describe('defaultBuildRunner', () => {
    it('returns status 0 when build succeeds', async () => {
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
    });

    it('returns status 1 when build has an error object', async () => {
      mockSpawnResult = () => ({
        status: null,
        error: new Error('boom'),
        stdout: '',
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.buildResult.status, 1);
    });

    it('returns status 1 when build exits with signal and no status', async () => {
      mockSpawnResult = () => ({
        status: null,
        signal: 'SIGTERM',
        stdout: '',
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.buildResult.status, 1);
    });

    it('returns status 0 when build status is null with no error/signal', async () => {
      mockSpawnResult = () => ({ status: null, stdout: '', stderr: '' });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
    });

    it('returns non-zero status when build fails', async () => {
      mockSpawnResult = () => ({ status: 2, stdout: '', stderr: '' });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.buildResult.status, 2);
    });
  });

  describe('defaultLintRunner', () => {
    it('returns status 0 when lint succeeds', async () => {
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
    });

    it('returns status 1 when lint has an error object', async () => {
      mockSpawnResult = () => ({
        status: null,
        error: new Error('lintboom'),
        stdout: '',
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.lintResult.status, 1);
    });

    it('returns status 1 when lint exits with signal', async () => {
      mockSpawnResult = () => ({
        status: null,
        signal: 'SIGKILL',
        stdout: '',
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.lintResult.status, 1);
    });

    it('returns status 0 when lint status null no error/signal', async () => {
      mockSpawnResult = () => ({ status: null, stdout: '', stderr: '' });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
    });

    it('returns non-zero status when lint fails', async () => {
      mockSpawnResult = () => ({ status: 4, stdout: '', stderr: '' });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.lintResult.status, 4);
    });
  });

  describe('main', () => {
    it('prints usage and exits 0 on --help', async () => {
      process.exit = (code) => {
        throw new Error(`EXIT:${code}`);
      };
      const { main } = await import('./shared-validation.gate.mjs');
      await assert.rejects(() => main(['--help']), /EXIT:0/);
      assert.ok(logs.some((l) => l.includes('shared-validation gate')));
    });

    it('emits standalone descriptor with --json and no --changed-files', async () => {
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main(['--json']);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
      assert.equal(JSON.parse(logs[0]).pass, true);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text with no --json and no --changed-files', async () => {
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main([]);
      assert.equal(result.pass, true);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('runs gate with --changed-files=value (all default runners pass)', async () => {
      mockDerivedTestFiles = [];
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main(['--json', '--changed-files=src/foo.ts']);
      assert.equal(result.pass, true);
    });

    it('emits FAIL text with fixHint when default runners fail', async () => {
      mockDerivedTestFiles = [];
      mockSpawnResult = () => ({ status: 1, stdout: '', stderr: '' });
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main(['--changed-files=src/foo.ts']);
      assert.equal(result.pass, false);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('accepts space-separated --changed-files value', async () => {
      mockDerivedTestFiles = [];
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main(['--changed-files', 'src/foo.ts']);
      assert.equal(result.pass, true);
    });

    it('splits --changed-files on commas and newlines', async () => {
      mockDerivedTestFiles = [];
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main([
        '--json',
        '--changed-files=src/a.ts,src/b.ts\r\nsrc/c.ts',
      ]);
      assert.equal(result.pass, true);
    });

    it('accepts --artifact-path=value', async () => {
      mockDerivedTestFiles = [];
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main([
        '--json',
        '--changed-files=src/foo.ts',
        '--artifact-path=coverage/custom.json',
      ]);
      assert.equal(result.pass, true);
      assert.ok(writtenArtifacts[0].target.includes('custom.json'));
    });

    it('accepts space-separated --artifact-path value', async () => {
      mockDerivedTestFiles = [];
      mockSpawnResult = () => ({ status: 0, stdout: '', stderr: '' });
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main([
        '--json',
        '--changed-files=src/foo.ts',
        '--artifact-path',
        'coverage/space.json',
      ]);
      assert.equal(result.pass, true);
      assert.ok(writtenArtifacts[0].target.includes('space.json'));
    });

    it('returns [] when --changed-files flag has no following value', async () => {
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main(['--json', '--changed-files']);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });
  });

  describe('default parameter coverage', () => {
    it('main() with no args uses process.argv.slice(2) default', async () => {
      const { main } = await import('./shared-validation.gate.mjs');
      const result = await main();
      assert.equal(result.pass, true);
    });

    it('runSharedValidationGate() with no args returns standalone descriptor', async () => {
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate();
      assert.equal(result.pass, true);
    });
  });

  describe('nullish and edge-case branches', () => {
    it('runs only non-mjs test files when no mjs files exist', async () => {
      mockDerivedTestFiles = ['src/a.test.ts'];
      mockExistsSync = () => true;
      mockSpawnResult = (cmd, args) => ({
        status: 0,
        stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.pass, true);
    });

    it('uses status=1 when groupResult.status is null (falsy non-zero)', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs'];
      mockExistsSync = () => true;
      mockSpawnResult = () => ({
        status: 0,
        stdout: JSON.stringify({ status: null, stdout: '', stderr: '' }),
        stderr: '',
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.testResult.status, 1);
    });

    it('covers nullish stdout/stderr in runJestGroup catch block', async () => {
      mockDerivedTestFiles = ['src/a.test.mjs'];
      mockExistsSync = () => true;
      mockSpawnResult = () => ({
        status: null,
        stdout: null,
        stderr: null,
      });
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.testResult.status, 1);
    });

    it('covers nullish stdout/stderr in defaultBuildRunner', async () => {
      mockSpawnResult = (cmd, args) => {
        if (cmd === 'npm' && args[1] === 'build') {
          return { status: 0, stdout: null, stderr: null };
        }
        return { status: 0, stdout: '', stderr: '' };
      };
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        lintRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.buildResult.stdout, '');
    });

    it('covers nullish stdout/stderr in defaultLintRunner', async () => {
      mockSpawnResult = (cmd, args) => {
        if (cmd === 'npm' && args[1] === 'lint') {
          return { status: 0, stdout: null, stderr: null };
        }
        return { status: 0, stdout: '', stderr: '' };
      };
      const { runSharedValidationGate } =
        await import('./shared-validation.gate.mjs');
      const result = await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
        buildRunner: async () => ({ status: 0, stdout: '', stderr: '' }),
      });
      assert.equal(result.evidence.lintResult.stdout, '');
    });
  });
});
