/**
 * @module shared-validation.gate.test
 * @description Red tests for the shared validation gate contract.
 *
 * The shared-validation gate runs tests and build once for a set of changed
 * files, then writes a structured JSON artifact that every specialist reviewer
 * receives. This avoids having each specialist independently run the same
 * suites and build.
 */
import {
  describe,
  expect,
  it,
  jest,
  beforeEach,
  afterEach,
} from '@jest/globals';
import path from 'node:path';
import { mkdtempSync, rmSync, readFileSync, existsSync } from 'node:fs';
import os from 'node:os';
import { spawnSync } from 'node:child_process';

jest.mock('node:child_process', () => ({
  spawnSync: jest.fn(),
}));

interface ValidationReport {
  pass: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  evidence: Record<string, any>;
  fixHint: string | null;
  owner: string;
}

interface RunnerResult {
  status: number;
  stdout: string;
  stderr: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const GATE_PATH = path.resolve(__dirname, 'shared-validation.gate.mjs');

// Strongly-typed mocked spawnSync for use with mockImplementation callbacks.
const mockSpawn = spawnSync as unknown as jest.Mock<any>;

/**
 * Lazily load the ESM gate module inside each test so Jest does not try to
 * transform its top-level await through the mjs-to-cjs transformer.
 */
async function loadGate() {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import('./shared-validation.gate.mjs');
}

describe('shared-validation.gate.mjs', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'shared-val-'));
  });

  afterEach(() => {
    process.exitCode = 0;
    jest.clearAllMocks();
    rmSync(tempDir, { recursive: true, force: true });
    rmSync(path.join(REPO_ROOT, 'coverage', 'shared-validation.json'), {
      force: true,
    });
  });

  it('exposes runSharedValidationGate and main exports', async () => {
    const gate = await loadGate();

    expect(typeof gate.runSharedValidationGate).toBe('function');
    expect(typeof gate.main).toBe('function');
  });

  it('passes when test, build, and lint runners all succeed', async () => {
    const { runSharedValidationGate } = await loadGate();
    const artifactPath = path.join(tempDir, 'artifact.json');

    const result = await runSharedValidationGate({
      changedFiles: ['src/foo.ts'],
      artifactPath,
      testRunner: async () => ({ status: 0, stdout: 'tests ok', stderr: '' }),
      buildRunner: async () => ({ status: 0, stdout: 'build ok', stderr: '' }),
      lintRunner: async () => ({ status: 0, stdout: 'lint ok', stderr: '' }),
    });

    expect(result).toEqual(
      expect.objectContaining({
        pass: true,
        owner: 'shared-validation',
        fixHint: null,
      }),
    );
    expect(result.evidence.changedFiles).toEqual(['src/foo.ts']);
    expect(result.evidence.testResult).toEqual(
      expect.objectContaining({ status: 0 }),
    );
    expect(result.evidence.buildResult).toEqual(
      expect.objectContaining({ status: 0 }),
    );
    expect(result.evidence.lintResult).toEqual(
      expect.objectContaining({ status: 0 }),
    );
  });

  describe('artifact writing', () => {
    let artifactPath: string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let artifact: Record<string, any>;

    beforeEach(async () => {
      const { runSharedValidationGate } = await loadGate();
      artifactPath = path.join(tempDir, 'artifact.json');
      await runSharedValidationGate({
        changedFiles: ['src/foo.ts', 'src/bar.ts'],
        artifactPath,
        testRunner: async () => ({ status: 0, stdout: 'tests ok', stderr: '' }),
        buildRunner: async () => ({
          status: 0,
          stdout: 'build ok',
          stderr: '',
        }),
        lintRunner: async () => ({ status: 0, stdout: 'lint ok', stderr: '' }),
      });
      artifact = JSON.parse(readFileSync(artifactPath, 'utf8'));
    });

    it('writes the artifact file to disk', () => {
      expect(existsSync(artifactPath)).toBe(true);
    });

    it('records the changed files in the artifact', () => {
      expect(artifact.changedFiles).toEqual(['src/foo.ts', 'src/bar.ts']);
    });

    it('records the test result in the artifact', () => {
      expect(artifact.testResult.status).toBe(0);
    });

    it('records the build result in the artifact', () => {
      expect(artifact.buildResult.status).toBe(0);
    });

    it('records the lint result in the artifact', () => {
      expect(artifact.lintResult.status).toBe(0);
    });

    it('uses the default artifact path when none is supplied', async () => {
      const defaultPath = path.join(
        REPO_ROOT,
        'coverage',
        'shared-validation.json',
      );
      const { runSharedValidationGate } = await loadGate();
      await runSharedValidationGate({
        changedFiles: ['src/foo.ts'],
        testRunner: async () => ({ status: 0, stdout: 'tests ok', stderr: '' }),
        buildRunner: async () => ({
          status: 0,
          stdout: 'build ok',
          stderr: '',
        }),
        lintRunner: async () => ({ status: 0, stdout: 'lint ok', stderr: '' }),
      });
      expect(existsSync(defaultPath)).toBe(true);
    });
  });

  it('fails when the test runner reports failing tests', async () => {
    const { runSharedValidationGate } = await loadGate();
    const artifactPath = path.join(tempDir, 'artifact.json');

    const result = await runSharedValidationGate({
      changedFiles: ['src/foo.ts'],
      artifactPath,
      testRunner: async () => ({
        status: 1,
        stdout: '',
        stderr: '1 test failed',
      }),
      buildRunner: async () => ({ status: 0, stdout: 'build ok', stderr: '' }),
      lintRunner: async () => ({ status: 0, stdout: 'lint ok', stderr: '' }),
    });

    expect(result.pass).toBe(false);
    expect(result.owner).toBe('shared-validation');
    expect(result.fixHint).toMatch(/shared validation failed/);
  });

  it('fails when the build runner reports a build error', async () => {
    const { runSharedValidationGate } = await loadGate();
    const artifactPath = path.join(tempDir, 'artifact.json');

    const result = await runSharedValidationGate({
      changedFiles: ['src/foo.ts'],
      artifactPath,
      testRunner: async () => ({ status: 0, stdout: 'tests ok', stderr: '' }),
      buildRunner: async () => ({
        status: 1,
        stdout: '',
        stderr: 'build error',
      }),
      lintRunner: async () => ({ status: 0, stdout: 'lint ok', stderr: '' }),
    });

    expect(result.pass).toBe(false);
    expect(result.evidence.buildResult).toEqual(
      expect.objectContaining({ status: 1 }),
    );
  });

  it('fails when the lint runner reports lint errors', async () => {
    const { runSharedValidationGate } = await loadGate();
    const artifactPath = path.join(tempDir, 'artifact.json');

    const result = await runSharedValidationGate({
      changedFiles: ['src/foo.ts'],
      artifactPath,
      testRunner: async () => ({ status: 0, stdout: 'tests ok', stderr: '' }),
      buildRunner: async () => ({ status: 0, stdout: 'build ok', stderr: '' }),
      lintRunner: async () => ({
        status: 1,
        stdout: '',
        stderr: 'lint error',
      }),
    });

    expect(result.pass).toBe(false);
    expect(result.evidence.lintResult).toEqual(
      expect.objectContaining({ status: 1 }),
    );
  });

  it('returns a standalone descriptor when no changed files are supplied', async () => {
    const { runSharedValidationGate } = await loadGate();

    const result = await runSharedValidationGate();

    expect(result.pass).toBe(true);
    expect(result.evidence.mode).toBe('standalone-descriptor');
  });

  it('emits a valid JSON contract from the CLI', () => {
    const realSpawnSync = (
      jest.requireActual(
        'node:child_process',
      ) as typeof import('node:child_process')
    ).spawnSync;
    const cliResult = realSpawnSync(process.execPath, [GATE_PATH, '--json'], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 60_000,
    });

    const report = JSON.parse(cliResult.stdout ?? '{}') as ValidationReport;

    expect(report).toEqual(
      expect.objectContaining({
        pass: expect.any(Boolean),
        owner: 'shared-validation',
        evidence: expect.objectContaining({
          mode: 'standalone-descriptor',
        }),
      }),
    );
  });

  describe('default Jest runner from main', () => {
    let result: ValidationReport;
    let logSpy: ReturnType<typeof jest.spyOn>;
    let logged: ValidationReport;

    beforeEach(async () => {
      const { main } = await loadGate();
      logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      } as unknown as ReturnType<typeof spawnSync>);

      result = (await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
        '--artifact-path',
        path.join(tempDir, 'main-artifact.json'),
      ])) as ValidationReport;

      logged = JSON.parse(logSpy.mock.calls[0][0] as string);
    });

    afterEach(() => {
      logSpy.mockRestore();
      (spawnSync as jest.Mock).mockClear();
    });

    it('returns a passing report', () => {
      expect(result.pass).toBe(true);
    });

    it('logs exactly once', () => {
      expect(logSpy).toHaveBeenCalledTimes(1);
    });

    it('logs a passing report', () => {
      expect(logged.pass).toBe(true);
    });

    it('logs the shared-validation owner', () => {
      expect(logged.owner).toBe('shared-validation');
    });

    it('invokes the default Jest runner', () => {
      expect(spawnSync).toHaveBeenCalled();
    });
  });

  describe('CLI non-JSON output', () => {
    afterEach(() => {
      jest.clearAllMocks();
    });

    it('logs PASS when validation succeeds', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main(['--changed-files', 'src/foo.md']);
      expect(logSpy).toHaveBeenCalledWith('PASS', 'shared-validation gate');
      logSpy.mockRestore();
    });

    it('logs FAIL when validation fails', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: JSON.stringify({
          status: 1,
          stdout: '',
          stderr: 'tests failed',
        }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main([
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      expect(logSpy).toHaveBeenCalledWith('FAIL', 'shared-validation gate');
      logSpy.mockRestore();
    });

    it('logs the fixHint after a failure', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: JSON.stringify({
          status: 1,
          stdout: '',
          stderr: 'tests failed',
        }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main([
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      expect(logSpy).toHaveBeenCalledWith(
        'fixHint:',
        expect.stringContaining('shared validation failed'),
      );
      logSpy.mockRestore();
    });
  });

  describe('default runner edge cases', () => {
    afterEach(() => {
      jest.clearAllMocks();
    });

    function successForTestRunner() {
      return {
        status: 0,
        stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      };
    }

    it('returns early when no test files are selected', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: '',
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main(['--json', '--changed-files', 'src/foo.md']);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(true);
    });

    it('returns early when derived test files are missing', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: '',
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main(['--json', '--changed-files', 'src/foo.ts']);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(true);
      expect(spawnSync).not.toHaveBeenCalledWith(
        process.execPath,
        expect.anything(),
      );
    });

    it('runs .mjs tests with experimental-vm-modules', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ status: 0, stdout: 'mjs ok', stderr: '' }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/hooks/post-write-reindex-hook.test.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(true);
      expect(report.evidence.testResult.stdout).toContain('mjs ok');
      const execCalls = (spawnSync as jest.Mock).mock.calls.filter(
        (call) => call[0] === process.execPath,
      );
      expect(execCalls.length).toBeGreaterThan(0);
    });

    it('runs .mjs and .ts tests in separate groups', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      let callCount = 0;
      (spawnSync as jest.Mock).mockImplementation(() => {
        callCount += 1;
        return {
          status: 0,
          stdout: JSON.stringify({
            status: 0,
            stdout: `group ${callCount}`,
            stderr: '',
          }),
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/hooks/post-write-reindex-hook.test.mjs,scripts/agent-customization/gates/shared-validation.gate.test.ts',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(true);
      const execCalls = (spawnSync as jest.Mock).mock.calls.filter(
        (call) => call[0] === process.execPath,
      );
      expect(execCalls).toHaveLength(2);
      expect(report.evidence.testResult.stdout).toContain('group 1');
      expect(report.evidence.testResult.stdout).toContain('group 2');
    });

    it('reports failure when any test group fails', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      let callCount = 0;
      (spawnSync as jest.Mock).mockImplementation(() => {
        callCount += 1;
        return {
          status: callCount === 1 ? 0 : 1,
          stdout: JSON.stringify({
            status: callCount === 1 ? 0 : 1,
            stdout: `group ${callCount}`,
            stderr: callCount === 1 ? '' : 'second group failed',
          }),
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/hooks/post-write-reindex-hook.test.mjs,scripts/agent-customization/gates/shared-validation.gate.test.ts',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(false);
      expect(report.evidence.testResult.status).toBe(1);
      expect(report.evidence.testResult.stderr).toContain(
        'second group failed',
      );
    });

    it('treats a null test group status as failure', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: JSON.stringify({ status: null, stdout: '', stderr: '' }),
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(false);
      expect(report.evidence.testResult.status).toBe(1);
    });

    it('reports a test runner error from parsed JSON', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string) => {
        if (cmd === process.execPath) {
          return {
            status: 0,
            stdout: JSON.stringify({
              stdout: '',
              stderr: '',
              error: 'spawn error',
            }),
            stderr: '',
            pid: 1,
            output: [],
            signal: null,
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.testResult.status).toBe(1);
    });

    it('falls back when the test runner returns non-JSON stdout', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string) => {
        if (cmd === process.execPath) {
          return {
            status: null,
            stdout: 'not json',
            stderr: 'bad',
            pid: 1,
            output: [],
            signal: null,
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.testResult.status).toBe(1);
      expect(report.evidence.testResult.stdout).toContain('not json');
      expect(report.evidence.testResult.stderr).toContain('bad');
    });

    it('falls back to empty strings when the test runner output is missing', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string) => {
        if (cmd === process.execPath) {
          return {
            status: null,
            stdout: undefined,
            stderr: undefined,
            pid: 1,
            output: [],
            signal: null,
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.testResult.status).toBe(1);
      expect(report.evidence.testResult.stdout).toContain(
        'scripts/agent-customization/gates/shared-validation.gate.test.ts',
      );
      expect(report.evidence.testResult.stderr).toContain(
        'scripts/agent-customization/gates/shared-validation.gate.test.ts',
      );
    });

    it('treats a build runner error as a failure', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string, args: string[]) => {
        if (cmd === process.execPath) {
          return successForTestRunner();
        }
        if (cmd === 'npm' && args[0] === 'run' && args[1] === 'build') {
          return {
            status: null,
            error: 'spawn error',
            signal: null,
            stdout: undefined,
            stderr: undefined,
            pid: 1,
            output: [],
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.buildResult.status).toBe(1);
    });

    it('treats a build runner signal as a failure', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string, args: string[]) => {
        if (cmd === process.execPath) {
          return successForTestRunner();
        }
        if (cmd === 'npm' && args[0] === 'run' && args[1] === 'build') {
          return {
            status: null,
            error: undefined,
            signal: 'SIGKILL',
            stdout: undefined,
            stderr: undefined,
            pid: 1,
            output: [],
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.buildResult.status).toBe(1);
    });

    it('treats a lint runner error as a failure', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string, args: string[]) => {
        if (cmd === process.execPath) {
          return successForTestRunner();
        }
        if (cmd === 'npm' && args[0] === 'run' && args[1] === 'lint') {
          return {
            status: null,
            error: 'spawn error',
            signal: null,
            stdout: undefined,
            stderr: undefined,
            pid: 1,
            output: [],
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.lintResult.status).toBe(1);
    });

    it('treats a lint runner signal as a failure', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string, args: string[]) => {
        if (cmd === process.execPath) {
          return successForTestRunner();
        }
        if (cmd === 'npm' && args[0] === 'run' && args[1] === 'lint') {
          return {
            status: null,
            error: undefined,
            signal: 'SIGTERM',
            stdout: undefined,
            stderr: undefined,
            pid: 1,
            output: [],
          };
        }
        return {
          status: 0,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.lintResult.status).toBe(1);
    });

    it('treats null status without error or signal as success', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      mockSpawn.mockImplementation((cmd: string) => {
        if (cmd === process.execPath) {
          return successForTestRunner();
        }
        return {
          status: null,
          stdout: 'ok',
          stderr: '',
          pid: 1,
          output: [],
          signal: null,
        };
      });
      await main([
        '--json',
        '--changed-files',
        'scripts/agent-customization/gates/shared-validation.gate.mjs',
      ]);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.pass).toBe(true);
    });
  });

  describe('argument parsing edge cases', () => {
    afterEach(() => {
      jest.clearAllMocks();
    });

    it('accepts --changed-files as a key-value pair', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: '',
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main(['--json', '--changed-files=src/foo.md']);
      const report = JSON.parse(
        logSpy.mock.calls[0][0] as string,
      ) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.changedFiles).toEqual(['src/foo.md']);
    });

    it('returns a standalone descriptor when no changed files are provided', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      const report = (await main(['--json'])) as ValidationReport;
      logSpy.mockRestore();
      expect(report.evidence.mode).toBe('standalone-descriptor');
    });

    it('creates the artifact directory when it does not exist', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: '',
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      const artifactPath = path.join(tempDir, 'missing', 'artifact.json');
      await main([
        '--json',
        '--changed-files',
        'src/foo.md',
        '--artifact-path',
        artifactPath,
      ]);
      logSpy.mockRestore();
      expect(existsSync(artifactPath)).toBe(true);
    });

    it('accepts --artifact-path as a key-value pair', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: '',
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      const artifactPath = path.join(tempDir, 'kv-artifact.json');
      await main([
        '--json',
        '--changed-files=src/foo.md',
        '--artifact-path=' + artifactPath,
      ]);
      logSpy.mockRestore();
      expect(existsSync(artifactPath)).toBe(true);
    });

    it('uses the default artifact path when --artifact-path is omitted', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      (spawnSync as jest.Mock).mockReturnValue({
        status: 0,
        stdout: '',
        stderr: '',
        pid: 1,
        output: [],
        signal: null,
      });
      await main(['--json', '--changed-files', 'src/foo.md']);
      logSpy.mockRestore();
      expect(
        existsSync(path.join(REPO_ROOT, 'coverage', 'shared-validation.json')),
      ).toBe(true);
    });

    it('uses default process.argv when main is called without arguments', async () => {
      const { main } = await loadGate();
      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await main();
      expect(logSpy).toHaveBeenCalledWith('PASS', 'shared-validation gate');
      logSpy.mockRestore();
    });
  });

  it('shows help and exits when --help is passed', async () => {
    const { main } = await loadGate();
    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {
      throw new Error('exit');
    });
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

    await expect(main(['--help'])).rejects.toThrow('exit');

    expect(exitSpy).toHaveBeenCalledWith(0);
    expect(logSpy).toHaveBeenCalledWith(
      expect.stringContaining('shared-validation gate'),
    );

    exitSpy.mockRestore();
    logSpy.mockRestore();
  });
});
