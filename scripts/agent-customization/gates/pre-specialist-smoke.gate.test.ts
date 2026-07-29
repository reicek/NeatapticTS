/**
 * @module pre-specialist-smoke.gate.test
 * @description Green tests for the pre-specialist smoke gate contract.
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';

jest.mock('node:child_process', () => ({
  spawnSync: jest.fn(),
}));

interface SmokeReport {
  pass: boolean;
  evidence: Record<string, unknown>;
  fixHint: string | null;
  owner: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const GATE_PATH = path.resolve(__dirname, 'pre-specialist-smoke.gate.mjs');

/**
 * Lazily load the ESM gate module inside each test so Jest does not try to
 * transform its top-level await through the mjs-to-cjs transformer.
 */
async function loadGate() {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import('./pre-specialist-smoke.gate.mjs');
}

describe('pre-specialist-smoke.gate.mjs', () => {
  afterEach(() => {
    process.exitCode = 0;
    jest.clearAllMocks();
  });

  it('passes when the injected test runner succeeds', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();

    const result = await runPreSpecialistSmokeGate({
      changedFiles: ['src/foo.ts'],
      runner: async () => ({ status: 0, stdout: '', stderr: '' }),
    });

    expect(result).toEqual(
      expect.objectContaining({
        pass: true,
        owner: 'pre-specialist-smoke',
        fixHint: null,
      }),
    );
  });

  it('fails when the injected test runner reports failing tests', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();

    const result = await runPreSpecialistSmokeGate({
      changedFiles: ['src/foo.ts'],
      runner: async () => ({
        status: 1,
        stdout: '',
        stderr: '1 test failed',
      }),
    });

    expect(result).toEqual(
      expect.objectContaining({
        pass: false,
        owner: 'pre-specialist-smoke',
      }),
    );
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
      timeout: 600_000,
    });

    const report = JSON.parse(cliResult.stdout ?? '{}') as SmokeReport;

    expect(report).toEqual(
      expect.objectContaining({
        pass: expect.any(Boolean),
        owner: 'pre-specialist-smoke',
        evidence: expect.objectContaining({
          mode: 'standalone-descriptor',
        }),
      }),
    );
  });

  it('maps .mjs sources to both .test.mjs and .test.ts candidates', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();
    const collected: string[] = [];

    await runPreSpecialistSmokeGate({
      changedFiles: ['scripts/agent-customization/gates/foo.mjs'],
      runner: async ({ testFiles }: { testFiles: string[] }) => {
        collected.push(...testFiles);
        return { status: 0, stdout: '', stderr: '' };
      },
    });

    expect(collected).toEqual(
      expect.arrayContaining([
        'scripts/agent-customization/gates/foo.test.mjs',
        'scripts/agent-customization/gates/foo.test.ts',
      ]),
    );
  });

  it('returns a standalone descriptor when no changed files are supplied', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();

    const result = await runPreSpecialistSmokeGate();

    expect(result.pass).toBe(true);
    expect(result.evidence.mode).toBe('standalone-descriptor');
  });

  it('passes when changed files have no testable source extension', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();

    const result = await runPreSpecialistSmokeGate({
      changedFiles: ['README.md', 'src/foo.png'],
    });

    expect(result.pass).toBe(true);
    expect(result.evidence.testFiles).toEqual([]);
    expect(result.fixHint).toBeNull();
  });

  it('keeps changed test files as-is', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();
    const collected: string[] = [];

    await runPreSpecialistSmokeGate({
      changedFiles: ['src/foo.test.ts'],
      runner: async ({ testFiles }: { testFiles: string[] }) => {
        collected.push(...testFiles);
        return { status: 0, stdout: '', stderr: '' };
      },
    });

    expect(collected).toEqual(['src/foo.test.ts']);
  });

  it('maps .cjs sources to .test.cjs and .test.ts candidates', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();
    const collected: string[] = [];

    await runPreSpecialistSmokeGate({
      changedFiles: ['src/foo.cjs'],
      runner: async ({ testFiles }: { testFiles: string[] }) => {
        collected.push(...testFiles);
        return { status: 0, stdout: '', stderr: '' };
      },
    });

    expect(collected).toEqual(
      expect.arrayContaining(['src/foo.test.cjs', 'src/foo.test.ts']),
    );
  });

  it('runs the default Jest runner and emits JSON from main', async () => {
    const { main } = await loadGate();
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    (spawnSync as jest.Mock).mockReturnValue({
      status: 0,
      stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
      stderr: '',
      pid: 1,
      output: [],
      signal: null,
    } as unknown as ReturnType<typeof spawnSync>);

    const result = await main(['--json', '--changed-files', 'src/foo.ts']);

    expect(result.pass).toBe(true);
    expect(logSpy).toHaveBeenCalledTimes(1);
    const logged = JSON.parse(logSpy.mock.calls[0][0] as string);
    expect(logged.pass).toBe(true);
    expect(logged.owner).toBe('pre-specialist-smoke');
    expect(spawnSync).toHaveBeenCalled();

    logSpy.mockRestore();
    (spawnSync as jest.Mock).mockClear();
  });

  it('emits plain PASS output from main when not given --json', async () => {
    const { main } = await loadGate();
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    (spawnSync as jest.Mock).mockReturnValue({
      status: 0,
      stdout: JSON.stringify({ status: 0, stdout: 'ok', stderr: '' }),
      stderr: '',
      pid: 1,
      output: [],
      signal: null,
    } as unknown as ReturnType<typeof spawnSync>);

    await main(['--changed-files', 'src/foo.ts']);

    expect(logSpy).toHaveBeenCalledWith('PASS', 'pre-specialist-smoke gate');

    logSpy.mockRestore();
    (spawnSync as jest.Mock).mockClear();
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
      expect.stringContaining('pre-specialist-smoke gate'),
    );

    exitSpy.mockRestore();
    logSpy.mockRestore();
  });

  it('falls back to raw spawn result when the default runner returns invalid JSON', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();
    (spawnSync as jest.Mock).mockReturnValue({
      status: 2,
      stdout: 'not json',
      stderr: 'spawn error',
      pid: 1,
      output: [],
      signal: null,
    } as unknown as ReturnType<typeof spawnSync>);

    const result = await runPreSpecialistSmokeGate({
      changedFiles: ['src/foo.ts'],
    });

    expect(result.pass).toBe(false);
    expect(result.evidence.exitStatus).toBe(2);
    expect(result.evidence.stderr).toBe('spawn error');
  });

  it('falls back to default values when the default runner result is incomplete', async () => {
    const { runPreSpecialistSmokeGate } = await loadGate();
    (spawnSync as jest.Mock).mockReturnValue(
      {} as unknown as ReturnType<typeof spawnSync>,
    );

    const result = await runPreSpecialistSmokeGate({
      changedFiles: ['src/foo.ts'],
    });

    expect(result.pass).toBe(false);
    expect(result.evidence.exitStatus).toBe(1);
    expect(result.evidence.stdout).toBe('');
    expect(result.evidence.stderr).toBe('');
  });

  it('parses --changed-files in equals form and newline separation', async () => {
    const { main } = await loadGate();
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    (spawnSync as jest.Mock).mockReturnValue({
      status: 0,
      stdout: JSON.stringify({ status: 0, stdout: '', stderr: '' }),
      stderr: '',
      pid: 1,
      output: [],
      signal: null,
    } as unknown as ReturnType<typeof spawnSync>);

    await main(['--json', '--changed-files=src/a.ts,src/b.ts\nsrc/c.ts']);

    const logged = JSON.parse(logSpy.mock.calls[0][0] as string);
    expect(logged.evidence.changedFiles).toEqual([
      'src/a.ts',
      'src/b.ts',
      'src/c.ts',
    ]);

    logSpy.mockRestore();
    (spawnSync as jest.Mock).mockClear();
  });

  it('parses an empty changed-files list from main', async () => {
    const { main } = await loadGate();
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});

    const result = await main(['--json']);

    expect(result.pass).toBe(true);
    expect(result.evidence.mode).toBe('standalone-descriptor');
    expect(logSpy).toHaveBeenCalledTimes(1);

    logSpy.mockRestore();
  });

  it('uses process.argv when no argv is passed and prints FAIL', async () => {
    const { main } = await loadGate();
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    (spawnSync as jest.Mock).mockReturnValue({
      status: 1,
      stdout: '',
      stderr: 'failing test',
      pid: 1,
      output: [],
      signal: null,
    } as unknown as ReturnType<typeof spawnSync>);

    const originalArgv = process.argv;
    process.argv = ['node', 'gate', '--changed-files', 'src/foo.ts'];

    try {
      const result = await main();

      expect(result.pass).toBe(false);
      expect(logSpy).toHaveBeenCalledWith('FAIL', 'pre-specialist-smoke gate');
      expect(logSpy).toHaveBeenCalledWith('fixHint:', expect.any(String));
    } finally {
      process.argv = originalArgv;
      logSpy.mockRestore();
      (spawnSync as jest.Mock).mockClear();
    }
  });
});
