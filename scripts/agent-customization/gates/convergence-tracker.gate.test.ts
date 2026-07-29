/**
 * @module convergence-tracker.gate.test
 * @description Red tests for the convergence-tracker gate contract.
 *
 * The convergence tracker reads a plan's `VALIDATION_EVIDENCE` section, counts
 * fix-loop iteration markers for a target slice, and escalates to `00-helping`
 * after 4 iterations without a green pass.
 */
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';

const actualReadFile = jest.requireActual('node:fs/promises').readFile;
jest.mock('node:fs/promises', () => ({
  ...jest.requireActual('node:fs/promises'),
  readFile: jest.fn((...args: unknown[]) => actualReadFile(...args)),
}));

const REPO_ROOT = path.resolve();
const GATE_PATH = './convergence-tracker.gate.mjs';
const GATE_ABS_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/convergence-tracker.gate.mjs',
);

interface ConvergenceReport {
  pass: boolean;
  evidence: Record<string, unknown>;
  fixHint: string | null;
  owner: string;
}

/**
 * Lazily load the ESM gate module inside each test so Jest does not try to
 * transform its top-level await through the mjs-to-cjs transformer.
 */
async function loadGate() {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import(GATE_PATH);
}

/**
 * Build a plan body that contains the required `## Latest validation evidence`
 * section and an optional list of fix-loop iteration markers.
 */
function buildPlanText(markers: string[] = []): string {
  const markerLines = markers.length
    ? markers.map((marker) => `- ${marker}`).join('\n')
    : '- No fix-loop iterations recorded.';

  return `## Latest validation evidence\n\n${markerLines}\n`;
}

describe('convergence-tracker.gate.mjs', () => {
  afterEach(() => {
    process.exitCode = 0;
  });

  describe('API contract', () => {
    it('exports a runConvergenceTrackerGate function', async () => {
      const gate = await loadGate();

      expect(typeof gate.runConvergenceTrackerGate).toBe('function');
    });

    it('returns OK with zero iterations when no fix-loop evidence exists', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const result = await runConvergenceTrackerGate({
        planText: buildPlanText(),
        sliceId: 'A6-red-tests',
      });

      expect(result).toMatchObject({
        pass: true,
        evidence: {
          sliceId: 'A6-red-tests',
          iterationCount: 0,
          status: 'OK',
        },
        fixHint: null,
        owner: 'convergence-tracker',
      });
    });

    it('counts fix-loop iteration markers for the target slice', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const result = await runConvergenceTrackerGate({
        planText: buildPlanText([
          'fix-loop: A6-red-tests iteration 1 status=failed',
          'fix-loop: A6-red-tests iteration 2 status=failed',
        ]),
        sliceId: 'A6-red-tests',
      });

      expect(result.evidence.iterationCount).toBe(2);
      expect(result.evidence.status).toBe('OK');
    });

    it('ignores iteration markers belonging to other slices', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const result = await runConvergenceTrackerGate({
        planText: buildPlanText([
          'fix-loop: A6-red-tests iteration 1 status=failed',
          'fix-loop: A6-impl iteration 1 status=failed',
          'fix-loop: A6-green iteration 1 status=passed',
        ]),
        sliceId: 'A6-red-tests',
      });

      expect(result.evidence.iterationCount).toBe(1);
    });

    it('returns ESCALATE status when iterations exceed 4 without a green pass', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const markers: string[] = [];
      for (let iteration = 1; iteration <= 5; iteration += 1) {
        markers.push(
          `fix-loop: A6-red-tests iteration ${iteration} status=failed`,
        );
      }

      const result = await runConvergenceTrackerGate({
        planText: buildPlanText(markers),
        sliceId: 'A6-red-tests',
      });

      expect(result.pass).toBe(false);
      expect(result.evidence.iterationCount).toBe(5);
      expect(result.evidence.status).toBe('ESCALATE');
    });

    it('suggests escalation to 00-helping when the slice is non-convergent', async () => {
      const { runConvergenceTrackerGate } = await loadGate();

      const result = await runConvergenceTrackerGate({
        planText: buildPlanText([
          'fix-loop: A6-red-tests iteration 1 status=failed',
          'fix-loop: A6-red-tests iteration 2 status=failed',
          'fix-loop: A6-red-tests iteration 3 status=failed',
          'fix-loop: A6-red-tests iteration 4 status=failed',
          'fix-loop: A6-red-tests iteration 5 status=failed',
        ]),
        sliceId: 'A6-red-tests',
      });

      expect(result.fixHint).toMatch(/00-helping/);
    });

    it('resets iteration count when a green pass marker is present', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const result = await runConvergenceTrackerGate({
        planText: buildPlanText([
          'fix-loop: A6-red-tests iteration 1 status=failed',
          'fix-loop: A6-red-tests iteration 2 status=failed',
          'fix-loop: A6-red-tests iteration 3 status=failed',
          'fix-loop: A6-red-tests iteration 4 status=failed',
          'fix-loop: A6-red-tests iteration 5 status=passed',
        ]),
        sliceId: 'A6-red-tests',
      });

      expect(result.pass).toBe(true);
      expect(result.evidence.iterationCount).toBe(0);
      expect(result.evidence.status).toBe('OK');
    });
  });

  describe('CLI contract', () => {
    it('emits a valid JSON gate contract from a real subprocess invocation', () => {
      const result = spawnSync(process.execPath, [GATE_ABS_PATH, '--json'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });

      const report = JSON.parse(result.stdout ?? '{}') as ConvergenceReport;

      expect(report).toEqual(
        expect.objectContaining({
          pass: expect.any(Boolean),
          owner: 'convergence-tracker',
          evidence: expect.any(Object),
        }),
      );
    });

    it('emits a gate contract instead of throwing when the plan path is missing', () => {
      const result = spawnSync(
        process.execPath,
        [
          GATE_ABS_PATH,
          '--json',
          '--plan=plans/this-plan-does-not-exist.plans.md',
          '--slice-id=A6-impl',
        ],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );

      const report = JSON.parse(result.stdout ?? '{}') as ConvergenceReport;

      expect(report).toMatchObject({
        pass: false,
        owner: 'convergence-tracker',
        evidence: {
          mode: 'read-error',
          planPath: 'plans/this-plan-does-not-exist.plans.md',
        },
        fixHint: expect.stringContaining('Could not read plan file'),
      });
      expect(result.status).toBe(1);
    });

    it('emits a gate contract instead of throwing when slice-id is missing', () => {
      const result = spawnSync(
        process.execPath,
        [GATE_ABS_PATH, '--json', '--plan=plans/orchestration-fixes.plans.md'],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );

      const report = JSON.parse(result.stdout ?? '{}') as ConvergenceReport;

      expect(report).toMatchObject({
        pass: false,
        owner: 'convergence-tracker',
        evidence: {
          mode: 'cli-error',
          error: 'Missing required --slice-id when --plan is provided.',
        },
      });
      expect(result.status).toBe(1);
    });
  });

  describe('CLI contract via in-process main', () => {
    afterEach(() => {
      process.exitCode = 0;
    });

    async function runMain(argv: string[]) {
      const { main } = await loadGate();
      const logs: unknown[][] = [];
      const logSpy = jest
        .spyOn(console, 'log')
        .mockImplementation((...args: unknown[]) => {
          logs.push(args);
        });
      const originalExitCode = process.exitCode;
      process.exitCode = 0;

      try {
        const result = await main(argv);
        return { result, logs, exitCode: process.exitCode };
      } finally {
        logSpy.mockRestore();
        process.exitCode = originalExitCode;
      }
    }

    it('prints a standalone descriptor when invoked with --json and no plan', async () => {
      const { result, logs, exitCode } = await runMain(['--json']);

      expect(result.pass).toBe(true);
      expect(result.evidence.mode).toBe('standalone-descriptor');
      expect(exitCode).toBe(0);
      expect(JSON.parse(String(logs[0][0]))).toMatchObject({
        pass: true,
        owner: 'convergence-tracker',
      });
    });

    it('prints plain PASS output when invoked without --json', async () => {
      const { result, logs, exitCode } = await runMain([]);

      expect(result.pass).toBe(true);
      expect(exitCode).toBe(0);
      expect(logs[0]).toEqual(['PASS', 'convergence-tracker gate']);
    });

    it('returns a read-error gate contract for a missing plan path', async () => {
      const { result, exitCode } = await runMain([
        '--json',
        '--plan=plans/this-plan-does-not-exist.plans.md',
        '--slice-id=A6-impl',
      ]);

      expect(result.pass).toBe(false);
      expect(result.evidence.mode).toBe('read-error');
      expect(exitCode).toBe(1);
    });

    it('returns a read-error gate contract when the read failure is an Error instance', async () => {
      (readFile as jest.Mock).mockRejectedValueOnce(
        new Error('mock read error'),
      );

      const { result, exitCode } = await runMain([
        '--json',
        '--plan=plans/orchestration-fixes.plans.md',
        '--slice-id=A6-impl',
      ]);

      expect(result.pass).toBe(false);
      expect(result.evidence.mode).toBe('read-error');
      expect(result.evidence.error).toBe('mock read error');
      expect(exitCode).toBe(1);

      (readFile as jest.Mock).mockClear();
    });

    it('returns a read-error gate contract when the read failure is not an Error', async () => {
      (readFile as jest.Mock).mockRejectedValueOnce('string read failure');

      const { result, exitCode } = await runMain([
        '--json',
        '--plan=plans/orchestration-fixes.plans.md',
        '--slice-id=A6-impl',
      ]);

      expect(result.pass).toBe(false);
      expect(result.evidence.mode).toBe('read-error');
      expect(result.evidence.error).toBe('string read failure');
      expect(exitCode).toBe(1);

      (readFile as jest.Mock).mockClear();
    });

    it('prints plain FAIL output for a read error when --json is omitted', async () => {
      const { result, logs, exitCode } = await runMain([
        '--plan=plans/this-plan-does-not-exist.plans.md',
        '--slice-id=A6-impl',
      ]);

      expect(result.pass).toBe(false);
      expect(exitCode).toBe(1);
      expect(logs[0]).toEqual(['FAIL', 'convergence-tracker gate']);
      expect(logs[1]).toEqual([
        'fixHint:',
        expect.stringContaining('Could not read plan file'),
      ]);
    });

    it('returns a CLI error when --plan is provided without --slice-id', async () => {
      const { result, exitCode } = await runMain([
        '--json',
        '--plan=plans/orchestration-fixes.plans.md',
      ]);

      expect(result.pass).toBe(false);
      expect(result.evidence.mode).toBe('cli-error');
      expect(result.evidence.error).toBe(
        'Missing required --slice-id when --plan is provided.',
      );
      expect(exitCode).toBe(1);
    });

    it('reads an actual plan and reports iteration evidence', async () => {
      const { result } = await runMain([
        '--json',
        '--plan=plans/orchestration-fixes.plans.md',
        '--slice-id=A6-impl',
      ]);

      expect(result.owner).toBe('convergence-tracker');
      expect(result.evidence).toMatchObject({
        sliceId: 'A6-impl',
        status: expect.any(String),
      });
    });

    it('supports space-separated --plan and --slice-id flags', async () => {
      const { result } = await runMain([
        '--json',
        '--plan',
        'plans/orchestration-fixes.plans.md',
        '--slice-id',
        'A6-impl',
      ]);

      expect(result.evidence).toMatchObject({
        sliceId: 'A6-impl',
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
        expect.stringContaining('convergence-tracker gate'),
      );

      exitSpy.mockRestore();
      logSpy.mockRestore();
    });
  });

  describe('top-level main guard', () => {
    it('executes the CLI when imported as the main module', async () => {
      const originalArgv = process.argv;
      const originalExitCode = process.exitCode;
      process.argv = ['node', GATE_ABS_PATH];
      jest.resetModules();

      const logs: unknown[][] = [];
      const logSpy = jest
        .spyOn(console, 'log')
        .mockImplementation((...args: unknown[]) => {
          logs.push(args);
        });

      try {
        await import(GATE_PATH);
      } finally {
        logSpy.mockRestore();
        process.argv = originalArgv;
        process.exitCode = originalExitCode;
      }

      expect(logs[0]).toEqual(['PASS', 'convergence-tracker gate']);
    });

    it('loads the module when process.argv[1] is undefined', async () => {
      const originalArgv = process.argv;
      process.argv = ['node'];
      jest.resetModules();

      try {
        const gate = await import(GATE_PATH);
        expect(gate.runConvergenceTrackerGate).toBeInstanceOf(Function);
      } finally {
        process.argv = originalArgv;
      }
    });
  });

  describe('edge coverage', () => {
    it('returns OK with zero iterations when the evidence section is missing', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const result = runConvergenceTrackerGate({
        planText: 'No validation evidence here.',
        sliceId: 'A6-impl',
      });

      expect(result).toMatchObject({
        pass: true,
        evidence: {
          sliceId: 'A6-impl',
          iterationCount: 0,
          status: 'OK',
        },
      });
    });

    it('stops extracting the evidence section at the next heading', async () => {
      const { runConvergenceTrackerGate } = await loadGate();
      const result = runConvergenceTrackerGate({
        planText:
          '## Latest validation evidence\n\n- fix-loop: A6-impl iteration 1 status=failed\n\n## Another section\n- fix-loop: A6-impl iteration 99 status=failed',
        sliceId: 'A6-impl',
      });

      expect(result.evidence.iterationCount).toBe(1);
    });
  });
});
