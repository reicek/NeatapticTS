/**
 * @module convergence-tracker.gate.test
 * @description Coverage tests for convergence-tracker.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const REPO_ROOT = process.cwd();

let mockPlanText;
let mockReadError;

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => ({
    json: argv.includes('--json'),
    help: argv.includes('--help') || argv.includes('-h'),
  }),
  readWorkspaceFile: async () => {
    if (mockReadError) throw mockReadError;
    return mockPlanText;
  },
  repoRoot: REPO_ROOT,
}));

describe('convergence-tracker gate', () => {
  let originalExit;
  let originalExitCode;
  let originalLog;
  let logs;

  beforeEach(() => {
    jest.resetModules();
    mockPlanText = '';
    mockReadError = null;
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

  describe('runConvergenceTrackerGate', () => {
    it('passes with iterationCount=0 when no evidence section exists', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const result = runConvergenceTrackerGate({
        planText: 'no evidence here',
        sliceId: 'A1-impl',
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.iterationCount, 0);
      assert.equal(result.evidence.status, 'OK');
    });

    it('passes and resets when a green-pass marker is found', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        '- fix-loop: A1-impl iteration 3 status=passed',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.iterationCount, 0);
      assert.equal(result.evidence.resetReason, 'green-pass marker found');
    });

    it('passes when failed iteration count is within limit (<=4)', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        '- fix-loop: A1-impl iteration 1 status=failed',
        '- fix-loop: A1-impl iteration 2 status=failed',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.iterationCount, 2);
      assert.equal(result.evidence.status, 'OK');
    });

    it('fails and escalates when failed iteration count exceeds 4', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        '- fix-loop: A1-impl iteration 1 status=failed',
        '- fix-loop: A1-impl iteration 2 status=failed',
        '- fix-loop: A1-impl iteration 3 status=failed',
        '- fix-loop: A1-impl iteration 4 status=failed',
        '- fix-loop: A1-impl iteration 5 status=failed',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      assert.equal(result.pass, false);
      assert.equal(result.evidence.iterationCount, 5);
      assert.equal(result.evidence.status, 'ESCALATE');
      assert.ok(result.fixHint.includes('Escalate to 00-helping'));
    });

    it('matches case-insensitively (PASSED, Failed)', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        '- fix-loop: A1-impl iteration 2 status=PASSED',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.iterationCount, 0);
    });

    it('does not match markers for a different sliceId', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        '- fix-loop: B2-impl iteration 5 status=failed',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.iterationCount, 0);
    });

    it('stops at the next heading when extracting the evidence section', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        '- fix-loop: A1-impl iteration 1 status=failed',
        '### Sub heading',
        '- fix-loop: A1-impl iteration 9 status=failed',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      // second marker is after next heading so excluded
      assert.equal(result.evidence.iterationCount, 1);
    });

    it('treats markers without bullet prefix as valid', async () => {
      const { runConvergenceTrackerGate } =
        await import('./convergence-tracker.gate.mjs');
      const planText = [
        '## Latest validation evidence',
        'fix-loop: A1-impl iteration 1 status=failed',
      ].join('\n');
      const result = runConvergenceTrackerGate({
        planText,
        sliceId: 'A1-impl',
      });
      assert.equal(result.evidence.iterationCount, 1);
    });
  });

  describe('main', () => {
    it('prints usage and exits 0 on --help', async () => {
      process.exit = (code) => {
        throw new Error(`EXIT:${code}`);
      };
      const { main } = await import('./convergence-tracker.gate.mjs');
      await assert.rejects(() => main(['--help']), /EXIT:0/);
      assert.ok(logs.some((l) => l.includes('convergence-tracker gate')));
    });

    it('uses process.argv.slice(2) when called with no arguments', async () => {
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
    });

    it('returns standalone descriptor when no --plan is provided (--json)', async () => {
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main(['--json']);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.mode, 'standalone-descriptor');
      assert.equal(JSON.parse(logs[0]).pass, true);
      assert.equal(process.exitCode, 0);
    });

    it('returns standalone descriptor with text output when no --plan', async () => {
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main([]);
      assert.equal(result.pass, true);
      assert.ok(logs.some((l) => l.includes('PASS')));
    });

    it('returns cli-error when --plan given without --slice-id', async () => {
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main(['--plan=plans/foo.plans.md', '--json']);
      assert.equal(result.pass, false);
      assert.equal(result.evidence.mode, 'cli-error');
      assert.equal(process.exitCode, 1);
    });

    it('returns cli-error with text output and fixHint when --plan without --slice-id', async () => {
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main(['--plan=plans/foo.plans.md']);
      assert.equal(result.pass, false);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('returns read-error when readWorkspaceFile throws (Error)', async () => {
      mockReadError = new Error('boom');
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main([
        '--plan=plans/foo.plans.md',
        '--slice-id=A1',
        '--json',
      ]);
      assert.equal(result.pass, false);
      assert.equal(result.evidence.mode, 'read-error');
      assert.equal(result.evidence.error, 'boom');
      assert.equal(process.exitCode, 1);
    });

    it('returns read-error when readWorkspaceFile throws (non-Error)', async () => {
      mockReadError = 'string error';
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main(['--plan=plans/foo.plans.md', '--slice-id=A1']);
      assert.equal(result.pass, false);
      assert.equal(result.evidence.error, 'string error');
    });

    it('delegates to runConvergenceTrackerGate with valid plan text', async () => {
      mockPlanText = [
        '## Latest validation evidence',
        '- fix-loop: A1-impl iteration 1 status=failed',
      ].join('\n');
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main([
        '--plan=plans/foo.plans.md',
        '--slice-id=A1-impl',
        '--json',
      ]);
      assert.equal(result.pass, true);
      assert.equal(result.evidence.iterationCount, 1);
      assert.equal(process.exitCode, 0);
    });

    it('delegates and emits FAIL text when iterations exceeded', async () => {
      mockPlanText = [
        '## Latest validation evidence',
        ...Array.from(
          { length: 5 },
          (_, i) => `- fix-loop: A1-impl iteration ${i + 1} status=failed`,
        ),
      ].join('\n');
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main([
        '--plan=plans/foo.plans.md',
        '--slice-id=A1-impl',
      ]);
      assert.equal(result.pass, false);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.equal(process.exitCode, 1);
    });

    it('accepts --plan value (space-separated)', async () => {
      mockPlanText = 'no evidence';
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main([
        '--plan',
        'plans/foo.plans.md',
        '--slice-id',
        'A1',
      ]);
      assert.equal(result.pass, true);
    });

    it('accepts --slice-id value (space-separated)', async () => {
      mockPlanText = 'no evidence';
      const { main } = await import('./convergence-tracker.gate.mjs');
      const result = await main([
        '--plan=plans/foo.plans.md',
        '--slice-id',
        'A1',
      ]);
      assert.equal(result.pass, true);
    });
  });
});
