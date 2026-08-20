import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  parseArgs: jest.fn(),
  readWorkspaceFile: jest.fn(),
  issue: jest.fn((severity, path, message) => ({ severity, path, message })),
  summarizeIssues: jest.fn((name, issues) => ({
    name,
    ok: issues.length === 0,
    issues,
    counts: {
      errors: issues.filter((i) => i.severity === 'error').length,
      warnings: 0,
    },
    summaryText: `${issues.length === 0 ? 'PASS' : 'FAIL'} ${name}`,
  })),
  writeReport: jest.fn(),
  printUsage: jest.fn(),
}));

let mockUtils;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  jest.clearAllMocks();
});

async function importModule() {
  try {
    await import('./run-skill-trigger-evals.mjs');
  } catch {
    // process.exit may throw
  }
}

describe('run-skill-trigger-evals', () => {
  it('prints help and exits 0 when --help is passed', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--help'];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    mockUtils.parseArgs.mockReturnValue({ help: true, json: false });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
  });

  it('grades passing trigger evals', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'how do I test?',
            shouldTrigger: true,
            observedTriggered: true,
            observedTarget: 'skill-a',
            observedNotes: 'notes',
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.total, 1);
    assert.strictEqual(report.summary.passed, 1);
    assert.strictEqual(report.summary.failed, 0);
    assert.strictEqual(report.summary.pending, 0);
  });

  it('grades failing trigger evals', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedTriggered: false,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.failed, 1);
    const errorCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'error',
    );
    assert.ok(errorCalls.some((c) => c[2].includes('failed')));
  });

  it('marks pending when observedTriggered is missing', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.pending, 1);
  });

  it('escalates pending to error in strict mode', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'run-skill-trigger-evals.mjs',
      '--json',
      '--strict',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: true,
    });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'error' && c[2].includes('pending'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('warns for pending in non-strict mode', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: false });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const warningCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'warning' && c[2].includes('pending'),
    );
    assert.ok(warningCalls.length >= 1);
  });

  it('flags error for unsupported observed field', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedBad: 'bad',
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('unsupported observed field'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('flags error when observedTriggered is not boolean', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedTriggered: 'yes',
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('observedTriggered must be boolean'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('flags error when observedTarget is not string', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedTarget: 123,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('observedTarget must be a string'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('flags error when observedNotes is not string', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedNotes: 123,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('observedNotes must be a string'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('flags error when no evals in fixture', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({ evals: [] }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('no evals'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('handles array fixture format', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify([
        {
          id: 'e1',
          target: 'skill-a',
          query: 'q',
          shouldTrigger: true,
          observedTriggered: true,
        },
      ]),
    );

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.total, 1);
  });

  it('uses UNKNOWN for eval without id in error messages', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedBad: 'bad',
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('UNKNOWN'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('handles fixture without evals property', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(JSON.stringify({}));

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('no evals'),
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('uses UNKNOWN for eval without id in all observed type errors', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-trigger-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            target: 'skill-a',
            query: 'q',
            shouldTrigger: true,
            observedTriggered: 'yes',
            observedTarget: 123,
            observedNotes: 123,
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter((c) =>
      c[2].includes('UNKNOWN'),
    );
    assert.ok(
      errorCalls.some((c) =>
        c[2].includes('observedTriggered must be boolean'),
      ),
    );
    assert.ok(
      errorCalls.some((c) => c[2].includes('observedTarget must be a string')),
    );
    assert.ok(
      errorCalls.some((c) => c[2].includes('observedNotes must be a string')),
    );
  });
});
