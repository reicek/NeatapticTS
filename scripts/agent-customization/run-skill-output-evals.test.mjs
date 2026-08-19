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
    await import('./run-skill-output-evals.mjs');
  } catch {
    // process.exit may throw
  }
}

describe('run-skill-output-evals', () => {
  it('prints help and exits 0 when --help is passed', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--help'];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    mockUtils.parseArgs.mockReturnValue({ help: true, json: false });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
  });

  it('grades passing evals and writes report', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            prompt: 'do thing',
            assertions: [
              { text: 'assert1', passed: true, evidence: 'ev1' },
              { text: 'assert2', passed: false, evidence: 'ev2' },
              { text: 'assert3', evidence: 'ev3' },
            ],
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    assert.ok(mockUtils.writeReport.mock.calls.length >= 1);
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.evals, 1);
    assert.strictEqual(report.summary.assertions, 3);
    assert.strictEqual(report.summary.passed, 1);
    assert.strictEqual(report.summary.failed, 1);
    assert.strictEqual(report.summary.pending, 1);
  });

  it('handles array fixture format', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify([
        {
          id: 'e1',
          target: 'skill-a',
          prompt: 'p',
          assertions: [{ text: 'a', passed: true }],
        },
      ]),
    );

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.evals, 1);
  });

  it('flags error when no evals in fixture', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({ evals: [] }),
    );

    await importModule();

    process.argv = origArgv;
    assert.ok(mockUtils.issue.mock.calls.length >= 1);
    const issueCall = mockUtils.issue.mock.calls[0];
    assert.strictEqual(issueCall[0], 'error');
    assert.ok(issueCall[2].includes('no evals'));
  });

  it('flags error for failed assertions', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            prompt: 'p',
            assertions: [{ text: 'a', passed: false }],
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'error',
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('flags warning for pending assertions', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            prompt: 'p',
            assertions: [{ text: 'a', evidence: 'e' }],
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const warningCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'warning',
    );
    assert.ok(warningCalls.length >= 1);
  });

  it('uses default input path when --input not provided', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({ evals: [] }),
    );

    await importModule();

    process.argv = origArgv;
    const readPath = mockUtils.readWorkspaceFile.mock.calls[0][0];
    assert.ok(readPath.includes('skill-output-evals.json'));
  });

  it('uses --input path when provided', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'run-skill-output-evals.mjs',
      '--json',
      '--input=custom.json',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      input: 'custom.json',
    });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({ evals: [] }),
    );

    await importModule();

    process.argv = origArgv;
    assert.strictEqual(
      mockUtils.readWorkspaceFile.mock.calls[0][0],
      'custom.json',
    );
  });

  it('sets exitCode to 1 when errors exist', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({ evals: [] }),
    );

    await importModule();

    process.argv = origArgv;
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('sets exitCode to 0 when no errors', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            prompt: 'p',
            assertions: [{ text: 'a', passed: true }],
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    assert.strictEqual(process.exitCode, 0);
    process.exitCode = origExitCode;
  });

  it('handles fixture without evals property', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(JSON.stringify({}));

    await importModule();

    process.argv = origArgv;
    assert.ok(mockUtils.issue.mock.calls.length >= 1);
    const issueCall = mockUtils.issue.mock.calls[0];
    assert.strictEqual(issueCall[0], 'error');
    assert.ok(issueCall[2].includes('no evals'));
  });

  it('handles eval case without assertions property', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'run-skill-output-evals.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      JSON.stringify({
        evals: [
          {
            id: 'e1',
            target: 'skill-a',
            prompt: 'p',
          },
        ],
      }),
    );

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.assertions, 0);
  });
});
