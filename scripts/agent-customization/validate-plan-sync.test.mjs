import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  extractDownstreamTrackers: jest.fn(),
  extractStatus: jest.fn(),
  issue: jest.fn((severity, path, message) => ({ severity, path, message })),
  parseArgs: jest.fn(),
  printUsage: jest.fn(),
  readWorkspaceFile: jest.fn(),
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
}));

let mockUtils;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  jest.clearAllMocks();
});

async function importModule() {
  try {
    await import('./validate-plan-sync.mjs');
  } catch {
    // process.exit may throw
  }
}

describe('validate-plan-sync', () => {
  it('prints help and exits 0 when --help is passed', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--help',
      '--plan=plans/test.plans.md',
    ];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    mockUtils.parseArgs.mockReturnValue({ help: true, json: false });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
  });

  it('exits 1 when --plan is not supplied', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    const origError = console.error;
    process.argv = ['node', 'validate-plan-sync.mjs', '--json'];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    console.error = () => {};
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    console.error = origError;
    // Should have called process.exit(1)
  });

  it('validates plan with matching references and no issues', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans/test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans/test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md')
        return 'Status: [WIP]\nSome plan content with test.plans.md reference.';
      if (p === 'plans/README.md')
        return 'Reference to test.plans.md and [WIP] and agent architecture, custom agents';
      if (p === 'plans/Roadmap.md')
        return 'Reference to test.plans.md and [WIP] and Standalone Meta-Workflow Lane';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue('WIP');
    mockUtils.extractDownstreamTrackers.mockResolvedValue([
      'plans/tracker.plans.md',
    ]);

    await importModule();

    process.argv = origArgv;
    process.exitCode = origExitCode;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.strictEqual(report.plan.path, 'plans/test.plans.md');
    assert.strictEqual(report.plan.status, 'WIP');
    assert.deepStrictEqual(report.downstreamTrackers, [
      'plans/tracker.plans.md',
    ]);
  });

  it('flags error when plan status is missing', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans/test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans/test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md') return 'content with test.plans.md';
      if (p === 'plans/README.md')
        return 'test.plans.md agent architecture, custom agents';
      if (p === 'plans/Roadmap.md')
        return 'test.plans.md Standalone Meta-Workflow Lane';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue(null);
    mockUtils.extractDownstreamTrackers.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    process.exitCode = origExitCode;
    const statusIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('missing a top-level status'),
    );
    assert.ok(statusIssue);
  });

  it('flags error when README missing plan reference', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans/test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans/test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md') return 'test.plans.md [WIP]';
      if (p === 'plans/README.md')
        return 'no ref here but agent architecture, custom agents';
      if (p === 'plans/Roadmap.md')
        return 'test.plans.md [WIP] Standalone Meta-Workflow Lane';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue('WIP');
    mockUtils.extractDownstreamTrackers.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    const missingRef = mockUtils.issue.mock.calls.find(
      (c) => c[1] === 'plans/README.md' && c[2].includes('Missing reference'),
    );
    assert.ok(missingRef);
  });

  it('flags error when Roadmap missing status marker', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans/test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans/test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md') return 'test.plans.md [WIP]';
      if (p === 'plans/README.md')
        return 'test.plans.md [WIP] agent architecture, custom agents';
      if (p === 'plans/Roadmap.md')
        return 'test.plans.md but no status here Standalone Meta-Workflow Lane';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue('WIP');
    mockUtils.extractDownstreamTrackers.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    const missingStatus = mockUtils.issue.mock.calls.find(
      (c) => c[1] === 'plans/Roadmap.md' && c[2].includes('Missing status'),
    );
    assert.ok(missingStatus);
  });

  it('flags warning when README missing trigger phrase', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans/test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans/test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md') return 'test.plans.md [WIP]';
      if (p === 'plans/README.md')
        return 'test.plans.md [WIP] no trigger phrase';
      if (p === 'plans/Roadmap.md')
        return 'test.plans.md [WIP] Standalone Meta-Workflow Lane';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue('WIP');
    mockUtils.extractDownstreamTrackers.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    const triggerIssue = mockUtils.issue.mock.calls.find(
      (c) => c[0] === 'warning' && c[2].includes('trigger phrase'),
    );
    assert.ok(triggerIssue);
  });

  it('flags error when Roadmap missing meta-workflow lane', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans/test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans/test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md') return 'test.plans.md [WIP]';
      if (p === 'plans/README.md')
        return 'test.plans.md [WIP] agent architecture, custom agents';
      if (p === 'plans/Roadmap.md') return 'test.plans.md [WIP] no lane here';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue('WIP');
    mockUtils.extractDownstreamTrackers.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    const laneIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('meta-workflow lane'),
    );
    assert.ok(laneIssue);
  });

  it('normalizes backslash plan paths', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-plan-sync.mjs',
      '--json',
      '--plan=plans\\test.plans.md',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      plan: 'plans\\test.plans.md',
    });
    mockUtils.readWorkspaceFile.mockImplementation(async (p) => {
      if (p === 'plans/test.plans.md') return 'test.plans.md [WIP]';
      if (p === 'plans/README.md')
        return 'test.plans.md [WIP] agent architecture, custom agents';
      if (p === 'plans/Roadmap.md')
        return 'test.plans.md [WIP] Standalone Meta-Workflow Lane';
      return '';
    });
    mockUtils.extractStatus.mockReturnValue('WIP');
    mockUtils.extractDownstreamTrackers.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.plan.path, 'plans/test.plans.md');
  });
});
