import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  parseArgs: jest.fn(),
  repoRoot: 'C:\\NeatapticTS',
  issue: jest.fn(),
  writeReport: jest.fn(),
  listMarkdownFiles: jest.fn(),
  readWorkspaceFile: jest.fn(),
  parseFrontmatter: jest.fn(),
  printUsage: jest.fn(),
  summarizeIssues: jest.fn(),
  extractMarkdownLinks: jest.fn(),
  fileExists: jest.fn(),
}));

jest.unstable_mockModule('node:fs/promises', () => ({
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  readdir: jest.fn(),
  readFile: jest.fn(),
  writeFile: jest.fn(),
  mkdir: jest.fn(),
  stat: jest.fn(),
  appendFile: jest.fn(),
  rm: jest.fn(),
  glob: jest.fn(),
}));

jest.unstable_mockModule('@libsql/client', () => ({
  createClient: jest.fn(),
}));

let mockUtils;
let mockFs;
let mockLibsql;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  mockFs = await import('node:fs/promises');
  mockLibsql = await import('@libsql/client');
  jest.clearAllMocks();
});

function captureConsole() {
  const logs = [];
  const origLog = console.log;
  console.log = (...args) => logs.push(args.join(' '));
  return {
    logs,
    restore() {
      console.log = origLog;
    },
  };
}

async function importModule(argv) {
  const origArgv = process.argv;
  const origExit = process.exit;
  process.argv = ['node', 'workflow-gap-audit.mjs', ...(argv || [])];
  process.exit = (code) => {
    throw new Error(`EXIT:${code}`);
  };
  try {
    await import('./workflow-gap-audit.mjs');
  } catch {
    // process.exit may throw
  }
  process.argv = origArgv;
  process.exit = origExit;
}

describe('workflow-gap-audit', () => {
  it('prints help and exits 0 when --help', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: true });
    const cap = captureConsole();

    await importModule(['--help']);

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('workflow-gap-audit')));
  });

  it('outputs JSON report with empty data', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.gateFailureFrequency.length, 0);
    assert.strictEqual(report.escalationCount, 0);
    assert.strictEqual(report.topFailingGate, null);
    assert.ok(report.recommendedActions.length > 0);
  });

  it('outputs human-readable report', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: false });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule([]);

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('Workflow Gap Audit')));
  });

  it('aggregates gate failures from learning log', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'cortex-index',
        agent: 'test-agent',
        sessionId: 'real-1',
      }),
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'cortex-index',
        agent: 'other-agent',
        sessionId: 'real-2',
      }),
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'agent-graph',
        agent: 'test-agent',
        sessionId: 'real-3',
      }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(report.gateFailureFrequency.length > 0);
    assert.strictEqual(report.gateFailureFrequency[0].gateId, 'cortex-index');
    assert.strictEqual(report.gateFailureFrequency[0].failureCount, 2);
    assert.strictEqual(report.topFailingGate, 'cortex-index');
  });

  it('filters test-session artifacts', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'test-gate',
        agent: 'a',
        sessionId: 'test-session-001',
      }),
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'real-gate',
        agent: 'b',
        sessionId: 'real-1',
        exceptionEvidence: { reason: 'real' },
      }),
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'test-gate2',
        agent: 'c',
        sessionId: 'real-2',
        exceptionEvidence: { reason: 'test-red-phase' },
      }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.gateFailureFrequency.length, 1);
    assert.strictEqual(report.gateFailureFrequency[0].gateId, 'real-gate');
  });

  it('counts escalations', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({ eventType: 'gate-escalation', sessionId: 's1' }),
      JSON.stringify({ category: 'gate-escalation', sessionId: 's2' }),
      JSON.stringify({ eventType: 'other', sessionId: 's3' }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.escalationCount, 2);
  });

  it('aggregates runtime enforcement evidence', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'runtime-action-prepass',
        actionId: 'a1',
        sessionId: 's1',
      }),
      JSON.stringify({
        eventType: 'runtime-action-prepass',
        actionId: 'a2',
        sessionId: 's1',
      }),
      JSON.stringify({
        eventType: 'runtime-action-postpass',
        actionId: 'a1',
        sessionId: 's1',
      }),
      JSON.stringify({ eventType: 'runtime-proof-mismatch', sessionId: 's1' }),
      JSON.stringify({ eventType: 'runtime-action-blocked', sessionId: 's1' }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.runtimeEnforcementEvidence.preActionPasses, 2);
    assert.strictEqual(report.runtimeEnforcementEvidence.postActionPasses, 1);
    assert.strictEqual(report.runtimeEnforcementEvidence.proofMismatches, 1);
    assert.strictEqual(report.runtimeEnforcementEvidence.blockedActions, 2);
    assert.deepStrictEqual(
      report.runtimeEnforcementEvidence.missingPostActionPairs,
      ['a2'],
    );
    assert.deepStrictEqual(
      report.runtimeEnforcementEvidence.postWithoutPrePairs,
      [],
    );
  });

  it('handles post-action without pre-action', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'runtime-action-postpass',
        actionId: 'x1',
        sessionId: 's1',
      }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.deepStrictEqual(
      report.runtimeEnforcementEvidence.postWithoutPrePairs,
      ['x1'],
    );
  });

  it('loads flow IDs from flows directory', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockResolvedValue([
      '04.scoped-fix.flow.yml',
      '00.workflow-gap-audit.flow.yml',
      'flow.schema.yml',
      'not-a-flow.txt',
    ]);
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(report.underusedFlows.length > 0);
    const flowIds = report.underusedFlows.map((f) => f.flowId);
    assert.ok(flowIds.includes('04.scoped-fix'));
    assert.ok(!flowIds.includes('flow.schema'));
  });

  it('computes underused flows with mentions', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockResolvedValue('');
    mockFs.readdir.mockResolvedValue(['04.test.flow.yml']);
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.underusedFlows[0].mentionCount, 0);
  });

  it('handles custom --window', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json', '--window=14']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.windowDays, 14);
  });

  it('handles invalid --window falling back to 7', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json', '--window=invalid']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.windowDays, 7);
  });

  it('handles --db path that does not exist', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.access.mockRejectedValue(new Error('ENOENT'));
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json', '--db=nonexistent.db']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.deepStrictEqual(report.agentSessionCounts, []);
  });

  it('handles malformed JSONL lines gracefully', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockResolvedValue(
      'not json\n{"eventType":"gate-exception","gateId":"g1","agent":"a","sessionId":"s1"}\n',
    );
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.gateFailureFrequency.length, 1);
  });

  it('builds recommended actions for gate failures', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'gate-exception',
        gateId: 'my-gate',
        agent: 'agent-a',
        sessionId: 's1',
      }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(
      report.recommendedActions.some((a) =>
        a.includes("Gate 'my-gate' failed"),
      ),
    );
  });

  it('builds recommended actions for escalations', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({ eventType: 'gate-escalation', sessionId: 's1' }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(report.recommendedActions.some((a) => a.includes('escalation')));
  });

  it('builds recommended actions for proof mismatches', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({ eventType: 'runtime-proof-mismatch', sessionId: 's1' }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(
      report.recommendedActions.some((a) => a.includes('proof mismatch')),
    );
  });

  it('builds recommended actions for missing post-action pairs', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'runtime-action-prepass',
        actionId: 'orphan',
        sessionId: 's1',
      }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(
      report.recommendedActions.some((a) =>
        a.includes('pre-action pass without'),
      ),
    );
  });

  it('builds recommended actions for post-without-pre pairs', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({
        eventType: 'runtime-action-postpass',
        actionId: 'lonely',
        sessionId: 's1',
      }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(
      report.recommendedActions.some((a) =>
        a.includes('post-action pass without'),
      ),
    );
  });

  it('builds recommended actions for underused flows', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockResolvedValue([
      '01.unused.flow.yml',
      '02.also-unused.flow.yml',
    ]);
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(
      report.recommendedActions.some((a) => a.includes('zero mention')),
    );
  });

  it('outputs good-health action when no issues', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockRejectedValue(new Error('ENOENT'));
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.ok(
      report.recommendedActions.some((a) => a.includes('gate health is good')),
    );
  });

  it('handles -h as help alias', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: true });
    const cap = captureConsole();

    await importModule(['-h']);

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('workflow-gap-audit')));
  });

  it('uses custom learning log path from env', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.readFile.mockResolvedValue('');
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const origEnv = process.env.NEATAPTIC_LEARNING_LOG_PATH;
    process.env.NEATAPTIC_LEARNING_LOG_PATH = 'C:\\custom\\log.jsonl';
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    process.env.NEATAPTIC_LEARNING_LOG_PATH = origEnv;
    assert.ok(mockFs.readFile.mock.calls.some((c) => c[0].includes('custom')));
  });

  it('uses unknown for missing gateId and agent', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const logLines = [
      JSON.stringify({ eventType: 'gate-exception', sessionId: 's1' }),
    ].join('\n');
    mockFs.readFile.mockResolvedValue(logLines);
    mockFs.readdir.mockRejectedValue(new Error('ENOENT'));
    const cap = captureConsole();

    await importModule(['--json']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.gateFailureFrequency[0].gateId, 'unknown');
  });

  describe('session store integration (success path)', () => {
    let cap;

    beforeEach(async () => {
      mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
      mockFs.access.mockResolvedValue(undefined);
      mockFs.readFile.mockResolvedValue('');
      mockFs.readdir.mockResolvedValue(['04.scoped-fix.flow.yml']);
      const mockDb = {
        execute: jest
          .fn()
          .mockResolvedValueOnce({
            rows: [{ agentName: 'agent-a', sessionCount: 3 }],
          })
          .mockResolvedValueOnce({
            rows: [
              {
                sessionId: 's1',
                agentName: 'agent-a',
                summary: 'no flow id here',
              },
            ],
          })
          .mockResolvedValueOnce({
            rows: [{ summary: 'worked on 04.scoped-fix' }],
          }),
        close: jest.fn().mockResolvedValue(undefined),
      };
      mockLibsql.createClient.mockReturnValue(mockDb);
      cap = captureConsole();
      await importModule(['--json', '--db=test.db']);
    });

    afterEach(() => {
      cap.restore();
    });

    it('loads agent session counts from session store', () => {
      const report = JSON.parse(cap.logs[0]);
      assert.strictEqual(report.agentSessionCounts.length, 1);
    });

    it('loads drift sessions from session store', () => {
      const report = JSON.parse(cap.logs[0]);
      assert.strictEqual(report.agentDriftSessions.length, 1);
    });

    it('computes underused flows with mention count from session summaries', () => {
      const report = JSON.parse(cap.logs[0]);
      assert.strictEqual(report.underusedFlows[0].mentionCount, 1);
    });

    it('builds recommended action for drift sessions', () => {
      const report = JSON.parse(cap.logs[0]);
      assert.ok(report.recommendedActions.some((a) => a.includes('flow ID')));
    });
  });

  it('returns empty when @libsql/client createClient throws', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.access.mockResolvedValue(undefined);
    mockFs.readFile.mockResolvedValue('');
    mockFs.readdir.mockResolvedValue([]);
    mockLibsql.createClient.mockImplementation(() => {
      throw new Error('createClient failed');
    });
    const cap = captureConsole();

    await importModule(['--json', '--db=test.db']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.deepStrictEqual(report.agentSessionCounts, []);
  });

  it('returns empty when db.execute throws', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.access.mockResolvedValue(undefined);
    mockFs.readFile.mockResolvedValue('');
    mockFs.readdir.mockResolvedValue([]);
    const mockDb = {
      execute: jest.fn().mockRejectedValue(new Error('query failed')),
      close: jest.fn().mockResolvedValue(undefined),
    };
    mockLibsql.createClient.mockReturnValue(mockDb);
    const cap = captureConsole();

    await importModule(['--json', '--db=test.db']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.deepStrictEqual(report.agentDriftSessions, []);
  });

  it('handles null summary and agentName in session store rows', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockFs.access.mockResolvedValue(undefined);
    mockFs.readFile.mockResolvedValue('');
    mockFs.readdir.mockResolvedValue([]);
    const mockDb = {
      execute: jest
        .fn()
        .mockResolvedValueOnce({
          rows: [{ agentName: null, sessionCount: 1 }],
        })
        .mockResolvedValueOnce({
          rows: [{ sessionId: 's1', agentName: null, summary: null }],
        })
        .mockResolvedValueOnce({
          rows: [{ summary: null }],
        }),
      close: jest.fn().mockResolvedValue(undefined),
    };
    mockLibsql.createClient.mockReturnValue(mockDb);
    const cap = captureConsole();

    await importModule(['--json', '--db=test.db']);

    cap.restore();
    const report = JSON.parse(cap.logs[0]);
    assert.strictEqual(report.agentDriftSessions[0].agentName, 'unknown');
  });
});
