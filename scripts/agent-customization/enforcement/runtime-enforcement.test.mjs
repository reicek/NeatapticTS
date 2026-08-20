import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: jest.fn(),
  writeFile: jest.fn(),
  mkdir: jest.fn(),
  appendFile: jest.fn(),
  readdir: jest.fn(),
  stat: jest.fn(),
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  rm: jest.fn(),
  glob: jest.fn(),
}));

let mod;

beforeAll(async () => {
  mod = await import('./runtime-enforcement.mjs');
});

beforeEach(() => {
  jest.clearAllMocks();
});

describe('runtime-enforcement', () => {
  describe('resolveSessionId', () => {
    it('uses session_id from input', () => {
      assert.strictEqual(
        mod.resolveSessionId({ session_id: 'abc-123' }),
        'abc-123',
      );
    });

    it('uses sessionId from input', () => {
      assert.strictEqual(
        mod.resolveSessionId({ sessionId: 'xyz-456' }),
        'xyz-456',
      );
    });

    it('uses env COPILOT_SESSION_ID', () => {
      const orig = process.env.COPILOT_SESSION_ID;
      process.env.COPILOT_SESSION_ID = 'env-session-1';
      try {
        assert.strictEqual(mod.resolveSessionId({}), 'env-session-1');
      } finally {
        if (orig === undefined) delete process.env.COPILOT_SESSION_ID;
        else process.env.COPILOT_SESSION_ID = orig;
      }
    });

    it('uses env GITHUB_COPILOT_SESSION_ID', () => {
      const orig = process.env.GITHUB_COPILOT_SESSION_ID;
      process.env.GITHUB_COPILOT_SESSION_ID = 'gh-session-1';
      try {
        assert.strictEqual(mod.resolveSessionId({}), 'gh-session-1');
      } finally {
        if (orig === undefined) delete process.env.GITHUB_COPILOT_SESSION_ID;
        else process.env.GITHUB_COPILOT_SESSION_ID = orig;
      }
    });

    it('uses env COPILOT_CLI_SESSION_ID', () => {
      const orig1 = process.env.COPILOT_SESSION_ID;
      const orig2 = process.env.GITHUB_COPILOT_SESSION_ID;
      const orig3 = process.env.COPILOT_CLI_SESSION_ID;
      delete process.env.COPILOT_SESSION_ID;
      delete process.env.GITHUB_COPILOT_SESSION_ID;
      process.env.COPILOT_CLI_SESSION_ID = 'cli-session-1';
      try {
        assert.strictEqual(mod.resolveSessionId({}), 'cli-session-1');
      } finally {
        if (orig1 !== undefined) process.env.COPILOT_SESSION_ID = orig1;
        if (orig2 !== undefined) process.env.GITHUB_COPILOT_SESSION_ID = orig2;
        if (orig3 === undefined) delete process.env.COPILOT_CLI_SESSION_ID;
        else process.env.COPILOT_CLI_SESSION_ID = orig3;
      }
    });

    it('falls back to cli-hook-session', () => {
      const orig1 = process.env.COPILOT_SESSION_ID;
      const orig2 = process.env.GITHUB_COPILOT_SESSION_ID;
      const orig3 = process.env.COPILOT_CLI_SESSION_ID;
      delete process.env.COPILOT_SESSION_ID;
      delete process.env.GITHUB_COPILOT_SESSION_ID;
      delete process.env.COPILOT_CLI_SESSION_ID;
      try {
        assert.strictEqual(mod.resolveSessionId({}), 'cli-hook-session');
      } finally {
        if (orig1 !== undefined) process.env.COPILOT_SESSION_ID = orig1;
        if (orig2 !== undefined) process.env.GITHUB_COPILOT_SESSION_ID = orig2;
        if (orig3 !== undefined) process.env.COPILOT_CLI_SESSION_ID = orig3;
      }
    });

    it('sanitizes special characters', () => {
      assert.strictEqual(
        mod.resolveSessionId({ session_id: 'abc/def@test' }),
        'abc_def_test',
      );
    });

    it('handles empty string session_id', () => {
      assert.strictEqual(
        mod.resolveSessionId({ session_id: '  ' }),
        'cli-hook-session',
      );
    });

    it('uses default argument when called with no args', () => {
      const orig1 = process.env.COPILOT_SESSION_ID;
      const orig2 = process.env.GITHUB_COPILOT_SESSION_ID;
      const orig3 = process.env.COPILOT_CLI_SESSION_ID;
      delete process.env.COPILOT_SESSION_ID;
      delete process.env.GITHUB_COPILOT_SESSION_ID;
      delete process.env.COPILOT_CLI_SESSION_ID;
      try {
        assert.strictEqual(mod.resolveSessionId(), 'cli-hook-session');
      } finally {
        if (orig1 !== undefined) process.env.COPILOT_SESSION_ID = orig1;
        if (orig2 !== undefined) process.env.GITHUB_COPILOT_SESSION_ID = orig2;
        if (orig3 !== undefined) process.env.COPILOT_CLI_SESSION_ID = orig3;
      }
    });
  });

  describe('requiresRuntimeProof', () => {
    it('returns true for edit', () => {
      assert.strictEqual(mod.requiresRuntimeProof('edit'), true);
    });

    it('returns true for powershell', () => {
      assert.strictEqual(mod.requiresRuntimeProof('powershell'), true);
    });

    it('returns true for task', () => {
      assert.strictEqual(mod.requiresRuntimeProof('task'), true);
    });

    it('returns true for apply_patch', () => {
      assert.strictEqual(mod.requiresRuntimeProof('apply_patch'), true);
    });

    it('returns true for create', () => {
      assert.strictEqual(mod.requiresRuntimeProof('create'), true);
    });

    it('returns false for grep', () => {
      assert.strictEqual(mod.requiresRuntimeProof('grep'), false);
    });

    it('returns false for undefined', () => {
      assert.strictEqual(mod.requiresRuntimeProof(undefined), false);
    });

    it('is case-insensitive', () => {
      assert.strictEqual(mod.requiresRuntimeProof('EDIT'), true);
      assert.strictEqual(mod.requiresRuntimeProof('PowerShell'), true);
    });
  });

  describe('inferActionClass', () => {
    it('returns write for edit', () => {
      assert.strictEqual(mod.inferActionClass('edit'), 'write');
    });

    it('returns write for create', () => {
      assert.strictEqual(mod.inferActionClass('create'), 'write');
    });

    it('returns write for apply_patch', () => {
      assert.strictEqual(mod.inferActionClass('apply_patch'), 'write');
    });

    it('returns execute for powershell', () => {
      assert.strictEqual(mod.inferActionClass('powershell'), 'execute');
    });

    it('returns execute for task', () => {
      assert.strictEqual(mod.inferActionClass('task'), 'execute');
    });

    it('returns null for grep', () => {
      assert.strictEqual(mod.inferActionClass('grep'), null);
    });

    it('returns null for undefined', () => {
      assert.strictEqual(mod.inferActionClass(undefined), null);
    });
  });

  describe('isRuntimeContextPreparation', () => {
    it('returns true when text references runtime-enforcement-context.mjs and --prepare', () => {
      assert.strictEqual(
        mod.isRuntimeContextPreparation(
          'node runtime-enforcement-context.mjs --prepare --flow-id=x',
        ),
        true,
      );
    });

    it('returns false when text does not reference preparation', () => {
      assert.strictEqual(
        mod.isRuntimeContextPreparation('some other command'),
        false,
      );
    });

    it('returns false for undefined', () => {
      assert.strictEqual(mod.isRuntimeContextPreparation(undefined), false);
    });
  });

  describe('getRuntimeContextPath', () => {
    it('returns a path ending with hook-context-<sessionId>.json', () => {
      const p = mod.getRuntimeContextPath('test-session');
      assert.ok(p.includes('hook-context-test-session.json'));
    });

    it('uses cli-hook-session when sessionId is null', () => {
      const p = mod.getRuntimeContextPath(null);
      assert.ok(p.includes('hook-context-cli-hook-session.json'));
    });
  });

  describe('ensureRuntimeStateDir', () => {
    it('calls mkdir with recursive true', async () => {
      const fs = await import('node:fs/promises');
      fs.mkdir.mockResolvedValue(undefined);
      await mod.ensureRuntimeStateDir();
      assert.strictEqual(fs.mkdir.mock.calls.length, 1);
      assert.strictEqual(fs.mkdir.mock.calls[0][1].recursive, true);
    });
  });

  describe('readRuntimeContext', () => {
    it('returns parsed JSON when file exists', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue(
        JSON.stringify({ schemaVersion: 1, sessionId: 'test' }),
      );
      const result = await mod.readRuntimeContext('test');
      assert.strictEqual(result.sessionId, 'test');
    });

    it('returns null when file does not exist', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const result = await mod.readRuntimeContext('missing');
      assert.strictEqual(result, null);
    });
  });

  describe('writeRuntimeContext', () => {
    it('writes carrier as JSON', async () => {
      const fs = await import('node:fs/promises');
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      await mod.writeRuntimeContext({ sessionId: 'test', data: 1 });
      assert.strictEqual(fs.writeFile.mock.calls.length, 1);
      const written = JSON.parse(fs.writeFile.mock.calls[0][1].trim());
      assert.strictEqual(written.sessionId, 'test');
    });
  });

  describe('initializeRuntimeContextCarrier', () => {
    it('creates a new carrier', async () => {
      const fs = await import('node:fs/promises');
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const carrier = await mod.initializeRuntimeContextCarrier(
        'test',
        'custom',
      );
      assert.strictEqual(carrier.sessionId, 'test');
      assert.strictEqual(carrier.carrierSource, 'custom');
      assert.strictEqual(carrier.preparedAction, null);
    });

    it('uses default carrierSource', async () => {
      const fs = await import('node:fs/promises');
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const carrier = await mod.initializeRuntimeContextCarrier('test');
      assert.strictEqual(carrier.carrierSource, 'session-start');
    });
  });

  describe('prepareRuntimeContext', () => {
    it('throws when flowId is missing', async () => {
      await assert.rejects(
        mod.prepareRuntimeContext({
          currentAgent: 'agent',
          delegatorChain: ['agent'],
          planPath: 'plans/test.md',
          allowedActionClass: 'write',
        }),
        /Missing required non-empty flowId/,
      );
    });

    it('throws when currentAgent is missing', async () => {
      await assert.rejects(
        mod.prepareRuntimeContext({
          flowId: 'flow1',
          delegatorChain: ['agent'],
          planPath: 'plans/test.md',
          allowedActionClass: 'write',
        }),
        /Missing required non-empty currentAgent/,
      );
    });

    it('throws when planPath is missing', async () => {
      await assert.rejects(
        mod.prepareRuntimeContext({
          flowId: 'flow1',
          currentAgent: 'agent',
          delegatorChain: ['agent'],
          allowedActionClass: 'write',
        }),
        /Missing required non-empty planPath/,
      );
    });

    it('throws when allowedActionClass is missing', async () => {
      await assert.rejects(
        mod.prepareRuntimeContext({
          flowId: 'flow1',
          currentAgent: 'agent',
          delegatorChain: ['agent'],
          planPath: 'plans/test.md',
        }),
        /Missing required non-empty allowedActionClass/,
      );
    });

    it('throws when delegatorChain is empty', async () => {
      await assert.rejects(
        mod.prepareRuntimeContext({
          flowId: 'flow1',
          currentAgent: 'agent',
          delegatorChain: [],
          planPath: 'plans/test.md',
          allowedActionClass: 'write',
        }),
        /delegatorChain must include at least one/,
      );
    });

    it('creates prepared action with all fields', async () => {
      const fs = await import('node:fs/promises');
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const carrier = await mod.prepareRuntimeContext({
        flowId: 'flow1',
        currentAgent: 'agent1',
        delegatorChain: ['orchestrator', 'agent1'],
        requiredSkills: ['skill1'],
        requiredSpecialists: ['spec1'],
        planPath: 'plans/test.md',
        activePhase: 'Phase 1',
        activeStep: 'Step 1',
        allowedActionClass: 'write',
        expectedToolName: 'edit',
      });
      assert.strictEqual(carrier.preparedAction.flowId, 'flow1');
      assert.strictEqual(carrier.preparedAction.currentAgent, 'agent1');
      assert.deepStrictEqual(carrier.preparedAction.delegatorChain, [
        'orchestrator',
        'agent1',
      ]);
      assert.strictEqual(carrier.preparedAction.activePhase, 'Phase 1');
      assert.strictEqual(carrier.preparedAction.activeStep, 'Step 1');
      assert.strictEqual(carrier.preparedAction.expectedToolName, 'edit');
    });

    it('nulls out empty activePhase and activeStep', async () => {
      const fs = await import('node:fs/promises');
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const carrier = await mod.prepareRuntimeContext({
        flowId: 'flow1',
        currentAgent: 'agent1',
        delegatorChain: ['agent1'],
        planPath: 'plans/test.md',
        allowedActionClass: 'write',
      });
      assert.strictEqual(carrier.preparedAction.activePhase, null);
      assert.strictEqual(carrier.preparedAction.activeStep, null);
      assert.strictEqual(carrier.preparedAction.expectedToolName, null);
    });
  });

  describe('clearPreparedRuntimeContext', () => {
    it('returns null when no carrier exists', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const result = await mod.clearPreparedRuntimeContext('test');
      assert.strictEqual(result, null);
    });

    it('clears prepared action when actionId matches', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue(
        JSON.stringify({
          sessionId: 'test',
          preparedAction: { actionId: 'act1' },
        }),
      );
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      const result = await mod.clearPreparedRuntimeContext('test', 'act1');
      assert.strictEqual(result.preparedAction, null);
    });

    it('does not clear when actionId does not match', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue(
        JSON.stringify({
          sessionId: 'test',
          preparedAction: { actionId: 'act1' },
        }),
      );
      const result = await mod.clearPreparedRuntimeContext('test', 'act2');
      assert.ok(result.preparedAction);
    });

    it('clears when actionId is null', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue(
        JSON.stringify({
          sessionId: 'test',
          preparedAction: { actionId: 'act1' },
        }),
      );
      fs.mkdir.mockResolvedValue(undefined);
      fs.writeFile.mockResolvedValue(undefined);
      const result = await mod.clearPreparedRuntimeContext('test');
      assert.strictEqual(result.preparedAction, null);
    });
  });

  describe('validatePreparedRuntimeContext', () => {
    it('delegates to diagnosePreparedRuntimeContext', () => {
      const result = mod.validatePreparedRuntimeContext({ toolName: 'grep' });
      assert.strictEqual(result.ok, true);
    });
  });

  describe('diagnosePreparedRuntimeContext', () => {
    it('returns ok for non-strict tools', () => {
      const result = mod.diagnosePreparedRuntimeContext({ toolName: 'grep' });
      assert.strictEqual(result.ok, true);
      assert.strictEqual(result.actionClass, null);
    });

    it('returns not ok when no carrier', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('Missing runtime enforcement context'));
    });

    it('returns not ok when carrier has no preparedAction', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: { sessionId: 'test', preparedAction: null },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('no prepared action'));
    });

    it('returns not ok when preparedAction missing flowId', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: { sessionId: 'test', preparedAction: { currentAgent: 'a' } },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('missing flowId'));
    });

    it('returns not ok when preparedAction missing currentAgent', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: { sessionId: 'test', preparedAction: { flowId: 'f' } },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('missing currentAgent'));
    });

    it('returns not ok when delegatorChain is empty', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: [],
          },
        },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('missing delegatorChain'));
    });

    it('returns not ok when delegatorChain is not array', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: 'x',
          },
        },
      });
      assert.strictEqual(result.ok, false);
    });

    it('returns not ok when planPath missing', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: ['a'],
          },
        },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('missing planPath'));
    });

    it('returns not ok when allowedActionClass missing', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: ['a'],
            planPath: 'plans/test.md',
          },
        },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('missing allowedActionClass'));
    });

    it('returns not ok when actionClass mismatch', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: ['a'],
            planPath: 'plans/test.md',
            allowedActionClass: 'execute',
          },
        },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('allows execute'));
    });

    it('returns not ok when expectedToolName mismatch', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: ['a'],
            planPath: 'plans/test.md',
            allowedActionClass: 'write',
            expectedToolName: 'create',
          },
        },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('expects tool create'));
    });

    it('returns not ok when planPath mismatch', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        planPath: 'plans/actual.md',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: ['a'],
            planPath: 'plans/expected.md',
            allowedActionClass: 'write',
          },
        },
      });
      assert.strictEqual(result.ok, false);
      assert.ok(result.reason.includes('does not match'));
    });

    it('returns ok when all validations pass', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        planPath: 'plans/test.md',
        carrier: {
          sessionId: 'test',
          preparedAction: {
            actionId: 'act1',
            flowId: 'f',
            currentAgent: 'a',
            delegatorChain: ['a'],
            planPath: 'plans/test.md',
            allowedActionClass: 'write',
          },
        },
      });
      assert.strictEqual(result.ok, true);
      assert.ok(result.preparedAction);
    });

    it('includes recovery hint with plan path', () => {
      const result = mod.diagnosePreparedRuntimeContext({
        toolName: 'edit',
        sessionId: 'test',
        planPath: 'plans/test.md',
      });
      assert.ok(result.recoveryHint.includes('--plan=plans/test.md'));
      assert.ok(result.recoveryHint.includes('--tool-name=edit'));
      assert.ok(result.recoveryHint.includes('--action-class=write'));
    });
  });

  describe('appendLearningEvent', () => {
    it('appends JSON line to learning log', async () => {
      const fs = await import('node:fs/promises');
      fs.appendFile.mockResolvedValue(undefined);
      await mod.appendLearningEvent({ type: 'test' });
      assert.strictEqual(fs.appendFile.mock.calls.length, 1);
      assert.ok(fs.appendFile.mock.calls[0][1].includes('"type":"test"'));
    });
  });

  describe('recordRuntimeActionEvent', () => {
    it('records structured event and returns it', async () => {
      const fs = await import('node:fs/promises');
      fs.appendFile.mockResolvedValue(undefined);
      const event = await mod.recordRuntimeActionEvent({
        eventType: 'test-event',
        sessionId: 'test',
        toolName: 'edit',
        actionClass: 'write',
        flowId: 'flow1',
        currentAgent: 'agent1',
        delegatorChain: ['a'],
        requiredSkills: ['s'],
        requiredSpecialists: ['sp'],
        planPath: 'plans/test.md',
        activePhase: 'P1',
        activeStep: 'S1',
        reason: 'test reason',
        status: 'pass',
      });
      assert.strictEqual(event.eventType, 'test-event');
      assert.strictEqual(event.category, 'runtime-enforcement');
      assert.strictEqual(event.sessionId, 'test');
      assert.strictEqual(event.toolName, 'edit');
      assert.strictEqual(event.flowId, 'flow1');
      assert.strictEqual(event.agent, 'agent1');
    });

    it('uses agent field when currentAgent is missing', async () => {
      const fs = await import('node:fs/promises');
      fs.appendFile.mockResolvedValue(undefined);
      const event = await mod.recordRuntimeActionEvent({
        eventType: 'test',
        agent: 'fallback-agent',
      });
      assert.strictEqual(event.agent, 'fallback-agent');
    });

    it('nulls out empty fields', async () => {
      const fs = await import('node:fs/promises');
      fs.appendFile.mockResolvedValue(undefined);
      const event = await mod.recordRuntimeActionEvent({
        eventType: 'test',
      });
      assert.strictEqual(event.toolName, null);
      assert.strictEqual(event.actionClass, null);
      assert.strictEqual(event.actionId, null);
    });
  });

  describe('loadLearningLogEvents', () => {
    it('returns empty array when file does not exist', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockRejectedValue(new Error('ENOENT'));
      const events = await mod.loadLearningLogEvents();
      assert.deepStrictEqual(events, []);
    });

    it('parses valid JSONL lines', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue('{"a":1}\n{"b":2}\n');
      const events = await mod.loadLearningLogEvents();
      assert.strictEqual(events.length, 2);
      assert.strictEqual(events[0].a, 1);
      assert.strictEqual(events[1].b, 2);
    });

    it('skips empty lines', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue('{"a":1}\n\n\n{"b":2}\n');
      const events = await mod.loadLearningLogEvents();
      assert.strictEqual(events.length, 2);
    });

    it('skips malformed JSON lines', async () => {
      const fs = await import('node:fs/promises');
      fs.readFile.mockResolvedValue('{"a":1}\n{bad json}\n{"b":2}\n');
      const events = await mod.loadLearningLogEvents();
      assert.strictEqual(events.length, 2);
    });
  });

  describe('countTrailingGateFailures', () => {
    it('counts trailing gate-exception events', () => {
      const events = [
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 2);
    });

    it('stops at reset event', () => {
      const events = [
        { sessionId: 's1', eventType: 'runtime-action-prepass' },
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 2);
    });

    it('handles session-id key', () => {
      const events = [
        { 'session-id': 's1', 'gate-id': 'g1', category: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 1);
    });

    it('filters events with gateId', () => {
      const events = [
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 1);
    });

    it('filters events with gate-id key', () => {
      const events = [
        { sessionId: 's1', 'gate-id': 'g1', category: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 1);
    });

    it('returns 0 when no matching events', () => {
      assert.strictEqual(mod.countTrailingGateFailures([], 's1'), 0);
    });

    it('handles postpass reset event', () => {
      const events = [
        { sessionId: 's1', eventType: 'runtime-action-postpass' },
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 1);
    });

    it('continues past non-reset non-exception events', () => {
      const events = [
        { sessionId: 's1', gateId: 'g1', eventType: 'other' },
        { sessionId: 's1', gateId: 'g1', eventType: 'gate-exception' },
      ];
      assert.strictEqual(mod.countTrailingGateFailures(events, 's1'), 1);
    });
  });

  describe('isFailureResetEvent', () => {
    it('returns true for runtime-action-prepass', () => {
      assert.strictEqual(
        mod.isFailureResetEvent({ eventType: 'runtime-action-prepass' }),
        true,
      );
    });

    it('returns true for runtime-action-postpass', () => {
      assert.strictEqual(
        mod.isFailureResetEvent({ eventType: 'runtime-action-postpass' }),
        true,
      );
    });

    it('returns false for other events', () => {
      assert.strictEqual(
        mod.isFailureResetEvent({ eventType: 'gate-exception' }),
        false,
      );
    });
  });

  describe('normalizeRepoPath', () => {
    it('converts backslashes to forward slashes', () => {
      assert.strictEqual(
        mod.normalizeRepoPath('plans\\test.md'),
        'plans/test.md',
      );
    });

    it('trims whitespace', () => {
      assert.strictEqual(
        mod.normalizeRepoPath('  plans/test.md  '),
        'plans/test.md',
      );
    });

    it('handles undefined', () => {
      assert.strictEqual(mod.normalizeRepoPath(undefined), '');
    });
  });

  describe('normalizeStringArray', () => {
    it('handles array input', () => {
      assert.deepStrictEqual(mod.normalizeStringArray(['a', 'b', 'c']), [
        'a',
        'b',
        'c',
      ]);
    });

    it('filters empty array entries', () => {
      assert.deepStrictEqual(mod.normalizeStringArray(['a', '', '  ', 'c']), [
        'a',
        'c',
      ]);
    });

    it('handles non-array non-string', () => {
      assert.deepStrictEqual(mod.normalizeStringArray(42), []);
      assert.deepStrictEqual(mod.normalizeStringArray(null), []);
      assert.deepStrictEqual(mod.normalizeStringArray(undefined), []);
    });

    it('handles empty string', () => {
      assert.deepStrictEqual(mod.normalizeStringArray(''), []);
      assert.deepStrictEqual(mod.normalizeStringArray('   '), []);
    });

    it('parses JSON array string', () => {
      assert.deepStrictEqual(mod.normalizeStringArray('["a", "b"]'), [
        'a',
        'b',
      ]);
    });

    it('falls back to comma split for malformed JSON', () => {
      assert.deepStrictEqual(mod.normalizeStringArray('[bad json, a]'), [
        '[bad json',
        'a]',
      ]);
    });

    it('handles comma-separated string', () => {
      assert.deepStrictEqual(mod.normalizeStringArray('a, b, c'), [
        'a',
        'b',
        'c',
      ]);
    });

    it('handles JSON array with non-string entries', () => {
      assert.deepStrictEqual(mod.normalizeStringArray('[1, 2, null]'), [
        '1',
        '2',
      ]);
    });

    it('handles array with null entries', () => {
      assert.deepStrictEqual(mod.normalizeStringArray(['a', null, 'b']), [
        'a',
        'b',
      ]);
    });
  });

  describe('LEARNING_LOG_PATH', () => {
    it('uses NEATAPTIC_LEARNING_LOG_PATH env var when set', async () => {
      const orig = process.env.NEATAPTIC_LEARNING_LOG_PATH;
      process.env.NEATAPTIC_LEARNING_LOG_PATH = 'custom-log-path';
      try {
        jest.resetModules();
        const mod2 = await import('./runtime-enforcement.mjs');
        assert.ok(mod2.LEARNING_LOG_PATH.includes('custom-log-path'));
      } finally {
        if (orig === undefined) delete process.env.NEATAPTIC_LEARNING_LOG_PATH;
        else process.env.NEATAPTIC_LEARNING_LOG_PATH = orig;
      }
    });
  });
});
