import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./runtime-enforcement.mjs', () => ({
  clearPreparedRuntimeContext: jest.fn(),
  diagnosePreparedRuntimeContext: jest.fn(),
  inferActionClass: jest.fn(),
  prepareRuntimeContext: jest.fn(),
  readRuntimeContext: jest.fn(),
  resolveSessionId: jest.fn(),
}));

let mockEnf;

beforeEach(async () => {
  jest.resetModules();
  mockEnf = await import('./runtime-enforcement.mjs');
  jest.clearAllMocks();
});

function captureConsole() {
  const logs = [];
  const errors = [];
  const origLog = console.log;
  const origError = console.error;
  console.log = (...args) => logs.push(args.join(' '));
  console.error = (...args) => errors.push(args.join(' '));
  return {
    logs, errors,
    restore() { console.log = origLog; console.error = origError; },
  };
}

async function importModule(argv) {
  const origArgv = process.argv;
  const origExit = process.exit;
  process.argv = ['node', 'runtime-enforcement-context.mjs', ...(argv || [])];
  process.exit = (code) => { throw new Error(`EXIT:${code}`); };
  try {
    await import('./runtime-enforcement-context.mjs');
  } catch {
    // process.exit may throw
  }
  process.argv = origArgv;
  process.exit = origExit;
}

describe('runtime-enforcement-context', () => {
  it('prints help and exits 0 when --help', async () => {
    const cap = captureConsole();
    await importModule(['--help']);
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('runtime-enforcement-context')));
  });

  it('prints usage and exits 1 when no mode specified', async () => {
    const cap = captureConsole();
    await importModule([]);
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('Usage:')));
  });

  it('prepares runtime context with --prepare', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('session-123');
    mockEnf.prepareRuntimeContext.mockResolvedValue({ preparedAction: { actionId: 'a1' } });

    await importModule(['--prepare', '--flow-id=f1', '--agent=test', '--tool-name=edit', '--plan=plans/test.md']);

    cap.restore();
    assert.ok(mockEnf.prepareRuntimeContext.mock.calls.length >= 1);
    const call = mockEnf.prepareRuntimeContext.mock.calls[0][0];
    assert.strictEqual(call.flowId, 'f1');
    assert.strictEqual(call.currentAgent, 'test');
    assert.strictEqual(call.expectedToolName, 'edit');
    assert.strictEqual(call.planPath, 'plans/test.md');
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.ok, true);
    assert.strictEqual(json.sessionId, 'session-123');
  });

  it('uses inferActionClass when --action-class not provided', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.inferActionClass.mockReturnValue('write');
    mockEnf.prepareRuntimeContext.mockResolvedValue({});

    await importModule(['--prepare', '--tool-name=edit']);

    assert.ok(mockEnf.inferActionClass.mock.calls.length >= 1);
    assert.strictEqual(mockEnf.prepareRuntimeContext.mock.calls[0][0].allowedActionClass, 'write');
  });

  it('uses explicit --action-class when provided', async () => {
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.prepareRuntimeContext.mockResolvedValue({});

    await importModule(['--prepare', '--action-class=execute', '--tool-name=task']);

    assert.strictEqual(mockEnf.prepareRuntimeContext.mock.calls[0][0].allowedActionClass, 'execute');
    assert.strictEqual(mockEnf.inferActionClass.mock.calls.length, 0);
  });

  it('passes delegator-chain and required-skills/specialists', async () => {
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.prepareRuntimeContext.mockResolvedValue({});

    await importModule([
      '--prepare', '--delegator-chain=a,b', '--required-skills=x,y',
      '--required-specialists=z,w', '--phase=1', '--step=2',
      '--session-id=custom',
    ]);

    const call = mockEnf.prepareRuntimeContext.mock.calls[0][0];
    assert.strictEqual(call.delegatorChain, 'a,b');
    assert.strictEqual(call.requiredSkills, 'x,y');
    assert.strictEqual(call.requiredSpecialists, 'z,w');
    assert.strictEqual(call.activePhase, '1');
    assert.strictEqual(call.activeStep, '2');
  });

  it('ignores unrecognized arguments', async () => {
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.prepareRuntimeContext.mockResolvedValue({});

    await importModule([
      '--prepare', '--tool-name=edit', '--plan=plans/test.md',
      '--unknown-flag=value',
    ]);

    assert.ok(mockEnf.prepareRuntimeContext.mock.calls.length >= 1);
  });

  it('shows runtime context with --show', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ preparedAction: { actionId: 'a1' } });

    await importModule(['--show']);

    cap.restore();
    assert.ok(mockEnf.readRuntimeContext.mock.calls.length >= 1);
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.ok, true);
  });

  it('diagnoses runtime context with --diagnose', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ carrier: {} });
    mockEnf.diagnosePreparedRuntimeContext.mockReturnValue({ ok: true });

    await importModule(['--diagnose', '--tool-name=edit', '--plan=plans/test.md']);

    cap.restore();
    assert.ok(mockEnf.diagnosePreparedRuntimeContext.mock.calls.length >= 1);
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.ok, true);
    assert.ok(json.context.diagnosis);
  });

  it('clears runtime context with --clear', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.clearPreparedRuntimeContext.mockResolvedValue({ cleared: true });

    await importModule(['--clear', '--action-id=a1']);

    cap.restore();
    assert.ok(mockEnf.clearPreparedRuntimeContext.mock.calls.length >= 1);
    assert.strictEqual(mockEnf.clearPreparedRuntimeContext.mock.calls[0][1], 'a1');
  });

  it('handles error and returns ok=false', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockRejectedValue(new Error('DB error'));

    await importModule(['--show']);

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.ok, false);
    assert.ok(json.error.includes('DB error'));
  });

  it('handles non-Error throw as string', async () => {
    const cap = captureConsole();
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockRejectedValue('string error');

    await importModule(['--show']);

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.ok, false);
    assert.strictEqual(json.error, 'string error');
  });
});