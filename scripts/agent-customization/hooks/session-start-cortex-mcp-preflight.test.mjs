import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('../enforcement/runtime-enforcement.mjs', () => ({
  clearPreparedRuntimeContext: jest.fn(),
  isRuntimeContextPreparation: jest.fn(),
  readRuntimeContext: jest.fn(),
  recordRuntimeActionEvent: jest.fn(),
  requiresRuntimeProof: jest.fn(),
  resolveSessionId: jest.fn(),
  validatePreparedRuntimeContext: jest.fn(),
  initializeRuntimeContextCarrier: jest.fn(),
}));

jest.unstable_mockModule('node:child_process', () => ({
  spawnSync: jest.fn(),
}));

jest.unstable_mockModule('node:fs', () => ({
  readFileSync: jest.fn(),
  existsSync: jest.fn(),
  writeFileSync: jest.fn(),
}));

let mockEnf;
let mockChildProc;
let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockEnf = await import('../enforcement/runtime-enforcement.mjs');
  mockChildProc = await import('node:child_process');
  mockFs = await import('node:fs');
  jest.clearAllMocks();
});

function captureStdout() {
  const output = [];
  const origWrite = process.stdout.write;
  process.stdout.write = (chunk) => { output.push(String(chunk)); return true; };
  return {
    output,
    restore() { process.stdout.write = origWrite; },
  };
}

function captureStderr() {
  const output = [];
  const origWrite = process.stderr.write;
  process.stderr.write = (chunk) => { output.push(String(chunk)); return true; };
  return {
    output,
    restore() { process.stderr.write = origWrite; },
  };
}

async function importModule() {
  const origExit = process.exit;
  process.exit = () => {};
  try {
    await import('./session-start-cortex-mcp-preflight.mjs');
  } catch {
    // import errors are safe to ignore
  }
  await new Promise((r) => setTimeout(r, 200));
  process.exit = origExit;
}

describe('session-start-cortex-mcp-preflight', () => {
  it('runs all preflight steps and writes success output', async () => {
    const capOut = captureStdout();
    const capErr = captureStderr();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": true}',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({ preparedAction: { actionId: 'a1' } });

    await importModule();

    capOut.restore();
    capErr.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
    assert.strictEqual(json.hookSpecificOutput.hookEventName, 'SessionStart');
    assert.ok(json.hookSpecificOutput.additionalContext.includes('runtime-context=prepared'));
  });

  it('handles preflight step failure', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const capErr = captureStderr();

    mockChildProc.spawnSync.mockReturnValue({
      status: 1,
      stdout: '',
      stderr: 'step error',
    });

    await importModule();

    process.exit = origExit;
    capErr.restore();
    assert.ok(capErr.output.some((o) => o.includes('session-start-index failed')));
  });

  it('handles preflight step failure with stdout only', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const capErr = captureStderr();

    mockChildProc.spawnSync.mockReturnValue({
      status: 1,
      stdout: 'stdout error',
      stderr: '',
    });

    await importModule();

    process.exit = origExit;
    capErr.restore();
    assert.ok(capErr.output.some((o) => o.includes('stdout error')));
  });

  it('handles preflight step failure with no output', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const capErr = captureStderr();

    mockChildProc.spawnSync.mockReturnValue({
      status: 1,
      stdout: '',
      stderr: '',
    });

    await importModule();

    process.exit = origExit;
    capErr.restore();
    assert.ok(capErr.output.some((o) => o.includes('exited with status')));
  });

  it('handles runtime context initialization without preparedAction', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('runtime-context=initialized'));
  });

  it('handles main() rejection', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const capErr = captureStderr();

    mockChildProc.spawnSync.mockImplementation(() => {
      throw new Error('spawn failed');
    });

    await importModule();

    process.exit = origExit;
    capErr.restore();
    assert.ok(capErr.output.some((o) => o.includes('runtime context initialization failed')));
  });

  it('parses stdout with ok boolean', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValueOnce({
      status: 0,
      stdout: '{"ok": true}',
      stderr: '',
    }).mockReturnValueOnce({
      status: 0,
      stdout: '{"ok": false}',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('ok'));
    assert.ok(json.hookSpecificOutput.additionalContext.includes('not-ok'));
  });

  it('parses stdout with status string', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"status": "running"}',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('running'));
  });

  it('parses stdout with documents/chunks', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"documents": 5, "chunks": 100}',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('5 docs / 100 chunks'));
  });

  it('parses stdout with steps array', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"steps": [{"name": "a", "status": "pass"}, {"name": "b", "status": "fail"}]}',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('a:pass'));
    assert.ok(json.hookSpecificOutput.additionalContext.includes('b:fail'));
  });

  it('parses non-JSON stdout as last line', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: 'line1\nline2\nlast',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('last'));
  });

  it('handles invalid JSON with braces', async () => {
    const capOut = captureStdout();

    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{invalid json}',
      stderr: '',
    });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    // Should fall through to last-line parsing
    assert.ok(json.hookSpecificOutput.additionalContext.length > 0);
  });

  it('summarizes null stdout as ok', async () => {
    const capOut = captureStdout();
    mockChildProc.spawnSync.mockReturnValue({ status: 0, stdout: null, stderr: '' });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('=ok'));
  });

  it('summarizes pass:false as fail', async () => {
    const capOut = captureStdout();
    mockChildProc.spawnSync.mockReturnValue({ status: 0, stdout: '{"pass": false}', stderr: '' });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('=fail'));
  });

  it('summarizes unrecognized JSON fields as ok', async () => {
    const capOut = captureStdout();
    mockChildProc.spawnSync.mockReturnValue({ status: 0, stdout: '{"unknown": true}', stderr: '' });
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.initializeRuntimeContextCarrier.mockResolvedValue({});
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('=ok'));
  });

  it('formats failure with null stderr, stdout, and status', async () => {
    const capErr = captureStderr();
    mockChildProc.spawnSync.mockReturnValue({ status: null, stdout: null, stderr: null });
    await importModule();
    capErr.restore();
    assert.ok(capErr.output.some((o) => o.includes('exited with status unknown')));
  });
});