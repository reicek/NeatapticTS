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
  process.stdout.write = (chunk) => {
    output.push(String(chunk));
    return true;
  };
  return {
    output,
    restore() {
      process.stdout.write = origWrite;
    },
  };
}

function captureStderr() {
  const output = [];
  const origWrite = process.stderr.write;
  process.stderr.write = (chunk) => {
    output.push(String(chunk));
    return true;
  };
  return {
    output,
    restore() {
      process.stderr.write = origWrite;
    },
  };
}

async function importModule() {
  try {
    await import('./refresh-cortex-after-write.mjs');
  } catch {
    // process.exit may throw
  }
}

describe('refresh-cortex-after-write', () => {
  it('continues when input is runtime context preparation', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(true);

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
    assert.strictEqual(json.hookSpecificOutput, undefined);
  });

  it('continues when no write or substantive tool', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(
      JSON.stringify({ tool_name: 'read_file' }),
    );
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
  });

  it('handles empty stdin', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue('');
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
  });

  it('handles invalid JSON stdin', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue('not json');
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
  });

  it('runs refresh steps for write tools', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": true}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
    assert.ok(
      json.hookSpecificOutput.additionalContext.includes('build-index'),
    );
  });

  it('runs workflow sync for substantive tools', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(
      JSON.stringify({ tool_name: 'powershell' }),
    );
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    // First call: workflow sync, subsequent: refresh steps
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"ok": true}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(
      json.hookSpecificOutput.additionalContext.includes(
        'workflow-update-sync',
      ),
    );
  });

  it('handles refresh step failure', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    // First call: gate exception recorder (status 0)
    // Second call: build-index step (status 1, fails)
    // Third call: build-browser-snapshot step (status 1)
    mockChildProc.spawnSync
      .mockReturnValueOnce({ status: 0, stdout: '', stderr: '' })
      .mockReturnValueOnce({ status: 1, stdout: '', stderr: 'build failed' })
      .mockReturnValue({ status: 1, stdout: '', stderr: 'build failed' });

    await importModule();
    // Wait for async main() to settle
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(capErr.output.some((o) => o.includes('build-index failed')));
  });

  it('handles workflow sync failure', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(
      JSON.stringify({ tool_name: 'powershell' }),
    );
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    // First call: workflow sync (status 1, fails)
    // Second call: gate exception recorder (status 0)
    mockChildProc.spawnSync
      .mockReturnValueOnce({ status: 1, stdout: '', stderr: 'sync failed' })
      .mockReturnValueOnce({ status: 0, stdout: '', stderr: '' });

    await importModule();
    // Wait for async main() to settle
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(
      capErr.output.some((o) => o.includes('workflow-update-sync failed')),
    );
  });

  it('handles runtime proof requirement - valid', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(true);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ carrier: {} });
    mockEnf.validatePreparedRuntimeContext.mockReturnValue({
      ok: true,
      preparedAction: {
        actionId: 'a1',
        allowedActionClass: 'write',
        flowId: 'f1',
        currentAgent: 'test',
        delegatorChain: [],
        requiredSkills: [],
        requiredSpecialists: [],
        planPath: 'plans/test.md',
        activePhase: '1',
        activeStep: '2',
      },
    });
    mockEnf.recordRuntimeActionEvent.mockResolvedValue();
    mockEnf.clearPreparedRuntimeContext.mockResolvedValue();
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": true}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(
      json.hookSpecificOutput.additionalContext.includes(
        'runtime-enforcement-context=pass',
      ),
    );
    assert.ok(mockEnf.clearPreparedRuntimeContext.mock.calls.length >= 1);
  });

  it('handles runtime proof mismatch', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(true);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ carrier: {} });
    mockEnf.validatePreparedRuntimeContext.mockReturnValue({
      ok: false,
      reason: 'mismatch',
      recoveryHint: 'try again',
      actionClass: 'write',
      preparedAction: { actionId: 'a1', flowId: 'f1', currentAgent: 'test' },
    });
    mockEnf.recordRuntimeActionEvent.mockResolvedValue();
    // Gate exception recorder
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '',
      stderr: '',
    });

    await importModule();
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(
      capErr.output.some((o) => o.includes('runtime-enforcement-context')),
    );
  });

  it('handles runtime proof mismatch without preparedAction', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(true);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ carrier: {} });
    mockEnf.validatePreparedRuntimeContext.mockReturnValue({
      ok: false,
      reason: 'no context',
      recoveryHint: 'prepare first',
      actionClass: 'write',
      preparedAction: null,
    });
    mockEnf.recordRuntimeActionEvent.mockResolvedValue();
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '',
      stderr: '',
    });

    await importModule();
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(capErr.output.some((o) => o.includes('no context')));
  });

  it('handles main() rejection', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockImplementation(() => {
      throw new Error('read failed');
    });

    await importModule();
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(
      capErr.output.some((o) => o.includes('runtime enforcement failed')),
    );
  });

  it('handles toolName via toolName camelCase', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ toolName: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": true}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    assert.ok(mockChildProc.spawnSync.mock.calls.length > 0);
  });

  it('handles refresh step failure with no stderr/stdout', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 1,
      stdout: null,
      stderr: null,
    });

    await importModule();
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(capErr.output.some((o) => o.includes('failed')));
  });

  it('parses stdout with ok boolean false', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"ok": false}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('not-ok'));
  });

  it('parses stdout with documents/chunks', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"documents": 3, "chunks": 50}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(
      json.hookSpecificOutput.additionalContext.includes('3 docs / 50 chunks'),
    );
  });

  it('parses stdout with steps array', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"steps": [{"name": "x", "status": "done"}]}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('x:done'));
  });

  it('handles non-JSON stdout as last line', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: 'line1\nlast line',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('last line'));
  });

  it('handles gate exception recorder failure', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    // First call: gate exception recorder (fails), subsequent: refresh step (fails)
    mockChildProc.spawnSync
      .mockReturnValueOnce({
        status: 1,
        stdout: '',
        stderr: 'recorder error',
      })
      .mockReturnValue({
        status: 1,
        stdout: '',
        stderr: 'step error',
      });

    await importModule();
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(
      capErr.output.some((o) => o.includes('failed to record gate exception')),
    );
  });

  it('handles empty stdout as ok', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('=ok'));
  });

  it('handles runtime proof with recordRuntimeActionEvent rejection on pass path', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(true);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ carrier: {} });
    mockEnf.validatePreparedRuntimeContext.mockReturnValue({
      ok: true,
      preparedAction: {
        actionId: 'a1',
        allowedActionClass: 'write',
        flowId: 'f1',
        currentAgent: 'test',
        delegatorChain: [],
        requiredSkills: [],
        requiredSpecialists: [],
        planPath: 'plans/test.md',
        activePhase: '1',
        activeStep: '2',
      },
    });
    mockEnf.recordRuntimeActionEvent.mockRejectedValue(new Error('log failed'));
    mockEnf.clearPreparedRuntimeContext.mockResolvedValue();
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": true}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    // Should still succeed despite logging failure (best-effort)
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
  });

  it('handles runtime proof with recordRuntimeActionEvent rejection on mismatch path', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();

    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(true);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockEnf.readRuntimeContext.mockResolvedValue({ carrier: {} });
    mockEnf.validatePreparedRuntimeContext.mockReturnValue({
      ok: false,
      reason: 'mismatch',
      recoveryHint: 'retry',
      actionClass: 'write',
      preparedAction: { actionId: 'a1', flowId: 'f1', currentAgent: 'test' },
    });
    mockEnf.recordRuntimeActionEvent.mockRejectedValue(new Error('log failed'));
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '',
      stderr: '',
    });

    await importModule();
    await new Promise((r) => setTimeout(r, 200));

    process.exit = origExit;
    capErr.restore();
    // Should still exit 2 despite logging failure (best-effort)
    assert.ok(exitCode !== null, 'process.exit was called');
    assert.ok(capErr.output.some((o) => o.includes('mismatch')));
  });

  it('parses stdout with status string', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"status": "healthy"}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('healthy'));
  });

  it('handles invalid JSON stdout with braces', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{bad json}',
      stderr: '',
    });

    await importModule();

    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    // Should fall through to last-line parsing
    assert.ok(json.hookSpecificOutput.additionalContext.length > 0);
  });

  it('collects candidate strings from array fields in hookInput', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(
      JSON.stringify({ tool_name: 'edit', tags: ['a', 'b'] }),
    );
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": true}',
      stderr: '',
    });
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
  });

  it('collects candidate strings from numeric fields in hookInput', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(
      JSON.stringify({ tool_name: 'powershell', count: 5 }),
    );
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"ok": true}',
      stderr: '',
    });
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.strictEqual(json.continue, true);
  });

  it('summarizes failure and formats failure with null status', async () => {
    const origExit = process.exit;
    let exitCode = null;
    process.exit = (code) => {
      exitCode = code;
    };
    const capErr = captureStderr();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync
      .mockReturnValueOnce({ status: 0, stdout: '', stderr: '' })
      .mockReturnValueOnce({ status: null, stdout: '', stderr: '' })
      .mockReturnValueOnce({ status: null, stdout: '', stderr: '' });
    await importModule();
    await new Promise((r) => setTimeout(r, 200));
    process.exit = origExit;
    capErr.restore();
    assert.ok(capErr.output.some((o) => o.includes('status unknown')));
  });

  it('summarizes parsed output with pass:false as fail', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"pass": false}',
      stderr: '',
    });
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('=fail'));
  });

  it('summarizes unrecognized JSON fields as ok', async () => {
    const capOut = captureStdout();
    mockFs.readFileSync.mockReturnValue(JSON.stringify({ tool_name: 'edit' }));
    mockEnf.isRuntimeContextPreparation.mockReturnValue(false);
    mockEnf.requiresRuntimeProof.mockReturnValue(false);
    mockEnf.resolveSessionId.mockReturnValue('s1');
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"unknown": true}',
      stderr: '',
    });
    await importModule();
    capOut.restore();
    const json = JSON.parse(capOut.output[0].trim());
    assert.ok(json.hookSpecificOutput.additionalContext.includes('=ok'));
  });
});
