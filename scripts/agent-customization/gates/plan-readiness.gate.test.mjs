/**
 * @module plan-readiness.gate.test
 * @description Coverage tests for plan-readiness.gate.mjs (top-level await gate).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

let mockOptions;
let mockReadWorkspaceFileResult;
let mockReadWorkspaceFileThrows;

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  readWorkspaceFile: async () => {
    if (mockReadWorkspaceFileThrows) throw mockReadWorkspaceFileThrows;
    return mockReadWorkspaceFileResult;
  },
  parseArgs: () => mockOptions,
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGate(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  let importError = null;
  try {
    jest.resetModules();
    await withArgv([process.execPath, 'dummy', ...argv], async () => {
      try {
        await import('./plan-readiness.gate.mjs');
      } catch (e) {
        importError = e;
      }
    });
  } finally {
    console.log = originalLog;
  }
  return { logs, importError };
}

describe('plan-readiness gate', () => {
  let originalExitCode;
  let originalExit;

  beforeEach(() => {
    mockOptions = { json: false, help: false, plan: 'plans/test.plans.md' };
    mockReadWorkspaceFileResult = '';
    mockReadWorkspaceFileThrows = null;
    originalExitCode = process.exitCode;
    originalExit = process.exit;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
    process.exit = originalExit;
  });

  it('prints usage and exits 0 with --help', async () => {
    mockOptions = { json: false, help: true, plan: 'plans/test.plans.md' };
    process.exit = (code) => {
      throw new Error(`EXIT_${code}`);
    };
    const { logs, importError } = await importGate(['--help']);
    assert.ok(importError);
    assert.ok(importError.message.includes('EXIT_0'));
    assert.ok(logs.some((l) => l.includes('plan-readiness gate')));
    assert.ok(logs.some((l) => l.includes('--plan')));
  });

  it('fails when readWorkspaceFile throws', async () => {
    mockReadWorkspaceFileThrows = new Error('ENOENT');
    mockOptions = { json: true, help: false, plan: 'plans/missing.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.ok(parsed.evidence.error);
    assert.equal(parsed.evidence.plan, 'plans/missing.plans.md');
    assert.equal(process.exitCode, 1);
  });

  it('fails when section is not found', async () => {
    mockReadWorkspaceFileResult = '# Plan\n\nJust some content without the section.\n';
    mockOptions = { json: true, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.sectionFound, false);
    assert.equal(parsed.evidence.greenLightFound, false);
    assert.equal(process.exitCode, 1);
  });

  it('fails when section exists but no green-light', async () => {
    mockReadWorkspaceFileResult =
      '# Plan\n\n## Latest validation evidence\n\nSome text without green light.\n';
    mockOptions = { json: true, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.sectionFound, true);
    assert.equal(parsed.evidence.greenLightFound, false);
    assert.equal(process.exitCode, 1);
  });

  it('passes when section has green-light: true', async () => {
    mockReadWorkspaceFileResult =
      '# Plan\n\n## Latest validation evidence\n\ngreen-light: true\n';
    mockOptions = { json: true, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.sectionFound, true);
    assert.equal(parsed.evidence.greenLightFound, true);
    assert.equal(process.exitCode, 0);
  });

  it('passes when section has status: green-light', async () => {
    mockReadWorkspaceFileResult =
      '# Plan\n\n## Latest validation evidence\n\nstatus: green-light\n';
    mockOptions = { json: true, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.greenLightFound, true);
    assert.equal(process.exitCode, 0);
  });

  it('emits PASS text without --json when pass=true', async () => {
    mockReadWorkspaceFileResult =
      '# Plan\n\n## Latest validation evidence\n\ngreen-light: true\n';
    mockOptions = { json: false, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate([]);
    assert.ok(logs.some((l) => l.includes('PASS')));
    assert.equal(process.exitCode, 0);
  });

  it('emits FAIL text with fixHint without --json when section missing', async () => {
    mockReadWorkspaceFileResult = '# Plan\nNo section here.\n';
    mockOptions = { json: false, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate([]);
    assert.ok(logs.some((l) => l.includes('FAIL')));
    assert.ok(logs.some((l) => l.includes('fixHint:')));
    assert.equal(process.exitCode, 1);
  });

  it('emits FAIL text with fixHint without --json when no green-light', async () => {
    mockReadWorkspaceFileResult =
      '# Plan\n\n## Latest validation evidence\n\nNo green light here.\n';
    mockOptions = { json: false, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate([]);
    assert.ok(logs.some((l) => l.includes('FAIL')));
    assert.ok(logs.some((l) => l.includes('fixHint:')));
    assert.equal(process.exitCode, 1);
  });

  it('emits JSON with --json when readWorkspaceFile throws', async () => {
    mockReadWorkspaceFileThrows = new Error('read error');
    mockOptions = { json: true, help: false, plan: 'plans/bad.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.ok(parsed.evidence.error.includes('read error'));
  });

  it('handles green-light=true with equals sign', async () => {
    mockReadWorkspaceFileResult =
      '# Plan\n\n## Latest validation evidence\n\ngreen-light=true\n';
    mockOptions = { json: true, help: false, plan: 'plans/test.plans.md' };
    const { logs } = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.greenLightFound, true);
  });
});