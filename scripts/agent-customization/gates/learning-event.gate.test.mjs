/**
 * @module learning-event.gate.test
 * @description Coverage tests for learning-event.gate.mjs (top-level await gate).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

let mockOptions;
let mockReadFileResult;
let mockReadFileThrows;
let mockRepoRoot;

jest.unstable_mockModule('node:fs/promises', () => ({
  readFile: async () => {
    if (mockReadFileThrows) throw mockReadFileThrows;
    return mockReadFileResult;
  },
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: () => mockOptions,
  repoRoot: mockRepoRoot,
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
  try {
    jest.resetModules();
    await withArgv([process.execPath, 'dummy', ...argv], async () => {
      await import('./learning-event.gate.mjs');
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('learning-event gate', () => {
  let originalExitCode;
  let originalEnv;

  beforeEach(() => {
    mockOptions = { json: false, help: false };
    mockReadFileResult = '';
    mockReadFileThrows = null;
    mockRepoRoot = '/fake/repo';
    originalExitCode = process.exitCode;
    originalEnv = process.env.NEATAPTIC_LEARNING_LOG_PATH;
    delete process.env.NEATAPTIC_LEARNING_LOG_PATH;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
    if (originalEnv !== undefined) {
      process.env.NEATAPTIC_LEARNING_LOG_PATH = originalEnv;
    } else {
      delete process.env.NEATAPTIC_LEARNING_LOG_PATH;
    }
  });

  it('fails with exists=false when readFile throws', async () => {
    mockReadFileThrows = new Error('ENOENT');
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.exists, false);
    assert.equal(process.exitCode, 1);
  });

  it('emits FAIL text without --json when readFile throws', async () => {
    mockReadFileThrows = new Error('ENOENT');
    mockOptions = { json: false, help: false };
    const logs = await importGate([]);
    assert.ok(logs.some((l) => l.includes('FAIL')));
    assert.ok(logs.some((l) => l.includes('fixHint:')));
    assert.equal(process.exitCode, 1);
  });

  it('passes with valid JSONL events and --json output', async () => {
    mockReadFileResult =
      '{"eventType":"gate-exception","description":"test"}\n';
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.exists, true);
    assert.equal(parsed.evidence.eventCount, 1);
    assert.equal(parsed.evidence.rawLineCount, 1);
    assert.deepEqual(parsed.evidence.categories, ['gate-exception']);
    assert.equal(process.exitCode, 0);
  });

  it('emits PASS text without --json when events exist', async () => {
    mockReadFileResult =
      '{"eventType":"gate-exception","description":"test"}\n';
    mockOptions = { json: false, help: false };
    const logs = await importGate([]);
    assert.ok(logs.some((l) => l.includes('PASS')));
    assert.equal(process.exitCode, 0);
  });

  it('handles mixed malformed and valid lines', async () => {
    mockReadFileResult = [
      'not-json',
      '{"eventType":"gate-exception","description":"test"}',
      '{"category":"cross-tier-helper","description":"test"}',
      '{"description":"no type field"}',
    ].join('\n');
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.eventCount, 3);
    assert.equal(parsed.evidence.rawLineCount, 4);
    assert.deepEqual(parsed.evidence.categories, [
      'gate-exception',
      'cross-tier-helper',
    ]);
  });

  it('fails when only malformed lines exist', async () => {
    mockReadFileResult = 'not-json\nalso not json\n';
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.eventCount, 0);
    assert.equal(parsed.evidence.rawLineCount, 2);
    assert.equal(process.exitCode, 1);
  });

  it('uses NEATAPTIC_LEARNING_LOG_PATH env var for custom path', async () => {
    process.env.NEATAPTIC_LEARNING_LOG_PATH = '/custom/learning-log.jsonl';
    mockReadFileResult =
      '{"eventType":"custom-event","description":"test"}\n';
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.ok(parsed.evidence.path.includes('custom'));
    assert.ok(parsed.evidence.path.includes('learning-log.jsonl'));
  });

  it('emits FAIL text without --json when only malformed lines', async () => {
    mockReadFileResult = 'not-json\n';
    mockOptions = { json: false, help: false };
    const logs = await importGate([]);
    assert.ok(logs.some((l) => l.includes('FAIL')));
    assert.ok(logs.some((l) => l.includes('fixHint:')));
    assert.equal(process.exitCode, 1);
  });

  it('handles events with neither eventType nor category (filter Boolean false)', async () => {
    mockReadFileResult = '{"description":"no type"}\n';
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.eventCount, 1);
    assert.deepEqual(parsed.evidence.categories, []);
  });

  it('handles empty file content (no lines)', async () => {
    mockReadFileResult = '\n   \n\n';
    mockOptions = { json: true, help: false };
    const logs = await importGate(['--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, false);
    assert.equal(parsed.evidence.eventCount, 0);
    assert.equal(parsed.evidence.rawLineCount, 0);
  });
});