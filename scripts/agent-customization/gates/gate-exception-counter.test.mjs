/**
 * @module gate-exception-counter.test
 * @description Coverage tests for gate-exception-counter.mjs.
 *
 * The helper has no exports and no import.meta.url guard — all code runs at
 * top level. We import it with mocked dependencies and capture console.log
 * output to verify behavior.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/gate-exception-counter.mjs',
);

let mockCountTrailingGateFailures;
let mockLoadLearningLogEvents;

jest.unstable_mockModule('../enforcement/runtime-enforcement.mjs', () => ({
  countTrailingGateFailures: (...args) =>
    mockCountTrailingGateFailures(...args),
  loadLearningLogEvents: (...args) => mockLoadLearningLogEvents(...args),
}));

async function importGate(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    process.argv = [process.execPath, GATE_PATH, ...argv];
    await import('./gate-exception-counter.mjs');
    await new Promise((r) => setTimeout(r, 200));
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('gate-exception-counter', () => {
  let originalExitCode;

  beforeEach(() => {
    mockCountTrailingGateFailures = jest.fn(() => 0);
    mockLoadLearningLogEvents = jest.fn(async () => []);
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  it('explicit failureCount < 3 → escalationTriggered=false, source=explicit', async () => {
    const logs = await importGate(['--failure-count=2', '--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(parsed.suggestedAgent, null);
    assert.equal(parsed.source, 'explicit');
    assert.equal(parsed.failureCount, 2);
  });

  it('explicit failureCount >= 3 → escalationTriggered=true, suggestedAgent=00-helping', async () => {
    const logs = await importGate(['--failure-count=3', '--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.escalationTriggered, true);
    assert.equal(parsed.suggestedAgent, '00-helping');
    assert.equal(parsed.source, 'explicit');
    assert.equal(parsed.failureCount, 3);
  });

  it('explicit failureCount > 3 → escalationTriggered=true', async () => {
    const logs = await importGate(['--failure-count=5']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.escalationTriggered, true);
    assert.equal(parsed.suggestedAgent, '00-helping');
    assert.equal(parsed.failureCount, 5);
  });

  it('--derive-from-learning-log with valid sessionId uses learning log count', async () => {
    mockCountTrailingGateFailures = jest.fn(() => 4);
    mockLoadLearningLogEvents = jest.fn(async () => [
      { eventType: 'gate-exception' },
    ]);

    const logs = await importGate([
      '--derive-from-learning-log',
      '--session-id=abc-123',
      '--json',
    ]);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.escalationTriggered, true);
    assert.equal(parsed.source, 'learning-log');
    assert.equal(parsed.failureCount, 4);
    assert.equal(parsed.sessionId, 'abc-123');
    assert.equal(mockLoadLearningLogEvents.mock.calls.length, 1);
    assert.equal(mockCountTrailingGateFailures.mock.calls[0][1], 'abc-123');
  });

  it('--derive-from-learning-log with empty sessionId → failureCount=0', async () => {
    const logs = await importGate([
      '--derive-from-learning-log',
      '--session-id=',
    ]);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(parsed.source, 'learning-log');
    assert.equal(mockLoadLearningLogEvents.mock.calls.length, 0);
  });

  it('--derive-from-learning-log with whitespace sessionId → trimmed, failureCount=0', async () => {
    const logs = await importGate([
      '--derive-from-learning-log',
      '--session-id=   ',
    ]);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(mockLoadLearningLogEvents.mock.calls.length, 0);
  });

  it('--failure-count=NaN (invalid) → failureCount=0', async () => {
    const logs = await importGate(['--failure-count=abc']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(parsed.source, 'explicit');
  });

  it('--json flag does not change output (always JSON)', async () => {
    const logs = await importGate(['--failure-count=1', '--json']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(parsed.failureCount, 1);
  });

  it('parses --gate-id= argument', async () => {
    const logs = await importGate(['--gate-id=my-gate', '--failure-count=0']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
  });

  it('parses --agent= argument', async () => {
    const logs = await importGate(['--agent=01-planning', '--failure-count=0']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
  });

  it('parses --session-id= argument for explicit mode', async () => {
    const logs = await importGate([
      '--session-id=session-xyz',
      '--failure-count=1',
    ]);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.sessionId, 'session-xyz');
    assert.equal(parsed.source, 'explicit');
  });

  it('unknown args are ignored', async () => {
    const logs = await importGate(['--unknown-flag', '--failure-count=0']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
    assert.equal(parsed.escalationTriggered, false);
  });

  it('default failureCount is 0 when no --failure-count provided', async () => {
    const logs = await importGate([]);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(parsed.source, 'explicit');
  });

  it('--derive-from-learning-log with learning log count < 3', async () => {
    mockCountTrailingGateFailures = jest.fn(() => 2);
    mockLoadLearningLogEvents = jest.fn(async () => []);

    const logs = await importGate([
      '--derive-from-learning-log',
      '--session-id=sess-1',
    ]);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 2);
    assert.equal(parsed.escalationTriggered, false);
    assert.equal(parsed.source, 'learning-log');
  });

  it('--derive-from-learning-log with non-string sessionId → failureCount=0', async () => {
    // Pass --session-id without value so it parses as empty string.
    // The typeof check branch for non-string is covered when sessionId
    // is undefined, but we can simulate it by passing --derive-from-learning-log
    // without any --session-id flag (sessionId defaults to empty string).
    const logs = await importGate(['--derive-from-learning-log']);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.failureCount, 0);
    assert.equal(parsed.escalationTriggered, false);
  });
});
