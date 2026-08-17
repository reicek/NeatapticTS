/**
 * @module implementation-artifact-paths.gate.test
 * @description Coverage tests for implementation-artifact-paths.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

describe('implementation-artifact-paths gate', () => {
  let originalLog;
  let logs;

  beforeEach(() => {
    logs = [];
    originalLog = console.log;
    console.log = (...args) => logs.push(args.map(String).join(' '));
  });

  afterEach(() => {
    console.log = originalLog;
    process.exitCode = 0;
  });

  it('emits JSON contract with --json', async () => {
    await withArgv([process.execPath, 'dummy', '--json'], async () => {
      await jest.isolateModulesAsync(async () => {
        await import('./implementation-artifact-paths.gate.mjs');
      });
    });
    assert.equal(logs.length, 1);
    const parsed = JSON.parse(logs[0]);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.evidence.gate, 'implementation-artifact-paths');
    assert.equal(process.exitCode, 0);
  });

  it('emits human-readable PASS without --json', async () => {
    await withArgv([process.execPath, 'dummy'], async () => {
      await jest.isolateModulesAsync(async () => {
        await import('./implementation-artifact-paths.gate.mjs');
      });
    });
    assert.ok(logs.some((l) => l.includes('PASS')));
    assert.equal(process.exitCode, 0);
  });
});