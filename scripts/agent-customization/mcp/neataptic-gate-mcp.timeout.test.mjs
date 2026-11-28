/**
 * @module neataptic-gate-mcp.timeout.test
 * @description Contract tests for the gate facade's timeout handling.
 *
 * The defect these tests lock down: the facade killed a legitimate gate run
 * at its flat spawn ceiling, the child produced no JSON, and the caller was
 * told "did not return valid JSON" — which reads like a broken gate rather
 * than a budget-sizing failure. The facade must (a) size the spawn budget
 * per gate and (b) report a timeout as a timeout.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

import { GATE_TIMEOUT_BUDGETS } from '../gate-timeout-budgets.mjs';

async function loadServerWithSpawn(spawnSync) {
  jest.unstable_mockModule('node:child_process', () => ({
    spawn: jest.fn(),
    spawnSync,
  }));
  jest.resetModules();
  return import('./neataptic-gate-mcp.mjs');
}

describe('neataptic-gate-mcp timeout handling', () => {
  afterEach(() => {
    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: jest.fn(),
    }));
    jest.resetModules();
    jest.clearAllMocks();
    process.exitCode = 0;
  });

  it('spawns the gate with the per-gate budget instead of a flat ceiling', async () => {
    const spawnSync = jest.fn(() => ({
      stdout: JSON.stringify({ pass: true, evidence: {} }),
      stderr: '',
      status: 0,
    }));
    const { createGateTools } = await loadServerWithSpawn(spawnSync);
    const runGate = createGateTools().find(
      (tool) => tool.name === 'run_gate_check',
    );

    await runGate.handler({ gate: 'slice-advancement' });

    const spawnCall = spawnSync.mock.calls[0];
    assert.equal(spawnCall[2].timeout, GATE_TIMEOUT_BUDGETS['slice-advancement']);
    assert.ok(
      GATE_TIMEOUT_BUDGETS['slice-advancement'] >
        GATE_TIMEOUT_BUDGETS['plan-sync'],
      'the consolidated gate must get a larger budget than a cheap gate',
    );
  });

  it('reports a timeout as a timeout instead of invalid JSON', async () => {
    const spawnSync = jest.fn(() => ({
      stdout: '',
      stderr: '',
      status: null,
      signal: 'SIGTERM',
      killed: true,
      error: Object.assign(new Error('spawnSync ETIMEDOUT'), {
        code: 'ETIMEDOUT',
      }),
    }));
    const { createGateTools } = await loadServerWithSpawn(spawnSync);
    const runGate = createGateTools().find(
      (tool) => tool.name === 'run_gate_check',
    );

    await assert.rejects(
      () => runGate.handler({ gate: 'slice-advancement' }),
      (error) => {
        assert.match(error.message, /timed out/i);
        assert.match(
          error.message,
          new RegExp(String(GATE_TIMEOUT_BUDGETS['slice-advancement'])),
        );
        assert.doesNotMatch(error.message, /did not return valid JSON/);
        return true;
      },
    );
  });

  it('still reports invalid JSON for genuine non-JSON gate output', async () => {
    const spawnSync = jest.fn(() => ({
      stdout: 'not-valid-json',
      stderr: 'boom',
      status: 1,
      signal: null,
    }));
    const { createGateTools } = await loadServerWithSpawn(spawnSync);
    const runGate = createGateTools().find(
      (tool) => tool.name === 'run_gate_check',
    );

    await assert.rejects(
      () => runGate.handler({ gate: 'plan-sync' }),
      (error) => {
        assert.match(error.message, /did not return valid JSON/);
        assert.match(error.message, /exit status: 1/);
        assert.match(error.message, /stderr: boom/);
        assert.doesNotMatch(error.message, /timed out/i);
        return true;
      },
    );
  });

  it('passes its own self-check, which now verifies budget coverage', async () => {
    jest.unstable_unmockModule('node:child_process');
    jest.resetModules();
    const { spawn } = await import('node:child_process');
    const path = await import('node:path');

    const serverPath = path.resolve(
      'scripts',
      'agent-customization',
      'mcp',
      'neataptic-gate-mcp.mjs',
    );

    const output = await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [serverPath, '--self-check', '--json'], {
        cwd: path.resolve('.'),
      });
      let text = '';
      child.stdout?.on('data', (chunk) => {
        text += chunk.toString();
      });
      child.stderr?.on('data', (chunk) => {
        text += chunk.toString();
      });
      child.on('error', reject);
      child.on('close', () => resolve(text));
    });

    const parsed = JSON.parse(output);
    assert.equal(parsed.ok, true, output);
  });
});
