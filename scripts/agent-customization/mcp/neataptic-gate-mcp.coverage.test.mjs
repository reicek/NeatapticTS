/**
 * @module neataptic-gate-mcp.coverage.test
 * @description Additional coverage tests for neataptic-gate-mcp.mjs.
 *
 * Covers the remaining uncovered lines: 204-205 (run_gate_check args loop)
 * and 404 (return after process.exit in help path).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

const UTILS_PATH = './mcp-utils.mjs';

async function loadServer() {
  return import('./neataptic-gate-mcp.mjs');
}

describe('neataptic-gate-mcp remaining coverage', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    jest.resetModules();
    jest.clearAllMocks();
    process.exitCode = 0;
  });

  it('builds extra CLI args from the args object in run_gate_check', async () => {
    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: jest.fn((cmd, args) => ({
        stdout: JSON.stringify({ pass: true, evidence: 'ok', args }),
        stderr: '',
        status: 0,
      })),
    }));
    jest.resetModules();

    const { createGateTools } = await loadServer();
    const tools = createGateTools();
    const runGate = tools.find((tool) => tool.name === 'run_gate_check');

    const result = await runGate.handler({
      gate: 'slice-advancement',
      args: {
        'slice-id': 'C3-impl',
        'changed-files': 'src/foo.ts',
      },
    });

    assert.equal(result.pass, true);
    assert.ok(result.args.includes('--slice-id=C3-impl'));
    assert.ok(result.args.includes('--changed-files=src/foo.ts'));

    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: jest.fn(() => ({
        stdout: '{}',
        stderr: '',
        status: 0,
      })),
    }));
  });

  it('covers the return statement after process.exit in the help path', async () => {
    const { main } = await loadServer();
    const logSpy = jest
      .spyOn(console, 'log')
      .mockImplementation(() => undefined);
    const exitSpy = jest
      .spyOn(process, 'exit')
      .mockImplementation(() => undefined);
    try {
      const result = await main(['--help']);
      assert.equal(result, undefined);
      assert.ok(logSpy.mock.calls.length > 0);
      assert.equal(exitSpy.mock.calls[0]?.[0], 0);
    } finally {
      logSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });
});