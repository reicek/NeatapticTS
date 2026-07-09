/**
 * @module neataptic-gate-mcp.test
 * @description Native-ESM coverage tests for neataptic-gate-mcp.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the gate MCP server.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const SERVER_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
);
const UTILS_PATH = './mcp-utils.mjs';

async function loadUtils() {
  return import(UTILS_PATH);
}

async function loadServer() {
  return import('./neataptic-gate-mcp.mjs');
}

async function buildMockServer(overrides = {}) {
  const [{ createGateTools }, { MCP_PROTOCOL_VERSION }] = await Promise.all([
    loadServer(),
    loadUtils(),
  ]);
  const tools = createGateTools();
  return {
    tools,
    async dispatch(request) {
      const override = overrides[request.method];
      if (override) {
        return override(request);
      }
      if (request.method === 'initialize') {
        return { protocolVersion: MCP_PROTOCOL_VERSION };
      }
      if (request.method === 'tools/list') {
        return { tools };
      }
      if (request.method === 'tools/call') {
        const name = request.params?.name;
        if (name === 'run_gate_check') {
          return { isError: false, structuredContent: { pass: true } };
        }
        if (name === 'query_tier_graph') {
          return {
            isError: false,
            structuredContent: { summary: { total: 1 } },
          };
        }
        if (name === 'query_customization_routing_table') {
          return {
            isError: false,
            structuredContent: {
              summary: { agents: 1 },
              freshness: { pass: true },
            },
          };
        }
      }
      throw new Error(`Unexpected mock method: ${request.method}`);
    },
  };
}

describe('neataptic-gate-mcp native-ESM coverage', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    jest.resetModules();
    jest.clearAllMocks();
    process.exitCode = 0;
  });

  it('exposes createGateTools with the expected tool surface', async () => {
    const { createGateTools } = await loadServer();
    const tools = createGateTools();
    const names = tools.map((tool) => tool.name).sort();
    assert.deepEqual(names, [
      'list_gates',
      'query_customization_routing_table',
      'query_tier_graph',
      'run_gate_check',
    ]);
  });

  it('passes a self-check against an in-process server', async () => {
    const { createGateTools, runGateSelfCheck } = await loadServer();
    const { createMcpServer } = await loadUtils();
    const server = createMcpServer({
      serverName: 'test-gate-mcp',
      serverVersion: '0.0.0',
      tools: createGateTools(),
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, true);
    assert.equal(report.name, 'gate-mcp self-check');
  });

  it('returns gate metadata from the list_gates tool handler', async () => {
    const { createGateTools } = await loadServer();
    const tools = createGateTools();
    const listGates = tools.find((tool) => tool.name === 'list_gates');
    const result = await listGates.handler();
    assert.ok(result.gates.some((gate) => gate.id === 'plan-sync'));
  });

  it('throws when run_gate_check receives non-JSON gate output', async () => {
    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: jest.fn(() => ({
        stdout: 'not-valid-json',
        stderr: 'boom',
        status: 0,
      })),
    }));
    jest.resetModules();

    const { createGateTools } = await loadServer();
    const tools = createGateTools();
    const runGate = tools.find((tool) => tool.name === 'run_gate_check');
    await assert.rejects(
      runGate.handler({ gate: 'learning-event' }),
      /did not return valid JSON/,
    );

    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: () => {
        throw new Error('restore real child_process in next test');
      },
    }));
  });

  it('throws with a fallback message when run_gate_check has no stderr', async () => {
    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: jest.fn(() => ({
        stdout: 'not-valid-json',
        stderr: undefined,
        status: 0,
      })),
    }));
    jest.resetModules();

    const { createGateTools } = await loadServer();
    const tools = createGateTools();
    const runGate = tools.find((tool) => tool.name === 'run_gate_check');
    await assert.rejects(
      runGate.handler({ gate: 'learning-event' }),
      /stderr: \(empty\)/,
    );

    jest.unstable_mockModule('node:child_process', () => ({
      spawn: jest.fn(),
      spawnSync: () => {
        throw new Error('restore real child_process in next test');
      },
    }));
  });

  it('reports a self-check issue when the protocol version mismatches', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      initialize: () => ({ protocolVersion: 'wrong-version' }),
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when the tool list is not an array', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/list': () => ({ tools: 'not-an-array' }),
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when the tool list array has the wrong length', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/list': () => ({ tools: [] }),
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when run_gate_check returns an error', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'run_gate_check') {
          return { isError: true };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when the gate result lacks a pass boolean', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'run_gate_check') {
          return { isError: false, structuredContent: {} };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when query_tier_graph returns an error', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'query_tier_graph') {
          return { isError: true };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when query_tier_graph has no valid total', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'query_tier_graph') {
          return {
            isError: false,
            structuredContent: { summary: { total: 0 } },
          };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when query_customization_routing_table returns an error', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'query_customization_routing_table') {
          return { isError: true };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when the routing table lacks a valid agent count', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'query_customization_routing_table') {
          return {
            isError: false,
            structuredContent: {
              summary: { agents: 0 },
              freshness: { pass: true },
            },
          };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('reports a self-check issue when the routing table freshness is missing', async () => {
    const { runGateSelfCheck } = await loadServer();
    const server = await buildMockServer({
      'tools/call': (request) => {
        if (request.params?.name === 'query_customization_routing_table') {
          return {
            isError: false,
            structuredContent: { summary: { agents: 1 } },
          };
        }
        return {
          isError: false,
          structuredContent: {
            pass: true,
            summary: { total: 1 },
            agents: 1,
            freshness: { pass: true },
          },
        };
      },
    });
    const report = await runGateSelfCheck({ server });
    assert.equal(report.ok, false);
  });

  it('prints MCP usage and exits when main() is called with --help', async () => {
    const { main } = await loadServer();
    const logSpy = jest
      .spyOn(console, 'log')
      .mockImplementation(() => undefined);
    const exitSpy = jest
      .spyOn(process, 'exit')
      .mockImplementation(() => undefined);
    try {
      await main(['--help']);
      assert.ok(logSpy.mock.calls.length > 0);
      assert.equal(exitSpy.mock.calls[0]?.[0], 0);
    } finally {
      logSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });

  it('runs a self-check through main() when --self-check is passed', async () => {
    const { main } = await loadServer();
    const selfCheckReport = { ok: true, name: 'gate-mcp self-check' };
    const runSelfCheckMock = jest.fn().mockResolvedValue(selfCheckReport);
    const originalExitCode = process.exitCode;
    try {
      await main(['--self-check'], { runSelfCheck: runSelfCheckMock });
      assert.equal(process.exitCode, 0);
      assert.equal(runSelfCheckMock.mock.calls.length, 1);
    } finally {
      process.exitCode = originalExitCode;
    }
  });

  it('sets a non-zero exit code when the self-check fails', async () => {
    const { main } = await loadServer();
    const runSelfCheckMock = jest.fn().mockResolvedValue({ ok: false });
    const originalExitCode = process.exitCode;
    try {
      await main(['--self-check'], { runSelfCheck: runSelfCheckMock });
      assert.equal(process.exitCode, 1);
    } finally {
      process.exitCode = originalExitCode;
    }
  });

  it('starts the stdio server through main() by default', async () => {
    const actualUtils = await loadUtils();
    const runStdioMock = jest.fn().mockResolvedValue(undefined);
    jest.unstable_mockModule(UTILS_PATH, () => ({
      ...actualUtils,
      runStdioMcpServer: runStdioMock,
    }));
    jest.resetModules();

    const { main } = await loadServer();
    await main([]);
    assert.equal(runStdioMock.mock.calls.length, 1);
  });

  it('catches a rejected main() in bootstrapMain()', async () => {
    const actualUtils = await loadUtils();
    const error = new Error('bootstrap failure');
    jest.unstable_mockModule(UTILS_PATH, () => ({
      ...actualUtils,
      runStdioMcpServer: jest.fn().mockRejectedValue(error),
    }));
    jest.resetModules();

    const { bootstrapMain } = await loadServer();
    const errorSpy = jest
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const originalExitCode = process.exitCode;
    try {
      process.exitCode = undefined;
      bootstrapMain();
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(process.exitCode, 1);
      assert.equal(errorSpy.mock.calls[0]?.[0], error);
    } finally {
      errorSpy.mockRestore();
      process.exitCode = originalExitCode ?? 0;
    }
  });

  it('resolves bootstrapMain without setting an exit code', async () => {
    const actualUtils = await loadUtils();
    jest.unstable_mockModule(UTILS_PATH, () => ({
      ...actualUtils,
      runStdioMcpServer: jest.fn().mockResolvedValue(undefined),
    }));
    jest.resetModules();

    const { bootstrapMain } = await loadServer();
    const errorSpy = jest
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const originalExitCode = process.exitCode;
    try {
      process.exitCode = undefined;
      bootstrapMain();
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal(process.exitCode, undefined);
      assert.equal(errorSpy.mock.calls.length, 0);
    } finally {
      errorSpy.mockRestore();
      process.exitCode = originalExitCode ?? 0;
    }
  });
});
