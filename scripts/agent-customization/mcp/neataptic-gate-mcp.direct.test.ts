import path from 'node:path';
/**
 * @module neataptic-gate-mcp.direct.test
 * @description Direct-import coverage tests for neataptic-gate-mcp.mjs.
 *
 * The server entry is also exercised end-to-end via spawn in
 * neataptic-gate-mcp.test.ts. These tests import the exported helpers so Jest
 * can instrument the source file.
 */
const SERVER_PATH = './neataptic-gate-mcp.mjs';
const UTILS_PATH = './mcp-utils.mjs';

type MockRequest = { method: string; params?: Record<string, unknown> };

/**
 * Build a fake MCP server for self-check error-branch testing.
 *
 * @param overrides - Method-specific dispatch overrides.
 * @returns A fake server with the same tool list as the real one.
 */
async function buildMockServer(
  overrides: Record<string, (request: MockRequest) => unknown> = {},
) {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  const [{ createGateTools }, { MCP_PROTOCOL_VERSION }] = await Promise.all([
    import(SERVER_PATH),
    import(UTILS_PATH),
  ]);
  const tools = createGateTools();
  return {
    tools,
    async dispatch(request: {
      method: string;
      params?: Record<string, unknown>;
    }) {
      const override = overrides[request.method];
      if (override) {
        return override(request);
      }
      // Default happy-path responses.
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

describe('neataptic-gate-mcp direct imports', () => {
  it('exposes createGateTools with the expected tool surface', async () => {
    const { createGateTools } = await import(SERVER_PATH);
    const tools = createGateTools();
    const names = tools.map((tool: { name: string }) => tool.name).toSorted();
    expect(names).toEqual([
      'get_slice_context',
      'list_gates',
      'query_customization_routing_table',
      'query_tier_graph',
      'run_gate_check',
    ]);
  });

  it('get_slice_context input schema in the gate MCP does not advertise a full option', async () => {
    const { createGateTools } = await import(SERVER_PATH);
    const tools = createGateTools();
    const tool = tools.find(
      (t: { name: string; inputSchema: Record<string, unknown> }) =>
        t.name === 'get_slice_context',
    );

    expect(tool?.inputSchema.properties).not.toHaveProperty('full');
    expect(tool?.inputSchema).toEqual(
      expect.objectContaining({
        required: ['slice_id'],
      }),
    );
  });

  it('get_slice_context returns compact through gate tool', async () => {
    const [{ writeFile, unlink }, { MCP_REPO_ROOT }] = await Promise.all([
      import('node:fs/promises'),
      import(UTILS_PATH),
    ]);
    const fileName = `__gate-direct-test-${Date.now()}-${Math.random()
      .toString(36)
      .slice(2)}.plans.md`;
    const absolutePath = path.join(MCP_REPO_ROOT, 'plans', fileName);
    await writeFile(
      absolutePath,
      `
# Test Plan

## Implementation phases

### Phase 1 — Test [WIP]

#### Step 01: Active step [WIP]

\`\`\`yaml
phase: 1
step: 1
title: '01: Active step'
slices:
  - slice_id: compact-gate-slice
    title: Compact gate slice
    status: '[WIP]'
    goal: compact gate goal
\`\`\`

## Validation gates

- none
`.trim(),
      'utf8',
    );
    try {
      const { createGateTools } = await import(SERVER_PATH);
      const tools = createGateTools();
      const tool = tools.find(
        (t: {
          name: string;
          handler: (args: Record<string, unknown>) => unknown;
        }) => t.name === 'get_slice_context',
      );
      const result = (await tool?.handler({
        slice_id: 'compact-gate-slice',
        plan_path: absolutePath,
      })) as Record<string, unknown>;
      expect(result.compact).toBe(true);
      expect(result).not.toHaveProperty('stepPacket');
      expect(result).not.toHaveProperty('sourceChunks');
      expect(result).toHaveProperty('title', 'Compact gate slice');
      expect(result).toHaveProperty('goal', 'compact gate goal');
    } finally {
      await unlink(absolutePath).catch(() => undefined);
    }
  });

  it('passes a self-check against an in-process server', async () => {
    const { createGateTools, runGateSelfCheck } = await import(SERVER_PATH);
    const { createMcpServer } = await import(UTILS_PATH);
    const server = createMcpServer({
      serverName: 'test-gate-mcp',
      serverVersion: '0.0.0',
      tools: createGateTools(),
    });
    const report = await runGateSelfCheck({ server });
    expect(report.ok).toBe(true);
    expect(report.name).toBe('gate-mcp self-check');
  });

  it('returns gate metadata from the list_gates tool handler', async () => {
    const { createGateTools } = await import(SERVER_PATH);
    const tools = createGateTools();
    const listGates = tools.find(
      (tool: { name: string }) => tool.name === 'list_gates',
    );
    const result = await listGates.handler();
    expect(
      result.gates.some((gate: { id: string }) => gate.id === 'plan-sync'),
    ).toBe(true);
  });

  it('throws when run_gate_check receives non-JSON gate output', () => {
    jest.resetModules();
    jest.doMock('node:child_process', () => ({
      spawnSync: jest.fn(() => ({
        stdout: 'not-valid-json',
        stderr: 'boom',
        status: 0,
      })),
    }));

    return jest.isolateModulesAsync(async () => {
      // @ts-ignore - tested module is authored in plain ESM without a declaration file.
      const { createGateTools } = await import(SERVER_PATH);
      const tools = createGateTools();
      const runGate = tools.find(
        (tool: { name: string }) => tool.name === 'run_gate_check',
      );
      await expect(runGate.handler({ gate: 'learning-event' })).rejects.toThrow(
        /did not return valid JSON/,
      );
    });

    // Remove the child_process mock so later tests exercise the real spawn path.
    jest.unmock('node:child_process');
    jest.resetModules();
  });

  it('reports a self-check issue when the protocol version mismatches', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      initialize: () => ({ protocolVersion: 'wrong-version' }),
    });
    const report = await runGateSelfCheck({ server });
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when the tool list is not an array', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/list': () => ({ tools: 'not-an-array' }),
    });
    const report = await runGateSelfCheck({ server });
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when the tool list array has the wrong length', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/list': () => ({ tools: [] }),
    });
    const report = await runGateSelfCheck({ server });
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when run_gate_check returns an error', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when the gate result lacks a pass boolean', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when query_tier_graph returns an error', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when query_tier_graph has no valid total', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when query_customization_routing_table returns an error', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when the routing table lacks a valid agent count', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('reports a self-check issue when the routing table freshness is missing', async () => {
    const { runGateSelfCheck } = await import(SERVER_PATH);
    const server = await buildMockServer({
      'tools/call': (request: { params?: { name?: string } }) => {
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
    expect(report.ok).toBe(false);
  });

  it('prints MCP usage and exits when main() is called with --help', async () => {
    jest.resetModules();
    // @ts-ignore - tested module is authored in plain ESM without a declaration file.
    const { main } = await import(SERVER_PATH);
    const logSpy = jest
      .spyOn(console, 'log')
      .mockImplementation(() => undefined);
    const exitSentinel = new Error('process.exit(0) mock');
    const exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {
      throw exitSentinel;
    });
    try {
      await expect(main(['--help'])).rejects.toBe(exitSentinel);
      expect(logSpy).toHaveBeenCalled();
      expect(exitSpy).toHaveBeenCalledWith(0);
    } finally {
      logSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });

  it('runs a self-check through main() when --self-check is passed', async () => {
    jest.resetModules();
    // @ts-ignore - tested module is authored in plain ESM without a declaration file.
    const { main } = await import(SERVER_PATH);
    const selfCheckReport = { ok: true, name: 'gate-mcp self-check' };
    const runSelfCheckMock = jest.fn().mockResolvedValue(selfCheckReport);
    try {
      await main(['--self-check'], { runSelfCheck: runSelfCheckMock });
      expect(process.exitCode).toBe(0);
      expect(runSelfCheckMock).toHaveBeenCalled();
    } finally {
      process.exitCode = 0;
    }
  });

  it('sets a non-zero exit code when the self-check fails', async () => {
    jest.resetModules();
    // @ts-ignore - tested module is authored in plain ESM without a declaration file.
    const { main } = await import(SERVER_PATH);
    const runSelfCheckMock = jest.fn().mockResolvedValue({ ok: false });
    try {
      await main(['--self-check'], { runSelfCheck: runSelfCheckMock });
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = 0;
    }
  });

  it('starts the stdio server through main() by default', async () => {
    jest.resetModules();
    const runStdioMock = jest.fn().mockResolvedValue(undefined);
    jest.doMock(UTILS_PATH, () => {
      // @ts-ignore - dynamic mock of ESM module.
      const actual = jest.requireActual(UTILS_PATH);
      return {
        ...actual,
        runStdioMcpServer: runStdioMock,
      };
    });

    await jest.isolateModulesAsync(async () => {
      // @ts-ignore - tested module is authored in plain ESM without a declaration file.
      const { main } = await import(SERVER_PATH);
      await main([]);
      expect(runStdioMock).toHaveBeenCalled();
    });
  });

  it('catches a rejected main() in bootstrapMain()', async () => {
    jest.resetModules();
    const error = new Error('bootstrap failure');
    jest.doMock(UTILS_PATH, () => {
      // @ts-ignore - dynamic mock of ESM module.
      const actual = jest.requireActual(UTILS_PATH);
      return {
        ...actual,
        runStdioMcpServer: jest.fn().mockRejectedValue(error),
      };
    });

    await jest.isolateModulesAsync(async () => {
      // @ts-ignore - tested module is authored in plain ESM without a declaration file.
      const { bootstrapMain } = await import(SERVER_PATH);
      const errorSpy = jest
        .spyOn(console, 'error')
        .mockImplementation(() => undefined);
      try {
        bootstrapMain();
        // Allow the microtask queue to flush so the rejection handler runs.
        await new Promise((resolve) => setImmediate(resolve));
        expect(process.exitCode).toBe(1);
        expect(errorSpy).toHaveBeenCalledWith(error);
      } finally {
        errorSpy.mockRestore();
        process.exitCode = 0;
      }
    });
  });

  it('resolves bootstrapMain without setting an exit code', async () => {
    jest.resetModules();
    jest.doMock(UTILS_PATH, () => {
      // @ts-ignore - dynamic mock of ESM module.
      const actual = jest.requireActual(UTILS_PATH);
      return {
        ...actual,
        runStdioMcpServer: jest.fn().mockResolvedValue(undefined),
      };
    });

    await jest.isolateModulesAsync(async () => {
      // @ts-ignore - tested module is authored in plain ESM without a declaration file.
      const { bootstrapMain } = await import(SERVER_PATH);
      const errorSpy = jest
        .spyOn(console, 'error')
        .mockImplementation(() => undefined);
      try {
        process.exitCode = undefined;
        bootstrapMain();
        await new Promise((resolve) => setImmediate(resolve));
        expect(process.exitCode).toBeUndefined();
        expect(errorSpy).not.toHaveBeenCalled();
      } finally {
        errorSpy.mockRestore();
        process.exitCode = 0;
      }
    });
  });
});
