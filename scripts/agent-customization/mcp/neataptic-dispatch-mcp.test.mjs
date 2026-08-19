import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('../inventory-customizations.mjs', () => ({
  runCustomizationInventory: jest.fn(),
}));

jest.unstable_mockModule('./mcp-utils.mjs', () => ({
  createMcpServer: jest.fn((opts) => ({
    name: opts.serverName,
    version: opts.serverVersion,
    tools: opts.tools,
    dispatch: jest.fn(async (req) => {
      if (req.method === 'initialize') {
        return {
          protocolVersion: '2024-11-05',
          capabilities: {},
          serverInfo: { name: opts.serverName, version: opts.serverVersion },
        };
      }
      if (req.method === 'tools/list') {
        return {
          tools: opts.tools.map((t) => ({
            name: t.name,
            description: t.description,
            inputSchema: t.inputSchema,
          })),
        };
      }
      if (req.method === 'tools/call') {
        const tool = opts.tools.find((t) => t.name === req.params.name);
        if (!tool)
          return {
            isError: true,
            content: [{ type: 'text', text: 'unknown tool' }],
          };
        try {
          const result = await tool.handler(req.params.arguments || {});
          return {
            structuredContent: result,
            content: [{ type: 'text', text: JSON.stringify(result) }],
          };
        } catch (error) {
          return {
            isError: true,
            content: [{ type: 'text', text: error.message }],
          };
        }
      }
      return {};
    }),
  })),
  createSelfCheckReport: jest.fn((name, issues, meta) => ({
    name,
    ok: issues.length === 0,
    issues,
    meta,
  })),
  createTool: jest.fn((opts) => ({
    name: opts.name,
    description: opts.description,
    inputSchema: opts.inputSchema,
    handler: opts.handler,
    annotations: opts.annotations,
  })),
  emitSelfCheckReport: jest.fn(),
  invokeServerRequest: jest.fn(),
  MCP_PROTOCOL_VERSION: '2024-11-05',
  parseMcpCliArgs: jest.fn(),
  printMcpUsage: jest.fn(),
  requireString: jest.fn((val, name) => {
    if (typeof val !== 'string' || !val.trim())
      throw new Error(`${name} is required`);
    return val;
  }),
  runStdioMcpServer: jest.fn(),
  selfCheckError: jest.fn((scope, message) => ({ scope, message })),
}));

jest.unstable_mockModule('../dispatch/build-dispatch-packet.mjs', () => ({
  ALLOWED_CALLER_TIERS: [0, 1, 2, 3, 4],
  ALLOWED_EDGES: [{ from: 0, to: 1 }],
  buildDispatchPacket: jest.fn(),
  DEFAULT_COMPLEXITY: 'moderate',
  PROMPT_LENGTH_MAX: 500,
  PROMPT_LENGTH_MAX_TRIVIAL: 200,
  TIER_LABELS: { 0: 'Agent Zero', 1: 'Phase Orchestrator' },
}));

let mockInventory;
let mockMcpUtils;
let mockDispatch;

beforeEach(async () => {
  jest.resetModules();
  mockInventory = await import('../inventory-customizations.mjs');
  mockMcpUtils = await import('./mcp-utils.mjs');
  mockDispatch = await import('../dispatch/build-dispatch-packet.mjs');
  jest.clearAllMocks();
});

async function importModule(argv) {
  const origArgv = process.argv;
  const origExit = process.exit;
  process.argv = ['node', 'neataptic-dispatch-mcp.mjs', ...(argv || [])];
  process.exit = (code) => {
    throw new Error(`EXIT:${code}`);
  };
  try {
    await import('./neataptic-dispatch-mcp.mjs');
  } catch {
    // process.exit may throw
  }
  process.argv = origArgv;
  process.exit = origExit;
}

describe('neataptic-dispatch-mcp', () => {
  it('prints help and exits 0 when --help', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({ help: true });

    await importModule(['--help']);

    assert.ok(mockMcpUtils.printMcpUsage.mock.calls.length >= 1);
  });

  it('starts stdio server by default', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: false,
    });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();

    await importModule([]);

    assert.ok(mockMcpUtils.runStdioMcpServer.mock.calls.length >= 1);
  });

  it('runs self-check when --self-check', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: true,
    });
    mockMcpUtils.invokeServerRequest.mockImplementation(async (server, req) => {
      return server.dispatch(req);
    });
    mockInventory.runCustomizationInventory.mockResolvedValue({
      agents: [
        {
          name: '01-planning',
          tier: '1',
          model: 'test',
          skills: [],
          agents: [],
          tools: [],
          userInvocable: true,
          path: '.github/agents/01-planning.agent.md',
        },
        {
          name: 'plan-scout',
          tier: '3',
          model: null,
          skills: [],
          agents: [],
          tools: [],
          userInvocable: false,
          path: '.github/agents/plan-scout.agent.md',
        },
      ],
    });
    mockDispatch.buildDispatchPacket.mockImplementation((req, agents) => {
      if (req.target_agent === '__nonexistent-agent__') {
        return { ok: false, dispatch_allowed: false, reason: 'Unknown agent' };
      }
      if (req.caller_tier > 1 && req.target_agent === '01-planning') {
        return {
          ok: false,
          dispatch_allowed: false,
          reason: 'Upward delegation not allowed',
        };
      }
      if (req.prompt && req.prompt.length > mockDispatch.PROMPT_LENGTH_MAX) {
        return {
          ok: false,
          dispatch_allowed: false,
          reason: 'Prompt too long',
          prompt_length: req.prompt.length,
          prompt_length_max: mockDispatch.PROMPT_LENGTH_MAX,
        };
      }
      return { ok: true, dispatch_allowed: true };
    });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
  });

  it('self-check records every issue branch when all responses are bad', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: true,
    });
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: 'WRONG' }) // initialize -> 228
      .mockResolvedValueOnce({ tools: [] }) // tools/list -> 244 (count), 261 (names)
      .mockResolvedValueOnce({
        // list_dispatchable_agents -> 275, 297
        isError: true,
        structuredContent: { agents: [{ name: 'z' }, { name: 'a' }] },
      })
      .mockResolvedValueOnce({
        // valid build_dispatch_packet -> 320, 330
        isError: true,
        structuredContent: { ok: false, dispatch_allowed: false },
      })
      .mockResolvedValueOnce({
        // upward build_dispatch_packet -> 347, 361
        isError: true,
        structuredContent: { ok: true, dispatch_allowed: true },
      })
      .mockResolvedValueOnce({
        // unknown build_dispatch_packet -> 378, 392
        isError: true,
        structuredContent: { ok: true, dispatch_allowed: true },
      })
      .mockResolvedValueOnce({
        // get_dispatch_policy -> 406, 419, 431, 443
        isError: true,
        structuredContent: {},
      })
      .mockResolvedValueOnce({
        // overlong prompt -> 472
        structuredContent: { ok: true, dispatch_allowed: true },
      });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
    const report = mockMcpUtils.createSelfCheckReport.mock.calls[0];
    const issues = report[1];
    // Every issue branch should have fired at least once.
    assert.ok(
      issues.length >= 16,
      `expected >=16 issues, got ${issues.length}`,
    );
  });

  it('self-check reports issue when list_dispatchable_agents returns empty agents', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: true,
    });
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: '2024-11-05' }) // initialize ok
      .mockResolvedValueOnce({
        // tools/list ok (3 tools, correct names)
        tools: [
          { name: 'build_dispatch_packet' },
          { name: 'get_dispatch_policy' },
          { name: 'list_dispatchable_agents' },
        ],
      })
      .mockResolvedValueOnce({
        // list_dispatchable_agents -> 285 (no agents array)
        isError: false,
        structuredContent: {},
      })
      .mockResolvedValueOnce({
        // valid dispatch ok
        structuredContent: { ok: true, dispatch_allowed: true },
      })
      .mockResolvedValueOnce({
        // upward dispatch ok
        structuredContent: { ok: false, dispatch_allowed: false },
      })
      .mockResolvedValueOnce({
        // unknown dispatch ok
        structuredContent: { ok: false, dispatch_allowed: false },
      })
      .mockResolvedValueOnce({
        // get_dispatch_policy ok
        structuredContent: {
          allowed_edges: [{ from: 0, to: 1 }],
          prompt_length_max: 500,
          prompt_length_max_trivial: 200,
        },
      })
      .mockResolvedValueOnce({
        // overlong prompt ok
        structuredContent: { ok: true, dispatch_allowed: true },
      });

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
    const report = mockMcpUtils.createSelfCheckReport.mock.calls[0];
    const issues = report[1];
    assert.ok(
      issues.some((i) =>
        /did not return a non-empty agent list/.test(i.message),
      ),
    );
  });

  it('list_dispatchable_agents handler returns sorted agents', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: false,
    });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockInventory.runCustomizationInventory.mockResolvedValue({
      agents: [
        {
          name: 'z-agent',
          tier: '3',
          skills: ['s1'],
          agents: ['a1'],
          tools: ['t1'],
          userInvocable: false,
          path: 'z.md',
        },
        {
          name: 'a-agent',
          tier: '1',
          model: 'm',
          skills: [],
          agents: [],
          tools: [],
          userInvocable: true,
          path: 'a.md',
        },
      ],
    });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const listTool = server.tools.find(
      (t) => t.name === 'list_dispatchable_agents',
    );
    const result = await listTool.handler({});
    assert.strictEqual(result.agents[0].name, 'a-agent');
    assert.strictEqual(result.agents[1].name, 'z-agent');
    assert.strictEqual(result.total, 2);
    assert.deepStrictEqual(result.tiers, { 1: 1, 3: 1 });
  });

  it('build_dispatch_packet handler delegates to buildDispatchPacket', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: false,
    });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockInventory.runCustomizationInventory.mockResolvedValue({ agents: [] });
    mockDispatch.buildDispatchPacket.mockReturnValue({
      ok: true,
      dispatch_allowed: true,
    });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const buildTool = server.tools.find(
      (t) => t.name === 'build_dispatch_packet',
    );
    const result = await buildTool.handler({
      target_agent: 'test',
      caller_tier: 1,
      prompt: 'hello',
    });
    assert.strictEqual(result.ok, true);
    assert.ok(mockDispatch.buildDispatchPacket.mock.calls.length >= 1);
  });

  it('get_dispatch_policy handler returns policy info', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: false,
    });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const policyTool = server.tools.find(
      (t) => t.name === 'get_dispatch_policy',
    );
    const result = await policyTool.handler({});
    assert.ok(Array.isArray(result.allowed_edges));
    assert.ok(typeof result.prompt_length_max === 'number');
  });

  it('list_dispatchable_agents handles missing agents property', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: false,
    });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockInventory.runCustomizationInventory.mockResolvedValue({});

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const listTool = server.tools.find(
      (t) => t.name === 'list_dispatchable_agents',
    );
    const result = await listTool.handler({});
    assert.deepStrictEqual(result.agents, []);
    assert.strictEqual(result.total, 0);
  });

  it('build_dispatch_packet handles missing agents property', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: false,
    });
    mockMcpUtils.runStdioMcpServer.mockResolvedValue();
    mockInventory.runCustomizationInventory.mockResolvedValue({});
    mockDispatch.buildDispatchPacket.mockReturnValue({
      ok: true,
      dispatch_allowed: true,
    });

    await importModule([]);

    const server = mockMcpUtils.createMcpServer.mock.calls[0][0];
    const buildTool = server.tools.find(
      (t) => t.name === 'build_dispatch_packet',
    );
    const result = await buildTool.handler({
      target_agent: 'test',
      caller_tier: 1,
      prompt: 'hello',
    });
    assert.strictEqual(result.ok, true);
  });

  it('self-check covers ?? {} branches when responses lack structuredContent', async () => {
    mockMcpUtils.parseMcpCliArgs.mockReturnValue({
      help: false,
      selfCheck: true,
    });
    mockInventory.runCustomizationInventory.mockResolvedValue({ agents: [] });
    mockMcpUtils.invokeServerRequest
      .mockResolvedValueOnce({ protocolVersion: '2024-11-05' }) // initialize ok
      .mockResolvedValueOnce({}) // tools/list: no tools array
      .mockResolvedValueOnce({}) // list_dispatchable_agents: no structuredContent
      .mockResolvedValueOnce({}) // valid build_dispatch_packet: no structuredContent
      .mockResolvedValueOnce({}) // upward build_dispatch_packet: no structuredContent
      .mockResolvedValueOnce({}) // unknown build_dispatch_packet: no structuredContent
      .mockResolvedValueOnce({}) // get_dispatch_policy: no structuredContent
      .mockResolvedValueOnce({}); // overlong prompt: no structuredContent

    await importModule(['--self-check']);

    assert.ok(mockMcpUtils.emitSelfCheckReport.mock.calls.length >= 1);
    const report = mockMcpUtils.createSelfCheckReport.mock.calls[0];
    const issues = report[1];
    assert.ok(
      issues.length >= 10,
      `expected >=10 issues, got ${issues.length}`,
    );
  });
});
