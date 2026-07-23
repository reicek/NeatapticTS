import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

const REPO_ROOT = path.resolve(__dirname, '../../../');

const GATE_SERVER_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/mcp/neataptic-gate-mcp.mjs',
);

const FACADE_SERVER_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/mcp/cortex-facade.mjs',
);

const TIER_TOOL_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/mcp/cortex-tier-tool.mjs',
);

const AGENT_DIR = path.resolve(REPO_ROOT, '.github/agents');

const TIER_1_AGENT_FILES = [
  '00-helping.agent.md',
  '01-planning.agent.md',
  '02-researching.agent.md',
  '03-red-testing.agent.md',
  '04-implementing.agent.md',
  '05-green-testing.agent.md',
  '06-documenting.agent.md',
  '07-logging.agent.md',
];

/**
 * JSON-RPC request shape for MCP over stdio.
 */
interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: Record<string, unknown>;
}

/**
 * JSON-RPC response shape for MCP over stdio.
 */
interface JsonRpcResponse {
  jsonrpc: '2.0';
  id?: number;
  result?: unknown;
  error?: { code: number; message: string };
}

/**
 * Minimal MCP tool descriptor.
 */
interface McpToolDescriptor {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
  handler: (
    argumentsObject: Record<string, unknown>,
  ) => Promise<unknown> | unknown;
}

/**
 * Payload returned by a successful `tools/list` MCP call.
 */
interface ToolListPayload {
  tools?: McpToolDescriptor[];
}

/**
 * Module instance returned by dynamically importing `cortex-tier-tool.mjs`.
 */
interface CortexTierToolModule {
  createSliceContextTool?: () => McpToolDescriptor;
  createTierGraphTool?: () => McpToolDescriptor;
}

/**
 * Shape returned by the `query_tier_graph` tool handler.
 */
interface TierGraphResult {
  agents: unknown[];
  violations: unknown[];
  validation: { issueCount: number };
}

/**
 * Parse result returned by `customization-utils.mjs`.
 */
interface FrontmatterParseResult {
  data: Record<string, unknown>;
}

/**
 * Spawn an MCP server over stdio and exchange JSON-RPC messages.
 *
 * Sends a standard initialize handshake followed by the caller's requests.
 * Resolves once the expected number of responses (initialize + caller requests)
 * have arrived, or after the timeout expires.
 *
 * @param serverPath - Absolute path to the server entry point.
 * @param argv - Additional CLI arguments for the server.
 * @param requests - JSON-RPC requests to send after the initialize handshake.
 * @param timeoutMs - Maximum time to wait for responses.
 * @returns The responses to the caller's requests (initialize response is dropped).
 */
async function runMcpSession(
  serverPath: string,
  argv: string[] = [],
  requests: JsonRpcRequest[] = [],
  timeoutMs = 10000,
): Promise<JsonRpcResponse[]> {
  const child = spawn(process.execPath, [serverPath, ...argv], {
    cwd: REPO_ROOT,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  const allResponses: JsonRpcResponse[] = [];
  const stdoutBuffer: string[] = [];
  const expectedResponseCount = requests.length + 1;

  return new Promise<JsonRpcResponse[]>((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill();
      reject(
        new Error(
          `Timed out after ${timeoutMs}ms waiting for ${expectedResponseCount} responses`,
        ),
      );
    }, timeoutMs);

    child.stdout?.on('data', (chunk: Buffer) => {
      stdoutBuffer.push(chunk.toString('utf8'));
      const lines = stdoutBuffer.join('').split('\n');
      stdoutBuffer.length = 0;
      const trailing = lines.pop();
      if (trailing !== undefined && trailing !== '') {
        stdoutBuffer.push(trailing);
      }

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          const parsed: JsonRpcResponse = JSON.parse(trimmed);
          allResponses.push(parsed);
        } catch {
          // Ignore non-JSON diagnostics written to stdout.
        }
      }

      if (allResponses.length >= expectedResponseCount) {
        clearTimeout(timeout);
        child.stdin?.end();
        resolve(allResponses.slice(1));
      }
    });

    child.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on('close', () => {
      clearTimeout(timeout);
      resolve(allResponses.slice(1));
    });

    child.stdin?.write(
      JSON.stringify({
        jsonrpc: '2.0',
        id: 0,
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'cortex-tier-tool-test', version: '0.1.0' },
        },
      }) + '\n',
    );

    for (const request of requests) {
      child.stdin?.write(JSON.stringify(request) + '\n');
    }
  });
}

/**
 * Extract the sorted list of tool names from a `tools/list` response.
 *
 * @param response - JSON-RPC response containing the tool list.
 * @returns Sorted array of tool names.
 */
function extractToolNames(response: JsonRpcResponse | undefined): string[] {
  if (response?.error || !response?.result) {
    return [];
  }

  const payload = response.result as ToolListPayload;
  return (payload.tools ?? []).map((tool) => tool.name).toSorted();
}

/**
 * Parse the frontmatter of a Tier-1 agent markdown file.
 *
 * @param fileName - Agent markdown file name.
 * @returns Parsed frontmatter data object.
 */
async function loadAgentFrontmatter(
  fileName: string,
): Promise<FrontmatterParseResult> {
  const filePath = path.join(AGENT_DIR, fileName);
  const raw = await readFile(filePath, 'utf8');
  const utils = await import(
    path.resolve(
      REPO_ROOT,
      'scripts/agent-customization/customization-utils.mjs',
    )
  );
  const parsed = utils.parseFrontmatter(raw, `.github/agents/${fileName}`);
  return { data: parsed.data as Record<string, unknown> };
}

/**
 * Determine whether an agent's frontmatter explicitly grants access to
 * `get_slice_context`.
 *
 * The contract under test requires the namespaced tool name to be present in
 * the agent's `tools` list. Wildcard access is intentionally not accepted for
 * this explicit-inclusion check.
 *
 * @param data - Parsed frontmatter data.
 * @returns True when the agent lists `neataptic-workflow-mcp/get_slice_context`.
 */
function agentExplicitlyListsSliceContext(
  data: Record<string, unknown>,
): boolean {
  const tools = Array.isArray(data.tools) ? data.tools : [];
  return tools.includes('neataptic-workflow-mcp/get_slice_context');
}

describe('cortex-tier-tool get_slice_context red contracts', () => {
  it('exports a createSliceContextTool factory from cortex-tier-tool.mjs', async () => {
    const mod = (await import(
      TIER_TOOL_PATH
    )) as unknown as CortexTierToolModule;
    expect(mod.createSliceContextTool).toEqual(expect.any(Function));
  });

  it('createSliceContextTool builds a tool descriptor named get_slice_context', async () => {
    const mod = (await import(
      TIER_TOOL_PATH
    )) as unknown as CortexTierToolModule;
    const factory = mod.createSliceContextTool;
    if (!factory) {
      throw new Error('createSliceContextTool is not exported');
    }
    const tool = factory();
    expect(tool.name).toBe('get_slice_context');
  });

  it('lists get_slice_context in the gate MCP server tool catalog', async () => {
    const [response] = await runMcpSession(
      GATE_SERVER_PATH,
      [],
      [{ jsonrpc: '2.0', id: 1, method: 'tools/list' }],
    );
    const names = extractToolNames(response);
    expect(names).toContain('get_slice_context');
  });

  it('lists get_slice_context in the lazy cortex facade tool catalog', async () => {
    const [response] = await runMcpSession(
      FACADE_SERVER_PATH,
      [],
      [{ jsonrpc: '2.0', id: 1, method: 'tools/list' }],
    );
    const names = extractToolNames(response);
    expect(names).toContain('get_slice_context');
  });
});

describe('Tier-1 agent get_slice_context tool-list inclusion', () => {
  it('explicitly lists get_slice_context in every Tier-1 agent frontmatter', async () => {
    const results = await Promise.all(
      TIER_1_AGENT_FILES.map(async (fileName) => {
        const { data } = await loadAgentFrontmatter(fileName);
        return {
          agent: fileName,
          listed: agentExplicitlyListsSliceContext(data),
        };
      }),
    );

    const missing = results
      .filter((entry) => !entry.listed)
      .map((entry) => entry.agent)
      .toSorted();

    expect(missing).toEqual([]);
  });
});

describe('cortex-tier-tool query_tier_graph coverage', () => {
  const fakeInventory = {
    generated_at: '2026-07-20T12:00:00Z',
    summary: { total: 3 },
    agents: [{ name: 'a' }],
    issues: [{ id: 1 }],
  };
  const fakeValidation = {
    ok: true,
    issues: [{ id: 1 }],
  };

  it('omits violations when includeViolations is false', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../validate-agent-graph.mjs', () => ({
        collectTierInventory: jest.fn().mockResolvedValue(fakeInventory),
        runValidateAgentGraph: jest.fn().mockResolvedValue(fakeValidation),
      }));

      const mod = (await import(
        TIER_TOOL_PATH
      )) as unknown as CortexTierToolModule;
      const factory = mod.createTierGraphTool;
      if (!factory) {
        throw new Error('createTierGraphTool is not exported');
      }

      const tool = factory();
      const result = (await tool.handler({
        includeViolations: false,
      })) as TierGraphResult;

      expect({
        hasAgents: result.agents.length === 1,
        violations: result.violations,
        issueCount: result.validation.issueCount,
      }).toEqual({
        hasAgents: true,
        violations: [],
        issueCount: 1,
      });
    });
  });

  it('omits agents when includeAgents is false', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../validate-agent-graph.mjs', () => ({
        collectTierInventory: jest.fn().mockResolvedValue(fakeInventory),
        runValidateAgentGraph: jest.fn().mockResolvedValue(fakeValidation),
      }));

      const mod = (await import(
        TIER_TOOL_PATH
      )) as unknown as CortexTierToolModule;
      const factory = mod.createTierGraphTool;
      if (!factory) {
        throw new Error('createTierGraphTool is not exported');
      }

      const tool = factory();
      const result = (await tool.handler({
        includeAgents: false,
      })) as TierGraphResult;

      expect({
        agents: result.agents,
        violations: result.violations,
        issueCount: result.validation.issueCount,
      }).toEqual({
        agents: [],
        violations: fakeValidation.issues,
        issueCount: 1,
      });
    });
  });

  it('includes violations when includeViolations is true', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../validate-agent-graph.mjs', () => ({
        collectTierInventory: jest.fn().mockResolvedValue(fakeInventory),
        runValidateAgentGraph: jest.fn().mockResolvedValue(fakeValidation),
      }));

      const mod = (await import(
        TIER_TOOL_PATH
      )) as unknown as CortexTierToolModule;
      const factory = mod.createTierGraphTool;
      if (!factory) {
        throw new Error('createTierGraphTool is not exported');
      }

      const tool = factory();
      const result = (await tool.handler({
        includeViolations: true,
      })) as TierGraphResult;

      expect({
        agents: result.agents,
        violations: result.violations,
        issueCount: result.validation.issueCount,
      }).toEqual({
        agents: fakeInventory.agents,
        violations: fakeValidation.issues,
        issueCount: 1,
      });
    });
  });

  it('includes both agents and violations by default', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../validate-agent-graph.mjs', () => ({
        collectTierInventory: jest.fn().mockResolvedValue(fakeInventory),
        runValidateAgentGraph: jest.fn().mockResolvedValue(fakeValidation),
      }));

      const mod = (await import(
        TIER_TOOL_PATH
      )) as unknown as CortexTierToolModule;
      const factory = mod.createTierGraphTool;
      if (!factory) {
        throw new Error('createTierGraphTool is not exported');
      }

      const tool = factory();
      const result = (await tool.handler({})) as TierGraphResult;

      expect({
        agents: result.agents,
        violations: result.violations,
        issueCount: result.validation.issueCount,
      }).toEqual({
        agents: fakeInventory.agents,
        violations: fakeValidation.issues,
        issueCount: 1,
      });
    });
  });
});

describe('cortex-tier-tool createSliceContextTool guard', () => {
  it('throws when the workflow tool set omits get_slice_context', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('./neataptic-workflow-mcp.mjs', () => ({
        createWorkflowTools: jest.fn().mockReturnValue([]),
      }));

      const mod = (await import(
        TIER_TOOL_PATH
      )) as unknown as CortexTierToolModule;
      const factory = mod.createSliceContextTool;
      if (!factory) {
        throw new Error('createSliceContextTool is not exported');
      }

      expect(factory).toThrow(
        'get_slice_context tool descriptor missing from neataptic-workflow-mcp tool set.',
      );
    });
  });
});
