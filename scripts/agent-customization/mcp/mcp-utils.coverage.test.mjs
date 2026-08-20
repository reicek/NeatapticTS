/**
 * @fileoverview Coverage tests for mcp-utils.mjs — non-stdio, non-spawn functions.
 * Covers CLI parsing, server creation, tool dispatch, formatting, tokenization, and helpers.
 */

import { jest } from '@jest/globals';

import {
  parseMcpCliArgs,
  requireExplicitPlanPath,
  resolveRepoRootPath,
  resolveExplicitPlanPath,
  printMcpUsage,
  emitSelfCheckReport,
  createSelfCheckReport,
  createTool,
  createMcpServer,
  invokeServerRequest,
  tokenizeShellSafeCommand,
  requireString,
  selfCheckError,
  formatToolResult,
  MCP_PROTOCOL_VERSION,
  MCP_REPO_ROOT,
} from './mcp-utils.mjs';

describe('parseMcpCliArgs', () => {
  it('returns all false with empty argv', () => {
    const result = parseMcpCliArgs([]);
    expect(result).toEqual({
      help: false,
      json: false,
      selfCheck: false,
      plan: undefined,
    });
  });

  it('parses --help and -h', () => {
    expect(parseMcpCliArgs(['--help']).help).toBe(true);
    expect(parseMcpCliArgs(['-h']).help).toBe(true);
  });

  it('parses --json', () => {
    expect(parseMcpCliArgs(['--json']).json).toBe(true);
  });

  it('parses --self-check', () => {
    expect(parseMcpCliArgs(['--self-check']).selfCheck).toBe(true);
  });

  it('parses --plan=<path>', () => {
    expect(parseMcpCliArgs(['--plan=plans/test.plans.md']).plan).toBe(
      'plans/test.plans.md',
    );
  });

  it('parses all flags at once', () => {
    const result = parseMcpCliArgs([
      '--help',
      '-h',
      '--json',
      '--self-check',
      '--plan=foo',
    ]);
    expect(result).toEqual({
      help: true,
      json: true,
      selfCheck: true,
      plan: 'foo',
    });
  });
});

describe('requireExplicitPlanPath', () => {
  it('returns trimmed path for valid string', () => {
    expect(requireExplicitPlanPath('  plans/test.plans.md  ')).toBe(
      'plans/test.plans.md',
    );
  });

  it('throws for undefined', () => {
    expect(() => requireExplicitPlanPath(undefined)).toThrow(
      'MCP entrypoints require --plan',
    );
  });

  it('throws for empty string', () => {
    expect(() => requireExplicitPlanPath('   ')).toThrow(
      'MCP entrypoints require --plan',
    );
  });

  it('throws for non-string', () => {
    expect(() => requireExplicitPlanPath(123)).toThrow(
      'MCP entrypoints require --plan',
    );
  });
});

describe('resolveRepoRootPath', () => {
  it('resolves repo-relative path', () => {
    const result = resolveRepoRootPath('scripts');
    expect(result).toBe(MCP_REPO_ROOT.replace(/[/\\]$/, '') + '\\scripts');
  });

  it('normalizes absolute path', () => {
    const result = resolveRepoRootPath('C:\\some\\path\\..');
    expect(result).toBe('C:\\some');
  });

  it('throws for non-string', () => {
    expect(() => resolveRepoRootPath(42)).toThrow(
      'path must be a non-empty string',
    );
  });
});

describe('resolveExplicitPlanPath', () => {
  it('returns absolutePath and displayPath for repo-relative path', () => {
    const result = resolveExplicitPlanPath('plans/test.plans.md');
    expect(result.absolutePath).toContain('plans');
    expect(result.absolutePath).toContain('test.plans.md');
    expect(result.displayPath).toBe('plans/test.plans.md');
  });

  it('returns absolute displayPath for path outside repo', () => {
    const result = resolveExplicitPlanPath('C:\\some\\outside\\path.md');
    expect(result.absolutePath).toBe('C:\\some\\outside\\path.md');
    expect(result.displayPath).toBe('C:/some/outside/path.md');
  });
});

describe('printMcpUsage', () => {
  it('prints usage text with tools', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    printMcpUsage({
      title: 'Test Server',
      entrypoint: 'test-server.mjs',
      summary: 'A test server.',
      tools: [
        { name: 'tool1', description: 'First tool' },
        { name: 'tool2', description: 'Second tool' },
      ],
    });
    const calls = logSpy.mock.calls.map((c) => c[0]);
    expect(calls.some((c) => c.includes('Test Server'))).toBe(true);
    expect(calls.some((c) => c.includes('tool1: First tool'))).toBe(true);
    expect(calls.some((c) => c.includes('tool2: Second tool'))).toBe(true);
    logSpy.mockRestore();
  });
});

describe('emitSelfCheckReport', () => {
  it('calls writeReport with json option', () => {
    // writeReport writes to stdout; spy on process.stdout.write
    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);
    emitSelfCheckReport(
      { name: 'test', issues: [], summary: 'ok' },
      { json: true },
    );
    expect(writeSpy).toHaveBeenCalled();
    const output = writeSpy.mock.calls.map((c) => c[0].toString()).join('');
    expect(output).toContain('"name"');
    writeSpy.mockRestore();
  });

  it('calls writeReport with non-json option', () => {
    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);
    emitSelfCheckReport(
      { name: 'test', issues: [], summary: 'ok' },
      { json: false },
    );
    expect(writeSpy).toHaveBeenCalled();
    writeSpy.mockRestore();
  });
});

describe('createSelfCheckReport', () => {
  it('merges summarizeIssues with details', () => {
    const report = createSelfCheckReport(
      'test',
      [{ severity: 'error', path: 'a', message: 'msg' }],
      { extra: 'detail' },
    );
    expect(report.name).toBe('test');
    expect(report.extra).toBe('detail');
    expect(report.ok).toBe(false);
    expect(report.counts.errors).toBe(1);
    expect(report.counts.warnings).toBe(0);
    expect(report.summaryText).toContain('FAIL');
  });

  it('works with no details', () => {
    const report = createSelfCheckReport('test', []);
    expect(report.name).toBe('test');
    expect(report.ok).toBe(true);
    expect(report.counts.errors).toBe(0);
    expect(report.summaryText).toContain('PASS');
  });
});

describe('createTool', () => {
  it('preserves existing inputSchema', () => {
    const schema = { type: 'object', properties: { foo: { type: 'string' } } };
    const tool = createTool({
      name: 'test',
      description: 'desc',
      inputSchema: schema,
      handler: () => null,
    });
    expect(tool.inputSchema).toBe(schema);
  });

  it('provides default inputSchema when absent', () => {
    const tool = createTool({
      name: 'test',
      description: 'desc',
      handler: () => null,
    });
    expect(tool.inputSchema).toEqual({
      type: 'object',
      properties: {},
      additionalProperties: false,
    });
  });
});

describe('createMcpServer and dispatch', () => {
  function makeServer(tools = []) {
    return createMcpServer({
      serverName: 'test-server',
      serverVersion: '1.0.0',
      tools,
    });
  }

  it('initialize returns protocolVersion, capabilities, serverInfo', async () => {
    const server = makeServer();
    const result = await server.dispatch({ method: 'initialize' });
    expect(result.protocolVersion).toBe(MCP_PROTOCOL_VERSION);
    expect(result.capabilities).toEqual({ tools: {} });
    expect(result.serverInfo).toEqual({
      name: 'test-server',
      version: '1.0.0',
    });
  });

  it('notifications/initialized returns null', async () => {
    const server = makeServer();
    const result = await server.dispatch({
      method: 'notifications/initialized',
    });
    expect(result).toBeNull();
  });

  it('ping returns empty object', async () => {
    const server = makeServer();
    const result = await server.dispatch({ method: 'ping' });
    expect(result).toEqual({});
  });

  it('tools/list returns listed tools without handler', async () => {
    const tool = createTool({
      name: 'myTool',
      description: 'A tool',
      handler: () => null,
    });
    const server = makeServer([tool]);
    const result = await server.dispatch({ method: 'tools/list' });
    expect(result.tools).toHaveLength(1);
    expect(result.tools[0].name).toBe('myTool');
    expect(result.tools[0].handler).toBeUndefined();
  });

  it('resources/list returns empty resources', async () => {
    const server = makeServer();
    const result = await server.dispatch({ method: 'resources/list' });
    expect(result).toEqual({ resources: [] });
  });

  it('prompts/list returns empty prompts', async () => {
    const server = makeServer();
    const result = await server.dispatch({ method: 'prompts/list' });
    expect(result).toEqual({ prompts: [] });
  });

  it('throws -32601 for unknown method', async () => {
    const server = makeServer();
    await expect(server.dispatch({ method: 'unknown/method' })).rejects.toThrow(
      'Unsupported method: unknown/method',
    );
  });

  describe('tools/call', () => {
    it('calls handler and returns formatted result', async () => {
      const tool = createTool({
        name: 'echo',
        description: 'Echo tool',
        handler: (args) => ({ echoed: args.value }),
      });
      const server = makeServer([tool]);
      const result = await server.dispatch({
        method: 'tools/call',
        params: { name: 'echo', arguments: { value: 'hello' } },
      });
      expect(result.isError).toBe(false);
      expect(result.structuredContent).toEqual({ echoed: 'hello' });
    });

    it('throws -32602 when params is missing', async () => {
      const server = makeServer();
      await expect(server.dispatch({ method: 'tools/call' })).rejects.toThrow(
        'tools/call requires params.',
      );
    });

    it('throws -32602 when params is not an object', async () => {
      const server = makeServer();
      await expect(
        server.dispatch({ method: 'tools/call', params: 'string' }),
      ).rejects.toThrow('tools/call requires params.');
    });

    it('throws -32602 when name is missing', async () => {
      const server = makeServer();
      await expect(
        server.dispatch({ method: 'tools/call', params: { arguments: {} } }),
      ).rejects.toThrow('tools/call requires a tool name.');
    });

    it('throws -32602 when name is not a string', async () => {
      const server = makeServer();
      await expect(
        server.dispatch({ method: 'tools/call', params: { name: 42 } }),
      ).rejects.toThrow('tools/call requires a tool name.');
    });

    it('throws -32602 for unknown tool name', async () => {
      const server = makeServer();
      await expect(
        server.dispatch({
          method: 'tools/call',
          params: { name: 'nonexistent' },
        }),
      ).rejects.toThrow('Unknown tool: nonexistent');
    });

    it('rethrows jsonRpcError from handler', async () => {
      const tool = createTool({
        name: 'throwy',
        description: 'Throws jsonRpcError',
        handler: () => {
          const err = new Error('Custom RPC error');
          err.jsonRpcCode = -32000;
          throw err;
        },
      });
      const server = makeServer([tool]);
      await expect(
        server.dispatch({ method: 'tools/call', params: { name: 'throwy' } }),
      ).rejects.toThrow('Custom RPC error');
    });

    it('returns error result for non-jsonRpcError throw from handler', async () => {
      const tool = createTool({
        name: 'throwy2',
        description: 'Throws regular Error',
        handler: () => {
          throw new Error('Handler error');
        },
      });
      const server = makeServer([tool]);
      const result = await server.dispatch({
        method: 'tools/call',
        params: { name: 'throwy2' },
      });
      expect(result.isError).toBe(true);
      expect(result.structuredContent.error).toBe('Handler error');
    });

    it('returns error result for non-Error throw from handler', async () => {
      const tool = createTool({
        name: 'throwy3',
        description: 'Throws string',
        handler: () => {
          throw 'string error';
        },
      });
      const server = makeServer([tool]);
      const result = await server.dispatch({
        method: 'tools/call',
        params: { name: 'throwy3' },
      });
      expect(result.isError).toBe(true);
      expect(result.structuredContent.error).toBe('string error');
    });

    it('passes empty object when arguments is not a plain object', async () => {
      const tool = createTool({
        name: 'argscheck',
        description: 'Checks args',
        handler: (args) => ({ received: args }),
      });
      const server = makeServer([tool]);
      const result = await server.dispatch({
        method: 'tools/call',
        params: { name: 'argscheck', arguments: 'not-an-object' },
      });
      expect(result.structuredContent).toEqual({ received: {} });
    });
  });
});

describe('invokeServerRequest', () => {
  it('dispatches with default id and no params when undefined', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });
    const result = await invokeServerRequest(server, { method: 'ping' });
    // ping returns {}
    expect(result).toEqual({});
  });

  it('dispatches with provided id and params', async () => {
    const tool = createTool({
      name: 'echo',
      description: 'echo',
      handler: (args) => ({ val: args.x }),
    });
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [tool],
    });
    const result = await invokeServerRequest(server, {
      method: 'tools/call',
      id: 'req-1',
      params: { name: 'echo', arguments: { x: 42 } },
    });
    expect(result.structuredContent).toEqual({ val: 42 });
  });
});

describe('tokenizeShellSafeCommand', () => {
  it('tokenizes a simple command', () => {
    expect(tokenizeShellSafeCommand('node -e hello')).toEqual([
      'node',
      '-e',
      'hello',
    ]);
  });

  it('tokenizes with double-quoted args', () => {
    expect(tokenizeShellSafeCommand('node -e "hello world"')).toEqual([
      'node',
      '-e',
      'hello world',
    ]);
  });

  it('tokenizes with single-quoted args', () => {
    expect(tokenizeShellSafeCommand("node -e 'hello world'")).toEqual([
      'node',
      '-e',
      'hello world',
    ]);
  });

  it('collapses multiple spaces', () => {
    expect(tokenizeShellSafeCommand('node    -e    hello')).toEqual([
      'node',
      '-e',
      'hello',
    ]);
  });

  it('throws for empty command', () => {
    expect(() => tokenizeShellSafeCommand('   ')).toThrow(
      'Validation command must not be empty',
    );
  });

  it('throws for shell metacharacters', () => {
    expect(() => tokenizeShellSafeCommand('node | grep')).toThrow(
      'Shell metacharacters are not allowed',
    );
  });

  it('throws for unterminated quote', () => {
    expect(() => tokenizeShellSafeCommand('node -e "hello')).toThrow(
      'unterminated quote',
    );
  });

  it('throws for unterminated single quote', () => {
    expect(() => tokenizeShellSafeCommand("node -e 'hello")).toThrow(
      'unterminated quote',
    );
  });

  it('throws for empty quotes producing no executable token', () => {
    expect(() => tokenizeShellSafeCommand("''")).toThrow(
      'produced no executable token',
    );
  });

  it('throws for empty double quotes producing no executable token', () => {
    expect(() => tokenizeShellSafeCommand('""')).toThrow(
      'produced no executable token',
    );
  });
});

describe('requireString', () => {
  it('returns trimmed string for valid input', () => {
    expect(requireString('  hello  ', 'field')).toBe('hello');
  });

  it('throws for non-string', () => {
    expect(() => requireString(42, 'field')).toThrow(
      'field must be a non-empty string',
    );
  });

  it('throws for empty string', () => {
    expect(() => requireString('   ', 'field')).toThrow(
      'field must be a non-empty string',
    );
  });

  it('throws for null', () => {
    expect(() => requireString(null, 'field')).toThrow(
      'field must be a non-empty string',
    );
  });
});

describe('selfCheckError', () => {
  it('creates an error-severity issue', () => {
    const result = selfCheckError('path/to/file', 'Something went wrong');
    expect(result.severity).toBe('error');
    expect(result.path).toBe('path/to/file');
    expect(result.message).toBe('Something went wrong');
  });
});

describe('formatToolResult', () => {
  it('passes through pre-formatted content array', () => {
    const preformatted = {
      content: [{ type: 'text', text: 'preformatted' }],
      structuredContent: { data: 1 },
      isError: false,
    };
    const result = formatToolResult(preformatted);
    expect(result.content).toEqual([{ type: 'text', text: 'preformatted' }]);
    expect(result.structuredContent).toEqual({ data: 1 });
    expect(result.isError).toBe(false);
  });

  it('wraps plain object as structuredContent with summary', () => {
    const result = formatToolResult({ key: 'value' });
    expect(result.structuredContent).toEqual({ key: 'value' });
    expect(result.content[0].type).toBe('text');
    expect(result.content[0].text).toContain('Compact workflow context');
    expect(result.isError).toBe(false);
  });

  it('wraps string value', () => {
    const result = formatToolResult('hello');
    expect(result.structuredContent).toEqual({ value: 'hello' });
    expect(result.isError).toBe(false);
  });

  it('wraps null value', () => {
    const result = formatToolResult(null);
    expect(result.structuredContent).toEqual({ value: null });
    expect(result.isError).toBe(false);
  });

  it('wraps undefined value', () => {
    const result = formatToolResult(undefined);
    expect(result.structuredContent).toEqual({ value: null });
    expect(result.isError).toBe(false);
  });

  it('builds slice-specific summary with slice_id only', () => {
    const result = formatToolResult({ slice_id: 'S1' });
    expect(result.content[0].text).toContain('slice S1');
    expect(result.content[0].text).not.toContain('(');
  });

  it('builds slice-specific summary with slice_id and phase', () => {
    const result = formatToolResult({ slice_id: 'S1', phase: 2 });
    expect(result.content[0].text).toContain('slice S1 (phase 2)');
  });

  it('builds slice-specific summary with slice_id and step_number', () => {
    const result = formatToolResult({ slice_id: 'S1', step_number: 3 });
    expect(result.content[0].text).toContain('slice S1 (step 3)');
  });

  it('builds slice-specific summary with slice_id, phase, and step_number', () => {
    const result = formatToolResult({
      slice_id: 'S1',
      phase: 2,
      step_number: 3,
    });
    expect(result.content[0].text).toContain('slice S1 (phase 2, step 3)');
  });
});
