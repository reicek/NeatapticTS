/**
 * @module mcp-utils.coverage.test
 * @description Coverage tests targeting scripts/agent-customization/mcp/mcp-utils.mjs —
 * exercises exported utilities, MCP server dispatch, shell-free command runner,
 * stdio parser, and all internal helper functions.
 */
import { jest } from '@jest/globals';
import { EventEmitter } from 'node:events';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

import {
  MCP_PROTOCOL_VERSION,
  MCP_REPO_ROOT,
  parseMcpCliArgs,
  requireExplicitPlanPath,
  resolveRepoRootPath,
  resolveExplicitPlanPath,
  printMcpUsage,
  emitSelfCheckReport,
  createSelfCheckReport,
  createTool,
  createMcpServer,
  runStdioMcpServer,
  invokeServerRequest,
  runShellFreeCommand,
  tokenizeShellSafeCommand,
  requireString,
  selfCheckError,
  formatToolResult,
} from '../../agent-customization/mcp/mcp-utils.mjs';

// ---------------------------------------------------------------------------
// parseMcpCliArgs
// ---------------------------------------------------------------------------
describe('mcp-utils — parseMcpCliArgs', () => {
  it('parses help, json, selfCheck flags', () => {
    expect(parseMcpCliArgs(['--help'])).toMatchObject({
      help: true,
      json: false,
      selfCheck: false,
    });
    expect(parseMcpCliArgs(['-h'])).toMatchObject({ help: true });
    expect(parseMcpCliArgs(['--json'])).toMatchObject({ json: true });
    expect(parseMcpCliArgs(['--self-check'])).toMatchObject({
      selfCheck: true,
    });
  });

  it('parses --plan= path', () => {
    expect(parseMcpCliArgs(['--plan=plans/x.md'])).toMatchObject({
      plan: 'plans/x.md',
    });
  });

  it('returns undefined plan when not present', () => {
    expect(parseMcpCliArgs([]).plan).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// requireExplicitPlanPath
// ---------------------------------------------------------------------------
describe('mcp-utils — requireExplicitPlanPath', () => {
  it('returns trimmed path when valid', () => {
    expect(requireExplicitPlanPath('  plans/x.md  ')).toBe('plans/x.md');
  });

  it('throws when not a string', () => {
    expect(() => requireExplicitPlanPath(undefined)).toThrow();
    expect(() => requireExplicitPlanPath(123)).toThrow();
  });

  it('throws when blank', () => {
    expect(() => requireExplicitPlanPath('   ')).toThrow();
  });
});

// ---------------------------------------------------------------------------
// resolveRepoRootPath
// ---------------------------------------------------------------------------
describe('mcp-utils — resolveRepoRootPath', () => {
  it('normalizes absolute paths', () => {
    const result = resolveRepoRootPath(path.resolve('/tmp', 'file.ts'));
    expect(path.isAbsolute(result)).toBe(true);
  });

  it('resolves relative paths against repo root', () => {
    const result = resolveRepoRootPath('scripts/foo.mjs');
    expect(path.isAbsolute(result)).toBe(true);
    expect(result).toContain('scripts');
  });

  it('throws for non-string', () => {
    expect(() => resolveRepoRootPath(null)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// resolveExplicitPlanPath
// ---------------------------------------------------------------------------
describe('mcp-utils — resolveExplicitPlanPath', () => {
  it('returns absolute and display paths for repo-relative plan', () => {
    const { absolutePath, displayPath } =
      resolveExplicitPlanPath('scripts/test.mjs');
    expect(path.isAbsolute(absolutePath)).toBe(true);
    expect(displayPath).toBe('scripts/test.mjs');
  });

  it('returns absolute displayPath when outside repo', () => {
    const external = path.resolve(
      __dirname,
      '..',
      '..',
      '..',
      'external-file.mjs',
    );
    const { displayPath } = resolveExplicitPlanPath(external);
    // When the path is outside the repo, displayPath is the normalized absolute path
    expect(displayPath).toBeTruthy();
  });

  it('handles path equal to repo root (empty repoRelativePath)', () => {
    const { displayPath } = resolveExplicitPlanPath(MCP_REPO_ROOT);
    expect(displayPath).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// printMcpUsage
// ---------------------------------------------------------------------------
describe('mcp-utils — printMcpUsage', () => {
  it('prints help text with tools', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    printMcpUsage({
      title: 'Test Server',
      entrypoint: 'test.mjs',
      summary: 'A test server',
      tools: [{ name: 'tool1', description: 'First tool' }],
    });
    expect(logSpy).toHaveBeenCalled();
    const allOutput = logSpy.mock.calls.map((c) => c[0]).join('\n');
    expect(allOutput).toContain('Test Server');
    expect(allOutput).toContain('tool1');
    logSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// emitSelfCheckReport
// ---------------------------------------------------------------------------
describe('mcp-utils — emitSelfCheckReport', () => {
  it('delegates to writeReport', () => {
    // emitSelfCheckReport calls writeReport which writes to stdout/console
    // Just verify it doesn't throw
    expect(() =>
      emitSelfCheckReport({ name: 'test' }, { json: false }),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// createSelfCheckReport
// ---------------------------------------------------------------------------
describe('mcp-utils — createSelfCheckReport', () => {
  it('merges issue summary with details', () => {
    const report = createSelfCheckReport(
      'test',
      [{ severity: 'error', path: 'a', message: 'b' }],
      { extra: true },
    );
    expect(report).toHaveProperty('name', 'test');
    expect(report).toHaveProperty('extra', true);
    expect(report).toHaveProperty('summaryText');
    expect(report).toHaveProperty('counts');
  });

  it('uses default empty details', () => {
    const report = createSelfCheckReport('test', []);
    expect(report).toHaveProperty('name', 'test');
  });
});

// ---------------------------------------------------------------------------
// createTool
// ---------------------------------------------------------------------------
describe('mcp-utils — createTool', () => {
  it('creates a tool with default inputSchema', () => {
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

  it('preserves caller-provided inputSchema', () => {
    const schema = { type: 'object', properties: { x: { type: 'string' } } };
    const tool = createTool({
      name: 'test',
      description: 'desc',
      inputSchema: schema,
      handler: () => null,
    });
    expect(tool.inputSchema).toBe(schema);
  });
});

// ---------------------------------------------------------------------------
// createMcpServer + dispatch
// ---------------------------------------------------------------------------
describe('mcp-utils — createMcpServer', () => {
  const tools = [
    createTool({
      name: 'echo',
      description: 'Echo tool',
      handler: async (args) => ({ value: args.text ?? 'default' }),
    }),
  ];

  let server;

  beforeEach(() => {
    server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools,
    });
  });

  it('exposes serverInfo and listed tools (without handler)', () => {
    expect(server.serverInfo).toEqual({ name: 'test', version: '1.0.0' });
    expect(server.tools).toHaveLength(1);
    expect(server.tools[0]).not.toHaveProperty('handler');
  });

  it('dispatches initialize', async () => {
    const result = await server.dispatch({ method: 'initialize', id: 1 });
    expect(result.protocolVersion).toBe(MCP_PROTOCOL_VERSION);
    expect(result.serverInfo).toEqual({ name: 'test', version: '1.0.0' });
  });

  it('dispatches notifications/initialized (returns null)', async () => {
    const result = await server.dispatch({
      method: 'notifications/initialized',
    });
    expect(result).toBeNull();
  });

  it('dispatches ping', async () => {
    const result = await server.dispatch({ method: 'ping', id: 2 });
    expect(result).toEqual({});
  });

  it('dispatches tools/list', async () => {
    const result = await server.dispatch({ method: 'tools/list', id: 3 });
    expect(result.tools).toHaveLength(1);
  });

  it('dispatches tools/call with arguments', async () => {
    const result = await server.dispatch({
      method: 'tools/call',
      id: 4,
      params: { name: 'echo', arguments: { text: 'hello' } },
    });
    expect(result.isError).toBe(false);
    expect(result.structuredContent).toEqual({ value: 'hello' });
  });

  it('dispatches tools/call without arguments (defaults to {})', async () => {
    const result = await server.dispatch({
      method: 'tools/call',
      id: 5,
      params: { name: 'echo' },
    });
    expect(result.isError).toBe(false);
    expect(result.structuredContent).toEqual({ value: 'default' });
  });

  it('throws on unsupported method', async () => {
    await expect(server.dispatch({ method: 'bogus', id: 6 })).rejects.toThrow();
  });

  it('throws on tools/call without params', async () => {
    await expect(
      server.dispatch({ method: 'tools/call', id: 7 }),
    ).rejects.toThrow();
  });

  it('throws on tools/call without name', async () => {
    await expect(
      server.dispatch({ method: 'tools/call', id: 8, params: { name: 123 } }),
    ).rejects.toThrow();
  });

  it('throws on unknown tool', async () => {
    await expect(
      server.dispatch({
        method: 'tools/call',
        id: 9,
        params: { name: 'bogus' },
      }),
    ).rejects.toThrow();
  });

  it('returns error result when handler throws', async () => {
    const errorTools = [
      createTool({
        name: 'boom',
        description: 'Always fails',
        handler: async () => {
          throw new Error('handler error');
        },
      }),
    ];
    const errorServer = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: errorTools,
    });
    const result = await errorServer.dispatch({
      method: 'tools/call',
      id: 10,
      params: { name: 'boom', arguments: {} },
    });
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toBe('handler error');
  });

  it('returns error result when handler throws non-Error', async () => {
    const errorTools = [
      createTool({
        name: 'boom',
        description: 'Always fails',
        handler: async () => {
          throw 'string error';
        },
      }),
    ];
    const errorServer = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: errorTools,
    });
    const result = await errorServer.dispatch({
      method: 'tools/call',
      id: 11,
      params: { name: 'boom', arguments: {} },
    });
    expect(result.isError).toBe(true);
    expect(result.content[0].text).toBe('string error');
  });

  it('dispatches resources/list', async () => {
    const result = await server.dispatch({ method: 'resources/list', id: 12 });
    expect(result).toEqual({ resources: [] });
  });

  it('dispatches prompts/list', async () => {
    const result = await server.dispatch({ method: 'prompts/list', id: 13 });
    expect(result).toEqual({ prompts: [] });
  });
});

// ---------------------------------------------------------------------------
// invokeServerRequest
// ---------------------------------------------------------------------------
describe('mcp-utils — invokeServerRequest', () => {
  it('dispatches a request with default id', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [
        createTool({ name: 'ping', description: 'd', handler: () => 'ok' }),
      ],
    });
    const result = await invokeServerRequest(server, { method: 'ping' });
    expect(result).toEqual({});
  });

  it('includes params when provided', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [
        createTool({ name: 'echo', description: 'd', handler: (a) => a }),
      ],
    });
    const result = await invokeServerRequest(server, {
      method: 'tools/call',
      params: { name: 'echo', arguments: { x: 1 } },
    });
    expect(result.isError).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// formatToolResult
// ---------------------------------------------------------------------------
describe('mcp-utils — formatToolResult', () => {
  it('passes through pre-formatted content array', () => {
    const result = formatToolResult({
      content: [{ type: 'text', text: 'pre' }],
      isError: false,
    });
    expect(result.content[0].text).toBe('pre');
    expect(result.isError).toBe(false);
  });

  it('wraps plain object in structuredContent', () => {
    const result = formatToolResult({ key: 'value' });
    expect(result.structuredContent).toEqual({ key: 'value' });
    expect(result.content[0].type).toBe('text');
    expect(result.isError).toBe(false);
  });

  it('wraps null result', () => {
    const result = formatToolResult(null);
    expect(result.structuredContent).toEqual({ value: null });
  });

  it('wraps non-object result', () => {
    const result = formatToolResult('plain string');
    expect(result.structuredContent).toEqual({ value: 'plain string' });
  });

  it('builds slice-specific summary when slice_id is present', () => {
    const result = formatToolResult({
      slice_id: 'S1',
      step_number: 3,
      phase: 'impl',
    });
    expect(result.content[0].text).toContain('S1');
    expect(result.content[0].text).toContain('phase impl');
    expect(result.content[0].text).toContain('step 3');
  });

  it('builds slice summary with only step_number', () => {
    const result = formatToolResult({ slice_id: 'S2', step_number: 5 });
    expect(result.content[0].text).toContain('S2');
    expect(result.content[0].text).toContain('step 5');
  });

  it('builds slice summary with only phase', () => {
    const result = formatToolResult({ slice_id: 'S3', phase: 'green' });
    expect(result.content[0].text).toContain('S3');
    expect(result.content[0].text).toContain('phase green');
  });

  it('builds slice summary with slice_id only', () => {
    const result = formatToolResult({ slice_id: 'S4' });
    expect(result.content[0].text).toContain('S4');
    expect(result.content[0].text).toContain('See structuredContent');
  });

  it('builds generic summary when no slice_id', () => {
    const result = formatToolResult({ key: 'val' });
    expect(result.content[0].text).toContain('See structuredContent');
  });
});

// ---------------------------------------------------------------------------
// tokenizeShellSafeCommand
// ---------------------------------------------------------------------------
describe('mcp-utils — tokenizeShellSafeCommand', () => {
  it('throws on empty quoted string (no executable token)', () => {
    expect(() => tokenizeShellSafeCommand('""')).toThrow('no executable token');
  });

  it('handles double spaces between tokens (flushShellSafeToken empty branch)', () => {
    expect(tokenizeShellSafeCommand('node  -e  1')).toEqual([
      'node',
      '-e',
      '1',
    ]);
  });

  it('tokenizes a simple command', () => {
    expect(tokenizeShellSafeCommand('node script.mjs')).toEqual([
      'node',
      'script.mjs',
    ]);
  });

  it('tokenizes with quoted arguments', () => {
    expect(tokenizeShellSafeCommand('node "my script.mjs"')).toEqual([
      'node',
      'my script.mjs',
    ]);
  });

  it('tokenizes with single-quoted arguments', () => {
    expect(tokenizeShellSafeCommand("node 'my script.mjs'")).toEqual([
      'node',
      'my script.mjs',
    ]);
  });

  it('throws on empty command', () => {
    expect(() => tokenizeShellSafeCommand('   ')).toThrow();
  });

  it('throws on shell metacharacters', () => {
    expect(() => tokenizeShellSafeCommand('node script.mjs | grep')).toThrow();
    expect(() => tokenizeShellSafeCommand('node script.mjs && echo')).toThrow();
    expect(() => tokenizeShellSafeCommand('node script.mjs; echo')).toThrow();
    expect(() => tokenizeShellSafeCommand('node script.mjs > out')).toThrow();
    expect(() => tokenizeShellSafeCommand('node script.mjs < in')).toThrow();
  });

  it('throws on unterminated quote', () => {
    expect(() => tokenizeShellSafeCommand('node "unterminated')).toThrow();
  });
});

// ---------------------------------------------------------------------------
// requireString
// ---------------------------------------------------------------------------
describe('mcp-utils — requireString', () => {
  it('returns trimmed string when valid', () => {
    expect(requireString('  hello  ', 'name')).toBe('hello');
  });

  it('throws when not a string', () => {
    expect(() => requireString(123, 'name')).toThrow();
    expect(() => requireString(null, 'name')).toThrow();
  });

  it('throws when empty', () => {
    expect(() => requireString('  ', 'name')).toThrow();
  });
});

// ---------------------------------------------------------------------------
// selfCheckError
// ---------------------------------------------------------------------------
describe('mcp-utils — selfCheckError', () => {
  it('creates an error issue', () => {
    const issue = selfCheckError('path/to/file', 'something wrong');
    expect(issue.severity).toBe('error');
    expect(issue.path).toBe('path/to/file');
    expect(issue.message).toBe('something wrong');
  });
});

// ---------------------------------------------------------------------------
// runShellFreeCommand
// ---------------------------------------------------------------------------
describe('mcp-utils — runShellFreeCommand', () => {
  it('runs a node command and captures stdout', async () => {
    const result = await runShellFreeCommand(
      'node -e "process.stdout.write(\'hello\')"',
    );
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe('hello');
    expect(result.executable).toBe('node');
  });

  it('captures stderr', async () => {
    const result = await runShellFreeCommand(
      'node -e "process.stderr.write(\'err\')"',
    );
    expect(result.exitCode).toBe(0);
    expect(result.stderr).toBe('err');
  });

  it('captures non-zero exit code', async () => {
    const result = await runShellFreeCommand('node -e "process.exit(42)"');
    expect(result.exitCode).toBe(42);
  });

  it('respects maxOutputBytes option', async () => {
    const result = await runShellFreeCommand(
      'node -e "process.stdout.write(\'x\'.repeat(100))"',
      { maxOutputBytes: 10 },
    );
    expect(result.truncated.stdout).toBe(true);
    expect(result.stdout.length).toBeLessThanOrEqual(30);
    expect(result.stdout).toContain('[output truncated]');
  });

  it('uses default output limit when maxOutputBytes is invalid', async () => {
    const result = await runShellFreeCommand(
      'node -e "process.stdout.write(\'hi\')"',
      { maxOutputBytes: -1 },
    );
    expect(result.stdout).toBe('hi');
  });

  it('rejects shell metacharacters', async () => {
    await expect(runShellFreeCommand('node -e "x" | grep')).rejects.toThrow();
  });

  it('returns structured result with all fields', async () => {
    const result = await runShellFreeCommand(
      'node -e "process.stdout.write(\'test\')"',
    );
    expect(result).toHaveProperty('command');
    expect(result).toHaveProperty('executable');
    expect(result).toHaveProperty('argv');
    expect(result).toHaveProperty('exitCode');
    expect(result).toHaveProperty('stdout');
    expect(result).toHaveProperty('stderr');
    expect(result).toHaveProperty('durationMs');
    expect(result).toHaveProperty('truncated');
  });
});

// ---------------------------------------------------------------------------
// runStdioMcpServer
// ---------------------------------------------------------------------------
describe('mcp-utils — runStdioMcpServer', () => {
  it('processes newline-delimited messages from stdin', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    // Create a mock stdin emitter
    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });

    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    // Send a ping request
    mockStdin.emit(
      'data',
      Buffer.from(
        JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' }) + '\n',
      ),
    );

    // Wait a tick for processing
    await new Promise((resolve) => setImmediate(resolve));

    // End stdin
    mockStdin.emit('end');

    await serverPromise;

    expect(writeSpy).toHaveBeenCalled();

    writeSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });

  it('processes content-length framed messages', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });

    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    const payload = JSON.stringify({ jsonrpc: '2.0', id: 2, method: 'ping' });
    const frame = `Content-Length: ${Buffer.byteLength(payload)}\r\n\r\n${payload}`;
    mockStdin.emit('data', Buffer.from(frame));

    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');

    await serverPromise;

    expect(writeSpy).toHaveBeenCalled();

    writeSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });

  it('handles invalid JSON in newline-delimited frame', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });

    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);
    const stderrSpy = jest
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    // Send invalid JSON
    mockStdin.emit('data', Buffer.from('{invalid json}\n'));

    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');

    await serverPromise;

    expect(writeSpy).toHaveBeenCalled();

    writeSpy.mockRestore();
    stderrSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });

  it('handles dispatch error in stdio message processing', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    // Override dispatch to throw
    const originalDispatch = server.dispatch;
    server.dispatch = async () => {
      throw new Error('dispatch error');
    };

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });

    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    mockStdin.emit(
      'data',
      Buffer.from(
        JSON.stringify({ jsonrpc: '2.0', id: 99, method: 'bogus' }) + '\n',
      ),
    );

    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');

    await serverPromise;

    expect(writeSpy).toHaveBeenCalled();

    server.dispatch = originalDispatch;
    writeSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });

  it('handles notification (no id) dispatch error', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    server.dispatch = async () => {
      throw new Error('notif error');
    };

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });

    const stderrSpy = jest
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    // Notification: no id field
    mockStdin.emit(
      'data',
      Buffer.from(JSON.stringify({ jsonrpc: '2.0', method: 'bogus' }) + '\n'),
    );

    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');

    await serverPromise;

    expect(stderrSpy).toHaveBeenCalled();

    stderrSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });
});

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
describe('mcp-utils — constants', () => {
  it('exports MCP_PROTOCOL_VERSION', () => {
    expect(MCP_PROTOCOL_VERSION).toBe('2024-11-05');
  });

  it('exports MCP_REPO_ROOT as an absolute path', () => {
    expect(path.isAbsolute(MCP_REPO_ROOT)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// invokeServerRequest — error paths
// ---------------------------------------------------------------------------
describe('mcp-utils — invokeServerRequest error paths', () => {
  it('throws JsonRpcError for unknown method', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });
    await expect(
      invokeServerRequest(server, { method: 'bogus' }),
    ).rejects.toThrow('Unsupported method: bogus');
  });

  it('re-throws JsonRpcError from tool handler (isJsonRpcError path)', async () => {
    const jsonRpcErr = new Error('Custom RPC error');
    jsonRpcErr.jsonRpcCode = -32000;
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [
        createTool({
          name: 'boom',
          description: 'd',
          handler: () => {
            throw jsonRpcErr;
          },
        }),
      ],
    });
    await expect(
      invokeServerRequest(server, {
        method: 'tools/call',
        params: { name: 'boom' },
      }),
    ).rejects.toThrow('Custom RPC error');
  });

  it('returns error result for non-JsonRpcError from tool handler', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [
        createTool({
          name: 'boom',
          description: 'd',
          handler: () => {
            throw new Error('handler error');
          },
        }),
      ],
    });
    const result = await invokeServerRequest(server, {
      method: 'tools/call',
      params: { name: 'boom' },
    });
    expect(result.isError).toBe(true);
  });

  it('returns error result for unknown tool', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });
    await expect(
      invokeServerRequest(server, {
        method: 'tools/call',
        params: { name: 'bogus' },
      }),
    ).rejects.toThrow('Unknown tool: bogus');
  });
});

// ---------------------------------------------------------------------------
// runShellFreeCommand — npx/npm on win32
// ---------------------------------------------------------------------------
describe('mcp-utils — runShellFreeCommand npx', () => {
  it('resolves npx on win32 through cmd.exe', async () => {
    if (process.platform !== 'win32') return;
    const result = await runShellFreeCommand('npx --version');
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toMatch(/^\d+\.\d+/);
  }, 30000);

  it('resolves non-node executable via fallback return (line 317)', async () => {
    if (process.platform !== 'win32') return;
    const result = await runShellFreeCommand('cmd /c echo hello');
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toBe('hello');
  }, 30000);
});

// ---------------------------------------------------------------------------
// runStdioMcpServer — additional parser edge cases
// ---------------------------------------------------------------------------
describe('mcp-utils — stdio parser edge cases', () => {
  function makeStdioServer(tools = []) {
    return createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools,
    });
  }

  function withMockStdin(callback) {
    return async () => {
      const server = makeStdioServer();
      const mockStdin = new EventEmitter();
      mockStdin.resume = () => {};
      const originalStdin = process.stdin;
      Object.defineProperty(process, 'stdin', {
        value: mockStdin,
        configurable: true,
      });
      const writeSpy = jest
        .spyOn(process.stdout, 'write')
        .mockImplementation(() => true);
      const stderrSpy = jest
        .spyOn(process.stderr, 'write')
        .mockImplementation(() => true);

      const serverPromise = runStdioMcpServer(server);
      await callback(mockStdin);
      await serverPromise;

      writeSpy.mockRestore();
      stderrSpy.mockRestore();
      Object.defineProperty(process, 'stdin', {
        value: originalStdin,
        configurable: true,
      });
    };
  }

  it(
    'handles leading line breaks before Content-Length frame',
    withMockStdin(async (mockStdin) => {
      const payload = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' });
      const frame = `\r\nContent-Length: ${Buffer.byteLength(payload)}\r\n\r\n${payload}`;
      mockStdin.emit('data', Buffer.from(frame));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles buffer with only line breaks (empty after dropLeadingLineBreaks)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit('data', Buffer.from('\r\n'));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles empty line in newline-delimited mode (continue on empty message)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit('data', Buffer.from(' \n'));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles incomplete Content-Length frame (parsedFrame null - body too short)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit('data', Buffer.from('Content-Length: 100\r\n\r\n{short}'));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles incomplete Content-Length header (no header separator found)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit('data', Buffer.from('Content-Length: 50\r\n\r'));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles invalid Content-Length value (throws, caught by queued handler)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit('data', Buffer.from('Content-Length: abc\r\n\r\n{}'));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles Content-Length header with no value (extractContentLength returns null)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit(
        'data',
        Buffer.from('Content-Length:\r\nX-Custom: value\r\n\r\n{}'),
      );
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles newline-delimited data without newline (lineEndIndex -1)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit('data', Buffer.from('hello world'));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles notifications/initialized with id (result null → writeJsonRpcSuccessResponse early return)',
    withMockStdin(async (mockStdin) => {
      mockStdin.emit(
        'data',
        Buffer.from(
          JSON.stringify({
            jsonrpc: '2.0',
            id: 42,
            method: 'notifications/initialized',
          }) + '\n',
        ),
      );
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );

  it(
    'handles extra header line before Content-Length (non-matching header skipped)',
    withMockStdin(async (mockStdin) => {
      const payload = JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'ping' });
      // First line is a non-matching header, second is Content-Length
      const frame = `X-Custom: value\r\nContent-Length: ${Buffer.byteLength(payload)}\r\n\r\n${payload}`;
      // startsWithContentLengthHeader won't match since it starts with X-Custom, so it falls through to newline mode
      // This tests the extractContentLength path with non-matching header lines
      mockStdin.emit('data', Buffer.from(frame));
      await new Promise((resolve) => setImmediate(resolve));
      mockStdin.emit('end');
    }),
  );
});

// ---------------------------------------------------------------------------
// writeStdioJsonRpcErrorResponse — branch coverage
// ---------------------------------------------------------------------------
describe('mcp-utils — writeStdioJsonRpcErrorResponse branches', () => {
  it('writes diagnostic for notification with non-Error throw (line 686 false branch)', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    server.dispatch = async () => {
      throw 'string error';
    };

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });
    const stderrSpy = jest
      .spyOn(process.stderr, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    // Notification: no id field, dispatch throws non-Error
    mockStdin.emit(
      'data',
      Buffer.from(JSON.stringify({ jsonrpc: '2.0', method: 'bogus' }) + '\n'),
    );
    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');
    await serverPromise;

    expect(stderrSpy).toHaveBeenCalled();
    stderrSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });

  it('writes error response with non-Error message for request with id (line 696 false branch)', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    server.dispatch = async () => {
      throw 'non-error dispatch';
    };

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });
    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    // Request with id, dispatch throws non-Error
    mockStdin.emit(
      'data',
      Buffer.from(
        JSON.stringify({ jsonrpc: '2.0', id: 55, method: 'bogus' }) + '\n',
      ),
    );
    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');
    await serverPromise;

    expect(writeSpy).toHaveBeenCalled();
    writeSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });

  it('writes error response with jsonRpcData when present (line 697 false branch)', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });

    const customError = new Error('custom error with data');
    customError.jsonRpcCode = -32001;
    customError.jsonRpcData = { detail: 'extra info' };
    server.dispatch = async () => {
      throw customError;
    };

    const mockStdin = new EventEmitter();
    mockStdin.resume = () => {};
    const originalStdin = process.stdin;
    Object.defineProperty(process, 'stdin', {
      value: mockStdin,
      configurable: true,
    });
    const writeSpy = jest
      .spyOn(process.stdout, 'write')
      .mockImplementation(() => true);

    const serverPromise = runStdioMcpServer(server);

    mockStdin.emit(
      'data',
      Buffer.from(
        JSON.stringify({ jsonrpc: '2.0', id: 77, method: 'bogus' }) + '\n',
      ),
    );
    await new Promise((resolve) => setImmediate(resolve));
    mockStdin.emit('end');
    await serverPromise;

    expect(writeSpy).toHaveBeenCalled();
    writeSpy.mockRestore();
    Object.defineProperty(process, 'stdin', {
      value: originalStdin,
      configurable: true,
    });
  });
});
