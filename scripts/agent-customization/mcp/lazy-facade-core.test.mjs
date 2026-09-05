import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { EventEmitter } from 'node:events';

jest.unstable_mockModule('./mcp-utils.mjs', () => ({
  emitSelfCheckReport: jest.fn(),
  formatToolResult: jest.fn((r) => r),
  MCP_REPO_ROOT: 'C:\\NeatapticTS',
  parseMcpCliArgs: jest.fn(),
  runStdioMcpServer: jest.fn(() => Promise.resolve()),
}));

jest.unstable_mockModule('node:child_process', () => ({
  spawn: jest.fn(),
}));

jest.unstable_mockModule('node:fs', () => ({
  existsSync: jest.fn(),
  readFileSync: jest.fn(),
}));

jest.unstable_mockModule('node:process', () => ({
  default: process,
  exitCode: undefined,
}));

const { createLazyFacade, runFacadeMain, resolveSpawnCommand } =
  await import('./lazy-facade-core.mjs');
const mcpUtils = await import('./mcp-utils.mjs');
const childProcess = await import('node:child_process');
const fs = await import('node:fs');

const baseConfig = {
  name: 'test-facade',
  target: 'test-target',
  title: 'Test Facade',
  version: '1.0.0',
  defaultSnapshotPath: 'C:\\snap\\test.json',
  defaultSpawnCommand: ['node', 'server.mjs'],
};

const cortexConfig = {
  name: 'cortex',
  target: 'cortex',
  title: 'NeatapticTS Repo Cortex',
  version: '0.1.0',
  defaultSnapshotPath: 'C:\\snap\\cortex.json',
  defaultSpawnCommand: ['node', 'scripts/mcp-semantic/repo-cortex-mcp.mjs'],
};

function makeSnapshot(tools = []) {
  return JSON.stringify({ tools });
}

function makeRouterSnapshot(name = 'test-facade') {
  return makeSnapshot([
    {
      name,
      description: 'Router tool',
      inputSchema: { required: ['operation'] },
    },
  ]);
}

function fakeChild() {
  const handlers = {};
  const stdout = new EventEmitter();
  stdout.on = jest.fn((ev, cb) => {
    handlers.data = cb;
  });
  stdout.pipe = jest.fn();
  stdout.removeAllListeners = jest.fn();
  const stderr = new EventEmitter();
  stderr.on = jest.fn();
  stderr.removeAllListeners = jest.fn();
  return {
    stdin: { write: jest.fn() },
    stdout,
    stderr,
    on: jest.fn((ev, cb) => {
      handlers[ev] = cb;
    }),
    kill: jest.fn(),
    _handlers: handlers,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  fs.existsSync.mockReturnValue(false);
  fs.readFileSync.mockReturnValue(makeRouterSnapshot());
  childProcess.spawn.mockReturnValue(fakeChild());
  mcpUtils.formatToolResult.mockImplementation((r) => r);
  mcpUtils.runStdioMcpServer.mockResolvedValue(undefined);
  mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: false });
  mcpUtils.emitSelfCheckReport.mockImplementation(() => {});
});

describe('createLazyFacade', () => {
  it('returns dispatch, server, and close', () => {
    const facade = createLazyFacade(baseConfig);
    assert.strictEqual(typeof facade.dispatch, 'function');
    assert.strictEqual(typeof facade.server.dispatch, 'function');
    assert.strictEqual(typeof facade.close, 'function');
    assert.strictEqual(facade.serverInfo.name, 'test-facade');
  });

  it('handles initialize', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
    });
    assert.strictEqual(res.result.serverInfo.name, 'test-facade');
    assert.strictEqual(res.id, 1);
  });

  it('handles notifications/initialized', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      method: 'notifications/initialized',
    });
    assert.strictEqual(res.result, null);
  });

  it('handles ping', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'ping',
    });
    assert.deepStrictEqual(res.result, {});
  });

  it('handles tools/list', async () => {
    fs.readFileSync.mockReturnValue(makeSnapshot([{ name: 'tool1' }]));
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 3,
      method: 'tools/list',
    });
    assert.deepStrictEqual(res.result.tools, [{ name: 'tool1' }]);
  });

  it('handles tools/list with null snapshot', async () => {
    fs.readFileSync.mockImplementation(() => {
      throw new Error('ENOENT');
    });
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 3,
      method: 'tools/list',
    });
    assert.deepStrictEqual(res.result.tools, []);
  });

  it('throws for unknown method', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 4,
      method: 'unknown/method',
    });
    assert.strictEqual(res.error.code, -32601);
    assert.ok(res.error.message.includes('Unsupported method'));
  });

  it('handles tools/call with local tool', async () => {
    const localHandler = jest.fn(() => ({ result: 'local-ok' }));
    const facade = createLazyFacade({
      ...baseConfig,
      localTools: [
        { name: 'local-tool', description: 'Local', handler: localHandler },
      ],
    });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 5,
      method: 'tools/call',
      params: { name: 'local-tool', arguments: { foo: 'bar' } },
    });
    assert.strictEqual(localHandler.mock.calls[0][0].foo, 'bar');
    assert.strictEqual(res.result.result, 'local-ok');
  });

  it('handles tools/call with unknown tool name', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 6,
      method: 'tools/call',
      params: { name: 'wrong-tool', arguments: {} },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('Unknown operation'));
  });

  it('handles tools/call with router tool but no operation', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 7,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: {} },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('operation'));
  });

  it('handles tools/call with router tool (single-tool mode)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);

    // Simulate the child responding to initialize then tools/call
    let idCounter = 0;
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { protocolVersion: '2024-11-05' },
              }) + '\n',
            ),
          );
          // notifications/initialized has no id, no response
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'hello' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 8,
      method: 'tools/call',
      params: {
        name: 'test-facade',
        arguments: { operation: 'search', args: { q: 'test' } },
      },
    });
    assert.ok(res.result.content[0].text.includes('hello'));
  });

  it('handles tools/call with native routing mode (native RPC method)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'resources/list') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { resources: [] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade({ ...baseConfig, routingMode: 'native' });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 9,
      method: 'tools/call',
      params: {
        name: 'test-facade',
        arguments: { operation: 'resources/list', args: {} },
      },
    });
    assert.deepStrictEqual(res.result.resources, []);
  });

  it('handles tools/call with native routing mode (tool call)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  content: [{ type: 'text', text: 'native-tool-result' }],
                },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade({ ...baseConfig, routingMode: 'native' });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 10,
      method: 'tools/call',
      params: {
        name: 'test-facade',
        arguments: { operation: 'some_native_tool', args: {} },
      },
    });
    assert.ok(res.result.content[0].text.includes('native-tool-result'));
  });

  it('applies facade prefix on error result from child', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  isError: true,
                  content: [{ type: 'text', text: 'inner error' }],
                },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 11,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.ok(res.result.content[0].text.includes('[test-facade]'));
    assert.ok(res.result.content[0].text.includes('inner error'));
  });

  it('handles child transport error on callTool', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                error: { message: 'tool failed' },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 12,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('tool failed'));
  });

  it('handles spawn failure', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 13,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('spawn failed'));
  });

  it('renders T4 guidance text when the Cortex server fails to spawn', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 20,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    const payload = JSON.parse(res.result.content[0].text);
    assert.strictEqual(res.result.isError, true);
    assert.ok(payload.guidance.includes('Cortex MCP server failed to start'));
    assert.ok(payload.guidance.includes('RAG search is UNAVAILABLE'));
    assert.ok(payload.guidance.includes('grep/glob/view'));
  });

  it('includes T4 manual recovery commands in the spawn failure guidance', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 21,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    const payload = JSON.parse(res.result.content[0].text);
    assert.ok(payload.guidance.includes('TURSO_DATABASE_URL'));
    assert.ok(payload.guidance.includes('npm run index:session-start'));
    assert.ok(payload.guidance.includes('npm run index:prewarm'));
    assert.ok(
      payload.guidance.includes(
        'node scripts/mcp-semantic/repo-cortex-mcp.mjs',
      ),
    );
  });

  it('fills the T4 error summary with the spawn failure message', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 26,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    const payload = JSON.parse(res.result.content[0].text);
    assert.ok(payload.guidance.includes('spawn failed'));
  });

  it('exposes self_heal block with action unavailable and full schema on spawn failure', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 22,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    const payload = JSON.parse(res.result.content[0].text);
    assert.strictEqual(payload.self_heal.action, 'unavailable');
    assert.ok(Array.isArray(payload.self_heal.manual_recovery));
    assert.ok(
      payload.self_heal.manual_recovery.includes('npm run index:session-start'),
    );
    assert.ok(
      payload.self_heal.manual_recovery.includes('npm run index:prewarm'),
    );
    assert.deepStrictEqual(
      Object.keys(payload.self_heal).sort(),
      [
        'action',
        'attempt',
        'cooldown_s',
        'est_duration_min',
        'guidance',
        'manual_recovery',
        'max_attempts',
        'next_allowed_at',
        'reason',
        'state',
      ].sort(),
    );
  });

  it('omits fallbackHint from the spawn failure payload', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 23,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    const payload = JSON.parse(res.result.content[0].text);
    assert.strictEqual(payload.fallbackHint, undefined);
  });

  it('propagates the original spawn error and does not retry spawn', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res1 = await facade.dispatch({
      jsonrpc: '2.0',
      id: 24,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    assert.ok(
      JSON.parse(res1.result.content[0].text).error.includes('spawn failed'),
    );
    assert.strictEqual(childProcess.spawn.mock.calls.length, 1);
    await facade.dispatch({
      jsonrpc: '2.0',
      id: 25,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    assert.strictEqual(childProcess.spawn.mock.calls.length, 1);
  });

  it('renders T4 guidance when Cortex spawn returns invalid child', async () => {
    childProcess.spawn.mockReturnValue(null);
    fs.readFileSync.mockReturnValue(makeRouterSnapshot('cortex'));
    const facade = createLazyFacade(cortexConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 27,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search' } },
    });
    const payload = JSON.parse(res.result.content[0].text);
    assert.strictEqual(res.result.isError, true);
    assert.strictEqual(payload.self_heal.action, 'unavailable');
    assert.ok(
      payload.guidance.includes('Spawn returned an invalid child process.'),
    );
    assert.strictEqual(payload.fallbackHint, undefined);
  });

  it('handles spawn returning invalid child', async () => {
    childProcess.spawn.mockReturnValue(null);
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 14,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles non-Error thrown in handleToolCall', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    // Make stdin.write throw a non-Error
    child.stdin.write.mockImplementation(() => {
      throw 'string error';
    });
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 15,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles child exit/error event', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    let initResolve;
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        // Don't respond - simulate the child crashing
        process.nextTick(() => {
          child._handlers.error(new Error('child crashed'));
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 16,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles child exit event with code and signal', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.exit(1, 'SIGTERM');
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 17,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('rejects sendChildRequest when child stdin becomes unavailable after init', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { protocolVersion: '2024-11-05' },
              }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'notifications/initialized') {
        // Null out stdin so the subsequent tools/call sendChildRequest hits
        // the `!child.stdin` guard (lines 569-570).
        child.stdin = null;
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 99,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('not available'));
  });

  it('handles sendChildRequest when child stdin not available', async () => {
    const child = fakeChild();
    child.stdin = null;
    childProcess.spawn.mockReturnValue(child);
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 18,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles error response with non-string message', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                error: { code: -1, details: 'complex' },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 19,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles malformed child output', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(Buffer.from('not valid json\n'));
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'ok' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 20,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.ok(res.result.content[0].text.includes('ok'));
  });

  it('handles empty lines in child output', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(Buffer.from('\n\n'));
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'ok' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 21,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.ok(res.result.content[0].text.includes('ok'));
  });

  it('handles partial data (no newline)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          // Send partial data without newline
          child._handlers.data(Buffer.from('{"jsonrpc":"2.0","id":'));
          // Then complete it
          child._handlers.data(
            Buffer.from(JSON.stringify(msg.id) + ',"result":{}}\n'),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'ok' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 22,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.ok(res.result.content[0].text.includes('ok'));
  });

  it('handles message with null id', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          // Send a message with id: null (should be ignored)
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: null, result: {} }) + '\n',
            ),
          );
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'ok' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 23,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.ok(res.result.content[0].text.includes('ok'));
  });

  it('handles message with unknown id', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          // Send a response with unknown id (should be ignored)
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: 999, result: {} }) + '\n',
            ),
          );
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'ok' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 24,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.ok(res.result.content[0].text.includes('ok'));
  });

  it('close() kills child and rejects pending', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation(() => {});

    const facade = createLazyFacade(baseConfig);
    // Start a call that will be pending
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 25,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });

    // Close before the child responds
    facade.close();

    const res = await callPromise;
    assert.strictEqual(res.result.isError, true);
    assert.ok(child.kill.mock.calls.length > 0);
  });

  it('close() when no child is running', async () => {
    const facade = createLazyFacade(baseConfig);
    await facade.close();
    // Should not throw
    assert.ok(true);
  });

  it('close() handles kill throwing', async () => {
    const child = fakeChild();
    child.kill.mockImplementation(() => {
      throw new Error('already dead');
    });
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation(() => {});

    const facade = createLazyFacade(baseConfig);
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 26,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    facade.close();
    const res = await callPromise;
    assert.strictEqual(res.result.isError, true);
  });

  it('serverDispatch returns bare result', async () => {
    const facade = createLazyFacade(baseConfig);
    const result = await facade.server.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
    });
    assert.strictEqual(result.serverInfo.name, 'test-facade');
  });

  it('buildSuccessResponse without id', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({ jsonrpc: '2.0', method: 'ping' });
    assert.strictEqual(res.id, undefined);
    assert.deepStrictEqual(res.result, {});
  });

  it('buildErrorResponse without id', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({ jsonrpc: '2.0', method: 'bad-method' });
    assert.strictEqual(res.id, undefined);
    assert.ok(res.error);
  });

  it('handles tools/call with non-object params', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 27,
      method: 'tools/call',
      params: 'not-an-object',
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('Unknown operation'));
  });

  it('handles tools/call with null params', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 28,
      method: 'tools/call',
      params: null,
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles tools/call with non-string tool name', async () => {
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 29,
      method: 'tools/call',
      params: { name: 123, arguments: {} },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(
      res.result.content[0].text.includes('Unknown operation: undefined'),
    );
  });

  it('uses custom snapshotPath and spawnCommand overrides', async () => {
    fs.readFileSync.mockReturnValue(makeRouterSnapshot());
    const facade = createLazyFacade({
      ...baseConfig,
      snapshotPath: 'C:\\custom\\snap.json',
      spawnCommand: ['npx', 'custom-server'],
    });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
    });
    assert.strictEqual(res.result.serverInfo.name, 'test-facade');
    // The snapshot path was used
    assert.ok(
      fs.readFileSync.mock.calls.some((c) => c[0] === 'C:\\custom\\snap.json'),
    );
  });

  it('handles localTools not being an array', async () => {
    const facade = createLazyFacade({ ...baseConfig, localTools: null });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 30,
      method: 'tools/call',
      params: { name: 'nonexistent', arguments: {} },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles applyFacadePrefixIfError with non-object result', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: 'string-result',
              }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: 'string-result',
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 31,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result, 'string-result');
  });

  it('handles applyFacadePrefixIfError with isError but no content array', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { isError: true },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 32,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
  });

  it('handles applyFacadePrefixIfError with isError but no text content', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  isError: true,
                  content: [{ type: 'image', data: 'xyz' }],
                },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 33,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
    // Content should be returned as-is (no text to prefix)
    assert.strictEqual(res.result.content[0].type, 'image');
  });

  it('handles ensureInit called twice (cached)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    let initCount = 0;
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        initCount++;
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'ok' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    await facade.dispatch({
      jsonrpc: '2.0',
      id: 34,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    await facade.dispatch({
      jsonrpc: '2.0',
      id: 35,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search2' } },
    });
    // initialize should only have been sent once
    assert.strictEqual(initCount, 1);
  });

  it('handles native routing mode without args field (RPC method)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'resources/list') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { resources: [] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade({ ...baseConfig, routingMode: 'native' });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 36,
      method: 'tools/call',
      params: {
        name: 'test-facade',
        arguments: { operation: 'resources/list' },
      },
    });
    assert.deepStrictEqual(res.result.resources, []);
  });

  it('handles native routing mode without args field (tool call)', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: {} }) + '\n',
            ),
          );
        });
      } else if (msg.method === 'tools/call') {
        process.nextTick(() => {
          child._handlers.data(
            Buffer.from(
              JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'no-args-result' }] },
              }) + '\n',
            ),
          );
        });
      }
    });

    const facade = createLazyFacade({ ...baseConfig, routingMode: 'native' });
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 37,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'some_tool' } },
    });
    assert.ok(res.result.content[0].text.includes('no-args-result'));
  });

  it('handles non-Error thrown by spawn', async () => {
    childProcess.spawn.mockImplementation(() => {
      throw 'spawn string error';
    });
    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 38,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('spawn string error'));
  });

  it('handles child exit with null code and null signal', async () => {
    const child = fakeChild();
    childProcess.spawn.mockReturnValue(child);
    child.stdin.write.mockImplementation((line) => {
      const msg = JSON.parse(line);
      if (msg.method === 'initialize') {
        process.nextTick(() => {
          child._handlers.exit(null, null);
        });
      }
    });

    const facade = createLazyFacade(baseConfig);
    const res = await facade.dispatch({
      jsonrpc: '2.0',
      id: 39,
      method: 'tools/call',
      params: { name: 'test-facade', arguments: { operation: 'search' } },
    });
    assert.strictEqual(res.result.isError, true);
    assert.ok(res.result.content[0].text.includes('code=null'));
    assert.ok(res.result.content[0].text.includes('signal=null'));
  });
});

describe('runFacadeMain', () => {
  it('runs selfCheck when --self-check', async () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: true, json: false });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot());

    runFacadeMain(baseConfig, ['--self-check']);

    assert.ok(mcpUtils.emitSelfCheckReport.mock.calls.length > 0);
  });

  it('runs stdio server when not self-check', () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: false });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot());

    runFacadeMain(baseConfig, []);

    assert.ok(mcpUtils.runStdioMcpServer.mock.calls.length > 0);
  });

  it('self-check sets exitCode=1 on failure', () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: true });
    fs.readFileSync.mockImplementation(() => {
      throw new Error('ENOENT');
    });

    const origExitCode = process.exitCode;
    runFacadeMain(baseConfig, ['--self-check']);
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('self-check passes when snapshot valid and spawn command present', () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: true });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot());

    const origExitCode = process.exitCode;
    runFacadeMain(baseConfig, ['--self-check']);
    assert.notStrictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('self-check fails when snapshot has no router tool', () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: true });
    fs.readFileSync.mockReturnValue(makeSnapshot([{ name: 'other-tool' }]));

    const origExitCode = process.exitCode;
    runFacadeMain(baseConfig, ['--self-check']);
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('self-check fails when spawn command is empty', () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: true });
    fs.readFileSync.mockReturnValue(makeRouterSnapshot());

    const origExitCode = process.exitCode;
    runFacadeMain({ ...baseConfig, defaultSpawnCommand: [] }, ['--self-check']);
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('self-check handles unparseable snapshot', () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: true });
    fs.readFileSync.mockReturnValue('not json');

    const origExitCode = process.exitCode;
    runFacadeMain(baseConfig, ['--self-check']);
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('runStdioMcpServer rejection sets exitCode=1', async () => {
    mcpUtils.parseMcpCliArgs.mockReturnValue({ selfCheck: false });
    mcpUtils.runStdioMcpServer.mockRejectedValue(new Error('server crashed'));
    fs.readFileSync.mockReturnValue(makeRouterSnapshot());

    const origExitCode = process.exitCode;
    const origError = console.error;
    console.error = () => {};
    runFacadeMain(baseConfig, []);
    // Wait for the promise to reject
    await new Promise((r) => process.nextTick(r));
    await new Promise((r) => process.nextTick(r));
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
    console.error = origError;
  });
});

describe('resolveSpawnCommand', () => {
  const origPlatform = process.platform;

  afterEach(() => {
    Object.defineProperty(process, 'platform', {
      value: origPlatform,
      configurable: true,
    });
  });

  it('passes through on non-win32', () => {
    Object.defineProperty(process, 'platform', {
      value: 'linux',
      configurable: true,
    });
    const result = resolveSpawnCommand('npx');
    assert.strictEqual(result.file, 'npx');
    assert.strictEqual(result.shell, false);
  });

  it('finds .exe on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    fs.existsSync.mockImplementation((p) => p.endsWith('.exe'));
    const result = resolveSpawnCommand('my-tool');
    assert.strictEqual(result.file, 'my-tool');
    assert.strictEqual(result.shell, false);
  });

  it('finds .cmd on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    fs.existsSync.mockImplementation((p) => p.endsWith('.cmd'));
    const result = resolveSpawnCommand('npx');
    assert.ok(result.file.endsWith('.cmd'));
    assert.strictEqual(result.shell, true);
  });

  it('finds .ps1 on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    fs.existsSync.mockImplementation((p) => p.endsWith('.ps1'));
    const result = resolveSpawnCommand('npx');
    assert.ok(result.file.endsWith('.ps1'));
    assert.strictEqual(result.shell, true);
  });

  it('finds .bat on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    fs.existsSync.mockImplementation((p) => p.endsWith('.bat'));
    const result = resolveSpawnCommand('npx');
    assert.ok(result.file.endsWith('.bat'));
    assert.strictEqual(result.shell, true);
  });

  it('falls back to shell:true when not found on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    fs.existsSync.mockReturnValue(false);
    const result = resolveSpawnCommand('unknown-cmd');
    assert.strictEqual(result.file, 'unknown-cmd');
    assert.strictEqual(result.shell, true);
  });

  it('skips empty PATH dirs on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    process.env.PATH = ';;C:\\bin;;';
    fs.existsSync.mockImplementation((p) => p === 'C:\\bin\\my-tool.exe');
    const result = resolveSpawnCommand('my-tool');
    assert.strictEqual(result.file, 'my-tool');
    assert.strictEqual(result.shell, false);
  });

  it('handles undefined PATH on Windows', () => {
    Object.defineProperty(process, 'platform', {
      value: 'win32',
      configurable: true,
    });
    const origPath = process.env.PATH;
    delete process.env.PATH;
    fs.existsSync.mockReturnValue(false);
    const result = resolveSpawnCommand('my-tool');
    assert.strictEqual(result.file, 'my-tool');
    assert.strictEqual(result.shell, true);
    process.env.PATH = origPath;
  });
});

describe('loadSnapshot', () => {
  it('logs error on read failure', () => {
    fs.readFileSync.mockImplementation(() => {
      throw new Error('read failed');
    });
    const origError = console.error;
    const errors = [];
    console.error = (...args) => errors.push(args.join(' '));
    const facade = createLazyFacade(baseConfig);
    console.error = origError;
    assert.ok(errors.some((e) => e.includes('Failed to load snapshot')));
    // Should still return a facade with null snapshot
    assert.ok(facade);
  });
});
