import { EventEmitter } from 'node:events';
import {
  mkdirSync,
  readFileSync,
  rmSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';
import type { Readable, Writable } from 'node:stream';
import { PassThrough } from 'node:stream';
import { spawn } from 'node:child_process';

jest.mock('node:child_process', () => ({
  spawn: jest.fn(),
}));

interface McpFacade {
  dispatch(request: JsonRpcRequest): Promise<JsonRpcResponse>;
  close(): Promise<void>;
}

interface JsonRpcRequest {
  jsonrpc: '2.0';
  id?: string | number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  id?: string | number;
  result?: unknown;
  error?: { code: number; message: string };
}

interface FakeChildOptions {
  crashAfterInit?: boolean;
  crashOnCall?: boolean;
  onInitialize?: () => void;
  onToolCall?: (name: string, args: Record<string, unknown>) => unknown;
  onMethodCall?: (method: string, params: Record<string, unknown>) => unknown;
}

const REPO_ROOT = path.resolve(__dirname, '../../../../');
const CORTEX_SNAPSHOT_PATH = path.join(
  REPO_ROOT,
  'files/mcp-facade/cortex-tool-snapshot.json',
);
const DEVTOOLS_SNAPSHOT_PATH = path.join(
  REPO_ROOT,
  'files/mcp-facade/devtools-tool-snapshot.json',
);

const CORTEX_SNAPSHOT = JSON.parse(readFileSync(CORTEX_SNAPSHOT_PATH, 'utf8'));
const DEVTOOLS_SNAPSHOT = JSON.parse(
  readFileSync(DEVTOOLS_SNAPSHOT_PATH, 'utf8'),
);

const CORTEX_FAKE_COMMAND = ['node', 'fake-cortex-server.mjs'];
const DEVTOOLS_FAKE_COMMAND = [
  'npx',
  '-y',
  'fake-devtools@latest',
  '--headless=true',
];

afterEach(async () => {
  jest.clearAllMocks();
});

describe('cortex lazy-load facade', () => {
  it('responds to tools/list with the lightweight snapshot and does not spawn the real server', async () => {
    const { createCortexFacade } = await importCortexFacade();
    const facade = createCortexFacade({ snapshotPath: CORTEX_SNAPSHOT_PATH });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
    });

    expect({
      toolNames: (
        (response.result as Record<string, unknown>)?.tools as Array<{
          name: string;
        }>
      )?.map((tool) => tool.name),
      spawnCalls: (spawn as jest.Mock).mock.calls.length,
    }).toEqual({
      toolNames: CORTEX_SNAPSHOT.tools.map(
        (tool: { name: string }) => tool.name,
      ),
      spawnCalls: 0,
    });

    await facade.close();
  });

  it('creates the cortex facade using the default options', async () => {
    const { createCortexFacade } = await importCortexFacade();
    const facade = createCortexFacade();

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
    });

    expect(
      (
        (response.result as Record<string, unknown>)?.tools as Array<{
          name: string;
        }>
      )?.map((tool) => tool.name),
    ).toEqual(CORTEX_SNAPSHOT.tools.map((tool: { name: string }) => tool.name));

    await facade.close();
  });

  it('responds to initialize locally without spawning the real server', async () => {
    const { createCortexFacade } = await importCortexFacade();
    const facade = createCortexFacade({ snapshotPath: CORTEX_SNAPSHOT_PATH });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'test', version: '0.1.0' },
      },
    });

    expect({
      protocolVersion: (response.result as Record<string, unknown>)
        ?.protocolVersion,
      serverName: (
        (response.result as Record<string, unknown>)?.serverInfo as Record<
          string,
          unknown
        >
      )?.name,
      listChanged: (
        (
          (response.result as Record<string, unknown>)?.capabilities as Record<
            string,
            unknown
          >
        )?.tools as Record<string, unknown>
      )?.listChanged,
      spawnCalls: (spawn as jest.Mock).mock.calls.length,
    }).toEqual({
      protocolVersion: '2024-11-05',
      serverName: 'cortex',
      listChanged: undefined,
      spawnCalls: 0,
    });

    await facade.close();
  });

  it('lazily spawns repo-cortex-mcp on the first tools/call and forwards the call', async () => {
    const { createCortexFacade } = await importCortexFacade();
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createCortexFacade({
      snapshotPath: CORTEX_SNAPSHOT_PATH,
      spawnCommand: CORTEX_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: 'cortex',
        arguments: { operation: 'search_corpus', args: { query: 'network' } },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const textContent =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const forwarded = JSON.parse(textContent) as Record<string, unknown>;

    const spawnCall = (spawn as jest.Mock).mock.calls[0] ?? [];
    expect({
      spawnCommand: [spawnCall[0], ...(spawnCall[1] ?? [])],
      forwardedTool: forwarded.name,
      forwardedQuery: (forwarded.args as Record<string, unknown>)?.query,
      isError: result.isError,
    }).toEqual({
      spawnCommand: CORTEX_FAKE_COMMAND,
      forwardedTool: 'search_corpus',
      forwardedQuery: 'network',
      isError: false,
    });

    await facade.close();
  });

  it('reuses the spawned transport for subsequent tools/call requests', async () => {
    const { createCortexFacade } = await importCortexFacade();
    const fakeChild = createFakeChildProcess({
      onToolCall: (name) => ({ reused: name }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createCortexFacade({
      snapshotPath: CORTEX_SNAPSHOT_PATH,
      spawnCommand: CORTEX_FAKE_COMMAND,
    });

    await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'cortex',
        arguments: { operation: 'search_corpus', args: { query: 'a' } },
      },
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: 'cortex',
        arguments: { operation: 'search_advanced', args: { query: 'b' } },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const textContent =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const forwarded = JSON.parse(textContent) as Record<string, unknown>;

    expect({
      spawnCalls: (spawn as jest.Mock).mock.calls.length,
      reusedTool: forwarded.reused,
      isError: result.isError,
    }).toEqual({
      spawnCalls: 1,
      reusedTool: 'search_advanced',
      isError: false,
    });

    await facade.close();
  });

  it('returns a structured error when the target spawn fails', async () => {
    const { createCortexFacade } = await importCortexFacade();
    (spawn as jest.Mock).mockImplementation(() => {
      throw new Error('spawn ENOENT');
    });

    const facade = createCortexFacade({
      snapshotPath: CORTEX_SNAPSHOT_PATH,
      spawnCommand: CORTEX_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'cortex',
        arguments: { operation: 'search_corpus', args: { query: 'network' } },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const errorText =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const errorPayload = JSON.parse(errorText) as Record<string, unknown>;

    expect({
      isError: result.isError,
      facade: errorPayload.facade,
      target: errorPayload.target,
      available: errorPayload.available,
      hasErrorMessage: typeof errorPayload.error === 'string',
    }).toEqual({
      isError: true,
      facade: 'cortex',
      target: 'cortex',
      available: false,
      hasErrorMessage: true,
    });

    await facade.close();
  });

  it('returns a structured error for unknown operations without spawning the target', async () => {
    const { createCortexFacade } = await importCortexFacade();
    const facade = createCortexFacade({ snapshotPath: CORTEX_SNAPSHOT_PATH });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: { name: 'not_a_real_tool', arguments: {} },
    });

    const result = (response.result as Record<string, unknown>) ?? {};

    expect({
      isError: result.isError,
      spawnCalls: (spawn as jest.Mock).mock.calls.length,
    }).toEqual({
      isError: true,
      spawnCalls: 0,
    });

    await facade.close();
  });
});

describe('devtools lazy-load facade', () => {
  it('responds to tools/list with the lightweight snapshot and does not spawn the real server', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
    });

    expect({
      toolNames: (
        (response.result as Record<string, unknown>)?.tools as Array<{
          name: string;
        }>
      )?.map((tool) => tool.name),
      spawnCalls: (spawn as jest.Mock).mock.calls.length,
    }).toEqual({
      toolNames: DEVTOOLS_SNAPSHOT.tools.map(
        (tool: { name: string }) => tool.name,
      ),
      spawnCalls: 0,
    });

    await facade.close();
  });

  it('creates the devtools facade using the default options', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const facade = createDevtoolsFacade();

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/list',
    });

    expect(
      (
        (response.result as Record<string, unknown>)?.tools as Array<{
          name: string;
        }>
      )?.map((tool) => tool.name),
    ).toEqual(
      DEVTOOLS_SNAPSHOT.tools.map((tool: { name: string }) => tool.name),
    );

    await facade.close();
  });

  it('lazily spawns devtools on the first tools/call and forwards the call', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: { operation: 'list_pages', args: {} },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const textContent =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const forwarded = JSON.parse(textContent) as Record<string, unknown>;

    const devtoolsSpawnCall = (spawn as jest.Mock).mock.calls[0] ?? [];
    expect({
      commandBasename: path.basename(String(devtoolsSpawnCall[0])),
      args: devtoolsSpawnCall[1] ?? [],
      forwardedTool: forwarded.name,
      isError: result.isError,
    }).toEqual({
      commandBasename: expect.stringMatching(/^npx(\.(cmd|ps1|bat))?$/),
      args: DEVTOOLS_FAKE_COMMAND.slice(1),
      forwardedTool: 'list_pages',
      isError: false,
    });

    await facade.close();
  });

  it('returns a structured error when the target spawn fails', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    (spawn as jest.Mock).mockImplementation(() => {
      throw new Error('npx install failed');
    });

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: { operation: 'list_pages', args: {} },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const errorText =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const errorPayload = JSON.parse(errorText) as Record<string, unknown>;

    expect({
      isError: result.isError,
      facade: errorPayload.facade,
      target: errorPayload.target,
      available: errorPayload.available,
      hasErrorMessage: typeof errorPayload.error === 'string',
    }).toEqual({
      isError: true,
      facade: 'devtools',
      target: 'devtools',
      available: false,
      hasErrorMessage: true,
    });

    await facade.close();
  });

  it('returns a structured error for unknown operations without spawning the target', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: { name: 'not_a_real_tool', arguments: {} },
    });

    const result = (response.result as Record<string, unknown>) ?? {};

    expect({
      isError: result.isError,
      spawnCalls: (spawn as jest.Mock).mock.calls.length,
    }).toEqual({
      isError: true,
      spawnCalls: 0,
    });

    await facade.close();
  });

  it('propagates a real-server tool error with a [devtools] prefix', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const fakeChild = createFakeChildProcess({
      onToolCall: () => ({
        content: [{ type: 'text', text: 'Chrome is not installed' }],
        isError: true,
      }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: { operation: 'take_screenshot', args: {} },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const textContent =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';

    expect({
      isError: result.isError,
      hasPrefix: textContent.startsWith('[devtools]'),
    }).toEqual({
      isError: true,
      hasPrefix: true,
    });

    await facade.close();
  });

  it('writes newline-delimited JSON frames to the child process', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const fakeChild = createFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const stdinWrites = captureStdinWrites(fakeChild.stdin);

    await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: { operation: 'list_pages', args: {} },
      },
    });

    const lastWrite = stdinWrites[stdinWrites.length - 1] ?? '';
    const isNdjson =
      !lastWrite.includes('Content-Length:') &&
      lastWrite.trimEnd().endsWith('}');

    expect(isNdjson).toBe(true);

    await facade.close();
  });

  it('forwards tools/list in native mode as a raw JSON-RPC method call', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const realTools = [
      { name: 'navigate' },
      { name: 'recordPerformanceTrace' },
    ];
    const fakeChild = createFakeChildProcess({
      onMethodCall: (method) =>
        method === 'tools/list' ? { tools: realTools } : undefined,
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const writes = captureStdinWrites(fakeChild.stdin);

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: { operation: 'tools/list' },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    expect(result.tools).toEqual(realTools);

    const methodFrame = writes
      .map((w) => JSON.parse(w) as Record<string, unknown>)
      .find((m) => m.method === 'tools/list');
    expect(methodFrame).toBeDefined();

    await facade.close();
  });

  it('forwards a native-mode tool call with omitted args as an empty object', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: { operation: 'navigate' },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const textContent =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const forwarded = JSON.parse(textContent) as Record<string, unknown>;

    expect(forwarded).toEqual({
      name: 'navigate',
      args: {},
    });

    await facade.close();
  });

  it('forwards a non-JSON-RPC operation in native mode as a real tools/call', async () => {
    const { createDevtoolsFacade } = await importDevtoolsFacade();
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);

    const facade = createDevtoolsFacade({
      snapshotPath: DEVTOOLS_SNAPSHOT_PATH,
      spawnCommand: DEVTOOLS_FAKE_COMMAND,
    });

    const response = await facade.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: {
        name: 'devtools',
        arguments: {
          operation: 'navigate',
          args: {
            url: 'http://localhost:8080/docs/examples/racing_curriculum/index.html',
          },
        },
      },
    });

    const result = (response.result as Record<string, unknown>) ?? {};
    const textContent =
      (result.content as Array<{ type: string; text: string }>)?.[0]?.text ??
      '';
    const forwarded = JSON.parse(textContent) as Record<string, unknown>;

    expect(forwarded).toEqual({
      name: 'navigate',
      args: {
        url: 'http://localhost:8080/docs/examples/racing_curriculum/index.html',
      },
    });

    await facade.close();
  });
});

function createFakeChildProcess(options: FakeChildOptions = {}) {
  const child = new EventEmitter() as unknown as ReturnType<typeof spawn> & {
    stdin: Writable;
    stdout: Readable;
    stderr: Readable;
  };
  const stdin = new PassThrough();
  const stdout = new PassThrough();
  const stderr = new PassThrough();

  child.stdin = stdin;
  child.stdout = stdout;
  child.stderr = stderr;
  child.kill = jest.fn(() => {
    stdin.end();
    stdout.end();
    stderr.end();
    child.emit('exit', 0, null);
    return true;
  }) as unknown as ReturnType<typeof spawn>['kill'];

  const pendingResponses: JsonRpcResponse[] = [];

  createFramingReader(stdin, (message) => {
    if (message.method === 'initialize') {
      options.onInitialize?.();
      pendingResponses.push({
        jsonrpc: '2.0',
        id: message.id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'fake-target', version: '0.0.1' },
        },
      });
      flushResponses();
      if (options.crashAfterInit) {
        setImmediate(() => child.emit('exit', 1, null));
      }
      return;
    }

    if (message.method === 'tools/call') {
      if (options.crashOnCall) {
        setImmediate(() => child.emit('exit', 1, null));
        return;
      }

      const name = String(message.params?.name ?? '');
      const args = (message.params?.arguments as Record<string, unknown>) ?? {};
      const toolResult = options.onToolCall?.(name, args);

      if (
        toolResult &&
        typeof toolResult === 'object' &&
        'isError' in toolResult
      ) {
        pendingResponses.push({
          jsonrpc: '2.0',
          id: message.id,
          result: toolResult,
        });
      } else {
        pendingResponses.push({
          jsonrpc: '2.0',
          id: message.id,
          result: {
            content: [{ type: 'text', text: JSON.stringify(toolResult) }],
            isError: false,
          },
        });
      }
      flushResponses();
      return;
    }

    if (options.onMethodCall) {
      const methodResult = options.onMethodCall(
        message.method,
        (message.params as Record<string, unknown>) ?? {},
      );
      pendingResponses.push({
        jsonrpc: '2.0',
        id: message.id,
        result: methodResult,
      });
      flushResponses();
      return;
    }
  });

  function flushResponses() {
    while (pendingResponses.length > 0) {
      const message = pendingResponses.shift();
      if (!message) continue;
      stdout.write(`${JSON.stringify(message)}\n`);
    }
  }

  return child;
}

function createFramingReader(
  stream: Readable,
  onMessage: (message: JsonRpcRequest) => void,
) {
  let buffer = '';

  stream.on('data', (chunk: Buffer) => {
    buffer += chunk.toString('utf8');

    while (buffer.length > 0) {
      const leadingWhitespace = buffer.match(/^[\r\n]+/);
      if (leadingWhitespace) {
        buffer = buffer.slice(leadingWhitespace[0].length);
        continue;
      }

      if (buffer.toLowerCase().startsWith('content-length:')) {
        const headerEnd = buffer.indexOf('\r\n\r\n');
        if (headerEnd === -1) break;
        const lengthMatch = buffer.match(/^Content-Length:\s*(\d+)/i);
        if (!lengthMatch) break;
        const bodyLength = parseInt(lengthMatch[1], 10);
        const bodyStart = headerEnd + 4;
        if (buffer.length < bodyStart + bodyLength) break;
        const body = buffer.slice(bodyStart, bodyStart + bodyLength);
        buffer = buffer.slice(bodyStart + bodyLength);
        try {
          onMessage(JSON.parse(body) as JsonRpcRequest);
        } catch {
          // Ignore malformed frames.
        }
      } else {
        const newlineIndex = buffer.indexOf('\n');
        if (newlineIndex === -1) break;
        const line = buffer.slice(0, newlineIndex);
        buffer = buffer.slice(newlineIndex + 1);
        if (!line.trim()) continue;
        try {
          onMessage(JSON.parse(line) as JsonRpcRequest);
        } catch {
          // Ignore non-JSON lines.
        }
      }
    }
  });
}

async function waitForStdinWrite(
  writes: string[],
  predicate: (message: Record<string, unknown>) => boolean,
): Promise<Record<string, unknown>> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const found = writes
      .map((write) => JSON.parse(write) as Record<string, unknown>)
      .find(predicate);
    if (found) {
      return found;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error('Timed out waiting for a matching stdin write.');
}

function captureStdinWrites(stdin: Writable): string[] {
  const writes: string[] = [];
  const originalWrite = stdin.write.bind(stdin);
  stdin.write = (
    chunk: unknown,
    encoding?: unknown,
    callback?: unknown,
  ): boolean => {
    writes.push(String(chunk));
    return originalWrite(
      chunk as string | Buffer,
      encoding as BufferEncoding,
      callback as (error?: Error | null) => void,
    );
  };
  return writes;
}

function createStallingFakeChildProcess() {
  const child = new EventEmitter() as unknown as ReturnType<typeof spawn> & {
    stdin: Writable;
    stdout: Readable;
    stderr: Readable;
  };
  const stdin = new PassThrough();
  const stdout = new PassThrough();
  const stderr = new PassThrough();

  child.stdin = stdin;
  child.stdout = stdout;
  child.stderr = stderr;
  child.kill = jest.fn(() => {
    stdin.end();
    stdout.end();
    stderr.end();
    child.emit('exit', 0, null);
    return true;
  }) as unknown as ReturnType<typeof spawn>['kill'];

  createFramingReader(stdin, (message) => {
    if (message.method === 'initialize') {
      stdout.write(
        `${JSON.stringify({
          jsonrpc: '2.0',
          id: message.id,
          result: {
            protocolVersion: '2024-11-05',
            capabilities: { tools: {} },
            serverInfo: { name: 'fake-target', version: '0.0.1' },
          },
        })}\n`,
      );
    }
  });

  return child;
}

describe('lazy-facade-core internals', () => {
  async function importLazyFacadeCore(): Promise<{
    createLazyFacade: (config: Record<string, unknown>) => McpFacade;
    runFacadeMain: (config: Record<string, unknown>, argv: string[]) => void;
    resolveSpawnCommand: (command: string) => {
      file: string;
      args: string[];
      shell: boolean;
    };
  }> {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore: .mjs module has no declaration file.
    return import('../lazy-facade-core.mjs');
  }

  function baseConfig(overrides: Record<string, unknown> = {}) {
    return {
      name: 'cortex',
      target: 'cortex',
      title: 'Cortex',
      version: '1.0.0',
      defaultSnapshotPath: CORTEX_SNAPSHOT_PATH,
      defaultSpawnCommand: CORTEX_FAKE_COMMAND,
      ...overrides,
    };
  }

  it('returns a JSON-RPC error for unsupported methods', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 1,
      method: 'foo/unsupported',
    })) as JsonRpcResponse;
    expect(response.error?.code).toBe(-32601);
    expect(response.id).toBe(1);
    await facade.close();
  });

  it('returns bare results through server dispatch', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig()) as unknown as {
      server: { dispatch: (request: JsonRpcRequest) => Promise<unknown> };
      close: () => Promise<void>;
    };
    const result = await facade.server.dispatch({
      jsonrpc: '2.0',
      id: 2,
      method: 'initialize',
    });
    expect(result).toMatchObject({
      protocolVersion: '2024-11-05',
      serverInfo: { name: 'cortex' },
    });
    await facade.close();
  });

  it('responds to ping and notifications/initialized', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const ping = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 3,
      method: 'ping',
    })) as JsonRpcResponse;
    expect(ping.result).toEqual({});
    const note = (await facade.dispatch({
      jsonrpc: '2.0',
      method: 'notifications/initialized',
    })) as JsonRpcResponse;
    expect(note.result).toBeNull();
    await facade.close();
  });

  it('returns a tool error for an unknown router tool name', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 4,
      method: 'tools/call',
      params: { name: 'unknown-tool', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Unknown operation: unknown-tool');
    await facade.close();
  });

  it('returns a tool error when the router payload omits operation', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 5,
      method: 'tools/call',
      params: { name: 'cortex', arguments: {} },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Router payload must include');
    await facade.close();
  });

  it('forwards a call with omitted args as an empty object', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const writes = captureStdinWrites(fakeChild.stdin);
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 6,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'search_corpus' } },
    })) as JsonRpcResponse;
    const resultText = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    const forwarded = JSON.parse(resultText) as Record<string, unknown>;
    expect(forwarded.args).toEqual({});
    const forwardedFrame = writes
      .map((w) => JSON.parse(w) as Record<string, unknown>)
      .find((m) => m.method === 'tools/call');
    expect(
      ((forwardedFrame?.params as Record<string, unknown>)?.arguments as
        Record<string, unknown> | undefined) ?? {},
    ).toEqual({});
    await facade.close();
  });

  it('falls back to an empty tool list when the snapshot cannot be loaded', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ defaultSnapshotPath: '/nonexistent/snapshot.json' }),
    );
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 7,
      method: 'tools/list',
    })) as JsonRpcResponse;
    expect(
      ((response.result as Record<string, unknown>).tools as
        unknown[] | undefined) ?? null,
    ).toEqual([]);
    await facade.close();
  });

  it('sets exit code 1 when self-check fails', async () => {
    const { runFacadeMain } = await importLazyFacadeCore();
    const originalExitCode = process.exitCode;
    try {
      process.exitCode = 0;
      runFacadeMain(
        baseConfig({
          defaultSnapshotPath: '/nonexistent/snapshot.json',
          defaultSpawnCommand: [],
        }),
        ['--self-check', '--json'],
      );
      expect(process.exitCode).toBe(1);
    } finally {
      process.exitCode = originalExitCode;
    }
  });

  it('starts the stdio server when runFacadeMain is not in self-check mode', async () => {
    const mockRunStdio = jest.fn().mockResolvedValue(undefined);
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../mcp-utils.mjs', () => {
        const actual = jest.requireActual('../mcp-utils.mjs');
        return { ...actual, runStdioMcpServer: mockRunStdio };
      });
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore: .mjs module has no declaration file.
      const { runFacadeMain } = await import('../lazy-facade-core.mjs');
      runFacadeMain(baseConfig(), []);
      expect(mockRunStdio).toHaveBeenCalledWith(
        expect.objectContaining({
          serverInfo: { name: 'cortex', version: '1.0.0' },
        }),
      );
    });
  });

  it('propagates a synchronous spawn failure as a structured tool error', async () => {
    (spawn as jest.Mock).mockImplementationOnce(() => {
      throw new Error('spawn exploded');
    });
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 8,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('spawn exploded');
    await facade.close();
  });

  it('returns a tool error when spawn returns an invalid child process', async () => {
    (spawn as jest.Mock).mockReturnValue({
      stdin: null,
      stdout: null,
      stderr: null,
      on: jest.fn(),
      kill: jest.fn(),
    });
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 9,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    await facade.close();
  });

  it('ignores data written to child stderr', async () => {
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    await facade.dispatch({
      jsonrpc: '2.0',
      id: 10,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    fakeChild.stderr.emit('data', Buffer.from('telemetry disclaimer'));
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 11,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'y' } },
    })) as JsonRpcResponse;
    expect(response.error).toBeUndefined();
    await facade.close();
  });

  it('returns a tool error when the child exits unexpectedly mid-call', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 12,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    await new Promise((resolve) => setImmediate(resolve));
    fakeChild.emit('exit', 1, null);
    const response = (await callPromise) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Child exited');
    await facade.close();
  });

  it('returns a tool error when the facade is closed mid-call', async () => {
    const fakeChild = createStallingFakeChildProcess();
    fakeChild.kill = jest.fn(() => {
      // End streams without emitting an exit event so the close path rejects
      // the pending call with the facade-closed message.
      fakeChild.stdin.end();
      (fakeChild.stdout as PassThrough).end();
      (fakeChild.stderr as PassThrough).end();
      return true;
    }) as unknown as ReturnType<typeof spawn>['kill'];
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 13,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    await facade.close();
    const response = (await callPromise) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Facade closed.');
  });

  it('ignores malformed child output', async () => {
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    (fakeChild.stdout as PassThrough).write('this is not json\n');
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 14,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.error).toBeUndefined();
    await facade.close();
  });

  it('buffers child stdout until a newline is received', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const writes = captureStdinWrites(fakeChild.stdin);
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 15,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    const toolFrame = await waitForStdinWrite(
      writes,
      (message) => message.method === 'tools/call',
    );
    const requestId = toolFrame.id as number;
    const responseJson = JSON.stringify({
      jsonrpc: '2.0',
      id: requestId,
      result: { content: [{ type: 'text', text: 'split' }], isError: false },
    });
    (fakeChild.stdout as PassThrough).write(responseJson.slice(0, 10));
    (fakeChild.stdout as PassThrough).write(`${responseJson.slice(10)}\n`);
    const response = (await callPromise) as JsonRpcResponse;
    expect(
      (
        (response.result as Record<string, unknown>).content as Array<{
          text: string;
        }>
      )[0].text,
    ).toBe('split');
    await facade.close();
  });

  it('tolerates empty lines in child stdout', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const writes = captureStdinWrites(fakeChild.stdin);
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 16,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    const toolFrame = await waitForStdinWrite(
      writes,
      (message) => message.method === 'tools/call',
    );
    const requestId = toolFrame.id as number;
    (fakeChild.stdout as PassThrough).write('\n\n');
    (fakeChild.stdout as PassThrough).write(
      `${JSON.stringify({
        jsonrpc: '2.0',
        id: requestId,
        result: { content: [{ type: 'text', text: 'ok' }], isError: false },
      })}\n`,
    );
    const response = (await callPromise) as JsonRpcResponse;
    expect(
      (
        (response.result as Record<string, unknown>).content as Array<{
          text: string;
        }>
      )[0].text,
    ).toBe('ok');
    await facade.close();
  });

  it('propagates a JSON-RPC error response from the child as a tool error', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const writes = captureStdinWrites(fakeChild.stdin);
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 17,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    const toolFrame = await waitForStdinWrite(
      writes,
      (message) => message.method === 'tools/call',
    );
    const requestId = toolFrame.id as number;
    (fakeChild.stdout as PassThrough).write(
      `${JSON.stringify({
        jsonrpc: '2.0',
        id: requestId,
        error: { message: 'real server error' },
      })}\n`,
    );
    const response = (await callPromise) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('real server error');
    await facade.close();
  });

  it('does not prefix an error result that has no text content', async () => {
    const fakeChild = createFakeChildProcess({
      onToolCall: () => ({ isError: true, content: [] }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({
        spawnCommand: CORTEX_FAKE_COMMAND,
        name: 'custom',
        target: 'custom',
      }),
    );
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 18,
      method: 'tools/call',
      params: { name: 'custom', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true, content: [] });
    await facade.close();
  });

  it('omits the id field for notification responses', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const note = (await facade.dispatch({
      jsonrpc: '2.0',
      method: 'notifications/initialized',
    })) as JsonRpcResponse;
    expect(note.id).toBeUndefined();
    await facade.close();
  });

  it('returns a tool error when the child transport is no longer available', async () => {
    const fakeChild = createFakeChildProcess({
      onToolCall: (name, args) => ({ name, args }),
    });
    fakeChild.kill = jest.fn(() => true) as unknown as ReturnType<
      typeof spawn
    >['kill'];
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    await facade.dispatch({
      jsonrpc: '2.0',
      id: 19,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'first' } },
    });
    fakeChild.stdin = null as unknown as Writable;
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 20,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'second' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Child transport is not available.');
    await facade.close();
  });

  it('returns a tool error for a missing router tool name', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 22,
      method: 'tools/call',
      params: { arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Unknown operation: undefined');
    await facade.close();
  });

  it('rejects pending calls with a facade-closed error', async () => {
    const fakeChild = createStallingFakeChildProcess();
    fakeChild.kill = jest.fn(() => {
      fakeChild.stdin.end();
      (fakeChild.stdout as PassThrough).end();
      (fakeChild.stderr as PassThrough).end();
      return true;
    }) as unknown as ReturnType<typeof spawn>['kill'];
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 25,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    await new Promise((resolve) => setImmediate(resolve));
    await facade.close();
    const response = (await callPromise) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Facade closed.');
  });

  it('handles a non-Error failure from the child transport', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const writes = captureStdinWrites(fakeChild.stdin);
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 26,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    await waitForStdinWrite(
      writes,
      (message) => message.method === 'tools/call',
    );
    fakeChild.emit('error', 'plain string failure');
    const response = (await callPromise) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('plain string failure');
    await facade.close();
  });

  it('sets exit code 1 when the stdio server fails to start', async () => {
    await jest.isolateModulesAsync(async () => {
      jest.doMock('../mcp-utils.mjs', () => {
        const actual = jest.requireActual('../mcp-utils.mjs');
        return {
          ...actual,
          runStdioMcpServer: jest
            .fn()
            .mockRejectedValue(new Error('stdio failed')),
        };
      });
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore: .mjs module has no declaration file.
      const { runFacadeMain } = await import('../lazy-facade-core.mjs');
      const originalExitCode = process.exitCode;
      process.exitCode = 0;
      runFacadeMain(baseConfig(), []);
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(process.exitCode).toBe(1);
      process.exitCode = originalExitCode;
    });
  });

  it('reports a failing self-check when the snapshot tools field is not an array', async () => {
    const tempPath = path.resolve(REPO_ROOT, 'tmp/invalid-tools-snapshot.json');
    writeFileSync(tempPath, JSON.stringify({ tools: 'not-an-array' }));
    const { runFacadeMain } = await importLazyFacadeCore();
    const originalExitCode = process.exitCode;
    process.exitCode = 0;
    runFacadeMain(baseConfig({ defaultSnapshotPath: tempPath }), [
      '--self-check',
    ]);
    expect(process.exitCode).toBe(1);
    process.exitCode = originalExitCode;
    unlinkSync(tempPath);
  });

  it('returns a tool error when router params is not an object', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 30,
      method: 'tools/call',
      params: 'not-an-object' as unknown as Record<string, unknown>,
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Unknown operation: undefined');
    await facade.close();
  });

  it('returns a tool error when router arguments is not an object', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 31,
      method: 'tools/call',
      params: {
        name: 'cortex',
        arguments: 'not-an-object' as unknown as Record<string, unknown>,
      },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('Router payload must include');
    await facade.close();
  });

  it('prefixes an error result whose content is not an array', async () => {
    const fakeChild = createFakeChildProcess({
      onToolCall: () => ({ isError: true, content: 'not-an-array' }),
    });
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({
        spawnCommand: CORTEX_FAKE_COMMAND,
        name: 'custom',
        target: 'custom',
      }),
    );
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 32,
      method: 'tools/call',
      params: { name: 'custom', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({
      isError: true,
      content: [],
    });
    await facade.close();
  });

  it('propagates a non-Error spawn failure as a structured tool error', async () => {
    (spawn as jest.Mock).mockImplementation(() => {
      throw 'spawn threw a string';
    });
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      id: 33,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    })) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('spawn threw a string');
    await facade.close();
  });

  it('propagates a child JSON-RPC error whose message is not a string', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const writes = captureStdinWrites(fakeChild.stdin);
    const callPromise = facade.dispatch({
      jsonrpc: '2.0',
      id: 34,
      method: 'tools/call',
      params: { name: 'cortex', arguments: { operation: 'x' } },
    });
    const toolFrame = await waitForStdinWrite(
      writes,
      (message) => message.method === 'tools/call',
    );
    const requestId = toolFrame.id as number;
    (fakeChild.stdout as PassThrough).write(
      `${JSON.stringify({
        jsonrpc: '2.0',
        id: requestId,
        error: { message: 123, code: -1 },
      })}\n`,
    );
    const response = (await callPromise) as JsonRpcResponse;
    expect(response.result).toMatchObject({ isError: true });
    const text = (
      (response.result as Record<string, unknown>).content as Array<{
        text: string;
      }>
    )[0].text;
    expect(text).toContain('123');
    await facade.close();
  });

  it('ignores child responses with an unknown or missing id', async () => {
    const fakeChild = createStallingFakeChildProcess();
    (spawn as jest.Mock).mockReturnValue(fakeChild);
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(
      baseConfig({ spawnCommand: CORTEX_FAKE_COMMAND }),
    );
    const timeout = new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error('did not ignore unknown response')),
        100,
      ),
    );
    const ignored = facade
      .dispatch({
        jsonrpc: '2.0',
        id: 35,
        method: 'tools/call',
        params: { name: 'cortex', arguments: { operation: 'x' } },
      })
      .then(() => {
        throw new Error('call should not resolve from an ignored response');
      });
    await waitForStdinWrite(
      captureStdinWrites(fakeChild.stdin),
      (message) => message.method === 'tools/call',
    );
    (fakeChild.stdout as PassThrough).write(
      `${JSON.stringify({
        jsonrpc: '2.0',
        id: 99999,
        result: { ignored: true },
      })}\n`,
    );
    await expect(Promise.race([ignored, timeout])).rejects.toThrow(
      'did not ignore unknown response',
    );
    await facade.close();
  });

  it('returns an error without an id field for a notification failure', async () => {
    const { createLazyFacade } = await importLazyFacadeCore();
    const facade = createLazyFacade(baseConfig());
    const response = (await facade.dispatch({
      jsonrpc: '2.0',
      method: 'unsupported/method',
    })) as JsonRpcResponse;
    expect(response.error).toMatchObject({ code: -32601 });
    expect(response.id).toBeUndefined();
    await facade.close();
  });

  it('returns shell:false for non-Windows platforms', async () => {
    const originalPlatform = Object.getOwnPropertyDescriptor(
      process,
      'platform',
    );
    Object.defineProperty(process, 'platform', { value: 'linux' });
    try {
      const { resolveSpawnCommand } = await importLazyFacadeCore();
      expect(resolveSpawnCommand('npx')).toEqual({
        file: 'npx',
        args: [],
        shell: false,
      });
    } finally {
      if (originalPlatform) {
        Object.defineProperty(process, 'platform', originalPlatform);
      }
    }
  });

  it('defaults PATH to empty when PATH is undefined on Windows', async () => {
    const { resolveSpawnCommand } = await importLazyFacadeCore();
    const originalPlatform = Object.getOwnPropertyDescriptor(
      process,
      'platform',
    );
    const originalPath = process.env.PATH;
    Object.defineProperty(process, 'platform', { value: 'win32' });
    delete process.env.PATH;
    try {
      const result = resolveSpawnCommand('npx');
      expect(result.file.endsWith('npx.cmd')).toBe(true);
      expect(path.dirname(result.file)).toBe(path.dirname(process.execPath));
      expect(result.shell).toBe(true);
    } finally {
      if (originalPlatform) {
        Object.defineProperty(process, 'platform', originalPlatform);
      }
      if (originalPath === undefined) {
        delete process.env.PATH;
      } else {
        process.env.PATH = originalPath;
      }
    }
  });

  it('resolves a Windows wrapper to an absolute path with shell:true', async () => {
    const { resolveSpawnCommand } = await importLazyFacadeCore();
    const tempDir = path.join(REPO_ROOT, 'tmp', `spawn-test-${Date.now()}`);
    const wrapperDir = path.join(tempDir, 'bin');
    mkdirSync(wrapperDir, { recursive: true });
    const wrapperPath = path.join(wrapperDir, 'npx.cmd');
    writeFileSync(wrapperPath, '@echo off\n', 'utf8');

    const originalPlatform = Object.getOwnPropertyDescriptor(
      process,
      'platform',
    );
    const originalPath = process.env.PATH;
    Object.defineProperty(process, 'platform', { value: 'win32' });
    process.env.PATH = wrapperDir;
    try {
      const result = resolveSpawnCommand('npx');
      expect(result.file).toBe(wrapperPath);
      expect(path.isAbsolute(result.file)).toBe(true);
      expect(result.shell).toBe(true);
    } finally {
      if (originalPlatform) {
        Object.defineProperty(process, 'platform', originalPlatform);
      }
      process.env.PATH = originalPath ?? '';
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it('falls back to the Node executable directory on Windows when PATH lacks the command', async () => {
    const { resolveSpawnCommand } = await importLazyFacadeCore();
    const originalPlatform = Object.getOwnPropertyDescriptor(
      process,
      'platform',
    );
    const originalPath = process.env.PATH;
    Object.defineProperty(process, 'platform', { value: 'win32' });
    process.env.PATH = '';
    try {
      const result = resolveSpawnCommand('npx');
      expect(result.file.endsWith('npx.cmd')).toBe(true);
      expect(path.dirname(result.file)).toBe(path.dirname(process.execPath));
      expect(result.shell).toBe(true);
    } finally {
      if (originalPlatform) {
        Object.defineProperty(process, 'platform', originalPlatform);
      }
      process.env.PATH = originalPath ?? '';
    }
  });

  it('falls back to shell:true with the original command on Windows when no wrapper is found', async () => {
    const { resolveSpawnCommand } = await importLazyFacadeCore();
    const originalPlatform = Object.getOwnPropertyDescriptor(
      process,
      'platform',
    );
    const originalPath = process.env.PATH;
    Object.defineProperty(process, 'platform', { value: 'win32' });
    process.env.PATH = '';
    const missingCommand = `this-command-does-not-exist-${Date.now()}`;
    try {
      const result = resolveSpawnCommand(missingCommand);
      expect(result).toEqual({
        file: missingCommand,
        args: [],
        shell: true,
      });
    } finally {
      if (originalPlatform) {
        Object.defineProperty(process, 'platform', originalPlatform);
      }
      process.env.PATH = originalPath ?? '';
    }
  });
});

describe('facade CLI self-check', () => {
  it('runs the cortex self-check when imported as the main module', async () => {
    const originalArgv = process.argv;
    const originalExitCode = process.exitCode;
    const wrapperPath = path.resolve(
      REPO_ROOT,
      'scripts/agent-customization/mcp/cortex-facade.mjs',
    );
    process.argv = ['node', wrapperPath, '--self-check', '--json'];

    try {
      await jest.isolateModulesAsync(async () => {
        await import('../cortex-facade.mjs');
      });
    } finally {
      process.argv = originalArgv;
    }

    expect(process.exitCode).not.toBe(1);
    process.exitCode = originalExitCode;
  });

  it('runs the devtools self-check when imported as the main module', async () => {
    const originalArgv = process.argv;
    const originalExitCode = process.exitCode;
    const wrapperPath = path.resolve(
      REPO_ROOT,
      'scripts/agent-customization/mcp/devtools-facade.mjs',
    );
    process.argv = ['node', wrapperPath, '--self-check', '--json'];

    try {
      await jest.isolateModulesAsync(async () => {
        await import('../devtools-facade.mjs');
      });
    } finally {
      process.argv = originalArgv;
    }

    expect(process.exitCode).not.toBe(1);
    process.exitCode = originalExitCode;
  });

  it('does not run main when process.argv[1] is missing for cortex', async () => {
    const originalArgv = process.argv;
    const originalExitCode = process.exitCode;
    process.argv = ['node'];

    try {
      await jest.isolateModulesAsync(async () => {
        await import('../cortex-facade.mjs');
      });
    } finally {
      process.argv = originalArgv;
    }

    expect(process.exitCode).not.toBe(1);
    process.exitCode = originalExitCode;
  });

  it('does not run main when process.argv[1] is missing for devtools', async () => {
    const originalArgv = process.argv;
    const originalExitCode = process.exitCode;
    process.argv = ['node'];

    try {
      await jest.isolateModulesAsync(async () => {
        await import('../devtools-facade.mjs');
      });
    } finally {
      process.argv = originalArgv;
    }

    expect(process.exitCode).not.toBe(1);
    process.exitCode = originalExitCode;
  });
});

async function importCortexFacade(): Promise<{
  createCortexFacade: (options?: Record<string, unknown>) => McpFacade;
}> {
  return import('../cortex-facade.mjs');
}

async function importDevtoolsFacade(): Promise<{
  createDevtoolsFacade: (options?: Record<string, unknown>) => McpFacade;
}> {
  return import('../devtools-facade.mjs');
}
