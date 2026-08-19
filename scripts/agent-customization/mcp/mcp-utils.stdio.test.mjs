/**
 * @fileoverview Coverage tests for stdio JSON-RPC processing in mcp-utils.mjs.
 * Uses PassThrough streams to mock process.stdin and spies for stdout/stderr.
 */

import { jest } from '@jest/globals';
import { PassThrough } from 'node:stream';
import { Buffer } from 'node:buffer';

import {
  createMcpServer,
  createTool,
  runStdioMcpServer,
  MCP_PROTOCOL_VERSION,
} from './mcp-utils.mjs';

/** Build a Content-Length–framed JSON-RPC message. */
function contentLengthFrame(json) {
  const body = JSON.stringify(json);
  return `Content-Length: ${Buffer.byteLength(body, 'utf8')}\r\n\r\n${body}`;
}

/** Helper to wait for async processing. */
function wait(ms = 50) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let mockStdin;
let stdoutSpy;
let stderrSpy;
let originalStdin;
let originalExitCode;

function makeServer() {
  return createMcpServer({
    serverName: 'stdio-test',
    serverVersion: '1.0.0',
    tools: [
      createTool({
        name: 'echo',
        description: 'Echo tool',
        handler: (args) => ({ echoed: args.value }),
      }),
      createTool({
        name: 'errorTool',
        description: 'Always throws Error',
        handler: () => {
          throw new Error('tool error');
        },
      }),
      createTool({
        name: 'jsonRpcErrorTool',
        description: 'Throws jsonRpcError with data',
        handler: () => {
          const err = new Error('custom rpc');
          err.jsonRpcCode = -32000;
          err.jsonRpcData = { detail: 'extra' };
          throw err;
        },
      }),
    ],
  });
}

beforeEach(() => {
  originalStdin = process.stdin;
  mockStdin = new PassThrough();
  Object.defineProperty(process, 'stdin', {
    value: mockStdin,
    writable: true,
    configurable: true,
  });
  stdoutSpy = jest
    .spyOn(process.stdout, 'write')
    .mockImplementation(() => true);
  stderrSpy = jest
    .spyOn(process.stderr, 'write')
    .mockImplementation(() => true);
  originalExitCode = process.exitCode;
  process.exitCode = undefined;
});

afterEach(() => {
  stdoutSpy.mockRestore();
  stderrSpy.mockRestore();
  process.exitCode = originalExitCode;
  Object.defineProperty(process, 'stdin', {
    value: originalStdin,
    writable: true,
    configurable: true,
  });
});

/** Get all stdout output as a string. */
function getStdout() {
  return stdoutSpy.mock.calls.map((c) => c[0].toString()).join('');
}

/** Get all stderr output as a string. */
function getStderr() {
  return stderrSpy.mock.calls.map((c) => c[0].toString()).join('');
}

/** Parse the first Content-Length response from stdout. */
function parseFirstResponse(output) {
  const match = output.match(/Content-Length:\s*\d+\r\n\r\n([\s\S]*)/);
  if (match) {
    return JSON.parse(match[1]);
  }
  // Try newline-delimited
  const lines = output.split('\n').filter((l) => l.trim());
  if (lines.length > 0) {
    return JSON.parse(lines[0]);
  }
  return null;
}

describe('runStdioMcpServer — Content-Length framing', () => {
  it('handles initialize request with Content-Length framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '1',
        method: 'initialize',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const output = getStdout();
    const response = parseFirstResponse(output);
    expect(response.jsonrpc).toBe('2.0');
    expect(response.id).toBe('1');
    expect(response.result.protocolVersion).toBe(MCP_PROTOCOL_VERSION);
  });

  it('handles tools/list request with Content-Length framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '2',
        method: 'tools/list',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.result.tools).toHaveLength(3);
  });

  it('handles tools/call request with Content-Length framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '3',
        method: 'tools/call',
        params: { name: 'echo', arguments: { value: 'hello' } },
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.result.structuredContent).toEqual({ echoed: 'hello' });
  });

  it('writes error response for invalid JSON with Content-Length framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    const body = '{ invalid json }';
    mockStdin.write(
      `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`,
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.error.code).toBe(-32700);
    expect(response.id).toBeNull();
  });

  it('handles notification (no id) — no response', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        method: 'notifications/initialized',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    expect(getStdout().trim()).toBe('');
  });

  it('handles null result with id — no response', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    // notifications/initialized returns null; with id, writeJsonRpcSuccessResponse skips
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '5',
        method: 'notifications/initialized',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    expect(getStdout().trim()).toBe('');
  });

  it('writes error response for tool error with Content-Length framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '6',
        method: 'tools/call',
        params: { name: 'errorTool' },
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    // errorTool throws a regular Error → createToolErrorResult → isError:true result
    expect(response.result.isError).toBe(true);
  });

  it('writes jsonRpcError response with jsonRpcCode and data', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '7',
        method: 'tools/call',
        params: { name: 'jsonRpcErrorTool' },
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.error.code).toBe(-32000);
    expect(response.error.data).toEqual({ detail: 'extra' });
  });

  it('writes diagnostic to stderr for dispatch error with no id', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        method: 'unknown/method',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('Unsupported method');
    expect(getStdout().trim()).toBe('');
  });

  it('handles multiple Content-Length messages in one chunk', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    const msg1 = contentLengthFrame({
      jsonrpc: '2.0',
      id: '10',
      method: 'ping',
    });
    const msg2 = contentLengthFrame({
      jsonrpc: '2.0',
      id: '11',
      method: 'ping',
    });
    mockStdin.write(msg1 + msg2);
    await wait();
    mockStdin.end();
    await promise;
    const output = getStdout();
    // Should have two responses
    const responses = output
      .split(/(?=Content-Length:)/)
      .filter((s) => s.trim());
    expect(responses.length).toBe(2);
  });

  it('handles partial Content-Length frame split across chunks', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    const frame = contentLengthFrame({
      jsonrpc: '2.0',
      id: '12',
      method: 'ping',
    });
    const midPoint = Math.floor(frame.length / 2);
    mockStdin.write(frame.slice(0, midPoint));
    await wait(20);
    mockStdin.write(frame.slice(midPoint));
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.id).toBe('12');
  });

  it('handles leading line breaks before Content-Length frame', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    const frame = contentLengthFrame({
      jsonrpc: '2.0',
      id: '13',
      method: 'ping',
    });
    mockStdin.write('\r\n\r\n' + frame);
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.id).toBe('13');
  });

  it('throws diagnostic for invalid Content-Length header value', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write('Content-Length: abc\r\n\r\n{}');
    await wait();
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('valid Content-Length');
    expect(process.exitCode).toBe(1);
  });

  it('handles Content-Length frame with extra headers', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    const body = JSON.stringify({ jsonrpc: '2.0', id: '14', method: 'ping' });
    // Put Content-Length first (so startsWithContentLengthHeader returns true),
    // then extra headers with empty values and non-content-length names
    const frame =
      `Content-Length:\r\n` +
      `Content-Type: application/json\r\n` +
      `:value\r\n` +
      `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`;
    mockStdin.write(frame);
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.id).toBe('14');
  });
});

describe('runStdioMcpServer — newline-delimited framing', () => {
  it('handles initialize request with newline framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', id: '20', method: 'initialize' }) + '\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    const output = getStdout();
    const lines = output.split('\n').filter((l) => l.trim());
    const response = JSON.parse(lines[0]);
    expect(response.id).toBe('20');
    expect(response.result.protocolVersion).toBe(MCP_PROTOCOL_VERSION);
  });

  it('handles CRLF line ending with newline framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', id: '21', method: 'ping' }) + '\r\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    const output = getStdout();
    const lines = output.split('\n').filter((l) => l.trim());
    const response = JSON.parse(lines[0]);
    expect(response.id).toBe('21');
    expect(response.result).toEqual({});
  });

  it('handles invalid JSON with newline framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write('{ invalid }\n');
    await wait();
    mockStdin.end();
    await promise;
    const output = getStdout();
    const lines = output.split('\n').filter((l) => l.trim());
    const response = JSON.parse(lines[0]);
    expect(response.error.code).toBe(-32700);
  });

  it('handles notification with newline framing — no response', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) +
        '\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    expect(getStdout().trim()).toBe('');
  });

  it('handles multiple newline-delimited messages', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', id: '30', method: 'ping' }) + '\n',
    );
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', id: '31', method: 'ping' }) + '\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    const lines = getStdout()
      .split('\n')
      .filter((l) => l.trim());
    expect(lines.length).toBe(2);
  });

  it('skips empty lines between messages', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write('\n\n');
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', id: '32', method: 'ping' }) + '\n',
    );
    mockStdin.write('\n\n');
    await wait();
    mockStdin.end();
    await promise;
    const lines = getStdout()
      .split('\n')
      .filter((l) => l.trim());
    expect(lines.length).toBe(1);
    const response = JSON.parse(lines[0]);
    expect(response.id).toBe('32');
  });

  it('handles tool error with newline framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({
        jsonrpc: '2.0',
        id: '33',
        method: 'tools/call',
        params: { name: 'errorTool' },
      }) + '\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    const lines = getStdout()
      .split('\n')
      .filter((l) => l.trim());
    const response = JSON.parse(lines[0]);
    expect(response.result.isError).toBe(true);
  });

  it('writes jsonRpcError with newline framing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({
        jsonrpc: '2.0',
        id: '34',
        method: 'tools/call',
        params: { name: 'jsonRpcErrorTool' },
      }) + '\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    const lines = getStdout()
      .split('\n')
      .filter((l) => l.trim());
    const response = JSON.parse(lines[0]);
    expect(response.error.code).toBe(-32000);
  });

  it('writes diagnostic for dispatch error with no id (newline framing)', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', method: 'unknown/method' }) + '\n',
    );
    await wait();
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('Unsupported method');
  });

  it('handles partial newline frame (no LF yet)', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    const msg = JSON.stringify({ jsonrpc: '2.0', id: '35', method: 'ping' });
    mockStdin.write(msg.slice(0, 10));
    await wait(20);
    mockStdin.write(msg.slice(10) + '\n');
    await wait();
    mockStdin.end();
    await promise;
    const lines = getStdout()
      .split('\n')
      .filter((l) => l.trim());
    const response = JSON.parse(lines[0]);
    expect(response.id).toBe('35');
  });
});

describe('runStdioMcpServer — error with jsonRpcCode but no jsonRpcData', () => {
  it('writes error response without data field when jsonRpcData is undefined', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [
        createTool({
          name: 'codeOnlyError',
          description: 'Throws jsonRpcError without data',
          handler: () => {
            const err = new Error('code only');
            err.jsonRpcCode = -32601;
            throw err;
          },
        }),
      ],
    });
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '40',
        method: 'tools/call',
        params: { name: 'codeOnlyError' },
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.error.code).toBe(-32601);
    expect(response.error.data).toBeUndefined();
  });
});

describe('runStdioMcpServer — non-Error throw in dispatch', () => {
  it('writes diagnostic for non-Error throw with no id', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });
    // Override dispatch to throw a non-Error
    server.dispatch = () => {
      throw 'string error';
    };
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        method: 'ping',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('string error');
  });

  it('writes error response for non-Error throw with id', async () => {
    const server = createMcpServer({
      serverName: 'test',
      serverVersion: '1.0.0',
      tools: [],
    });
    server.dispatch = () => {
      throw 'string error';
    };
    const promise = runStdioMcpServer(server);
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '41',
        method: 'ping',
      }),
    );
    await wait();
    mockStdin.end();
    await promise;
    const response = parseFirstResponse(getStdout());
    expect(response.error.code).toBe(-32603);
    expect(response.error.message).toBe('string error');
  });
});

describe('runStdioMcpServer — non-Error in parser catch', () => {
  it('handles non-Error throw from parser.push', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    // Send a Content-Length frame with non-numeric value to trigger parser throw
    mockStdin.write('Content-Length: NaN\r\n\r\n{}');
    await wait();
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('valid Content-Length');
    expect(process.exitCode).toBe(1);
  });
});

describe('runStdioMcpServer — mixed framing', () => {
  it('handles Content-Length frame followed by newline-delimited frame', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    // Content-Length frame
    const clFrame = contentLengthFrame({
      jsonrpc: '2.0',
      id: '50',
      method: 'ping',
    });
    // Newline-delimited frame after
    const nlFrame =
      JSON.stringify({ jsonrpc: '2.0', id: '51', method: 'ping' }) + '\n';
    mockStdin.write(clFrame + '\r\n' + nlFrame);
    await wait();
    mockStdin.end();
    await promise;
    const output = getStdout();
    // First response is Content-Length framed, second is newline-delimited
    expect(output).toContain('Content-Length');
  });
});

describe('runStdioMcpServer — whitespace-only line in newline mode', () => {
  it('skips whitespace-only lines between messages', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    // Valid message followed by a whitespace-only line then another message
    mockStdin.write(
      JSON.stringify({ jsonrpc: '2.0', id: '60', method: 'ping' }) + '\n \n',
    );
    await wait();
    mockStdin.end();
    await promise;
    const lines = getStdout()
      .split('\n')
      .filter((l) => l.trim());
    expect(lines.length).toBe(1);
    const response = JSON.parse(lines[0]);
    expect(response.id).toBe('60');
  });
});

describe('runStdioMcpServer — partial Content-Length header (no separator)', () => {
  it('waits for full header separator before processing', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    // Send only the header line without the blank-line separator
    mockStdin.write('Content-Length: 10\r\nX-Custom: value\r\n');
    await wait(30);
    // End without completing — parser returned null, no message processed
    mockStdin.end();
    await promise;
    expect(getStdout().trim()).toBe('');
  });
});

describe('runStdioMcpServer — Content-Length with empty value', () => {
  it('throws diagnostic for Content-Length header with no value', async () => {
    const server = makeServer();
    const promise = runStdioMcpServer(server);
    mockStdin.write('Content-Length:\r\n\r\n{}');
    await wait();
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('valid Content-Length');
    expect(process.exitCode).toBe(1);
  });
});

describe('runStdioMcpServer — non-Error in parser queued catch', () => {
  it('handles non-Error throw propagating through parser.push via stdout failure', async () => {
    const server = makeServer();
    // Override stdout.write to throw a non-Error string.
    // This causes writeJsonRpcMessage to throw inside processStdioServerMessage's
    // catch block, which propagates through parser.push to the queued .catch().
    stdoutSpy.mockImplementation(() => {
      throw 'write error';
    });
    const promise = runStdioMcpServer(server);
    // Send a ping with an id — dispatch succeeds, but writeJsonRpcMessage throws
    mockStdin.write(
      contentLengthFrame({
        jsonrpc: '2.0',
        id: '99',
        method: 'ping',
      }),
    );
    await wait(100);
    mockStdin.end();
    await promise;
    expect(getStderr()).toContain('write error');
    expect(process.exitCode).toBe(1);
  });
});
