import { spawn } from 'node:child_process';
import { Buffer } from 'node:buffer';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  issue,
  summarizeIssues,
  writeReport,
} from '../customization-utils.mjs';

export const MCP_PROTOCOL_VERSION = '2024-11-05';
export const MCP_REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

const DEFAULT_OUTPUT_LIMIT = 24_576;
const CONTENT_LENGTH_FRAME = 'content-length';
const CONTENT_LENGTH_HEADER = 'content-length';
const HEADER_SEPARATOR = Buffer.from('\r\n\r\n');
const NEWLINE_DELIMITED_FRAME = 'newline-delimited';
const SHELL_METACHARACTER_PATTERN = /[|&;<>]/;

/**
 * Parse the bounded CLI flags shared by the direct-MCP entrypoints.
 *
 * @param {string[]} argv - Raw CLI arguments without the node executable and script path.
 * @returns {{ help: boolean, json: boolean, selfCheck: boolean, plan?: string }} Parsed options.
 */
export function parseMcpCliArgs(argv) {
  return {
    help: argv.includes('--help') || argv.includes('-h'),
    json: argv.includes('--json'),
    selfCheck: argv.includes('--self-check'),
    plan: argv.find((argument) => argument.startsWith('--plan='))?.slice('--plan='.length),
  };
}

/**
 * Require an explicit plan path for direct-MCP entrypoints.
 *
 * @param {string | undefined} planPath - CLI plan path.
 * @returns {string} Trimmed plan path.
 */
export function requireExplicitPlanPath(planPath) {
  if (typeof planPath !== 'string' || !planPath.trim()) {
    throw new Error('MCP entrypoints require --plan=<path>; no default plan path is assumed.');
  }

  return planPath.trim();
}

/**
 * Resolve a repo-relative or absolute path against the repository root.
 *
 * @param {string} candidatePath - Repo-relative or absolute path.
 * @returns {string} Absolute path.
 */
export function resolveRepoRootPath(candidatePath) {
  const normalizedCandidatePath = requireString(candidatePath, 'path');
  return path.isAbsolute(normalizedCandidatePath)
    ? path.normalize(normalizedCandidatePath)
    : path.resolve(MCP_REPO_ROOT, normalizedCandidatePath);
}

/**
 * Resolve and normalize an explicit plan path for MCP consumers.
 *
 * @param {string | undefined} planPath - CLI plan path.
 * @returns {{ absolutePath: string, displayPath: string }} Resolved plan path details.
 */
export function resolveExplicitPlanPath(planPath) {
  const explicitPlanPath = requireExplicitPlanPath(planPath);
  const absolutePath = resolveRepoRootPath(explicitPlanPath);
  const repoRelativePath = path.relative(MCP_REPO_ROOT, absolutePath);
  const isRepoRelativePath = repoRelativePath !== ''
    && !repoRelativePath.startsWith('..')
    && !path.isAbsolute(repoRelativePath);

  return {
    absolutePath,
    displayPath: normalizePath(isRepoRelativePath ? repoRelativePath : absolutePath),
  };
}

/**
 * Print script help for a direct-MCP entrypoint.
 *
 * @param {{ title: string, entrypoint: string, summary: string, tools: Array<{ name: string, description: string }> }} config - Help text configuration.
 * @returns {void}
 */
export function printMcpUsage({ title, entrypoint, summary, tools }) {
  console.log(`${title}\n`);
  console.log(`${summary}\n`);
  console.log('Usage:');
  console.log(`  node ${entrypoint} --plan=<path> [--self-check] [--json]`);
  console.log(`  node ${entrypoint} --help`);
  console.log('');
  console.log('Modes:');
  console.log('  --plan=<path> Required workflow plan path resolved from the repository root.');
  console.log('  --help        Show this help text.');
  console.log('  --json        Write machine-readable self-check output.');
  console.log('  --self-check  Run bounded local checks instead of starting the stdio server.');
  console.log('  default       Start the stdio MCP server on stdin/stdout using the explicit plan path.');
  console.log('');
  console.log('Tools:');

  for (const tool of tools) {
    console.log(`  - ${tool.name}: ${tool.description}`);
  }
}

/**
 * Emit a self-check report using the repository's standard report writer.
 *
 * @param {Record<string, unknown>} report - Report payload.
 * @param {{ json: boolean }} options - CLI output options.
 * @returns {void}
 */
export function emitSelfCheckReport(report, options) {
  writeReport(report, { json: options.json });
}

/**
 * Build a standard self-check report.
 *
 * @param {string} name - Report name.
 * @param {Array<{ severity: string, path: string, message: string }>} issues - Collected issues.
 * @param {Record<string, unknown>} details - Additional details to merge into the report.
 * @returns {Record<string, unknown>} Report payload.
 */
export function createSelfCheckReport(name, issues, details = {}) {
  return {
    ...summarizeIssues(name, issues),
    ...details,
  };
}

/**
 * Create a single MCP tool descriptor.
 *
 * @param {{ name: string, description: string, inputSchema?: Record<string, unknown>, annotations?: Record<string, unknown>, handler: (argumentsObject: Record<string, unknown>) => Promise<unknown> | unknown }} tool - Tool descriptor with handler.
 * @returns {{ name: string, description: string, inputSchema: Record<string, unknown>, annotations?: Record<string, unknown>, handler: (argumentsObject: Record<string, unknown>) => Promise<unknown> | unknown }} Tool descriptor.
 */
export function createTool(tool) {
  return {
    ...tool,
    inputSchema: tool.inputSchema ?? {
      type: 'object',
      properties: {},
      additionalProperties: false,
    },
  };
}

/**
 * Create a dependency-light MCP server that supports initialize, tools/list, and tools/call.
 *
 * @param {{ serverName: string, serverVersion: string, tools: Array<{ name: string, description: string, inputSchema: Record<string, unknown>, annotations?: Record<string, unknown>, handler: (argumentsObject: Record<string, unknown>) => Promise<unknown> | unknown }> }} config - Server configuration.
 * @returns {{ serverInfo: { name: string, version: string }, tools: Array<Record<string, unknown>>, dispatch: (request: Record<string, unknown>) => Promise<unknown> }} Server implementation.
 */
export function createMcpServer({ serverName, serverVersion, tools }) {
  const toolRegistry = new Map(tools.map((tool) => [tool.name, tool]));
  const listedTools = tools.map(({ handler, ...tool }) => tool);

  return {
    serverInfo: { name: serverName, version: serverVersion },
    tools: listedTools,
    async dispatch(request) {
      switch (request.method) {
        case 'initialize':
          return {
            protocolVersion: MCP_PROTOCOL_VERSION,
            capabilities: {
              tools: {},
            },
            serverInfo: { name: serverName, version: serverVersion },
          };

        case 'notifications/initialized':
          return null;

        case 'ping':
          return {};

        case 'tools/list':
          return { tools: listedTools };

        case 'tools/call':
          return await callTool(request.params);

        case 'resources/list':
          return { resources: [] };

        case 'prompts/list':
          return { prompts: [] };

        default:
          throw createJsonRpcError(-32601, `Unsupported method: ${String(request.method)}`);
      }
    },
  };

  async function callTool(params) {
    if (!params || typeof params !== 'object') {
      throw createJsonRpcError(-32602, 'tools/call requires params.');
    }

    const toolName = typeof params.name === 'string' ? params.name : null;
    if (!toolName) {
      throw createJsonRpcError(-32602, 'tools/call requires a tool name.');
    }

    const tool = toolRegistry.get(toolName);
    if (!tool) {
      throw createJsonRpcError(-32602, `Unknown tool: ${toolName}`);
    }

    try {
      const handlerResult = await tool.handler(isPlainObject(params.arguments) ? params.arguments : {});
      return formatToolResult(handlerResult);
    } catch (error) {
      return createToolErrorResult(error);
    }
  }
}

/**
 * Run a dependency-light stdio MCP server using Content-Length or newline-delimited JSON-RPC.
 *
 * @param {{ serverInfo: { name: string, version: string }, dispatch: (request: Record<string, unknown>) => Promise<unknown> }} server - Server implementation.
 * @returns {Promise<void>} Resolves when stdin closes.
 */
export async function runStdioMcpServer(server) {
  const parser = createStdioJsonRpcParser(async ({ messageBuffer, framing }) => {
    let request;

    try {
      request = JSON.parse(messageBuffer.toString('utf8'));
    } catch {
      writeJsonRpcMessage({
        jsonrpc: '2.0',
        id: null,
        error: { code: -32700, message: 'Invalid JSON payload.' },
      }, framing);
      return;
    }

    try {
      const result = await server.dispatch(request);
      if (request.id !== undefined && result !== null) {
        writeJsonRpcMessage({
          jsonrpc: '2.0',
          id: request.id,
          result,
        }, framing);
      }
    } catch (error) {
      if (request?.id === undefined) {
        writeDiagnostic(error instanceof Error ? error.message : String(error));
        return;
      }

      writeJsonRpcMessage({
        jsonrpc: '2.0',
        id: request.id,
        error: {
          code: error?.jsonRpcCode ?? -32603,
          message: error instanceof Error ? error.message : String(error),
          ...(error?.jsonRpcData === undefined ? {} : { data: error.jsonRpcData }),
        },
      }, framing);
    }
  });

  let pending = Promise.resolve();
  process.stdin.on('data', (chunk) => {
    pending = pending
      .then(() => parser.push(chunk))
      .catch((error) => {
        writeDiagnostic(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
      });
  });

  process.stdin.resume();
  await new Promise((resolve) => process.stdin.on('end', resolve));
}

/**
 * Invoke the same server dispatch path used by stdio mode.
 *
 * @param {{ dispatch: (request: Record<string, unknown>) => Promise<unknown> }} server - Server implementation.
 * @param {{ method: string, params?: Record<string, unknown>, id?: string | number }} request - JSON-RPC request payload.
 * @returns {Promise<unknown>} Dispatch result.
 */
export async function invokeServerRequest(server, request) {
  return server.dispatch({
    jsonrpc: '2.0',
    id: request.id ?? 'self-check',
    method: request.method,
    ...(request.params === undefined ? {} : { params: request.params }),
  });
}

/**
 * Run an exact allow-listed command without using a shell.
 *
 * @param {string} commandString - Exact command string from the active step packet.
 * @returns {Promise<{ command: string, executable: string, argv: string[], exitCode: number, stdout: string, stderr: string, durationMs: number, truncated: { stdout: boolean, stderr: boolean } }>} Process result.
 */
export async function runShellFreeCommand(commandString, options = {}) {
  const tokens = tokenizeShellSafeCommand(commandString);
  const [requestedExecutable, ...argv] = tokens;
  const executable = /^(node|node\.exe)$/iu.test(requestedExecutable) ? process.execPath : requestedExecutable;
  const startTime = Date.now();
  const maxOutputBytes = Number.isFinite(options.maxOutputBytes) && options.maxOutputBytes > 0
    ? options.maxOutputBytes
    : DEFAULT_OUTPUT_LIMIT;

  return await new Promise((resolve, reject) => {
    const childProcess = spawn(executable, argv, {
      cwd: MCP_REPO_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
    });

    let stdout = '';
    let stderr = '';
    let stdoutBytes = 0;
    let stderrBytes = 0;
    let stdoutTruncated = false;
    let stderrTruncated = false;

    childProcess.stdout.on('data', (chunk) => {
      const chunkText = chunk.toString('utf8');
      if (stdoutBytes < maxOutputBytes) {
        const remainingBytes = maxOutputBytes - stdoutBytes;
        const truncatedChunk = chunkText.slice(0, remainingBytes);
        stdout += truncatedChunk;
      } else {
        stdoutTruncated = true;
      }

      stdoutBytes += Buffer.byteLength(chunkText);
      stdoutTruncated ||= stdoutBytes > maxOutputBytes;
    });

    childProcess.stderr.on('data', (chunk) => {
      const chunkText = chunk.toString('utf8');
      if (stderrBytes < maxOutputBytes) {
        const remainingBytes = maxOutputBytes - stderrBytes;
        const truncatedChunk = chunkText.slice(0, remainingBytes);
        stderr += truncatedChunk;
      } else {
        stderrTruncated = true;
      }

      stderrBytes += Buffer.byteLength(chunkText);
      stderrTruncated ||= stderrBytes > maxOutputBytes;
    });

    childProcess.on('error', reject);
    childProcess.on('close', (exitCode) => {
      resolve({
        command: commandString,
        executable: requestedExecutable,
        argv,
        exitCode: exitCode ?? 1,
        stdout: finalizeCapturedOutput(stdout, stdoutTruncated),
        stderr: finalizeCapturedOutput(stderr, stderrTruncated),
        durationMs: Date.now() - startTime,
        truncated: {
          stdout: stdoutTruncated,
          stderr: stderrTruncated,
        },
      });
    });
  });
}

/**
 * Tokenize a command string for shell-free execution.
 *
 * @param {string} commandString - Command string to tokenize.
 * @returns {string[]} Tokenized executable and arguments.
 */
export function tokenizeShellSafeCommand(commandString) {
  const normalizedCommand = commandString.trim();
  if (!normalizedCommand) {
    throw new Error('Validation command must not be empty.');
  }

  if (SHELL_METACHARACTER_PATTERN.test(normalizedCommand)) {
    throw new Error('Shell metacharacters are not allowed in validation commands.');
  }

  const tokens = [];
  let currentToken = '';
  let activeQuote = null;

  for (const character of normalizedCommand) {
    if (activeQuote) {
      if (character === activeQuote) {
        activeQuote = null;
      } else {
        currentToken += character;
      }
      continue;
    }

    if (character === '"' || character === "'") {
      activeQuote = character;
      continue;
    }

    if (/\s/u.test(character)) {
      if (currentToken) {
        tokens.push(currentToken);
        currentToken = '';
      }
      continue;
    }

    currentToken += character;
  }

  if (activeQuote) {
    throw new Error('Validation command contains an unterminated quote.');
  }

  if (currentToken) {
    tokens.push(currentToken);
  }

  if (tokens.length === 0) {
    throw new Error('Validation command produced no executable token.');
  }

  return tokens;
}

/**
 * Assert that a value is a string.
 *
 * @param {unknown} value - Value to validate.
 * @param {string} fieldName - Human-readable field name.
 * @returns {string} Validated string.
 */
export function requireString(value, fieldName) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${fieldName} must be a non-empty string.`);
  }

  return value.trim();
}

/**
 * Build a standard issue object for self-check reports.
 *
 * @param {string} path - Issue path.
 * @param {string} message - Issue message.
 * @returns {{ severity: string, path: string, message: string }} Issue.
 */
export function selfCheckError(path, message) {
  return issue('error', path, message);
}

function createJsonRpcError(code, message, data) {
  const error = new Error(message);
  error.jsonRpcCode = code;
  error.jsonRpcData = data;
  return error;
}

function formatToolResult(handlerResult) {
  if (isPlainObject(handlerResult) && Array.isArray(handlerResult.content)) {
    return {
      isError: false,
      ...handlerResult,
    };
  }

  const structuredContent = isPlainObject(handlerResult) ? handlerResult : { value: handlerResult ?? null };
  return {
    content: [
      {
        type: 'text',
        text: JSON.stringify(structuredContent, null, 2),
      },
    ],
    structuredContent,
    isError: false,
  };
}

function createToolErrorResult(error) {
  const message = error instanceof Error ? error.message : String(error);
  return {
    content: [
      {
        type: 'text',
        text: message,
      },
    ],
    structuredContent: {
      error: message,
    },
    isError: true,
  };
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function writeJsonRpcMessage(payload, framing = CONTENT_LENGTH_FRAME) {
  const serializedPayload = JSON.stringify(payload);

  if (framing === NEWLINE_DELIMITED_FRAME) {
    process.stdout.write(`${serializedPayload}\n`);
    return;
  }

  const contentLength = Buffer.byteLength(serializedPayload, 'utf8');
  process.stdout.write(`Content-Length: ${contentLength}\r\n\r\n${serializedPayload}`);
}

function writeDiagnostic(message) {
  process.stderr.write(`${message}\n`);
}

function createStdioJsonRpcParser(onMessage) {
  let buffer = Buffer.alloc(0);

  return {
    async push(chunk) {
      buffer = Buffer.concat([buffer, Buffer.from(chunk)]);

      while (buffer.length > 0) {
        buffer = dropLeadingLineBreaks(buffer);
        if (buffer.length === 0) {
          return;
        }

        if (startsWithContentLengthHeader(buffer)) {
          const parsedFrame = readContentLengthFrame(buffer);
          if (parsedFrame === null) {
            return;
          }

          buffer = parsedFrame.remainingBuffer;
          await onMessage({
            messageBuffer: parsedFrame.messageBuffer,
            framing: CONTENT_LENGTH_FRAME,
          });
          continue;
        }

        const lineEndIndex = buffer.indexOf('\n');
        if (lineEndIndex === -1) {
          return;
        }

        const messageBuffer = trimTrailingCarriageReturn(buffer.subarray(0, lineEndIndex));
        buffer = buffer.subarray(lineEndIndex + 1);
        if (messageBuffer.toString('utf8').trim() === '') {
          continue;
        }

        await onMessage({
          messageBuffer,
          framing: NEWLINE_DELIMITED_FRAME,
        });
      }
    },
  };
}

function startsWithContentLengthHeader(buffer) {
  const leadingText = buffer.subarray(0, Math.min(buffer.length, 64)).toString('utf8');
  return /^content-length\s*:/iu.test(leadingText);
}

function readContentLengthFrame(buffer) {
  const headerEndIndex = buffer.indexOf(HEADER_SEPARATOR);
  if (headerEndIndex === -1) {
    return null;
  }

  const headerText = buffer.subarray(0, headerEndIndex).toString('utf8');
  const contentLength = extractContentLength(headerText);
  if (contentLength === null) {
    throw new Error('Received an MCP message without a valid Content-Length header.');
  }

  const bodyStartIndex = headerEndIndex + HEADER_SEPARATOR.length;
  const bodyEndIndex = bodyStartIndex + contentLength;
  if (buffer.length < bodyEndIndex) {
    return null;
  }

  return {
    messageBuffer: buffer.subarray(bodyStartIndex, bodyEndIndex),
    remainingBuffer: buffer.subarray(bodyEndIndex),
  };
}

function dropLeadingLineBreaks(buffer) {
  let firstContentIndex = 0;
  while (firstContentIndex < buffer.length && (buffer[firstContentIndex] === 10 || buffer[firstContentIndex] === 13)) {
    firstContentIndex += 1;
  }

  return firstContentIndex === 0 ? buffer : buffer.subarray(firstContentIndex);
}

function trimTrailingCarriageReturn(buffer) {
  return buffer.at(-1) === 13 ? buffer.subarray(0, -1) : buffer;
}

function extractContentLength(headerText) {
  const headerLines = headerText.split(/\r?\n/u);
  for (const headerLine of headerLines) {
    const [rawName, rawValue] = headerLine.split(':');
    if (!rawName || !rawValue) {
      continue;
    }

    if (rawName.trim().toLowerCase() !== CONTENT_LENGTH_HEADER) {
      continue;
    }

    const parsedValue = Number.parseInt(rawValue.trim(), 10);
    return Number.isFinite(parsedValue) ? parsedValue : null;
  }

  return null;
}

function finalizeCapturedOutput(output, truncated) {
  if (!truncated) {
    return output;
  }

  return `${output}\n[output truncated]`;
}

function normalizePath(value) {
  return value.replaceAll(path.sep, '/');
}