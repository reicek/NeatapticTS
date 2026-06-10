/**
 * @module mcp-utils
 * @description Dependency-light MCP server utilities for the NeatapticTS workflow and validation servers.
 *
 * Provides everything needed to stand up a minimalist stdio JSON-RPC 2.0 MCP
 * server without the full `@modelcontextprotocol/sdk` package: framing detection,
 * streaming parser, tool registry, dispatch, and shell-free command execution.
 *
 * @remarks
 * ### Stdio Framing Detection and Parser Flow
 *
 * ```mermaid
 * flowchart TD
 *   A[stdin data chunk] --> B[Append to buffer]
 *   B --> C[Drop leading CRLF/LF bytes]
 *   C --> D{Starts with Content-Length header?}
 *   D -- yes --> E[readContentLengthFrame]
 *   E --> F{Full body received?}
 *   F -- no  --> G[Wait for more data]
 *   F -- yes --> H[onMessage - content-length frame]
 *   D -- no  --> I{LF found in buffer?}
 *   I -- no  --> G
 *   I -- yes --> J[onMessage - newline-delimited frame]
 *   H & J --> K[JSON.parse request]
 *   K --> L[server.dispatch]
 *   L --> M[writeJsonRpcMessage response]
 * ```
 */
import { spawn } from 'node:child_process';
import { Buffer } from 'node:buffer';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  issue,
  summarizeIssues,
  writeReport,
} from '../customization-utils.mjs';

/**
 * MCP protocol version negotiated during the JSON-RPC 2.0 `initialize` handshake
 * and echoed in every server capability response sent to the host.
 */
export const MCP_PROTOCOL_VERSION = '2024-11-05';
/** Absolute path to the repository root, resolved from this module's location. */
export const MCP_REPO_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../..',
);

/** Default maximum captured output bytes per stream when running shell-free commands. */
const DEFAULT_OUTPUT_LIMIT = 24_576;
/** Framing mode identifier for Content-Length–delimited JSON-RPC messages (used by VS Code). */
const CONTENT_LENGTH_FRAME = 'content-length';
/** HTTP-style header name used when reading Content-Length frames. */
const CONTENT_LENGTH_HEADER = 'content-length';
/** CRLF double-newline that separates the header block from the message body. */
const HEADER_SEPARATOR = Buffer.from('\r\n\r\n');
/** Framing mode identifier for newline-delimited JSON-RPC messages (one object per line). */
const NEWLINE_DELIMITED_FRAME = 'newline-delimited';
/** Rejects shell metacharacters before shell-free command execution to prevent injection. */
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
    plan: argv
      .find((argument) => argument.startsWith('--plan='))
      ?.slice('--plan='.length),
  };
}

/**
 * Validate and return an explicit plan path for direct-MCP entrypoints,
 * throwing a descriptive error when no path was supplied or the value is blank.
 *
 * @param {string | undefined} planPath - CLI plan path.
 * @returns {string} Trimmed plan path.
 */
export function requireExplicitPlanPath(planPath) {
  if (typeof planPath !== 'string' || !planPath.trim()) {
    throw new Error(
      'MCP entrypoints require --plan=<path>; no default plan path is assumed.',
    );
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
  const isRepoRelativePath =
    repoRelativePath !== '' &&
    !repoRelativePath.startsWith('..') &&
    !path.isAbsolute(repoRelativePath);

  return {
    absolutePath,
    displayPath: normalizePath(
      isRepoRelativePath ? repoRelativePath : absolutePath,
    ),
  };
}

/**
 * Print structured help text for a direct-MCP entrypoint, listing available
 * modes, required CLI flags, and the registered tool names with their descriptions.
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
  console.log(
    '  --plan=<path> Required workflow plan path resolved from the repository root.',
  );
  console.log('  --help        Show this help text.');
  console.log('  --json        Write machine-readable self-check output.');
  console.log(
    '  --self-check  Run bounded local checks instead of starting the stdio server.',
  );
  console.log(
    '  default       Start the stdio MCP server on stdin/stdout using the explicit plan path.',
  );
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
 * Build a standard self-check report that merges an issue summary with
 * caller-supplied diagnostics into one JSON-serializable payload.
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
 * Create a single MCP tool descriptor with a validated `inputSchema`, merging
 * caller defaults so every tool advertises a consistent JSON Schema contract to the host.
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
  const serverInfo = { name: serverName, version: serverVersion };
  const toolRegistry = new Map(tools.map((tool) => [tool.name, tool]));
  const listedTools = tools.map(({ handler, ...tool }) => tool);
  const methodHandlers = createMcpMethodHandlerMap({
    listedTools,
    serverInfo,
    toolRegistry,
  });

  return {
    serverInfo,
    tools: listedTools,
    async dispatch(request) {
      const methodHandler = methodHandlers.get(request.method);
      if (!methodHandler) {
        throw createJsonRpcError(
          -32601,
          `Unsupported method: ${String(request.method)}`,
        );
      }

      return await methodHandler(request);
    },
  };
}

/**
 * Run a dependency-light stdio MCP server using Content-Length or newline-delimited JSON-RPC.
 *
 * @param {{ serverInfo: { name: string, version: string }, dispatch: (request: Record<string, unknown>) => Promise<unknown> }} server - Server implementation.
 * @returns {Promise<void>} Resolves when stdin closes.
 */
export async function runStdioMcpServer(server) {
  const parser = createStdioJsonRpcParser((framedMessage) =>
    processStdioServerMessage(server, framedMessage),
  );

  process.stdin.on('data', createQueuedStdinParserHandler(parser));

  process.stdin.resume();
  await waitForStdinEnd();
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
 * Resolve an executable and argument list to a spawn-safe target for the current platform.
 *
 * On Windows, `npx` and `npm` are `.cmd` batch scripts co-located with `node.exe`.
 * Windows cannot directly execute `.cmd` files via `spawn` with `shell: false` —
 * they require `cmd.exe` as the interpreter. This resolver detects that case and
 * wraps the call in an explicit `cmd.exe /c <absolute-cmd-path> args...` invocation,
 * keeping `shell: false` in the `spawn` options so cmd.exe itself is spawned as a
 * native executable rather than through an environment-supplied shell.
 *
 * Using `ComSpec` (or the `cmd.exe` fallback) instead of `shell: true` preserves the
 * security invariant: the shell interpreter is fixed and not injectable from PATH.
 *
 * @param {string} requestedExecutable - Executable token from the tokenized command.
 * @param {string[]} requestedArgv - Argument list for the executable.
 * @returns {{ executable: string, argv: string[] }} Spawn-safe executable and argv.
 */
function resolveSpawnTarget(requestedExecutable, requestedArgv) {
  if (/^(node|node\.exe)$/iu.test(requestedExecutable)) {
    return { executable: process.execPath, argv: requestedArgv };
  }

  if (
    process.platform === 'win32' &&
    /^(npx|npm)(\.cmd)?$/iu.test(requestedExecutable)
  ) {
    const baseName = requestedExecutable.toLowerCase().replace(/\.cmd$/iu, '');
    const cmdScriptPath = path.join(
      path.dirname(process.execPath),
      `${baseName}.cmd`,
    );
    const comSpec = process.env['ComSpec'] ?? 'cmd.exe';
    return {
      executable: comSpec,
      argv: ['/c', cmdScriptPath, ...requestedArgv],
    };
  }

  return { executable: requestedExecutable, argv: requestedArgv };
}

/**
 * Run an exact allow-listed command without invoking a shell, capturing bounded
 * stdout and stderr output and returning a structured process result.
 *
 * The executable is spawned directly via `child_process.spawn` with `shell: false`.
 * Shell metacharacters in the command string are rejected before any process is started.
 *
 * @param {string} commandString - Exact command string from the active step packet.
 * @param {{ maxOutputBytes?: number }} [options] - Optional output capture limits.
 * @returns {Promise<{ command: string, executable: string, argv: string[], exitCode: number, stdout: string, stderr: string, durationMs: number, truncated: { stdout: boolean, stderr: boolean } }>} Process result.
 */
export async function runShellFreeCommand(commandString, options = {}) {
  const tokens = tokenizeShellSafeCommand(commandString);
  const [requestedExecutable, ...requestedArgv] = tokens;
  const { executable, argv } = resolveSpawnTarget(
    requestedExecutable,
    requestedArgv,
  );
  const startTime = Date.now();
  const maxOutputBytes =
    Number.isFinite(options.maxOutputBytes) && options.maxOutputBytes > 0
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
 * Tokenize a command string into an executable name and argument array, rejecting
 * shell metacharacters and unterminated quotes before any process is spawned.
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
    throw new Error(
      'Shell metacharacters are not allowed in validation commands.',
    );
  }

  const tokens = scanShellSafeTokens(normalizedCommand);

  if (tokens.length === 0) {
    throw new Error('Validation command produced no executable token.');
  }

  return tokens;
}

/**
 * Assert that a value is a non-empty string and return its trimmed form,
 * throwing a descriptive error that names the field when the assertion fails.
 *
 * @param {unknown} value - Value to validate.
 * @param {string} fieldName - Human-readable field name used in the error message.
 * @returns {string} Validated and trimmed string.
 */
export function requireString(value, fieldName) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${fieldName} must be a non-empty string.`);
  }

  return value.trim();
}

/**
 * Build a severity-`'error'` issue object for inclusion in a self-check report,
 * conforming to the shared issue schema consumed by `summarizeIssues`.
 *
 * @param {string} path - Issue path.
 * @param {string} message - Issue message.
 * @returns {{ severity: string, path: string, message: string }} Issue.
 */
export function selfCheckError(path, message) {
  return issue('error', path, message);
}

/**
 * Create a standard JSON-RPC error object detectable by the dispatch loop.
 *
 * The `jsonRpcCode` and optional `jsonRpcData` properties are read by the
 * dispatch loop to build a well-formed JSON-RPC error envelope rather than
 * a generic internal-error response.
 *
 * @param {number} code - JSON-RPC error code (e.g. -32601 for method-not-found).
 * @param {string} message - Human-readable error description.
 * @param {unknown} [data] - Optional additional error data.
 * @returns {Error} Error instance with `jsonRpcCode` and `jsonRpcData` properties.
 */
function createJsonRpcError(code, message, data) {
  const error = new Error(message);
  error.jsonRpcCode = code;
  error.jsonRpcData = data;
  return error;
}

function createMcpMethodHandlerMap({ listedTools, serverInfo, toolRegistry }) {
  return new Map([
    ['initialize', () => createInitializeResult(serverInfo)],
    ['notifications/initialized', () => null],
    ['ping', () => ({})],
    ['tools/list', () => ({ tools: listedTools })],
    [
      'tools/call',
      (request) => callRegisteredTool(toolRegistry, request.params),
    ],
    ['resources/list', () => ({ resources: [] })],
    ['prompts/list', () => ({ prompts: [] })],
  ]);
}

function createInitializeResult(serverInfo) {
  return {
    protocolVersion: MCP_PROTOCOL_VERSION,
    capabilities: {
      tools: {},
    },
    serverInfo,
  };
}

async function callRegisteredTool(toolRegistry, params) {
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
    const handlerArguments = isPlainObject(params.arguments)
      ? params.arguments
      : {};
    const handlerResult = await tool.handler(handlerArguments);
    return formatToolResult(handlerResult);
  } catch (error) {
    if (isJsonRpcError(error)) {
      throw error;
    }

    return createToolErrorResult(error);
  }
}

function isJsonRpcError(error) {
  return Boolean(
    error &&
    typeof error === 'object' &&
    'jsonRpcCode' in error &&
    typeof error.jsonRpcCode === 'number',
  );
}

/**
 * Wrap a tool handler return value in the MCP `tools/call` response envelope.
 *
 * If the handler already returned a pre-formatted content array it is passed
 * through unchanged. Plain objects are serialized to a JSON text block and
 * also exposed as `structuredContent` for type-safe callers.
 *
 * @param {unknown} handlerResult - Raw value returned by the tool handler.
 * @returns {{ content: Array<{ type: string, text: string }>, structuredContent: object, isError: false }} Formatted tool result.
 */
function formatToolResult(handlerResult) {
  if (isPlainObject(handlerResult) && Array.isArray(handlerResult.content)) {
    return {
      isError: false,
      ...handlerResult,
    };
  }

  const structuredContent = isPlainObject(handlerResult)
    ? handlerResult
    : { value: handlerResult ?? null };
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

/**
 * Build an error tool-call response from a caught exception.
 *
 * @param {unknown} error - Caught error value.
 * @returns {{ content: Array<{ type: string, text: string }>, structuredContent: { error: string }, isError: true }} Error tool result.
 */
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

/**
 * Return `true` when the value is a plain non-null non-array object.
 *
 * @param {unknown} value - Value to test.
 * @returns {boolean} Whether the value is a plain object.
 */
function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function processStdioServerMessage(server, { messageBuffer, framing }) {
  const request = parseStdioJsonRpcRequest(messageBuffer, framing);
  if (!request) {
    return;
  }

  try {
    const result = await server.dispatch(request);
    writeJsonRpcSuccessResponse(request, result, framing);
  } catch (error) {
    writeStdioJsonRpcErrorResponse(request, error, framing);
  }
}

function parseStdioJsonRpcRequest(messageBuffer, framing) {
  try {
    return JSON.parse(messageBuffer.toString('utf8'));
  } catch {
    writeJsonRpcMessage(
      {
        jsonrpc: '2.0',
        id: null,
        error: { code: -32700, message: 'Invalid JSON payload.' },
      },
      framing,
    );
    return null;
  }
}

function writeJsonRpcSuccessResponse(request, result, framing) {
  if (request.id === undefined || result === null) {
    return;
  }

  writeJsonRpcMessage(
    {
      jsonrpc: '2.0',
      id: request.id,
      result,
    },
    framing,
  );
}

function writeStdioJsonRpcErrorResponse(request, error, framing) {
  if (request?.id === undefined) {
    writeDiagnostic(error instanceof Error ? error.message : String(error));
    return;
  }

  writeJsonRpcMessage(
    {
      jsonrpc: '2.0',
      id: request.id,
      error: {
        code: error?.jsonRpcCode ?? -32603,
        message: error instanceof Error ? error.message : String(error),
        ...(error?.jsonRpcData === undefined
          ? {}
          : { data: error.jsonRpcData }),
      },
    },
    framing,
  );
}

function createQueuedStdinParserHandler(parser) {
  let pending = Promise.resolve();

  return (chunk) => {
    pending = pending
      .then(() => parser.push(chunk))
      .catch((error) => {
        writeDiagnostic(error instanceof Error ? error.message : String(error));
        process.exitCode = 1;
      });
  };
}

function waitForStdinEnd() {
  return new Promise((resolve) => process.stdin.on('end', resolve));
}

function scanShellSafeTokens(normalizedCommand) {
  const tokenizerState = {
    activeQuote: null,
    currentToken: '',
    tokens: [],
  };

  for (const character of normalizedCommand) {
    consumeShellSafeTokenCharacter(tokenizerState, character);
  }

  if (tokenizerState.activeQuote) {
    throw new Error('Validation command contains an unterminated quote.');
  }

  if (tokenizerState.currentToken) {
    tokenizerState.tokens.push(tokenizerState.currentToken);
  }

  return tokenizerState.tokens;
}

function consumeShellSafeTokenCharacter(tokenizerState, character) {
  if (tokenizerState.activeQuote) {
    if (character === tokenizerState.activeQuote) {
      tokenizerState.activeQuote = null;
      return;
    }

    tokenizerState.currentToken += character;
    return;
  }

  if (character === '"' || character === "'") {
    tokenizerState.activeQuote = character;
    return;
  }

  if (/\s/u.test(character)) {
    flushShellSafeToken(tokenizerState);
    return;
  }

  tokenizerState.currentToken += character;
}

function flushShellSafeToken(tokenizerState) {
  if (!tokenizerState.currentToken) {
    return;
  }

  tokenizerState.tokens.push(tokenizerState.currentToken);
  tokenizerState.currentToken = '';
}

/**
 * Serialize and write a JSON-RPC message to stdout using the detected framing.
 *
 * @param {Record<string, unknown>} payload - JSON-RPC envelope to serialize.
 * @param {string} [framing] - Framing mode (`content-length` or `newline-delimited`).
 * @returns {void}
 */
function writeJsonRpcMessage(payload, framing = CONTENT_LENGTH_FRAME) {
  const serializedPayload = JSON.stringify(payload);

  if (framing === NEWLINE_DELIMITED_FRAME) {
    process.stdout.write(`${serializedPayload}\n`);
    return;
  }

  const contentLength = Buffer.byteLength(serializedPayload, 'utf8');
  process.stdout.write(
    `Content-Length: ${contentLength}\r\n\r\n${serializedPayload}`,
  );
}

/**
 * Write a diagnostic message to stderr.
 *
 * Used for notification processing errors and non-fatal dispatch failures
 * where no JSON-RPC response is required or appropriate.
 *
 * @param {string} message - Diagnostic message text.
 * @returns {void}
 */
function writeDiagnostic(message) {
  process.stderr.write(`${message}\n`);
}

/**
 * Create a stateful incremental parser for the MCP stdio JSON-RPC protocol.
 *
 * Accumulates binary chunks pushed via `parser.push(chunk)` and detects the
 * framing format automatically: Content-Length–delimited (used by VS Code and
 * the official MCP SDK) or newline-delimited (used by some lightweight clients).
 *
 * @param {(params: { messageBuffer: Buffer, framing: string }) => Promise<void>} onMessage - Callback invoked with each complete message buffer and its detected framing type.
 * @returns {{ push: (chunk: Buffer | Uint8Array) => Promise<void> }} Incremental parser.
 */
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

        const messageBuffer = trimTrailingCarriageReturn(
          buffer.subarray(0, lineEndIndex),
        );
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

/**
 * Return `true` when the buffer begins with a Content-Length HTTP-style header.
 *
 * Checks only the first 64 bytes to keep the probe cheap. Case-insensitive to
 * handle non-conforming clients that capitalize the header differently.
 *
 * @param {Buffer} buffer - Input buffer.
 * @returns {boolean} Whether the buffer starts with a content-length header.
 */
function startsWithContentLengthHeader(buffer) {
  const leadingText = buffer
    .subarray(0, Math.min(buffer.length, 64))
    .toString('utf8');
  return /^content-length\s*:/iu.test(leadingText);
}

/**
 * Attempt to read one complete Content-Length–delimited frame from the buffer.
 *
 * Returns `null` when the buffer does not yet contain the full message body
 * so the caller can wait for more data chunks without losing buffered bytes.
 *
 * @param {Buffer} buffer - Accumulated input buffer.
 * @returns {{ messageBuffer: Buffer, remainingBuffer: Buffer } | null} Parsed frame, or `null` if incomplete.
 * @throws {Error} When a Content-Length header is present but its value is invalid.
 */
function readContentLengthFrame(buffer) {
  const headerEndIndex = buffer.indexOf(HEADER_SEPARATOR);
  if (headerEndIndex === -1) {
    return null;
  }

  const headerText = buffer.subarray(0, headerEndIndex).toString('utf8');
  const contentLength = extractContentLength(headerText);
  if (contentLength === null) {
    throw new Error(
      'Received an MCP message without a valid Content-Length header.',
    );
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

/**
 * Advance past any leading CR (0x0D) or LF (0x0A) bytes in the buffer.
 *
 * The MCP stdio framing often includes trailing line-break separators between
 * messages. Stripping them avoids false framing-detection failures when the
 * parser resumes after writing a response.
 *
 * @param {Buffer} buffer - Input buffer.
 * @returns {Buffer} Buffer slice starting at the first non-line-break byte.
 */
function dropLeadingLineBreaks(buffer) {
  let firstContentIndex = 0;
  while (
    firstContentIndex < buffer.length &&
    (buffer[firstContentIndex] === 10 || buffer[firstContentIndex] === 13)
  ) {
    firstContentIndex += 1;
  }

  return firstContentIndex === 0 ? buffer : buffer.subarray(firstContentIndex);
}

/**
 * Remove a trailing CR byte (0x0D) from a buffer slice if present.
 *
 * Needed when splitting on LF alone, since CRLF-terminated lines leave a
 * dangling CR that would corrupt the JSON parse.
 *
 * @param {Buffer} buffer - Input buffer slice ending at a LF.
 * @returns {Buffer} Buffer without a trailing carriage return.
 */
function trimTrailingCarriageReturn(buffer) {
  return buffer.at(-1) === 13 ? buffer.subarray(0, -1) : buffer;
}

/**
 * Parse the numeric value of a `Content-Length` header from raw header text.
 *
 * @param {string} headerText - Raw header section text before the double CRLF.
 * @returns {number | null} Parsed content length, or `null` when absent or invalid.
 */
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

/**
 * Append a truncation notice to captured output when the byte limit was reached.
 *
 * @param {string} output - Captured output text.
 * @param {boolean} truncated - Whether the output was truncated at the byte limit.
 * @returns {string} Output string with `[output truncated]` appended if needed.
 */
function finalizeCapturedOutput(output, truncated) {
  if (!truncated) {
    return output;
  }

  return `${output}\n[output truncated]`;
}

function normalizePath(value) {
  return value.replaceAll(path.sep, '/');
}
