/**
 * @module lazy-facade-core
 * @description Shared lazy-load MCP facade machinery for the `cortex` and
 * `devtools` stdio proxy servers.
 *
 * Each facade exposes a single router tool to the host.  The router accepts
 * `{ operation, args? }` and lazily spawns the real heavy MCP server on the first
 * `tools/call`.  After the lazy spawn it reuses the cached child stdio transport
 * for the lifetime of the facade process.
 *
 * The child transport speaks newline-delimited JSON-RPC, which both real targets
 * accept (Repo Cortex natively, Chrome DevTools MCP via its bundled SDK
 * transport).  The host side is served by {@link runStdioMcpServer}, which
 * auto-detects Content-Length vs. NDJSON framing and replies in the same format.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

import {
  emitSelfCheckReport,
  MCP_REPO_ROOT,
  parseMcpCliArgs,
  runStdioMcpServer,
} from './mcp-utils.mjs';

/**
 * Build a single-router lazy-load facade for a single MCP server.
 *
 * @param {object} config - Facade configuration.
 * @param {string} config.name - Short facade server key (e.g. `cortex`).  This is also the name of the one router tool exposed to the host.
 * @param {string} config.target - Real server name used in diagnostics.
 * @param {string} config.title - Human-readable server title for `initialize`.
 * @param {string} config.version - Facade version string.
 * @param {string} config.defaultSnapshotPath - Default absolute path to the static tool snapshot JSON.
 * @param {string[]} config.defaultSpawnCommand - Default command/argv used to spawn the real server.
 * @param {Record<string, string>} [config.defaultEnv] - Extra environment variables merged into `process.env` for the child.
 * @param {string} [config.snapshotPath] - Optional override for `defaultSnapshotPath`.
 * @param {string[]} [config.spawnCommand] - Optional override for `defaultSpawnCommand`.
 * @param {Record<string, string>} [config.env] - Optional env overrides applied on top of `defaultEnv`.
 * @param {'single-tool'|'native'} [config.routingMode='single-tool'] - How the router tool forwards calls.
 *   `single-tool` wraps every operation as a `tools/call` to the real server's
 *   single named tool (used by `cortex`).  `native` forwards known MCP JSON-RPC
 *   methods directly and treats all other operations as real native tool calls
 *   (used by `devtools`).
 * @returns {{
 *   dispatch: (request: Record<string, unknown>) => Promise<Record<string, unknown>>,
 *   server: { serverInfo: { name: string, version: string }, dispatch: (request: Record<string, unknown>) => Promise<unknown> },
 *   close: () => Promise<void>,
 * }}
 */
export function createLazyFacade(config) {
  const serverInfo = { name: config.name, version: config.version };
  const snapshotPath = config.snapshotPath ?? config.defaultSnapshotPath;
  const spawnCommand = config.spawnCommand ?? config.defaultSpawnCommand;
  const env = { ...config.defaultEnv, ...config.env };
  const routingMode = config.routingMode ?? 'single-tool';

  const initializeResult = {
    protocolVersion: '2024-11-05',
    capabilities: { tools: {} },
    serverInfo: {
      name: config.name,
      title: config.title,
      version: config.version,
    },
  };

  const snapshot = loadSnapshot(snapshotPath);
  const routerToolName = config.name;

  const childTransport = createChildTransport({
    spawnCommand,
    env,
    repoRoot: MCP_REPO_ROOT,
  });

  /**
   * Direct dispatch that returns a full JSON-RPC response envelope.  This is the
   * shape expected by the lazy-facade unit tests; the stdio server path uses
   * `facade.server.dispatch`, which returns only the result payload.
   *
   * @param {Record<string, unknown>} request - JSON-RPC request object.
   * @returns {Promise<Record<string, unknown>>} JSON-RPC response envelope.
   */
  async function dispatch(request) {
    try {
      const result = await handleMethod(request, {
        initializeResult,
        snapshot,
        routerToolName,
        childTransport,
        facadeName: config.name,
        targetName: config.target,
        routingMode,
      });
      return buildSuccessResponse(request, result);
    } catch (error) {
      return buildErrorResponse(request, error);
    }
  }

  /**
   * Server-style dispatch that returns the bare JSON-RPC result.  Used by
   * `runStdioMcpServer` to wrap its own framing and envelope.
   *
   * @param {Record<string, unknown>} request - JSON-RPC request object.
   * @returns {Promise<unknown>} JSON-RPC result payload.
   */
  async function serverDispatch(request) {
    return handleMethod(request, {
      initializeResult,
      snapshot,
      routerToolName,
      childTransport,
      facadeName: config.name,
      targetName: config.target,
      routingMode,
    });
  }

  return {
    serverInfo,
    dispatch,
    server: { serverInfo, dispatch: serverDispatch },
    close: async () => {
      childTransport.close();
    },
  };
}

/**
 * Route a JSON-RPC request to the correct local or proxy handler.
 *
 * @param {Record<string, unknown>} request - JSON-RPC request object.
 * @param {object} context - Method context.
 */
async function handleMethod(request, context) {
  switch (request.method) {
    case 'initialize':
      return context.initializeResult;
    case 'notifications/initialized':
      return null;
    case 'ping':
      return {};
    case 'tools/list':
      return { tools: context.snapshot?.tools ?? [] };
    case 'tools/call':
      return handleToolCall(request, context);
    default:
      throw createJsonRpcError(
        -32601,
        `Unsupported method: ${String(request.method)}`,
      );
  }
}

/**
 * Parse CLI flags and either run the stdio server or emit a self-check report.
 *
 * @param {object} config - Facade configuration passed through to {@link createLazyFacade}.
 * @param {string[]} argv - `process.argv.slice(2)`.
 */
export function runFacadeMain(config, argv) {
  const options = parseMcpCliArgs(argv);

  if (options.selfCheck) {
    runSelfCheck(config, options);
    return;
  }

  const facade = createLazyFacade(config);
  runStdioMcpServer(facade.server).catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

/**
 * Run the bounded local self-check without touching the real server.
 *
 * @param {object} config - Facade configuration.
 * @param {{ json: boolean }} options - CLI output options.
 */
function runSelfCheck(config, options) {
  let snapshotValid = false;
  let canParseSnapshot = false;
  let snapshotToolCount = 0;

  try {
    const raw = readFileSync(config.defaultSnapshotPath, 'utf8');
    canParseSnapshot = true;
    const parsed = JSON.parse(raw);
    const tools = Array.isArray(parsed.tools) ? parsed.tools : [];
    snapshotToolCount = tools.length;
    const routerTool = tools.find(
      (tool) => isPlainObject(tool) && tool.name === config.name,
    );
    snapshotValid =
      snapshotToolCount === 1 &&
      routerTool != null &&
      isPlainObject(routerTool.inputSchema) &&
      Array.isArray(routerTool.inputSchema.required) &&
      routerTool.inputSchema.required.includes('operation');
  } catch {
    // Snapshot is missing or unreadable; the report will surface that.
  }

  const report = {
    facade: config.name,
    target: config.target,
    snapshotPath: path
      .relative(MCP_REPO_ROOT, config.defaultSnapshotPath)
      .replace(/\\/g, '/'),
    snapshotToolCount,
    snapshotValid,
    canParseSnapshot,
    spawnCommand: config.defaultSpawnCommand.join(' '),
    pass:
      snapshotValid &&
      canParseSnapshot &&
      config.defaultSpawnCommand.length > 0,
  };

  emitSelfCheckReport(report, options);
  if (!report.pass) {
    process.exitCode = 1;
  }
}

/**
 * Read and parse a tool snapshot JSON file.
 *
 * @param {string} snapshotPath - Absolute path to the snapshot.
 * @returns {{ tools: Array<{ name: string, description: string, inputSchema: object }> } | null}
 */
function loadSnapshot(snapshotPath) {
  try {
    const raw = readFileSync(snapshotPath, 'utf8');
    return JSON.parse(raw);
  } catch (error) {
    console.error(`Failed to load snapshot ${snapshotPath}: ${error.message}`);
    return null;
  }
}

/**
 * Handle a host `tools/call` request.
 *
 * Only the router tool (named after the facade itself) is accepted.  The router
 * payload must contain `{ operation: string, args?: object | null }`.  The
 * operation is forwarded to the real server lazily, and the response is returned
 * to the host.
 *
 * @param {Record<string, unknown>} request - JSON-RPC request.
 * @param {object} context - Method context.
 */
async function handleToolCall(request, context) {
  const params = isPlainObject(request.params) ? request.params : {};
  const toolName = typeof params.name === 'string' ? params.name : null;
  const routerArgs = isPlainObject(params.arguments) ? params.arguments : {};
  const operation =
    typeof routerArgs.operation === 'string' ? routerArgs.operation : null;

  if (toolName !== context.routerToolName) {
    return createToolErrorResult({
      facade: context.facadeName,
      target: context.targetName,
      available: true,
      error: `Unknown operation: ${toolName ?? 'undefined'}`,
    });
  }

  if (!operation) {
    return createToolErrorResult({
      facade: context.facadeName,
      target: context.targetName,
      available: true,
      error: `Router payload must include { operation: string, args? }.`,
    });
  }

  const nativeRpcMethods = new Set([
    'initialize',
    'notifications/initialized',
    'ping',
    'tools/list',
    'tools/call',
    'resources/list',
    'resources/read',
    'prompts/list',
    'prompts/get',
  ]);

  try {
    let result;
    if (context.routingMode === 'native') {
      if (nativeRpcMethods.has(operation)) {
        result = await context.childTransport.callMethod(
          operation,
          routerArgs.args ?? {},
        );
      } else {
        result = await context.childTransport.callTool(
          operation,
          routerArgs.args ?? {},
        );
      }
    } else {
      result = await context.childTransport.callTool(
        operation,
        routerArgs.args ?? {},
      );
    }
    return applyFacadePrefixIfError(result, context.facadeName);
  } catch (error) {
    return createToolErrorResult({
      facade: context.facadeName,
      target: context.targetName,
      available: false,
      error: error instanceof Error ? error.message : String(error),
      fallbackHint: `Start the real ${context.targetName} server manually or check the spawn command.`,
    });
  }
}

/**
 * Prefix the first text content of a real-server error response so it is
 * attributable to the facade in logs.
 *
 * @param {unknown} result - Raw tool result from the child.
 * @param {string} facadeName - Short facade name.
 * @returns {unknown}
 */
function applyFacadePrefixIfError(result, facadeName) {
  if (!isPlainObject(result) || result.isError !== true) {
    return result;
  }

  const prefix = `[${facadeName}] `;
  const content = Array.isArray(result.content) ? [...result.content] : [];
  const firstTextIndex = content.findIndex(
    (item) =>
      isPlainObject(item) &&
      item.type === 'text' &&
      typeof item.text === 'string',
  );

  if (firstTextIndex !== -1) {
    content[firstTextIndex] = {
      ...content[firstTextIndex],
      text: prefix + content[firstTextIndex].text,
    };
  }

  return { ...result, content };
}

/**
 * Build an MCP `tools/call` error result from a structured diagnostic payload.
 *
 * @param {Record<string, unknown>} payload - Diagnostic payload.
 */
function createToolErrorResult(payload) {
  return {
    isError: true,
    content: [
      {
        type: 'text',
        text: JSON.stringify(payload, null, 2),
      },
    ],
  };
}

/**
 * Create a lazy child stdio transport that spawns the real server on first use.
 *
 * @param {object} options - Transport options.
 * @param {string[]} options.spawnCommand - Spawn argv.
 * @param {Record<string, string>} options.env - Extra env vars for the child.
 * @param {string} options.repoRoot - Repository root used as `cwd`.
 */
/**
 * Resolve a bare command name to an executable file on Windows.
 * Node's `spawn` on Windows does not resolve `.cmd`/`.ps1` scripts when
 * `shell: false`, so we detect the correct wrapper. On other platforms the
 * command is used as-is.
 *
 * @param {string} command - First element of the spawn argv.
 * @returns {{ file: string, args: string[], shell: boolean }} Resolved spawn options.
 */
export function resolveSpawnCommand(command) {
  if (process.platform !== 'win32') {
    return { file: command, args: [], shell: false };
  }

  const envPath = process.env.PATH ?? '';
  const searchDirs = envPath.split(path.delimiter);
  // On Windows, `npx` is often installed next to the Node executable rather
  // than on PATH in bundled / CI environments.  Fall back to that directory
  // when the command is not found in PATH.
  searchDirs.push(path.dirname(process.execPath));

  for (const dir of searchDirs) {
    if (!dir) continue;
    const base = path.join(dir, command);
    if (existsSync(`${base}.exe`)) {
      // .exe binaries can be spawned directly; keep the original command so
      // cross-platform contract tests stay stable.
      return { file: command, args: [], shell: false };
    }
    const wrapperExtensions = ['.cmd', '.ps1', '.bat'];
    for (const ext of wrapperExtensions) {
      const wrapperPath = `${base}${ext}`;
      if (existsSync(wrapperPath)) {
        // Node 25+ rejects .cmd/.ps1/.bat scripts with shell:false on Windows.
        // Return the absolute wrapper path and let the system shell resolve it.
        return { file: wrapperPath, args: [], shell: true };
      }
    }
  }
  return { file: command, args: [], shell: true };
}

function createChildTransport({ spawnCommand, env, repoRoot }) {
  /** @type {import('node:child_process').ChildProcess | null} */
  let child = null;
  let initPromise = null;
  let active = false;
  let nextId = 1;
  const pending = new Map();
  let buffer = '';

  function ensureInit() {
    if (initPromise) {
      return initPromise;
    }

    initPromise = (async () => {
      try {
        const command = resolveSpawnCommand(spawnCommand[0]);
        child = spawn(
          command.file,
          command.args.concat(spawnCommand.slice(1)),
          {
            cwd: repoRoot,
            stdio: ['pipe', 'pipe', 'pipe'],
            env: { ...process.env, ...env },
            shell: command.shell,
          },
        );
      } catch (error) {
        throw new Error(
          `Failed to spawn ${spawnCommand.join(' ')}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }

      if (!child || !child.stdin || !child.stdout || !child.stderr) {
        throw new Error('Spawn returned an invalid child process.');
      }

      child.stdout.on('data', handleChildData);
      child.stderr.on('data', () => {
        // Real servers may print legal/telemetry disclaimers to stderr.
        // Diagnostics are intentionally dropped here to keep the proxy quiet.
      });
      child.on('error', handleChildCrash);
      child.on('exit', (code, signal) => {
        handleChildCrash(
          new Error(
            `Child exited (code=${code ?? 'null'}, signal=${signal ?? 'null'}).`,
          ),
        );
      });

      await sendChildRequest('initialize', {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'lazy-facade', version: '0.1.0' },
      });

      child.stdin.write(
        `${JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' })}\n`,
      );
      active = true;
    })();

    return initPromise;
  }

  function handleChildData(chunk) {
    buffer += chunk.toString('utf8');

    while (buffer.length > 0) {
      const newlineIndex = buffer.indexOf('\n');
      if (newlineIndex === -1) {
        return;
      }

      const line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      if (!line.trim()) {
        continue;
      }

      try {
        const message = JSON.parse(line);
        if (message.id != null && pending.has(message.id)) {
          const { resolve, reject } = pending.get(message.id);
          pending.delete(message.id);
          if (isPlainObject(message.error)) {
            reject(
              new Error(
                typeof message.error.message === 'string'
                  ? message.error.message
                  : JSON.stringify(message.error),
              ),
            );
          } else {
            resolve(message.result);
          }
        }
      } catch {
        // Ignore malformed child output.
      }
    }
  }

  function handleChildCrash(error) {
    active = false;
    for (const { reject } of pending.values()) {
      reject(error);
    }
    pending.clear();
    child?.stdout?.removeAllListeners('data');
    child?.stderr?.removeAllListeners('data');
    try {
      child?.kill();
    } catch {
      // Already dead; ignore.
    }
    child = null;
    initPromise = null;
  }

  function sendChildRequest(method, params) {
    return new Promise((resolve, reject) => {
      if (!child || !child.stdin) {
        reject(new Error('Child transport is not available.'));
        return;
      }

      const id = nextId++;
      pending.set(id, { resolve, reject });
      child.stdin.write(
        `${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`,
      );
    });
  }

  async function callMethod(method, params) {
    await ensureInit();
    return sendChildRequest(method, params);
  }

  async function callTool(toolName, args) {
    return callMethod('tools/call', { name: toolName, arguments: args });
  }

  function close() {
    active = false;
    if (child) {
      try {
        child.kill();
      } catch {
        // Ignore.
      }
      child = null;
    }
    for (const { reject } of pending.values()) {
      reject(new Error('Facade closed.'));
    }
    pending.clear();
    initPromise = null;
  }

  return { callTool, callMethod, close };
}

/**
 * Wrap a successful result in a JSON-RPC response envelope.
 *
 * @param {Record<string, unknown>} request - Original request.
 * @param {unknown} result - Method result.
 */
function buildSuccessResponse(request, result) {
  const response = { jsonrpc: '2.0', result };
  if (request.id !== undefined) {
    response.id = request.id;
  }
  return response;
}

/**
 * Wrap a thrown error in a JSON-RPC error response envelope.
 *
 * @param {Record<string, unknown>} request - Original request.
 * @param {unknown} error - Thrown error.
 */
function buildErrorResponse(request, error) {
  const response = {
    jsonrpc: '2.0',
    error: {
      code: error.jsonRpcCode,
      message: error.message,
    },
  };
  if (request.id !== undefined) {
    response.id = request.id;
  }
  return response;
}

/**
 * Create a JSON-RPC error object that the mcp-utils stdio loop recognizes.
 *
 * @param {number} code - JSON-RPC error code.
 * @param {string} message - Error message.
 */
function createJsonRpcError(code, message) {
  const error = new Error(message);
  error.jsonRpcCode = code;
  return error;
}

/**
 * Return `true` when the value is a plain non-null non-array object.
 *
 * @param {unknown} value
 */
function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
