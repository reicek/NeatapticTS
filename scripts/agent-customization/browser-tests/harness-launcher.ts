import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * @file Browser testing harness — minimal local static-file server for hidden
 * `docs/browser-tests/` smoke scenarios.
 *
 * The server is rooted at the caller's chosen directory and serves the scenario
 * page plus any relative assets (the IIFE bundle, module scripts, etc.). Callers
 * must invoke {@link LocalServerHandle.teardown} when the scenario is complete so
 * the port is released and sockets are closed.
 */

/**
 * Default port for the local NeatapticTS documentation/test server.
 */
export const DEFAULT_SERVER_PORT = 8080;

/**
 * Default scenario path served under the local server root.
 */
export const DEFAULT_SCENARIO_PATH =
  '/docs/browser-tests/webgpu-inference-smoke.html';

/**
 * Default timeout (in milliseconds) to wait for the HTTP server to report that
 * it is ready.
 */
export const DEFAULT_SERVER_START_TIMEOUT_MS = 10_000;

/**
 * Options for {@link launchLocalServer}.
 */
export interface LaunchLocalServerOptions {
  /** Working directory from which the local server is started. */
  cwd: string;
  /** TCP port for the local server. Defaults to {@link DEFAULT_SERVER_PORT}. */
  port?: number;
  /**
   * URL path (relative to the server root) for the primary scenario page.
   * Defaults to {@link DEFAULT_SCENARIO_PATH}.
   */
  scenarioPath?: string;
  /**
   * Maximum time to wait for the server to become ready.
   * Defaults to {@link DEFAULT_SERVER_START_TIMEOUT_MS}.
   */
  startTimeoutMs?: number;
}

/**
 * Handle returned by {@link launchLocalServer}. Holds the resolved URLs and a
 * teardown function for the started HTTP server.
 */
export interface LocalServerHandle {
  /** Base URL of the running local server, e.g. `http://localhost:8080`. */
  serverUrl: string;
  /** Full URL of the scenario page under the local server. */
  scenarioUrl: string;
  /** Resolve once the server has been stopped and all sockets are closed. */
  teardown: () => Promise<void>;
}

/**
 * Minimal static-file content types used by the docs/test server.
 */
const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

/**
 * Resolve a request URL to a safe file path under the server root.
 *
 * Returns `null` when the requested path escapes the root directory.
 *
 * @param root - Server root directory.
 * @param requestUrl - Incoming request URL.
 * @returns Absolute file path or `null`.
 */
function resolveFilePath(root: string, requestUrl: string): string | null {
  const { pathname } = new URL(requestUrl, 'http://localhost');
  const safePath = path.normalize(path.join(root, pathname));
  if (!safePath.startsWith(path.normalize(root) + path.sep)) {
    return null;
  }
  return safePath;
}

/**
 * Serve a single static file request.
 *
 * @param root - Server root directory.
 * @param request - Incoming HTTP request.
 * @param response - Outgoing HTTP response.
 */
async function serveStaticFile(
  root: string,
  request: import('node:http').IncomingMessage,
  response: import('node:http').ServerResponse,
): Promise<void> {
  const filePath = resolveFilePath(root, request.url ?? '/');
  if (!filePath) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }

  const extension = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[extension] ?? 'application/octet-stream';

  try {
    const content = await readFile(filePath);
    response.writeHead(200, {
      'Content-Type': contentType,
      'Cache-Control': 'no-cache',
    });
    response.end(content);
  } catch {
    response.writeHead(404);
    response.end('Not found');
  }
}

/**
 * Start a minimal local static-file server for the NeatapticTS docs and
 * browser test scenarios.
 *
 * The server is rooted at {@link options.cwd} and exposes the requested
 * scenario page at {@link scenarioUrl}. Callers must invoke
 * {@link LocalServerHandle.teardown} when the server is no longer needed.
 *
 * @param options - Server launch configuration.
 * @returns A promise resolving to the server handle.
 * @throws {Error} if the server fails to start or does not become ready within
 * the configured timeout.
 *
 * @example
 * ```ts
 * const { serverUrl, scenarioUrl, teardown } = await launchLocalServer({
 *   cwd: '/path/to/repo',
 *   port: 8080,
 * });
 * try {
 *   // Navigate to scenarioUrl and run a browser scenario.
 * } finally {
 *   await teardown();
 * }
 * ```
 */
export function launchLocalServer(
  options: LaunchLocalServerOptions,
): Promise<LocalServerHandle> {
  const {
    cwd,
    port = DEFAULT_SERVER_PORT,
    scenarioPath = DEFAULT_SCENARIO_PATH,
    startTimeoutMs = DEFAULT_SERVER_START_TIMEOUT_MS,
  } = options;

  const serverUrl = `http://localhost:${port}`;
  const scenarioUrl = `${serverUrl}${scenarioPath}`;
  const root = path.resolve(cwd);

  return new Promise((resolve, reject) => {
    let settled = false;

    const server = createServer((request, response) => {
      void serveStaticFile(root, request, response);
    });

    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      server.closeAllConnections?.();
      server.close(() => {
        reject(error);
      });
    };

    const startTimeout = setTimeout(() => {
      fail(
        new Error(
          `Local HTTP server did not become ready within ${startTimeoutMs}ms`,
        ),
      );
    }, startTimeoutMs);

    server.on('error', (error) => {
      if (settled) return;
      clearTimeout(startTimeout);
      reject(
        new Error(`Local HTTP server failed to start: ${error.message}`, {
          cause: error,
        }),
      );
    });

    server.listen(port, '127.0.0.1', () => {
      if (settled) return;
      clearTimeout(startTimeout);
      settled = true;

      resolve({
        serverUrl,
        scenarioUrl,
        teardown: () =>
          new Promise((res) => {
            server.closeAllConnections?.();
            server.close(() => {
              res();
            });
          }),
      });
    });
  });
}
