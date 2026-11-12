/**
 * @module devtools-facade
 * @description Lazy-load MCP facade for the Chrome DevTools MCP server.
 *
 * Exposes a single router tool (`devtools`) that accepts `{ operation, args? }`
 * and only spawns the real `chrome-devtools-mcp` package on the first
 * `tools/call`.  This avoids pulling up a Chromium instance during editor
 * startup and defers telemetry prompts until a browser operation is actually
 * requested.
 *
 * ## Attach mode (the browserUrl option)
 *
 * When a host already runs a PREPARED visible Chrome instance with remote
 * debugging enabled (launch one via {@link launchVisibleChrome}, or the
 * Start-Process command in the workaround note below), the facade ATTACHES
 * the real MCP server to that instance instead of spawning its own second
 * Chrome:
 *
 * ```js
 * const facade = createDevtoolsFacade({ browserUrl: 'http://127.0.0.1:9222' });
 * ```
 *
 * In attach mode the facade, in order:
 *   1. probes the CDP HTTP readiness endpoint `<browserUrl>/json/version`
 *      through {@link waitForCdpReadiness} BEFORE the child transport's first
 *      spawn (an HTTP 200 with a JSON body carrying the browser identity is
 *      READY), and
 *   2. appends `--browserUrl=<browserUrl>` ADDITIVELY to the spawned
 *      `chrome-devtools-mcp` command so the MCP attaches to the prepared
 *      instance instead of launching its own second browser.
 *
 * A bare TCP-socket check (`Get-NetTCPConnection -LocalPort 9222 -State
 * Listen`) is NOT a readiness signal: the listener can be up while CDP HTTP
 * is not serving, which fed the FALSE-NEGATIVE readiness probe of the
 * 2026-11-10 harness finding (b). The checkable readiness probe is:
 *
 * ```sh
 * curl http://127.0.0.1:9222/json/version
 * ```
 *
 * (`200` + a JSON browser-identity body = READY; anything else keeps the
 * probe polling until its budget expires. On expiry the attach flow does not
 * abort — it logs a console warning and attaches anyway, so the MCP attach
 * surface itself is expected to surface the failure.)
 */
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createLazyFacade, runFacadeMain } from './lazy-facade-core.mjs';

// Derive the script directory so the snapshot JSON can be loaded relative to
// this module, both in a real ESM runtime and under Jest's CommonJS transform.
const scriptDir = path.dirname(fileURLToPath(import.meta.url));

const DEFAULT_SNAPSHOT_PATH = path.join(
  scriptDir,
  'devtools-tool-snapshot.json',
);

/**
 * IMPORTANT: Visible browser window workaround.
 *
 * The `--headless=false` flag below asks Chrome DevTools MCP to launch Chrome
 * in a visible window, but it does NOT reliably produce a visible window in all
 * environments (some sessions still spawn a headless or background instance).
 *
 * Reliable workaround for real GPU/performance validation:
 *   1. Manually launch Chrome with remote debugging enabled:
 *      PowerShell:
 *        Start-Process "chrome.exe" -ArgumentList "--remote-debugging-port=9222","--user-data-dir=C:\temp\chrome-debug","<url>"
 *      Or call `launchVisibleChrome(url)` exported by this module.
 *   2. Attach the DevTools MCP to that PREPARED instance through the facade:
 *      `createDevtoolsFacade({ browserUrl: 'http://127.0.0.1:9222' })`.  The
 *      facade polls the CDP readiness endpoint BEFORE the MCP spawn and
 *      passes `--browserUrl=http://127.0.0.1:9222` to the spawned MCP, so it
 *      attaches to the already-foregrounded browser (the "attach mode"
 *      documented at the top of this module).
 *   3. This is REQUIRED for valid WebGPU/GPU measurements — headless browsers
 *      may not engage the GPU process properly.
 */
const DEFAULT_SPAWN_COMMAND = [
  'npx',
  '-y',
  'chrome-devtools-mcp@1.6.0',
  '--headless=false',
  '--usage-statistics=false',
  '--performance-crux=false',
  '--allow-unrestricted-paths=true',
];
const DEFAULT_ENV = {
  CI: '1',
  CHROME_DEVTOOLS_MCP_NO_USAGE_STATISTICS: '1',
};

/**
 * Create a lazy-load facade for the Chrome DevTools MCP server.
 *
 * Passing a non-empty `browserUrl` string switches the facade into ATTACH
 * mode: the spawn command gains an ADDITIVE `--browserUrl=<browserUrl>` tail
 * flag (with `browserUrl` unset the spawn command is byte-identical to
 * `DEFAULT_SPAWN_COMMAND`), and the first `tools/call` is gated behind the
 * CDP HTTP readiness probe — the attach flow documented at the top of this
 * module.
 *
 * @param {object} [options] - Facade options.
 * @param {string} [options.snapshotPath] - Override the default snapshot path.
 * @param {string[]} [options.spawnCommand] - Override the default spawn command.
 * @param {string} [options.browserUrl] - The CDP base endpoint URL of a
 *   PREPARED Chrome instance to attach to (e.g. `http://127.0.0.1:9222`).
 *   Unset (the default) keeps the default launch-mode behaviour untouched.
 * @returns {ReturnType<typeof createLazyFacade>} Facade instance; in attach
 *   mode its `server.dispatch` is the readiness-gated dispatcher.
 * @throws {Error} When `browserUrl` is set to a value that is not a usable
 *   http(s) URL string (silently spawning a second Chrome would be the bug
 *   attach mode exists to prevent).
 */
export function createDevtoolsFacade(options = {}) {
  const browserUrl = parseBrowserUrl(options.browserUrl);
  const facade = createLazyFacade({
    name: 'devtools',
    target: 'devtools',
    title: 'Chrome DevTools MCP',
    version: '0.1.0',
    defaultSnapshotPath: DEFAULT_SNAPSHOT_PATH,
    defaultSpawnCommand: DEFAULT_SPAWN_COMMAND,
    defaultEnv: DEFAULT_ENV,
    routingMode: 'native',
    ...options,
    ...(browserUrl
      ? {
          spawnCommand: appendBrowserUrlFlag(options.spawnCommand, browserUrl),
        }
      : {}),
  });
  return wireAttachReadinessGate(facade, browserUrl);
}

/**
 * Launch a visible Chrome instance with remote debugging enabled.
 *
 * This is the reliable way to obtain a visible browser window for real GPU/
 * performance validation when `--headless=false` does not work. The returned
 * process is detached so this script can exit without closing Chrome.
 *
 * @param {string} url - URL to open in Chrome.
 * @param {object} [options] - Launch options.
 * @param {string} [options.chromePath] - Chrome executable path. Defaults to
 *   `process.env.CHROME_PATH` or `chrome.exe` (Windows) / `google-chrome`
 *   (other platforms).
 * @param {number} [options.remotePort=9222] - Remote debugging port.
 * @param {string} [options.userDataDir] - Chrome user data directory. Defaults
 *   to `process.env.CHROME_USER_DATA_DIR` or `C:\temp\chrome-debug` on Windows,
 *   `/tmp/chrome-debug` elsewhere.
 * @returns {{pid: number | undefined, port: number, url: string, args: string[]}}
 */
export function launchVisibleChrome(url, options = {}) {
  const isWin = process.platform === 'win32';
  const chromePath =
    options.chromePath ||
    process.env.CHROME_PATH ||
    (isWin ? 'chrome.exe' : 'google-chrome');
  const remotePort = options.remotePort || 9222;
  const userDataDir =
    options.userDataDir ||
    process.env.CHROME_USER_DATA_DIR ||
    (isWin ? 'C:\\temp\\chrome-debug' : '/tmp/chrome-debug');

  const args = [
    `--remote-debugging-port=${remotePort}`,
    `--user-data-dir=${userDataDir}`,
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    url,
  ];

  const proc = spawn(chromePath, args, {
    detached: true,
    stdio: 'ignore',
    shell: isWin,
  });
  proc.unref();

  return { pid: proc.pid, port: remotePort, url, args };
}

// ---------------------------------------------------------------------------
// THE ATTACH FLOW: launchVisibleChrome -> createDevtoolsFacade({ browserUrl })
// -> waitForCdpReadiness (the exported CDP /json/version HTTP probe) -> the
// first facade.server.dispatch delegating to the lazy transport, whose first
// spawn carries the ADDITIVE --browserUrl=<browserUrl> flag. There is NO bare
// TCP-socket check anywhere in this flow: a listening port is not CDP
// readiness (the false-negative harness lesson) — the HTTP identity is.
// ---------------------------------------------------------------------------

/**
 * The CDP HTTP readiness probe budget the attach gate uses when constructing
 * the attach-mode facade.
 *
 * `timeoutMs` bounds the whole poll (a Chrome cold boot on the visible
 * instance comfortably fits inside it); `pollIntervalMs` is the pause between
 * consecutive `/json/version` GETs. The budget is deliberately generous: on
 * expiry the attach flow still attaches (with a console warning) instead of
 * failing the facade, so a slow-but-alive endpoint degrades gracefully.
 */
const DEFAULT_CDP_PROBE_OPTIONS = {
  timeoutMs: 3_000,
  pollIntervalMs: 100,
};

/**
 * The bounded per-probe fetch cap: one GET /json/version may never hang the
 * whole probe loop (the deadline governs total budget, this governs a single
 * wedged connection).
 */
const CDP_PROBE_FETCH_TIMEOUT_MS = 1_500;

/**
 * The NAMED error {@link waitForCdpReadiness} throws when the CDP
 * `/json/version` endpoint does not become READY within the probe budget.
 * The name lets callers discriminate the timeout from incidental probe
 * failures (connection refused while Chrome boots, non-200 answers), which
 * the probe loop consumes as "not READY yet".
 */
export class CdpReadinessTimeoutError extends Error {
  /**
   * @param {string} versionUrl - The probed /json/version endpoint.
   * @param {number} timeoutMs - The exhausted probe budget in ms.
   * @param {ErrorOptions} [options] - Forwarded to Error (enables cause).
   */
  constructor(versionUrl, timeoutMs, options = {}) {
    super(
      `CDP readiness probe gave up after ${timeoutMs}ms: ${versionUrl} never returned 200 with the browser identity.`,
      options,
    );
    this.name = 'CdpReadinessTimeoutError';
    this.versionUrl = versionUrl;
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Wait for a Chrome DevTools Protocol HTTP endpoint to become READY by
 * polling its `/json/version` identity.
 *
 * The endpoint is READY when a probe returns HTTP 200 with a JSON body
 * carrying the browser identity (non-empty `Browser` and
 * `webSocketDebuggerUrl` strings). Connection refusals, non-200 statuses and
 * identity-less bodies mean "not READY yet": the loop keeps polling until the
 * budget expires (the named {@link CdpReadinessTimeoutError}) or the identity
 * resolves, whichever comes first. This HTTP identity probe REPLACES the
 * false-negative TCP-socket channel check recorded in the 2026-11-10 harness
 * finding (b) — a listening port is not a serving CDP endpoint.
 *
 * @param {string} versionUrl - The CDP version endpoint to poll, typically
 *   `<browserUrl>/json/version`.
 * @param {object} [options] - Probe tuning.
 * @param {number} [options.timeoutMs=3000] - Whole-probe budget in ms.
 * @param {number} [options.pollIntervalMs=100] - Pause between probes in ms.
 * @returns {Promise<object>} The parsed /json/version browser identity.
 * @throws {CdpReadinessTimeoutError} When the endpoint stays not-READY for
 *   the whole `timeoutMs` budget.
 * @example
 * const identity = await waitForCdpReadiness('http://127.0.0.1:9222/json/version');
 * // identity.Browser === 'Chrome/141.0.0.0'
 */
export async function waitForCdpReadiness(versionUrl, options = {}) {
  const probeOptions = { ...DEFAULT_CDP_PROBE_OPTIONS, ...options };
  const deadline = Date.now() + probeOptions.timeoutMs;

  for (;;) {
    const identity = await probeCdpVersion(versionUrl);
    if (identity !== null) {
      return identity;
    }
    if (Date.now() >= deadline) {
      throw new CdpReadinessTimeoutError(versionUrl, probeOptions.timeoutMs);
    }
    await delay(probeOptions.pollIntervalMs);
  }
}

/**
 * Probe a CDP `/json/version` endpoint exactly ONCE over the HTTP channel and
 * return the parsed browser identity, or null when the endpoint is not READY
 * (connection failure, non-200, or an identity-less body). The bounded
 * per-probe fetch keeps one wedged connection from hanging the facade
 * dispatch.
 *
 * @param {string} versionUrl - The /json/version endpoint to probe.
 * @returns {Promise<object | null>} The parsed identity when READY, null
 *   otherwise.
 */
async function probeCdpVersion(versionUrl) {
  try {
    const response = await fetch(versionUrl, {
      signal: AbortSignal.timeout(CDP_PROBE_FETCH_TIMEOUT_MS),
    });
    if (!response.ok) {
      return null;
    }
    const identity = await response.json();
    return isCdpBrowserIdentity(identity) ? identity : null;
  } catch {
    // Not READY yet (connection refused while Chrome boots, probe timeout...).
    return null;
  }
}

/**
 * Whether a parsed /json/version body carries the browser identity the CDP
 * HTTP channel must produce before the attach flow may proceed: a plain
 * object with a non-empty `Browser` and a non-empty `webSocketDebuggerUrl`
 * string.
 *
 * @param {unknown} identity - The parsed probe body.
 * @returns {boolean} Whether the body carries the browser identity.
 */
function isCdpBrowserIdentity(identity) {
  return (
    isJsonObject(identity) &&
    typeof identity.Browser === 'string' &&
    identity.Browser !== '' &&
    typeof identity.webSocketDebuggerUrl === 'string' &&
    identity.webSocketDebuggerUrl !== ''
  );
}

/**
 * Whether a value is a plain JSON object (not an array, not null, not a
 * primitive).
 *
 * @param {unknown} value - Any value.
 * @returns {boolean}
 */
function isJsonObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * A tiny ms timer promise (the readiness poll pause).
 *
 * @param {number} ms - The pause length in ms.
 * @returns {Promise<void>}
 */
function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/**
 * Wire the attach-mode readiness gate onto a freshly created facade.
 *
 * The gate wraps `facade.server.dispatch` exactly once: the first host
 * `tools/call` runs the CDP readiness probe and only then delegates to the
 * lazy transport, whose first use spawns the MCP server — the probe-THEN-
 * attach ordering the harness homes measure by call order. The gate settles
 * after its first probe (later requests bypass it); locally answered
 * requests (initialize / ping / tools/list) bypass it entirely. If the probe
 * never reports READY, the attach still proceeds with a console warning — a
 * wedged endpoint reproduces the pre-attach behaviour instead of failing the
 * facade, and the MCP attach surface itself is expected to surface the
 * failure.
 *
 * @param {ReturnType<typeof createLazyFacade>} facade - The fresh facade.
 * @param {string | null} browserUrl - The normalised attach URL, or null in
 *   launch mode (the facade is handed back untouched).
 * @returns {ReturnType<typeof createLazyFacade>} The facade handed to the
 *   host; in attach mode `facade.server.dispatch` is the gated dispatcher.
 */
function wireAttachReadinessGate(facade, browserUrl) {
  if (browserUrl === null || typeof facade?.server?.dispatch !== 'function') {
    // Launch mode (or a bare-shell facade): byte-for-byte today's behaviour.
    return facade;
  }
  const serverDispatch = facade.server.dispatch.bind(facade.server);
  const versionUrl = `${browserUrl.replace(/\/+$/, '')}/json/version`;
  let attachGateSettled = false;

  facade.server.dispatch = async (request) => {
    if (attachGateSettled || !isSpawnTriggeringRequest(request)) {
      return serverDispatch(request);
    }
    // THE ATTACH LAW: the readiness probe runs BEFORE the MCP spawn (this
    // first delegated tools/call is the transport's first use, i.e. the
    // spawn trigger).
    attachGateSettled = true;
    try {
      await waitForCdpReadiness(versionUrl, {
        timeoutMs: DEFAULT_CDP_PROBE_OPTIONS.timeoutMs,
        pollIntervalMs: DEFAULT_CDP_PROBE_OPTIONS.pollIntervalMs,
      });
    } catch (error) {
      console.warn(
        `[devtools] CDP readiness probe gave up after ${DEFAULT_CDP_PROBE_OPTIONS.timeoutMs}ms (${versionUrl}: ${error?.message}); attaching to the MCP anyway — the attach surface is expected to surface the failure on its own.`,
      );
    }
    return serverDispatch(request);
  };

  return facade;
}

/**
 * Whether a JSON-RPC message actually reaches the child transport: a host
 * `tools/call` is the only message that may lazily spawn the MCP server;
 * initialize / ping / tools/list are answered locally.
 *
 * @param {unknown} request - The incoming JSON-RPC message.
 * @returns {boolean}
 */
function isSpawnTriggeringRequest(request) {
  return (
    typeof request === 'object' &&
    request !== null &&
    request.method === 'tools/call'
  );
}

/**
 * Normalise the `browserUrl` facade option into the attach-mode endpoint.
 * `undefined` / `null` (the option not set) selects default launch mode.
 *
 * @param {unknown} value - The raw option value.
 * @returns {string | null} The trimmed attach URL, or null for launch mode.
 * @throws {Error} When the option is SET to something that is not a usable
 *   http(s) URL string — silently launching a second Chrome in that case is
 *   exactly the bug attach mode exists to prevent.
 */
function parseBrowserUrl(value) {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    throw new Error(
      `devtools facade: browserUrl must be a string http(s) URL of the prepared Chrome instance, received ${typeof value}.`,
    );
  }
  const browserUrl = value.trim();
  if (!/^https?:\/\//i.test(browserUrl)) {
    throw new Error(
      `devtools facade: browserUrl must be an http(s) URL (e.g. "http://127.0.0.1:9222"), received "${value}".`,
    );
  }
  return browserUrl;
}

/**
 * Build the attach-mode spawn command: the caller-provided (or default)
 * command with the ADDITIVE `--browserUrl=<browserUrl>` flag appended last.
 *
 * @param {string[]} [customSpawnCommand] - The caller's spawn command, or
 *   undefined for `DEFAULT_SPAWN_COMMAND`.
 * @param {string} browserUrl - The normalised attach endpoint.
 * @returns {string[]} The attach-mode spawn command.
 */
function appendBrowserUrlFlag(customSpawnCommand, browserUrl) {
  return [
    ...(customSpawnCommand ?? DEFAULT_SPAWN_COMMAND),
    `--browserUrl=${browserUrl}`,
  ];
}

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) ===
    path.resolve(scriptDir, 'devtools-facade.mjs');

if (isMain) {
  runFacadeMain(
    {
      name: 'devtools',
      target: 'devtools',
      title: 'Chrome DevTools MCP',
      version: '0.1.0',
      defaultSnapshotPath: DEFAULT_SNAPSHOT_PATH,
      defaultSpawnCommand: DEFAULT_SPAWN_COMMAND,
      defaultEnv: DEFAULT_ENV,
      routingMode: 'native',
    },
    process.argv.slice(2),
  );
}
