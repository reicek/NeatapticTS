/**
 * @module devtools-facade
 * @description Lazy-load MCP facade for the Chrome DevTools MCP server.
 *
 * Exposes a single router tool (`devtools`) that accepts `{ operation, args? }`
 * and only spawns the real `chrome-devtools-mcp` package on the first
 * `tools/call`.  This avoids pulling up a Chromium instance during editor
 * startup and defers telemetry prompts until a browser operation is actually
 * requested.
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
 *   2. Once Chrome is running, connect the DevTools MCP to the existing
 *      browser instance on port 9222 (the MCP will reuse it when it can attach
 *      to the remote debugging port).
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
 * @param {object} [options] - Facade options.
 * @param {string} [options.snapshotPath] - Override the default snapshot path.
 * @param {string[]} [options.spawnCommand] - Override the default spawn command.
 * @returns {ReturnType<typeof createLazyFacade>} Facade instance.
 */
export function createDevtoolsFacade(options = {}) {
  return createLazyFacade({
    name: 'devtools',
    target: 'devtools',
    title: 'Chrome DevTools MCP',
    version: '0.1.0',
    defaultSnapshotPath: DEFAULT_SNAPSHOT_PATH,
    defaultSpawnCommand: DEFAULT_SPAWN_COMMAND,
    defaultEnv: DEFAULT_ENV,
    routingMode: 'native',
    ...options,
  });
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
    },
    process.argv.slice(2),
  );
}
