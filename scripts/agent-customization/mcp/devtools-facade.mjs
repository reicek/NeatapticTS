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
const DEFAULT_SPAWN_COMMAND = [
  'npx',
  '-y',
  'chrome-devtools-mcp@1.4.0',
  '--headless=true',
  '--usage-statistics=false',
  '--performance-crux=false',
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
