/**
 * @module cortex-facade
 * @description Lazy-load MCP facade for the Repo Cortex server.
 *
 * Exposes a single router tool (`cortex`) that accepts `{ operation, args? }`
 * and only spawns the real `scripts/mcp-semantic/repo-cortex-mcp.mjs` process on
 * the first `tools/call`.  This keeps editor MCP startup fast and avoids paying
 * the ONNX embedding/model load cost until a semantic query is actually
 * needed.
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createLazyFacade, runFacadeMain } from './lazy-facade-core.mjs';

// Derive the script directory so the snapshot JSON can be loaded relative to
// this module, both in a real ESM runtime and under Jest's CommonJS transform.
const scriptDir = path.dirname(fileURLToPath(import.meta.url));

const DEFAULT_SNAPSHOT_PATH = path.join(scriptDir, 'cortex-tool-snapshot.json');
const DEFAULT_SPAWN_COMMAND = [
  'node',
  'scripts/mcp-semantic/repo-cortex-mcp.mjs',
];

/**
 * Create a lazy-load facade for the Repo Cortex MCP server.
 *
 * @param {object} [options] - Facade options.
 * @param {string} [options.snapshotPath] - Override the default snapshot path.
 * @param {string[]} [options.spawnCommand] - Override the default spawn command.
 * @returns {ReturnType<typeof createLazyFacade>} Facade instance.
 */
export function createCortexFacade(options = {}) {
  return createLazyFacade({
    name: 'cortex',
    target: 'cortex',
    title: 'NeatapticTS Repo Cortex',
    version: '0.1.0',
    defaultSnapshotPath: DEFAULT_SNAPSHOT_PATH,
    defaultSpawnCommand: DEFAULT_SPAWN_COMMAND,
    ...options,
  });
}

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) ===
    path.resolve(scriptDir, 'cortex-facade.mjs');

if (isMain) {
  runFacadeMain(
    {
      name: 'cortex',
      target: 'cortex',
      title: 'NeatapticTS Repo Cortex',
      version: '0.1.0',
      defaultSnapshotPath: DEFAULT_SNAPSHOT_PATH,
      defaultSpawnCommand: DEFAULT_SPAWN_COMMAND,
    },
    process.argv.slice(2),
  );
}
