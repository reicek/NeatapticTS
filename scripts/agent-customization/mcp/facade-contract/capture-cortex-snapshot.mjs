/**
 * Capture a JSON-RPC-shaped snapshot of the real repo-cortex-mcp server.
 *
 * Uses the server's exported createRepoCortexMcpServer and dispatches the same
 * requests a stdio host would send: initialize, tools/list, and a few safe
 * read-only tools/call requests against the default corpus database.
 */
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createRepoCortexMcpServer } from '../../../mcp-semantic/repo-cortex-mcp.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const snapshotPath = path.join(__dirname, 'cortex-real-snapshot.json');

const server = createRepoCortexMcpServer();

async function dispatch(method, params) {
  return server.dispatch({ jsonrpc: '2.0', id: 1, method, params });
}

const snapshot = {
  meta: {
    source: 'scripts/mcp-semantic/repo-cortex-mcp.mjs',
    capturedAt: new Date().toISOString(),
    protocolVersion: '2024-11-05',
  },
  initialize: await dispatch('initialize', { protocolVersion: '2024-11-05' }),
  toolsList: await dispatch('tools/list', {}),
  calls: {},
};

// Safe read-only calls against the default corpus database.
for (const [callName, toolName, args] of [
  ['indexStats', 'index_stats', { include_metadata_coverage: false }],
  ['listFamilies', 'list_families', {}],
  [
    'searchCorpus',
    'search_corpus',
    { query: 'network', limit: 1, compact: true },
  ],
]) {
  try {
    snapshot.calls[callName] = await dispatch('tools/call', {
      name: toolName,
      arguments: args,
    });
  } catch (error) {
    snapshot.calls[callName] = {
      isError: true,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

writeFileSync(snapshotPath, JSON.stringify(snapshot, null, 2));
console.log(`Wrote ${snapshotPath}`);
