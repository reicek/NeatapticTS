import { existsSync, statSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { invokeServerRequest, MCP_REPO_ROOT } from '../mcp/mcp-utils.mjs';
import { CORTEX_FIX_HINT, resolveDatabasePath } from '../../mcp-semantic/tools/cortex-db.mjs';
import { createRepoCortexMcpServer } from '../../mcp-semantic/repo-cortex-mcp.mjs';

const OWNER = '05-green-testing';

export async function runCortexMcpSmoke(options = {}) {
  const databasePath = resolveDatabasePath(options.databasePath);
  const evidence = { databasePath };

  if (!existsSync(databasePath) || statSync(databasePath).size === 0) {
    return failReport(evidence, 'Semantic index is missing or empty.');
  }

  const server = createRepoCortexMcpServer({ databasePath });
  const statsResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: { name: 'index_stats', arguments: {} },
  });
  const stats = statsResult.structuredContent;
  evidence.stats = stats;

  if (statsResult.isError || !stats || Number(stats.total_chunks) < 1) {
    return failReport(evidence, 'index_stats did not report indexed chunks.');
  }

  const searchResult = await invokeServerRequest(server, {
    method: 'tools/call',
    params: { name: 'search_corpus', arguments: { query: 'NEAT activation', limit: 3 } },
  });
  const results = searchResult.structuredContent?.results ?? [];
  evidence.searchResults = results.length;

  if (searchResult.isError || results.length < 1) {
    return failReport(evidence, 'search_corpus did not return results for NEAT activation.');
  }

  return { pass: true, evidence, fixHint: null, owner: OWNER };
}

function failReport(evidence, message) {
  return { pass: false, evidence: { ...evidence, message }, fixHint: CORTEX_FIX_HINT, owner: OWNER };
}

function parseArgs(argv) {
  return {
    help: argv.includes('--help') || argv.includes('-h'),
    json: argv.includes('--json'),
    databasePath: argv.find((argument) => argument.startsWith('--databasePath='))?.slice('--databasePath='.length),
  };
}

function printUsage() {
  console.log([
    'Repo Cortex MCP smoke gate',
    '',
    'Usage:',
    '  node scripts/agent-customization/gates/cortex-mcp-smoke.mjs [--json] [--databasePath=<path>]',
    '  node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --help',
    '',
    'Options:',
    '  --json                 Emit machine-readable gate JSON.',
    '  --databasePath=<path>  Override the semantic index database path.',
  ].join('\n'));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const report = await runCortexMcpSmoke(options);
  const text = options.json ? JSON.stringify(report, null, 2) : `${report.pass ? 'PASS' : 'FAIL'} cortex-mcp-smoke`;
  console.log(text);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.chdir(MCP_REPO_ROOT);
  await main();
}