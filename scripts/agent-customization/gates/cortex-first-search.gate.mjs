#!/usr/bin/env node
/**
 * @description Verify the real prerequisites for the Cortex-first search policy.
 * The gate passes only when the semantic index validates as fresh and the Repo
 * Cortex MCP can answer a `search_corpus` request. It does not claim to verify
 * runtime tool-call order inside agents.
 *
 * @param {boolean} [--json] - Emit the standard gate JSON contract.
 * @param {string}  [--databasePath=<path>] - Override the semantic index database path.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 when the Cortex-first prerequisites are green, 1 otherwise.
 */
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { defaultDatabasePath } from '../../../rag-index/init-schema.mjs';
import { validateDatabase } from '../../../rag-index/validate-index.mjs';
import { parseArgs } from '../customization-utils.mjs';
import { runCortexMcpSmoke } from './cortex-mcp-smoke.mjs';

const OWNER = 'repo-cortex-workflow';
const DEFAULT_FIX_HINT =
  'Run `node rag-index/build-index.mjs` and confirm the Repo Cortex MCP is reachable before relying on Cortex-first search.';

export async function runCortexFirstSearchGate(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const indexValidator = options.indexValidator ?? validateDatabase;
  const mcpSmoke = options.mcpSmoke ?? runCortexMcpSmoke;

  const indexReport = await indexValidator({ databasePath });
  const corpusMcpReport = await mcpSmoke({ databasePath });
  const pass = Boolean(indexReport.pass) && Boolean(corpusMcpReport.pass);

  return {
    pass,
    evidence: {
      database_path: databasePath,
      index_documents: Number(indexReport.documents ?? 0),
      index_chunks: Number(indexReport.chunks ?? 0),
      index_fresh: Boolean(indexReport.pass),
      corpus_mcp_alive: Boolean(corpusMcpReport.pass),
      corpus_search_results: Number(
        corpusMcpReport.evidence?.searchResults ?? 0,
      ),
    },
    fixHint: pass ? null : resolveFixHint(indexReport, corpusMcpReport),
    owner: OWNER,
  };
}

function parseCliOptions(argv) {
  const base = parseArgs(argv);

  return {
    ...base,
    databasePath: argv
      .find((argument) => argument.startsWith('--databasePath='))
      ?.slice('--databasePath='.length),
  };
}

function printUsage() {
  console.log(
    [
      'Cortex-first search prerequisite gate',
      '',
      'Usage:',
      '  node scripts/agent-customization/gates/cortex-first-search.gate.mjs [--json] [--databasePath=<path>]',
      '  node scripts/agent-customization/gates/cortex-first-search.gate.mjs --help',
      '',
      'Options:',
      '  --json                 Emit machine-readable gate JSON.',
      '  --databasePath=<path>  Override the semantic index database path.',
      '  --help                 Show this help.',
    ].join('\n'),
  );
}

function resolveFixHint(indexReport, corpusMcpReport) {
  return indexReport.fixHint ?? corpusMcpReport.fixHint ?? DEFAULT_FIX_HINT;
}

async function main() {
  const options = parseCliOptions(process.argv.slice(2));

  if (options.help) {
    printUsage();
    return;
  }

  const report = await runCortexFirstSearchGate(options);
  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(
      report.pass
        ? 'PASS cortex-first-search gate'
        : 'FAIL cortex-first-search gate',
    );
    if (!report.pass) console.log(`fixHint: ${report.fixHint}`);
  }

  process.exitCode = report.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  await main();
}
