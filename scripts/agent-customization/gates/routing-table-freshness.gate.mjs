#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { fileExists, parseArgs, readWorkspaceFile } from '../customization-utils.mjs';
import {
  collectCustomizationRoutingTable,
  extractRoutingTableSourceHash,
  ROUTING_TABLE_PATH,
} from '../generate-agent-skill-routing-table.mjs';

const options = parseArgs(process.argv.slice(2));

export async function runRoutingTableFreshnessGate() {
  const expectedTable = await collectCustomizationRoutingTable();

  if (!(await fileExists(ROUTING_TABLE_PATH))) {
    return {
      pass: false,
      evidence: {
        tablePath: ROUTING_TABLE_PATH,
        exists: false,
        expectedHash: expectedTable.sourceHash,
        sourceFileCount: expectedTable.sourceFiles.length,
      },
      fixHint: 'Run `npm run agents:routing-table` to create the canonical generated routing table.',
      owner: 'generate-agent-skill-routing-table.mjs',
    };
  }

  const currentMarkdown = await readWorkspaceFile(ROUTING_TABLE_PATH);
  const currentHash = extractRoutingTableSourceHash(currentMarkdown);
  const matchesExpectedBody = currentMarkdown === expectedTable.markdown;
  const pass = currentHash === expectedTable.sourceHash && matchesExpectedBody;

  return {
    pass,
    evidence: {
      tablePath: ROUTING_TABLE_PATH,
      exists: true,
      expectedHash: expectedTable.sourceHash,
      currentHash,
      matchesExpectedBody,
      sourceFileCount: expectedTable.sourceFiles.length,
      agentCount: expectedTable.agentRows.length,
      skillCount: expectedTable.skillRows.length,
    },
    fixHint: pass
      ? 'The canonical generated routing table is fresh.'
      : 'Run `npm run agents:routing-table` to regenerate `.github/agent-skill-routing-table.md` from current agent and skill files.',
    owner: 'generate-agent-skill-routing-table.mjs',
  };
}

async function main() {
  const result = await runRoutingTableFreshnessGate();

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', 'routing-table-freshness gate');
    if (!result.pass) console.log('fixHint:', result.fixHint);
  }

  process.exitCode = result.pass ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}