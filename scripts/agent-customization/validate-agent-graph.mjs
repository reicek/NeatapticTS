#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs, printUsage, writeReport } from './customization-utils.mjs';
import {
  collectTierInventory,
  runValidateAgentGraph,
} from './tier-graph-utils.mjs';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Validate NeatapticTS custom agent delegation graph.',
    usage: 'node scripts/agent-customization/validate-agent-graph.mjs [--json]',
  });
  process.exit(0);
}

export {
  collectTierInventory,
  runValidateAgentGraph,
} from './tier-graph-utils.mjs';

async function main() {
  const report = await runValidateAgentGraph();
  writeReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}