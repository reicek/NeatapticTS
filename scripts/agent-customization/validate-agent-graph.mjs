#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs, printUsage, writeReport } from './customization-utils.mjs';
import {
  collectTierInventory,
  runValidateAgentGraph,
} from './tier-graph-utils.mjs';

export {
  collectTierInventory,
  runValidateAgentGraph,
} from './tier-graph-utils.mjs';

export async function main() {
  const options = parseArgs(process.argv.slice(2));

  if (options.help) {
    printUsage({
      title: 'Validate NeatapticTS custom agent delegation graph.',
      usage:
        'node scripts/agent-customization/validate-agent-graph.mjs [--json]',
    });
    process.exit(0);
  }

  const report = await runValidateAgentGraph();
  writeReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
}

export function handleMainError(error) {
  console.error(error);
  process.exitCode = 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  main().then(() => {}, handleMainError);
}
