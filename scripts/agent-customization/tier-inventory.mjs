#!/usr/bin/env node
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs, printUsage, writeReport } from './customization-utils.mjs';
import { collectTierInventory } from './validate-agent-graph.mjs';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  printUsage({
    title: 'Inventory NeatapticTS agent delegation tiers.',
    usage: 'node scripts/agent-customization/tier-inventory.mjs [--json]',
  });
  process.exit(0);
}

export async function runTierInventory({ workspaceRoot = process.cwd() } = {}) {
  const inventory = await collectTierInventory({ workspaceRoot });

  return {
    name: 'agent tier inventory',
    ok: inventory.violations.length === 0,
    issues: inventory.violations,
    summaryText: [
      `${inventory.violations.length === 0 ? 'PASS' : 'FAIL'} agent tier inventory`,
      `total=${inventory.summary.total}`,
      `tier1=${inventory.summary.by_tier[1]}`,
      `tier2=${inventory.summary.by_tier[2]}`,
      `tier3=${inventory.summary.by_tier[3]}`,
      `tier4=${inventory.summary.by_tier[4]}`,
      `violations=${inventory.summary.violation_count}`,
    ].join(' '),
    ...inventory,
  };
}

async function main() {
  const report = await runTierInventory();
  writeReport(report, options);
  process.exitCode = report.ok ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  await main();
}