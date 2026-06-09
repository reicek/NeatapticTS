#!/usr/bin/env node
/**
 * Tier-1 gate: tier-enforcement (MCP-compatible entry point)
 *
 * Thin forwarder so the MCP server's `<gate-id>.gate.mjs` naming convention
 * resolves correctly. The canonical implementation lives in
 * `tier-enforcement-gate.mjs`; this file simply re-exports and delegates to it.
 *
 * The MCP server (`neataptic-gate-mcp.mjs`) constructs gate paths as:
 *   `gates/${gateId}.gate.mjs`
 * The gate was originally shipped as `tier-enforcement-gate.mjs`, creating a
 * mismatch. Adding this forwarder fixes the MCP lookup without renaming the
 * original file or breaking any existing references.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/tier-enforcement.gate.mjs [--json]
 */

import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { parseArgs } from '../customization-utils.mjs';
import { runTierEnforcementGate } from './tier-enforcement-gate.mjs';

export { runTierEnforcementGate };

const options = parseArgs(process.argv.slice(2));

async function main() {
  const report = await runTierEnforcementGate();

  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(report.pass ? 'PASS' : 'FAIL', 'tier-enforcement gate');
    if (!report.pass) console.log('fixHint:', report.fixHint);
  }

  process.exitCode = report.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  await main();
}
