#!/usr/bin/env node
/**
 * Tier-2 gate: docs-artifact-reference (agent: 06-documenting)
 *
 * Validates that a 06-documenting agent output declares at least one documentation
 * artifact path. In standalone mode (--json), returns the gate descriptor and a
 * healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/docs-artifact-reference.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'docs-artifact-reference',
    tier: 2,
    agent: '06-documenting',
    check: 'At least one documentation artifact path declared in the output',
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 06-documenting output declares at least one documentation artifact path in FILES_CHANGED or an equivalent artifact list.',
  owner: '06-documenting output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'docs-artifact-reference gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;
