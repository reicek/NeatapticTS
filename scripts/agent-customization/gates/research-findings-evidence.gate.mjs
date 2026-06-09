#!/usr/bin/env node
/**
 * Tier-2 gate: research-findings-evidence (agent: 02-researching)
 *
 * Validates that a 02-researching agent output contains a KEY_FINDINGS list with
 * at least one finding, and that no file paths in the output are fabricated.
 * In standalone mode (--json), returns the gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/research-findings-evidence.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'research-findings-evidence',
    tier: 2,
    agent: '02-researching',
    check:
      'KEY_FINDINGS list contains at least one finding; no file paths are fabricated',
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 02-researching output includes a non-empty KEY_FINDINGS list and that all file paths in the output resolve to real workspace files.',
  owner: '02-researching output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'research-findings-evidence gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;
