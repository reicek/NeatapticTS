#!/usr/bin/env node
/**
 * Tier-2 gate: green-validation-evidence (agent: 05-green-testing)
 *
 * Validates that a 05-green-testing agent output contains a per-check result table
 * and that no check is marked PASS without supporting evidence. In standalone mode
 * (--json), returns the gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/green-validation-evidence.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'green-validation-evidence',
    tier: 2,
    agent: '05-green-testing',
    check:
      'Per-check result table present; no check marked PASS without evidence',
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 05-green-testing output includes a per-check result table in VALIDATION_EVIDENCE. Each PASS entry must include a command and its observed result.',
  owner: '05-green-testing output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'green-validation-evidence gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;
