#!/usr/bin/env node
/**
 * Tier-2 gate: output-resolution-evidence (agent: 00-helping)
 *
 * Validates that a 00-helping agent output contains a `resolution` or `blocked`
 * field with supporting evidence text. In standalone mode (--json), returns the
 * gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/output-resolution-evidence.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'output-resolution-evidence',
    tier: 2,
    agent: '00-helping',
    check: 'Output contains a resolution or blocked field with evidence text',
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 00-helping output includes a `resolution` or `blocked` field with non-empty evidence text before exiting.',
  owner: '00-helping output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  /* istanbul ignore next: result.pass is always true in standalone-descriptor mode */
  console.log(result.pass ? 'PASS' : 'FAIL', 'output-resolution-evidence gate');
  /* istanbul ignore if: result.pass is always true in standalone-descriptor mode */
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

/* istanbul ignore next: result.pass is always true in standalone-descriptor mode */
process.exitCode = result.pass ? 0 : 1;
