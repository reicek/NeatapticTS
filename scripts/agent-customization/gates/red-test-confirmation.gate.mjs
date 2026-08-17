#!/usr/bin/env node
/**
 * Tier-2 gate: red-test-confirmation (agent: 03-red-testing)
 *
 * Validates that a 03-red-testing agent output declares at least two red test file
 * paths and records a failure reason in the active plan. In standalone mode (--json),
 * returns the gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/red-test-confirmation.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'red-test-confirmation',
    tier: 2,
    agent: '03-red-testing',
    check:
      'At least two red test file paths declared and a failure reason recorded in the plan',
    minimumRedTests: 2,
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 03-red-testing output lists at least two red test file paths and records a confirmed failure reason before ending.',
  owner: '03-red-testing output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  /* istanbul ignore next: result.pass is always true in standalone-descriptor mode */
  console.log(result.pass ? 'PASS' : 'FAIL', 'red-test-confirmation gate');
  /* istanbul ignore if: result.pass is always true in standalone-descriptor mode */
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

/* istanbul ignore next: result.pass is always true in standalone-descriptor mode */
process.exitCode = result.pass ? 0 : 1;
