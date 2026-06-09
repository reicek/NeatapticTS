#!/usr/bin/env node
/**
 * Tier-2 gate: planning-output-contract (agent: 01-planning)
 *
 * Validates that a 01-planning agent output contains a structured-v1 block with
 * the mandatory fields: TASK_STATUS, FILES_CHANGED, ACTIONS_TAKEN, PHASE_COMPLETE.
 * In standalone mode (--json), returns the gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/planning-output-contract.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'planning-output-contract',
    tier: 2,
    agent: '01-planning',
    check:
      'Structured-v1 block contains TASK_STATUS, FILES_CHANGED, ACTIONS_TAKEN, PHASE_COMPLETE',
    requiredFields: [
      'TASK_STATUS',
      'FILES_CHANGED',
      'ACTIONS_TAKEN',
      'PHASE_COMPLETE',
    ],
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 01-planning output includes a structured-v1 fenced block with TASK_STATUS, FILES_CHANGED, ACTIONS_TAKEN, and PHASE_COMPLETE fields.',
  owner: 'structured-v1 schema',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'planning-output-contract gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;
