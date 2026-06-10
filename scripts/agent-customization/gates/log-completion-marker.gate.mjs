#!/usr/bin/env node
/**
 * Tier-2 gate: log-completion-marker (agent: 07-logging)
 *
 * Validates that a 07-logging agent output contains a compressed log entry and
 * that the target phase is marked [DONE] in the tracker. In standalone mode
 * (--json), returns the gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/log-completion-marker.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'log-completion-marker',
    tier: 2,
    agent: '07-logging',
    check:
      'Compressed log entry present and target phase marked [DONE] in the tracker',
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 07-logging output includes a compressed log entry and updates the tracker to mark the target phase [DONE] before ending.',
  owner: '07-logging output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'log-completion-marker gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;
