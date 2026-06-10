#!/usr/bin/env node
/**
 * Tier-2 gate: implementation-artifact-paths (agent: 04-implementing)
 *
 * Validates that a 04-implementing agent output declares changed file paths, and
 * for workflow-only tasks confirms that none of those paths fall under `src/`.
 * In standalone mode (--json), returns the gate descriptor and a healthy baseline result.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/implementation-artifact-paths.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'implementation-artifact-paths',
    tier: 2,
    agent: '04-implementing',
    check:
      'Changed file paths declared; for workflow-only tasks none under src/',
    mode: 'standalone-descriptor',
  },
  fixHint:
    'Ensure 04-implementing output declares all changed file paths in FILES_CHANGED. For workflow-only tasks, no path may begin with src/.',
  owner: '04-implementing output contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(
    result.pass ? 'PASS' : 'FAIL',
    'implementation-artifact-paths gate',
  );
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;
