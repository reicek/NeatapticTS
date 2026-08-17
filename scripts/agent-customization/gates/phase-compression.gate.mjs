#!/usr/bin/env node
/**
 * Tier-2 gate: phase-compression (agents: 01-planning, 07-logging)
 *
 * Validates that any phase newly marked [DONE] in the active plan tracker has
 * had its detailed history compressed to a concise coverage note before the
 * next phase is kicked off or the workstream is closed. Prevents verbose
 * session transcripts from accumulating in active plan files.
 *
 * Enforcement contract: when a phase transitions to [DONE], the owning agent
 * must compress the phase record to a short coverage note (no per-step
 * transcripts, no raw validation output) before returning structured-v1 output.
 *
 * In standalone mode (--json), returns the gate descriptor and a healthy
 * baseline result. Runtime evaluation is delegated to the owning agent's
 * structured-v1 VALIDATION_EVIDENCE section.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/phase-compression.gate.mjs [--json]
 */

import { parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = {
  pass: true,
  evidence: {
    gate: 'phase-compression',
    tier: 2,
    agents: ['01-planning', '07-logging'],
    check:
      'Any phase marked [DONE] must have its history compressed to a concise coverage note in the tracker before the next phase starts or the workstream closes',
    mode: 'standalone-descriptor',
    enforcementPoints: [
      '01.phase-kickoff — requires compressed record of the previous [DONE] phase',
      '07.tracker-closure — requires all [DONE] phases have compressed history',
    ],
  },
  fixHint:
    'Compress the completed phase history: replace verbose step transcripts and raw validation output with a short coverage note (e.g., "[DONE] Phase N: <summary of what was achieved>"). Then re-run this gate.',
  owner: '01-planning / 07-logging phase-compression contract',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  /* istanbul ignore next: result.pass is always true in standalone-descriptor mode */
  console.log(result.pass ? 'PASS' : 'FAIL', 'phase-compression gate');
  /* istanbul ignore if: result.pass is always true in standalone-descriptor mode */
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

/* istanbul ignore next: result.pass is always true in standalone-descriptor mode */
process.exitCode = result.pass ? 0 : 1;
