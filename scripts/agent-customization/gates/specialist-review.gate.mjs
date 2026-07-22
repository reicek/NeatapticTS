#!/usr/bin/env node
/**
 * Tier-1 gate: specialist-review
 *
 * Validates that the orchestrator dispatched specialist review agents
 * (Tier-3 scouts/specialists) to review 04-implementing output BEFORE
 * dispatching 05-green-testing. This gate checks the active plan's
 * VALIDATION_EVIDENCE section for evidence of specialist review.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/specialist-review.gate.mjs [--json] [--plan=path]
 */

import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from '../customization-utils.mjs';

const REPO_ROOT = path.resolve(
  path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Z]:)/, '$1'),
  '..',
  '..',
  '..',
);

const options = parseArgs(process.argv.slice(2));

const planDir = path.join(REPO_ROOT, 'plans');
const planArg = options.plan ? path.resolve(REPO_ROOT, options.plan) : null;

function findActivePlans() {
  if (planArg) {
    return [planArg];
  }
  return readdirSync(planDir)
    .filter((f) => f.endsWith('.plans.md'))
    .map((f) => path.join(planDir, f));
}

function checkPlanForSpecialistReview(planPath) {
  let content;
  try {
    content = readFileSync(planPath, 'utf8');
  } catch {
    return {
      plan: path.basename(planPath),
      found: false,
      reason: 'file not readable',
    };
  }

  if (!content.includes('[WIP]')) {
    return {
      plan: path.basename(planPath),
      found: false,
      reason: 'no [WIP] sections',
    };
  }

  // Look for specialist review evidence markers in VALIDATION_EVIDENCE
  const specialistMarkers = [
    /specialist[_\s-]review/i,
    /implementation-pattern-scout/i,
    /nge-core-scout/i,
    /performance-trace-specialist/i,
    /coverage-scout/i,
    /nge-benchmark-scout/i,
    /browser-runtime-scout/i,
    /visualizer-scout/i,
    /APPROVE/i,
    /REQUEST_CHANGES/i,
  ];

  const evidenceSection = content.match(
    /VALIDATION_EVIDENCE[\s\S]*?(?=##|PlanUpdate|$)/i,
  );

  if (!evidenceSection) {
    return {
      plan: path.basename(planPath),
      found: false,
      reason: 'no VALIDATION_EVIDENCE section found',
    };
  }

  const evidenceText = evidenceSection[0];
  const matchedMarkers = specialistMarkers.filter((m) => m.test(evidenceText));

  // Require at least 3 markers to confirm specialist review evidence
  const found = matchedMarkers.length >= 3;

  return {
    plan: path.basename(planPath),
    found,
    markers: matchedMarkers.map((m) => m.source),
    reason: found
      ? 'specialist review evidence found in VALIDATION_EVIDENCE'
      : 'no specialist review evidence found',
  };
}

const activePlans = findActivePlans();
const wipPlans = activePlans.filter((p) => {
  try {
    return readFileSync(p, 'utf8').includes('[WIP]');
  } catch {
    return false;
  }
});

if (wipPlans.length === 0) {
  const noWipResult = {
    pass: true,
    evidence: {
      gate: 'specialist-review',
      tier: 1,
      check: 'No [WIP] plans found — specialist review gate not applicable.',
      wipPlans: 0,
    },
    fixHint:
      'No action needed. When a plan is [WIP] with implementation slices, ensure 3+ Tier-3 specialists review each 04-implementing slice before dispatching 05-green-testing.',
    owner: 'orchestrator (Agent Zero)',
  };
  if (options.json) {
    console.log(JSON.stringify(noWipResult, null, 2));
  } else {
    console.log('PASS', 'specialist-review gate — no [WIP] plans');
  }
  process.exitCode = 0;
} else {
  const results = wipPlans.map(checkPlanForSpecialistReview);
  const allFound = results.every((r) => r.found);

  const result = {
    pass: allFound,
    evidence: {
      gate: 'specialist-review',
      tier: 1,
      check:
        'Each [WIP] plan with implementation slices must have specialist review evidence in VALIDATION_EVIDENCE before 05-green-testing is dispatched.',
      wipPlans: wipPlans.length,
      planResults: results,
    },
    fixHint: allFound
      ? 'Specialist review evidence confirmed.'
      : "Dispatch 3+ Tier-3 specialists from different relevant viewpoints (e.g., implementation-pattern-scout, nge-core-scout, performance-trace-specialist) to review each 04-implementing slice BEFORE dispatching 05-green-testing. Record APPROVE/REQUEST_CHANGES verdicts in the plan's VALIDATION_EVIDENCE section.",
    owner: 'orchestrator (Agent Zero)',
  };

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', 'specialist-review gate');
    if (!result.pass) {
      for (const r of results) {
        if (!r.found) {
          console.log(`  ${r.plan}: ${r.reason}`);
        }
      }
      console.log('fixHint:', result.fixHint);
    }
  }
  process.exitCode = result.pass ? 0 : 1;
}
