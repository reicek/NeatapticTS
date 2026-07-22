#!/usr/bin/env node
/**
 * Tier-1 gate: plan-readiness
 *
 * Checks that an active plan has been independently verified by a fresh
 * 01-planning agent and contains a green-light marker in its
 * ## Latest validation evidence section.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/plan-readiness.gate.mjs --plan=plans/<Plan>.plans.md [--json]
 */

import { readWorkspaceFile, parseArgs } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

if (options.help) {
  console.log(`plan-readiness gate

Usage:
  node scripts/agent-customization/gates/plan-readiness.gate.mjs --plan=plans/<Plan>.plans.md [--json]

Options:
  --plan   Path to the plan file to check (repo-relative).
  --json   Emit machine-readable gate contract JSON.`);
  process.exit(0);
}

const planPath = options.plan;
const result = await runPlanReadinessGate(planPath);

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'plan-readiness gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

async function runPlanReadinessGate(planPath) {
  let text;
  try {
    text = await readWorkspaceFile(planPath);
  } catch (error) {
    return {
      pass: false,
      evidence: {
        error: String(error),
        plan: planPath,
      },
      fixHint: `Ensure ${planPath} exists and is readable.`,
      owner: '01-planning',
    };
  }

  const sectionMatch = text.match(
    /(?:^|\n)## Latest validation evidence\s*\r?\n([\s\S]*?)(?=\r?\n## |$)/,
  );
  const hasSection = sectionMatch !== null;
  const sectionText = hasSection ? sectionMatch[1] : '';

  const greenLightPatterns = [
    /green-light\s*[:=]\s*true/i,
    /status\s*[:=]\s*green-light/i,
  ];

  const hasGreenLight = greenLightPatterns.some((pattern) =>
    pattern.test(sectionText),
  );

  if (!hasSection) {
    return {
      pass: false,
      evidence: {
        plan: planPath,
        sectionFound: false,
        greenLightFound: false,
        preview: text.slice(0, 200),
      },
      fixHint:
        'Dispatch a fresh 01-planning verification agent to validate the plan and create a ## Latest validation evidence section with green-light: true or status: green-light.',
      owner: '01-planning',
    };
  }

  if (!hasGreenLight) {
    return {
      pass: false,
      evidence: {
        plan: planPath,
        sectionFound: true,
        greenLightFound: false,
        sectionPreview: sectionText.slice(0, 400),
      },
      fixHint:
        'Verification did not record a green light. Dispatch a fresh 01-planning verification agent to patch blockers and record green-light: true or status: green-light in the ## Latest validation evidence section.',
      owner: '01-planning',
    };
  }

  return {
    pass: true,
    evidence: {
      plan: planPath,
      sectionFound: true,
      greenLightFound: true,
      sectionPreview: sectionText.slice(0, 200),
    },
    fixHint:
      'Plan has a recorded green light from independent 01-planning verification.',
    owner: '01-planning',
  };
}
