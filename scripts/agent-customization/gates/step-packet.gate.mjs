#!/usr/bin/env node
/**
 * Tier-1 gate: step-packet
 *
 * Checks that all active [WIP] step YAML blocks in plans/ contain the required
 * fields (phase, step, agent, status, next_step) and the required surrounding
 * sections (Stop conditions, Required validation).
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/step-packet.gate.mjs [--json]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs, repoRoot } from '../customization-utils.mjs';

const REQUIRED_YAML_FIELDS = ['phase', 'step', 'agent', 'status', 'next_step'];
const YAML_BLOCK_PATTERN = /```yaml\r?\n([\s\S]*?)```/g;
const CONTEXT_WINDOW = 4_000;

const options = parseArgs(process.argv.slice(2));

const result = await runStepPacketGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'step-packet gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

async function runStepPacketGate() {
  // Step 1: Discover plan files in plans/ (not completed/).
  let planFiles = [];
  try {
    const entries = await readdir(path.join(repoRoot, 'plans'));
    planFiles = entries
      .filter((name) => name.endsWith('.plans.md'))
      .map((name) => `plans/${name}`);
  } catch (error) {
    return {
      pass: false,
      evidence: { error: String(error) },
      fixHint: 'Ensure the plans/ directory is readable.',
      owner: 'validate-plan-phase-packets.mjs',
    };
  }

  // Step 2: Scan each plan for WIP yaml step blocks and validate required fields.
  const violations = [];
  const stepsChecked = [];

  for (const planFile of planFiles) {
    let text = '';
    try {
      text = await readFile(path.join(repoRoot, planFile), 'utf8');
    } catch {
      continue;
    }

    YAML_BLOCK_PATTERN.lastIndex = 0;
    let match;
    while ((match = YAML_BLOCK_PATTERN.exec(text)) !== null) {
      const block = match[1];
      // Only check WIP step packets.
      if (
        !block.includes('status: "[WIP]"') &&
        !block.includes("status: '[WIP]'")
      )
        continue;

      const stepId = `${planFile}:yaml@${match.index}`;
      stepsChecked.push(stepId);

      // Check required yaml fields.
      for (const field of REQUIRED_YAML_FIELDS) {
        if (!block.includes(`${field}:`)) {
          violations.push({ stepId, missingField: field });
        }
      }

      // Check for required prose sections in the surrounding context.
      const surroundingContext = text.slice(
        match.index,
        match.index + CONTEXT_WINDOW,
      );
      if (
        !surroundingContext.includes('Stop conditions') &&
        !surroundingContext.includes('stop conditions')
      ) {
        violations.push({ stepId, missingSection: 'Stop conditions' });
      }
      if (
        !surroundingContext.includes('Required validation') &&
        !surroundingContext.includes('required validation')
      ) {
        violations.push({ stepId, missingSection: 'Required validation' });
      }
    }
  }

  const pass = violations.length === 0;

  return {
    pass,
    evidence: {
      stepsChecked,
      violations,
      plansScanned: planFiles.length,
    },
    fixHint: pass
      ? 'All active WIP step packets have required fields and sections.'
      : `Fix missing fields or sections in WIP step packets: ${violations.map((violation) => violation.missingField ?? violation.missingSection).join(', ')}`,
    owner: 'validate-plan-phase-packets.mjs',
  };
}
