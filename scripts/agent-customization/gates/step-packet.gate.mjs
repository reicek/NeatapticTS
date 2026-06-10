#!/usr/bin/env node
/**
 * Tier-1 gate: step-packet
 *
 * Checks that all active [WIP] step YAML blocks in plans/ contain the required
 * fields (phase, step, goal, status, next_step) and the required surrounding
 * sections (Stop conditions, Required validation).
 *
 * Validates that:
 * - 'goal' is required and must be one of the allowed values
 * - 'agent' is deprecated and rejected as a violation
 * - 'tdd_sequence' values are valid when present
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/step-packet.gate.mjs [--json]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs, repoRoot } from '../customization-utils.mjs';

/** Fields required in every WIP step packet. */
const REQUIRED_YAML_FIELDS = ['phase', 'step', 'goal', 'status', 'next_step'];

/** Allowed values for the 'goal' routing field. */
const VALID_GOAL_VALUES = [
  'planning',
  'researching',
  'red-testing',
  'implementing',
  'green-testing',
  'documenting',
  'logging',
  'helping',
];

/** Allowed values for the optional 'tdd_sequence' field. */
const VALID_TDD_SEQUENCE_VALUES = ['red-green', 'green-only'];

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

/**
 * Extracts the value of a YAML field from a block string.
 * Strips surrounding single or double quotes from the value.
 *
 * @param block - The YAML block string to search
 * @param field - The field name to extract
 * @returns The extracted value (quotes stripped) or undefined if not found
 */
function extractYamlValue(block, field) {
  const regex = new RegExp(`^${field}:\\s*(.+)$`, 'm');
  const match = block.match(regex);
  if (!match) return undefined;
  let raw = match[1].trim();
  if (
    (raw.startsWith("'") && raw.endsWith("'")) ||
    (raw.startsWith('"') && raw.endsWith('"'))
  ) {
    raw = raw.slice(1, -1);
  }
  return raw;
}

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

      // Check required yaml fields (non-routing).
      for (const field of REQUIRED_YAML_FIELDS) {
        if (!block.includes(`${field}:`)) {
          violations.push({ stepId, missingField: field });
        }
      }

      // Check for deprecated 'agent' field — no longer accepted.
      const hasAgent = block.includes('agent:');

      if (hasAgent) {
        violations.push({
          stepId,
          deprecatedField: 'agent',
          message:
            'The "agent" field is deprecated and no longer accepted; use "goal" instead.',
        });
      }

      // Validate goal value if present.
      const goalValue = extractYamlValue(block, 'goal');
      if (goalValue !== undefined && !VALID_GOAL_VALUES.includes(goalValue)) {
        violations.push({
          stepId,
          invalidField: 'goal',
          invalidValue: goalValue,
          allowedValues: [...VALID_GOAL_VALUES],
        });
      }

      // Validate tdd_sequence value if present.
      if (block.includes('tdd_sequence:')) {
        const tddValue = extractYamlValue(block, 'tdd_sequence');
        if (
          tddValue !== undefined &&
          !VALID_TDD_SEQUENCE_VALUES.includes(tddValue)
        ) {
          violations.push({
            stepId,
            invalidField: 'tdd_sequence',
            invalidValue: tddValue,
            allowedValues: [...VALID_TDD_SEQUENCE_VALUES],
          });
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
      : `Fix missing or deprecated fields or sections in WIP step packets: ${violations.map((violation) => violation.missingField ?? violation.missingSection ?? violation.invalidField ?? violation.deprecatedField).join(', ')}`,
    owner: 'validate-plan-phase-packets.mjs',
  };
}
