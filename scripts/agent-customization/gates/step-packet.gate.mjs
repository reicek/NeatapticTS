#!/usr/bin/env node
/**
 * Tier-1 gate: step-packet
 *
 * Checks every active [WIP] phase/step YAML block in plans/ for the required
 * fields of the new workflow format. Phase-level WIP blocks must be
 * phase-kickoff packets (expansion: steps, auto_expand: false). Step-level WIP
 * blocks must carry the required routing fields and, when they expand into
 * slices, a valid slice list.
 *
 * Legacy blocks without expansion: or with agent:/agent_file: are rejected.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/step-packet.gate.mjs [--json]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import {
  parseArgs,
  parsePlanYamlBlock,
  repoRoot,
} from '../customization-utils.mjs';

const VALID_GOAL_VALUES = new Set([
  'planning',
  'researching',
  'red-testing',
  'implementing',
  'green-testing',
  'documenting',
  'logging',
  'helping',
]);

const VALID_TDD_SEQUENCE_VALUES = new Set(['red-green', 'green-only']);
const STEP_EXPANSION_VALUES = new Set(['none', 'slices']);
const SLICE_GOAL_VALUES = new Set([
  'red-testing',
  'implementing',
  'green-testing',
  'helping',
]);

const YAML_BLOCK_PATTERN = /```yaml\r?\n([\s\S]*?)```/g;
const MIGRATION_HINT =
  'Legacy format detected. Run: node scripts/agent-customization/migrate-plan-format.mjs --plan=<plan-file>';

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
      owner: 'step-packet.gate.mjs',
    };
  }

  const violations = [];
  const blocksChecked = [];

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
      const rawBlock = match[1];
      const statusValue = extractStatusFromYaml(rawBlock);
      if (statusValue !== 'WIP') continue;

      const blockId = `${planFile}:yaml@${match.index}`;
      let metadata;
      try {
        metadata = parsePlanYamlBlock(rawBlock);
      } catch (error) {
        violations.push({
          blockId,
          parseError: String(error),
        });
        continue;
      }

      blocksChecked.push(blockId);

      if (isLegacyBlock(metadata)) {
        violations.push({
          blockId,
          legacy: true,
          message: MIGRATION_HINT,
        });
        continue;
      }

      if (metadata.step !== undefined) {
        validateStepBlock(metadata, blockId, violations);
      } else if (metadata.phase !== undefined) {
        validatePhaseBlock(metadata, blockId, violations);
      } else {
        violations.push({
          blockId,
          missingField: 'phase',
        });
      }
    }
  }

  const pass = violations.length === 0;

  return {
    pass,
    evidence: {
      blocksChecked,
      violations,
      plansScanned: planFiles.length,
    },
    fixHint: pass
      ? 'All active WIP phase/step packets conform to the new format.'
      : `Fix new-format violations in WIP phase/step packets: ${violations
          .map((violation) =>
            [
              violation.missingField,
              violation.invalidField,
              violation.deprecatedField,
              violation.parseError,
              violation.message,
            ]
              .filter(Boolean)
              .join(' '),
          )
          .join('; ')}`,
    owner: 'step-packet.gate.mjs',
  };
}

function extractStatusFromYaml(rawBlock) {
  const match =
    /^status:\s*['"]?\[?(?<status>PLANNED|WIP|DONE)\]?['"]?\s*(?:#.*)?$/mu.exec(
      rawBlock,
    );
  return match?.groups?.status ?? null;
}

function isLegacyBlock(metadata) {
  if (!metadata || typeof metadata !== 'object') return false;
  if (metadata.agent !== undefined || metadata.agent_file !== undefined)
    return true;
  if (metadata.expansion === undefined) return true;
  return false;
}

function validatePhaseBlock(metadata, blockId, violations) {
  const requiredFields = [
    'phase',
    'title',
    'status',
    'goal',
    'expansion',
    'auto_expand',
    'mode',
    'source_of_truth',
    'copy_paste',
    'next_phase',
    'skills',
    'validation',
    'acceptance_criteria',
    'placeholder_steps',
  ];

  for (const field of requiredFields) {
    if (!Object.hasOwn(metadata, field)) {
      violations.push({ blockId, missingField: field, level: 'phase' });
    }
  }

  if (metadata.goal !== 'planning') {
    violations.push({
      blockId,
      invalidField: 'goal',
      invalidValue: metadata.goal,
      expected: 'planning',
      level: 'phase',
    });
  }

  if (metadata.expansion !== 'steps') {
    violations.push({
      blockId,
      invalidField: 'expansion',
      invalidValue: metadata.expansion,
      expected: 'steps',
      level: 'phase',
    });
  }

  if (metadata.auto_expand !== false) {
    violations.push({
      blockId,
      invalidField: 'auto_expand',
      invalidValue: metadata.auto_expand,
      expected: false,
      level: 'phase',
    });
  }

  if (!Array.isArray(metadata.skills) || metadata.skills.length === 0) {
    violations.push({
      blockId,
      missingField: 'skills',
      message: 'skills must be a non-empty list',
      level: 'phase',
    });
  }

  if (!Array.isArray(metadata.validation) || metadata.validation.length === 0) {
    violations.push({
      blockId,
      missingField: 'validation',
      message: 'validation must be a non-empty list',
      level: 'phase',
    });
  }
}

function validateStepBlock(metadata, blockId, violations) {
  const requiredFields = [
    'phase',
    'step',
    'title',
    'status',
    'goal',
    'mode',
    'source_of_truth',
    'copy_paste',
    'next_step',
    'skills',
    'validation',
    'acceptance_criteria',
  ];

  for (const field of requiredFields) {
    if (!Object.hasOwn(metadata, field)) {
      violations.push({ blockId, missingField: field, level: 'step' });
    }
  }

  if (metadata.agent !== undefined) {
    violations.push({
      blockId,
      deprecatedField: 'agent',
      message:
        'The "agent" field is deprecated and no longer accepted; use "goal".',
    });
  }

  if (metadata.agent_file !== undefined) {
    violations.push({
      blockId,
      deprecatedField: 'agent_file',
      message: 'The "agent_file" field is deprecated and no longer accepted.',
    });
  }

  if (!VALID_GOAL_VALUES.has(metadata.goal)) {
    violations.push({
      blockId,
      invalidField: 'goal',
      invalidValue: metadata.goal,
      allowedValues: [...VALID_GOAL_VALUES],
    });
  }

  if (
    metadata.tdd_sequence !== undefined &&
    !VALID_TDD_SEQUENCE_VALUES.has(metadata.tdd_sequence)
  ) {
    violations.push({
      blockId,
      invalidField: 'tdd_sequence',
      invalidValue: metadata.tdd_sequence,
      allowedValues: [...VALID_TDD_SEQUENCE_VALUES],
    });
  }

  if (
    metadata.expansion !== undefined &&
    !STEP_EXPANSION_VALUES.has(metadata.expansion)
  ) {
    violations.push({
      blockId,
      invalidField: 'expansion',
      invalidValue: metadata.expansion,
      allowedValues: [...STEP_EXPANSION_VALUES],
    });
  }

  if (!Array.isArray(metadata.skills) || metadata.skills.length === 0) {
    violations.push({
      blockId,
      missingField: 'skills',
      message: 'skills must be a non-empty list',
    });
  }

  if (!Array.isArray(metadata.validation) || metadata.validation.length === 0) {
    violations.push({
      blockId,
      missingField: 'validation',
      message: 'validation must be a non-empty list',
    });
  }

  if (
    !Array.isArray(metadata.acceptance_criteria) ||
    metadata.acceptance_criteria.length === 0
  ) {
    violations.push({
      blockId,
      missingField: 'acceptance_criteria',
      message: 'acceptance_criteria must be a non-empty list',
    });
  }

  if (metadata.expansion === 'slices') {
    if (metadata.auto_expand !== true) {
      violations.push({
        blockId,
        invalidField: 'auto_expand',
        invalidValue: metadata.auto_expand,
        expected: true,
        message: 'expansion: slices requires auto_expand: true',
      });
    }

    if (!metadata.tdd_sequence) {
      violations.push({
        blockId,
        missingField: 'tdd_sequence',
        message: 'expansion: slices requires a tdd_sequence',
      });
    }

    if (!Array.isArray(metadata.slices) || metadata.slices.length === 0) {
      violations.push({
        blockId,
        missingField: 'slices',
        message: 'expansion: slices requires a non-empty slices list',
      });
    } else {
      validateSlices(
        metadata.slices,
        metadata.tdd_sequence,
        blockId,
        violations,
      );
    }
  }
}

function validateSlices(slices, tddSequence, blockId, violations) {
  const expectedGoals =
    tddSequence === 'green-only'
      ? ['implementing', 'green-testing']
      : ['red-testing', 'implementing', 'green-testing'];

  for (const [index, slice] of slices.entries()) {
    const sliceId = slice?.slice_id ?? `slice-${index}`;
    const requiredKeys = [
      'slice_id',
      'title',
      'status',
      'goal',
      'estimate_hours',
      'files_to_change',
      'acceptance_criteria',
      'parallelizable',
      'dependencies',
    ];

    for (const key of requiredKeys) {
      if (!Object.hasOwn(slice ?? {}, key)) {
        violations.push({
          blockId,
          sliceId,
          missingField: key,
          message: `slice ${index} missing key ${key}`,
        });
      }
    }

    if (slice && !SLICE_GOAL_VALUES.has(slice.goal)) {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'goal',
        invalidValue: slice.goal,
        allowedValues: [...SLICE_GOAL_VALUES],
        message: `slice ${index} has invalid goal`,
      });
    }

    const expectedGoal = expectedGoals[index];
    if (slice && expectedGoal && slice.goal !== expectedGoal) {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'goal',
        invalidValue: slice.goal,
        expected: expectedGoal,
        message: `slice ${index} expected goal ${expectedGoal}`,
      });
    }

    if (slice && !Array.isArray(slice.files_to_change)) {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'files_to_change',
        message: `slice ${index} files_to_change must be a list`,
      });
    }

    if (slice && !Array.isArray(slice.acceptance_criteria)) {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'acceptance_criteria',
        message: `slice ${index} acceptance_criteria must be a list`,
      });
    }

    if (slice && typeof slice.parallelizable !== 'boolean') {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'parallelizable',
        message: `slice ${index} parallelizable must be a boolean`,
      });
    }

    if (slice && !Array.isArray(slice.dependencies)) {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'dependencies',
        message: `slice ${index} dependencies must be a list`,
      });
    }

    if (slice && typeof slice.estimate_hours !== 'number') {
      violations.push({
        blockId,
        sliceId,
        invalidField: 'estimate_hours',
        message: `slice ${index} estimate_hours must be a number`,
      });
    }
  }
}
