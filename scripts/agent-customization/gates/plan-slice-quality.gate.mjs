#!/usr/bin/env node
/**
 * Tier-1 gate: plan-slice-quality
 *
 * Checks that every active [WIP] step packet with slices has all slices
 * within the 4-hour estimate limit. Oversized slices must be broken down
 * before the plan can pass verification.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/plan-slice-quality.gate.mjs [--json]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import {
  parseArgs,
  parsePlanYamlBlock,
  repoRoot,
} from '../customization-utils.mjs';

const SLICE_HOURS_LIMIT = 4;
const MAX_SLICES_PER_STEP = 5;
const YAML_BLOCK_PATTERN = /```yaml\r?\n([\s\S]*?)```/g;

const options = parseArgs(process.argv.slice(2));
const result = await runPlanSliceQualityGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'plan-slice-quality gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

async function runPlanSliceQualityGate() {
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
      owner: 'plan-slice-quality.gate.mjs',
    };
  }

  const violations = [];
  const plansChecked = [];

  for (const planFile of planFiles) {
    let text = '';
    try {
      text = await readFile(path.join(repoRoot, planFile), 'utf8');
    } catch {
      continue;
    }

    plansChecked.push(planFile);
    YAML_BLOCK_PATTERN.lastIndex = 0;
    let match;

    while ((match = YAML_BLOCK_PATTERN.exec(text)) !== null) {
      const rawBlock = match[1];
      const statusMatch =
        /^status:\s*['"]?\[?(?<status>PLANNED|WIP|DONE)\]?['"]?\s*(?:#.*)?$/mu.exec(
          rawBlock,
        );
      if (statusMatch?.groups?.status !== 'WIP') continue;

      let metadata;
      try {
        metadata = parsePlanYamlBlock(rawBlock);
      } catch {
        continue;
      }

      if (metadata.step === undefined) continue;
      if (!Array.isArray(metadata.slices) || metadata.slices.length === 0)
        continue;

      // Check slice count per step (max 5)
      if (metadata.slices.length > MAX_SLICES_PER_STEP) {
        violations.push({
          plan: planFile,
          stepId: `step-${metadata.step}`,
          sliceCount: metadata.slices.length,
          limit: MAX_SLICES_PER_STEP,
          message: `Step ${metadata.step} has ${metadata.slices.length} slices, exceeding the ${MAX_SLICES_PER_STEP}-slice-per-step limit. Split it into multiple smaller steps.`,
        });
      }

      for (const slice of metadata.slices) {
        const sliceId = slice?.slice_id ?? 'unknown';
        if (
          typeof slice?.estimate_hours === 'number' &&
          slice.estimate_hours > SLICE_HOURS_LIMIT
        ) {
          violations.push({
            plan: planFile,
            sliceId,
            estimateHours: slice.estimate_hours,
            limit: SLICE_HOURS_LIMIT,
            message: `Slice ${sliceId} has estimate_hours ${slice.estimate_hours}, exceeding the ${SLICE_HOURS_LIMIT}-hour limit. Break it into smaller slices (ideally 2-3 hours each).`,
          });
        }
      }
    }
  }

  const pass = violations.length === 0;

  return {
    pass,
    evidence: {
      plansChecked,
      violations,
      limit: SLICE_HOURS_LIMIT,
    },
    fixHint: pass
      ? 'All WIP plan slices are within the 4-hour estimate limit and 5-slice-per-step limit.'
      : `Issues found: ${violations.map((v) => v.message).join('; ')}`,
    owner: 'plan-slice-quality.gate.mjs',
  };
}
