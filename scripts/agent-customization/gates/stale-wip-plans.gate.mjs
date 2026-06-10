#!/usr/bin/env node
/**
 * Tier-1 gate: stale-wip-plans
 *
 * Detects plans whose top-level **Status:** is [WIP] but whose
 * ## Implementation phases section contains only [DONE] phase/step markers.
 * This is the "missed closure" condition: all implementation work is done but
 * the tracker was never compressed and moved to plans/completed/.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/stale-wip-plans.gate.mjs [--json]
 *
 * Detection logic:
 *   1. Scan all non-perpetual plan files in plans/ (excludes completed/, README,
 *      Roadmap, and plans without an ## Implementation phases section).
 *   2. For each WIP plan, extract the Implementation phases section.
 *   3. Collect phase/step status markers from:
 *      - Markdown headings: lines starting with # that contain [DONE|WIP|PLANNED]
 *      - YAML step-packet fields: lines matching status: '[DONE|WIP|PLANNED]'
 *   4. If ALL collected markers are [DONE] and at least one exists, the plan is stale.
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { extractStatus, parseArgs, repoRoot } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = await runStaleWipGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'stale-wip-plans gate');
  if (!result.pass) {
    console.log('fixHint:', result.fixHint);
    for (const stale of result.evidence.stalePlans) {
      console.log(' -', stale.plan, `(${stale.phaseCount} phases all [DONE])`);
    }
  }
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

/**
 * Collects phase and step status markers from the implementation phases section
 * of a plan file.
 *
 * Matches:
 * - Heading lines (`# heading`) containing `[DONE]`, `[WIP]`, or `[PLANNED]`
 * - YAML step-packet status lines: `status: '[DONE|WIP|PLANNED]'`
 *
 * @param implSection - Text content starting from `## Implementation phases`.
 * @returns Array of status strings ('DONE' | 'WIP' | 'PLANNED').
 */
function collectPhaseMarkers(implSection) {
  const phaseStatuses = [];
  const statusRe = /\[(DONE|WIP|PLANNED)\]/;

  for (const line of implSection.split('\n')) {
    const trimmed = line.trim();

    // Markdown heading with inline status tag
    if (trimmed.startsWith('#')) {
      const match = statusRe.exec(trimmed);
      if (match) phaseStatuses.push(match[1]);
      continue;
    }

    // YAML step-packet status field (e.g. `status: '[DONE]'`)
    if (/^status:\s*'\[(?:DONE|WIP|PLANNED)\]'/.test(trimmed)) {
      const match = statusRe.exec(trimmed);
      if (match) phaseStatuses.push(match[1]);
    }
  }

  return phaseStatuses;
}

/**
 * Inspects a single plan file and returns a stale-WIP descriptor when the plan
 * qualifies (status WIP but all implementation markers are DONE), or null when
 * the plan is legitimately WIP or cannot be evaluated.
 *
 * @param planPath - Repo-relative path to the plan file.
 * @param planText - Full text content of the plan file.
 * @returns Stale descriptor or null.
 */
function detectStalePlan(planPath, planText) {
  const status = extractStatus(planText);
  if (status !== 'WIP') return null;

  // Step 1: Locate the Implementation phases section.
  const implIdx = planText.indexOf('## Implementation phases');
  if (implIdx === -1) return null; // Design spec or perpetual plan with no phase list — skip.

  const implSection = planText.slice(implIdx);

  // Step 2: Collect all status markers visible in the section.
  const phaseStatuses = collectPhaseMarkers(implSection);
  if (phaseStatuses.length === 0) return null; // No markers found — cannot evaluate.

  // Step 3: If any non-DONE marker exists, the plan is legitimately WIP.
  const hasOpenWork = phaseStatuses.some((marker) => marker !== 'DONE');
  if (hasOpenWork) return null;

  return {
    plan: planPath,
    topLevelStatus: 'WIP',
    phaseCount: phaseStatuses.length,
    finding: `All ${phaseStatuses.length} phase/step marker(s) are [DONE] but **Status:** is still [WIP]`,
  };
}

async function runStaleWipGate() {
  // Step 1: Discover root-level Markdown tracker files in plans/ (not completed/).
  let planFiles = [];
  try {
    const entries = await readdir(path.join(repoRoot, 'plans'), {
      withFileTypes: true,
    });
    planFiles = entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter(
        (name) =>
          name.endsWith('.md') &&
          !name.endsWith('.logs.md') &&
          !['README.md', 'Roadmap.md'].includes(name),
      )
      .map((name) => `plans/${name}`);
  } catch (error) {
    return {
      pass: false,
      evidence: { error: String(error), scannedDir: 'plans/' },
      fixHint: 'Ensure the plans/ directory is readable.',
      owner: 'stale-wip-plans.gate.mjs',
    };
  }

  // Step 2: Evaluate each plan file for the stale-WIP condition.
  const stalePlans = [];
  const checked = [];

  for (const planFile of planFiles) {
    let text;
    try {
      text = await readFile(path.join(repoRoot, planFile), 'utf8');
    } catch {
      continue; // Skip unreadable files.
    }

    checked.push(planFile);
    const stale = detectStalePlan(planFile, text);
    if (stale) stalePlans.push(stale);
  }

  const pass = stalePlans.length === 0;

  return {
    pass,
    evidence: {
      stalePlans,
      plansChecked: checked.length,
      plansFound: planFiles.length,
    },
    fixHint: pass
      ? 'No stale WIP plans detected — all active plans have open work remaining.'
      : `${stalePlans.length} plan(s) have Status: [WIP] but all implementation phases are [DONE]. ` +
        `For each stale plan: (1) change **Status:** to [DONE], (2) compress history to concise closure notes, ` +
        `(3) update the paired .logs.md, (4) move both files to plans/completed/, ` +
        `(5) update plans/README.md and plans/Roadmap.md entries. ` +
        `Stale plans: ${stalePlans.map((s) => s.plan).join(', ')}`,
    owner: 'stale-wip-plans.gate.mjs',
  };
}
