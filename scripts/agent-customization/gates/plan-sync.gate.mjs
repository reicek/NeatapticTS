#!/usr/bin/env node
/**
 * Tier-1 gate: plan-sync
 *
 * Checks that all active [WIP] plan files in plans/ are registered in
 * plans/README.md and plans/Roadmap.md with coherent status markers.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/plan-sync.gate.mjs [--json]
 */

import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseArgs, repoRoot } from '../customization-utils.mjs';

const options = parseArgs(process.argv.slice(2));

const result = await runPlanSyncGate();

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(result.pass ? 'PASS' : 'FAIL', 'plan-sync gate');
  if (!result.pass) console.log('fixHint:', result.fixHint);
}

process.exitCode = result.pass ? 0 : 1;

// ---------------------------------------------------------------------------

async function runPlanSyncGate() {
  // Step 1: Load the registration index files.
  let readmeText = '';
  let roadmapText = '';

  try {
    readmeText = await readFile(path.join(repoRoot, 'plans', 'README.md'), 'utf8');
    roadmapText = await readFile(path.join(repoRoot, 'plans', 'Roadmap.md'), 'utf8');
  } catch (error) {
    return {
      pass: false,
      evidence: { error: String(error), readmeFound: false, roadmapFound: false },
      fixHint: 'Ensure plans/README.md and plans/Roadmap.md both exist.',
      owner: 'validate-plan-sync.mjs',
    };
  }

  // Step 2: Discover plan files in the active plans/ directory (not completed/).
  let planFiles = [];
  try {
    const entries = await readdir(path.join(repoRoot, 'plans'));
    planFiles = entries
      .filter((name) => name.endsWith('.plans.md'))
      .map((name) => `plans/${name}`);
  } catch (error) {
    return {
      pass: false,
      evidence: { error: String(error), scannedDir: 'plans/' },
      fixHint: 'Ensure the plans/ directory is readable.',
      owner: 'validate-plan-sync.mjs',
    };
  }

  // Step 3: Identify which plans carry a [WIP] status marker.
  const wipPlans = [];
  for (const planFile of planFiles) {
    try {
      const text = await readFile(path.join(repoRoot, planFile), 'utf8');
      if (text.includes('[WIP]')) {
        wipPlans.push(planFile);
      }
    } catch {
      // Skip unreadable plan files.
    }
  }

  // Step 4: Check each WIP plan against the registration files.
  const missingFromReadme = wipPlans.filter((planFile) => {
    const basename = planFile.replace('plans/', '');
    return !readmeText.includes(basename);
  });

  const missingFromRoadmap = wipPlans.filter((planFile) => {
    const basename = planFile.replace('plans/', '');
    return !roadmapText.includes(basename);
  });

  const pass = missingFromReadme.length === 0 && missingFromRoadmap.length === 0;

  return {
    pass,
    evidence: {
      wipPlans,
      missingFromReadme,
      missingFromRoadmap,
      plansChecked: planFiles.length,
    },
    fixHint: pass
      ? 'All WIP plans are correctly registered in README and Roadmap.'
      : `Register missing plans in plans/README.md and plans/Roadmap.md: ${[...missingFromReadme, ...missingFromRoadmap].join(', ')}`,
    owner: 'validate-plan-sync.mjs',
  };
}
