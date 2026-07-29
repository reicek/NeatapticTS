#!/usr/bin/env node
/**
 * Tier-1 gate: convergence-tracker
 *
 * Reads a plan's `## Latest validation evidence` section, counts fix-loop
 * iteration markers for a target slice, and escalates to `00-helping` after 4
 * failed iterations without a green pass.
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string|null, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/convergence-tracker.gate.mjs --json
 *   node scripts/agent-customization/gates/convergence-tracker.gate.mjs --json --plan=plans/orchestration-fixes.plans.md --slice-id=A6-impl
 */

import { pathToFileURL } from 'node:url';

import { parseArgs, readWorkspaceFile } from '../customization-utils.mjs';

const OWNER = 'convergence-tracker';
const MAX_FAILED_ITERATIONS = 4;
const EVIDENCE_HEADING = '## Latest validation evidence';

/* istanbul ignore next */
const isMain = import.meta.url === pathToFileURL(process.argv[1] ?? '').href;

/**
 * CLI entry point for the convergence tracker gate.
 *
 * @param {string[]} [argv] - Raw CLI arguments (defaults to `process.argv.slice(2)`).
 * @returns {Promise<object>} The standard gate contract.
 */
export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);

  if (options.help) {
    console.log(`convergence-tracker gate

Usage:
  node scripts/agent-customization/gates/convergence-tracker.gate.mjs --json
  node scripts/agent-customization/gates/convergence-tracker.gate.mjs --json --plan=plans/orchestration-fixes.plans.md --slice-id=A6-impl

Options:
  --json               Write machine-readable JSON to stdout.
  --plan               Path to a repo-relative Markdown plan file.
  --slice-id           Slice identifier to evaluate.
  --help, -h           Show this help text.`);
    process.exit(0);
  }

  const sliceId = parseSliceId(argv);
  const planPath = parsePlanFlag(argv);

  if (!planPath) {
    const result = standaloneDescriptor();
    emit(options.json, result);
    return result;
  }

  if (!sliceId) {
    const result = {
      pass: false,
      evidence: {
        mode: 'cli-error',
        owner: OWNER,
        error: 'Missing required --slice-id when --plan is provided.',
      },
      fixHint: 'Re-run with --slice-id=<slice-id>.',
      owner: OWNER,
    };
    emit(options.json, result);
    return result;
  }

  let planText;
  try {
    planText = await readWorkspaceFile(planPath);
  } catch (error) {
    const result = {
      pass: false,
      evidence: {
        mode: 'read-error',
        owner: OWNER,
        planPath,
        error: error instanceof Error ? error.message : String(error),
      },
      fixHint: `Could not read plan file at ${planPath}. Verify the path exists and is readable.`,
      owner: OWNER,
    };
    emit(options.json, result);
    return result;
  }

  const result = runConvergenceTrackerGate({ planText, sliceId });

  emit(options.json, result);
  return result;
}

/* istanbul ignore next */
if (isMain) {
  main();
}

// ---------------------------------------------------------------------------

/**
 * Runs the convergence tracker gate.
 *
 * Scans the plan text for `fix-loop: <sliceId> iteration <n> status=<status>`
 * markers under `## Latest validation evidence`. If any matching marker has
 * `status=passed` (case-insensitive), the iteration count resets to 0 and the
 * gate passes. Otherwise, if the count exceeds 4, the gate fails with an
 * escalation hint pointing to `00-helping`.
 *
 * @param {object} params - Gate parameters.
 * @param {string} params.planText - Raw plan Markdown text.
 * @param {string} params.sliceId - Slice identifier to evaluate.
 * @returns {object} Standard gate contract.
 */
export function runConvergenceTrackerGate({ planText, sliceId }) {
  const section = extractLatestValidationEvidence(planText);
  const markers = parseFixLoopMarkers(section, sliceId);

  const passed = markers.some(
    (marker) => marker.status.toLowerCase() === 'passed',
  );

  if (passed) {
    return {
      pass: true,
      evidence: {
        sliceId,
        iterationCount: 0,
        status: 'OK',
        resetReason: 'green-pass marker found',
      },
      fixHint: null,
      owner: OWNER,
    };
  }

  const iterationCount = markers.length;

  if (iterationCount > MAX_FAILED_ITERATIONS) {
    return {
      pass: false,
      evidence: {
        sliceId,
        iterationCount,
        status: 'ESCALATE',
      },
      fixHint: `Slice ${sliceId} has failed ${iterationCount} fix-loop iterations without a green pass. Escalate to 00-helping for orchestration-level intervention.`,
      owner: OWNER,
    };
  }

  return {
    pass: true,
    evidence: {
      sliceId,
      iterationCount,
      status: 'OK',
    },
    fixHint: null,
    owner: OWNER,
  };
}

/**
 * Returns a standalone descriptor used when the gate is invoked without a plan.
 *
 * @returns {object} Standard gate contract in descriptor mode.
 */
function standaloneDescriptor() {
  return {
    pass: true,
    evidence: {
      mode: 'standalone-descriptor',
      owner: OWNER,
      description:
        'Convergence tracker gate: counts fix-loop iteration markers for a slice in the latest validation evidence and escalates to 00-helping after 4 failed iterations without a green pass.',
    },
    fixHint: 'n/a',
    owner: OWNER,
  };
}

/**
 * Emits the gate result to stdout.
 *
 * @param {boolean} json - Whether to emit JSON.
 * @param {object} result - Gate contract.
 */
function emit(json, result) {
  if (json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log(result.pass ? 'PASS' : 'FAIL', `${OWNER} gate`);
    if (!result.pass) console.log('fixHint:', result.fixHint);
  }

  process.exitCode = result.pass ? 0 : 1;
}

/**
 * Extracts the body of the `## Latest validation evidence` section from plan
 * text. Returns the empty string if the section is missing.
 *
 * @param {string} planText - Raw plan Markdown text.
 * @returns {string} Section body, or empty string if not found.
 */
function extractLatestValidationEvidence(planText) {
  const index = planText.indexOf(EVIDENCE_HEADING);
  if (index === -1) {
    return '';
  }

  const afterHeading = planText.slice(index + EVIDENCE_HEADING.length);
  const nextHeading = afterHeading.search(/^#{1,6}\s+/m);
  const section =
    nextHeading === -1 ? afterHeading : afterHeading.slice(0, nextHeading);

  return section;
}

/**
 * Parses fix-loop iteration markers for the given slice from a section body.
 *
 * Marker format (bullets tolerated):
 *   fix-loop: <slice-id> iteration <n> status=<failed|passed>
 *
 * @param {string} section - Section body text.
 * @param {string} sliceId - Slice identifier to match.
 * @returns {Array<{iteration: number, status: string}>} Matched markers.
 */
function parseFixLoopMarkers(section, sliceId) {
  const lines = section.split(/\r?\n/);
  const markers = [];
  const escapedSliceId = sliceId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const pattern = new RegExp(
    `^\\s*(?:-\\s+)?fix-loop:\\s*${escapedSliceId}\\s+iteration\\s+(\\d+)\\s+status=(\\S+)`,
    'iu',
  );

  for (const line of lines) {
    const match = pattern.exec(line);
    if (match) {
      markers.push({
        iteration: parseInt(match[1], 10),
        status: match[2],
      });
    }
  }

  return markers;
}

/**
 * Parses `--plan` from raw CLI arguments.
 *
 * @param {string[]} argv - Raw CLI arguments.
 * @returns {string|undefined} Repo-relative plan path, or undefined if not provided.
 */
function parsePlanFlag(argv) {
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg.startsWith('--plan=')) {
      return arg.slice('--plan='.length);
    }
    if (arg === '--plan' && argv[index + 1] !== undefined) {
      return argv[++index];
    }
  }
  return undefined;
}

/**
 * Parses `--slice-id` from raw CLI arguments.
 *
 * @param {string[]} argv - Raw CLI arguments.
 * @returns {string|undefined} Slice identifier, or undefined if not provided.
 */
function parseSliceId(argv) {
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg.startsWith('--slice-id=')) {
      return arg.slice('--slice-id='.length);
    }
    if (arg === '--slice-id' && argv[index + 1] !== undefined) {
      return argv[++index];
    }
  }
  return undefined;
}
