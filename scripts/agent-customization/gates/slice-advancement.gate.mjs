#!/usr/bin/env node
/**
 * Tier-1 master gate: slice-advancement
 *
 * Consolidates per-slice gate checks into a single call with smart internal
 * routing based on slice severity classification. This reduces the
 * 6+ gate critical path to one gate invocation per slice.
 *
 * TRIVIAL slices (docs, comments, formatting, plan-only):
 *   plan-sync + step-packet + plan-slice-quality + plan-command-lint
 *
 * FULL slices (source code logic changes):
 *   all TRIVIAL gates + shared-validation + code-coverage + specialist-review
 *   + convergence-tracker (if fix-loop markers exist)
 *
 * Gate contract: { pass: boolean, evidence: object, fixHint: string, owner: string }
 *
 * Usage:
 *   node scripts/agent-customization/gates/slice-advancement.gate.mjs \
 *     --json \
 *     --slice-id=C3-impl \
 *     --changed-files=src/foo.ts,README.md
 */

import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { parseArgs } from '../customization-utils.mjs';

const REPO_ROOT = path.resolve(
  path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Z]:)/, '$1'),
  '..',
  '..',
  '..',
);

const GATES_DIR = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'gates',
);

const options = parseArgs(process.argv.slice(2));

// parseArgs doesn't handle --slice-id or --changed-files, so parse them directly
const rawArgv = process.argv.slice(2);
const sliceId = (() => {
  const arg = rawArgv.find((a) => a.startsWith('--slice-id='));
  return arg ? arg.slice('--slice-id='.length) : '';
})();
const changedFilesRaw = (() => {
  const arg = rawArgv.find((a) => a.startsWith('--changed-files='));
  return arg ? arg.slice('--changed-files='.length) : '';
})();

if (!options.json && options.help !== undefined) {
  console.log(`slice-advancement gate — master consolidated gate

Usage:
  node slice-advancement.gate.mjs --json --slice-id=<id> --changed-files=f1,f2

Options:
  --json               Output JSON gate contract
  --slice-id=<id>      Slice identifier (required)
  --changed-files=f1,f2  Comma-separated changed file paths
  --help               Show this help text
`);
  process.exitCode = 0;
}

if (!sliceId) {
  const result = {
    pass: false,
    evidence: {
      gate: 'slice-advancement',
      tier: 1,
      error: 'Missing required --slice-id parameter',
    },
    fixHint:
      'Provide --slice-id=<slice-id> and --changed-files=<file1>,<file2>,...',
    owner: 'orchestrator (Agent Zero)',
  };
  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    console.log('FAIL slice-advancement gate — missing --slice-id');
  }
  process.exitCode = 1;
}

/**
 * Run a sub-gate and return its parsed JSON result.
 * @param {string} gateName - gate script filename (without .gate.mjs)
 * @param {string[]} extraArgs - additional CLI args
 * @returns {{ gate: string, pass: boolean, evidence: object, fixHint: string|null, raw: string }}
 */
function runSubGate(gateName, extraArgs = []) {
  const gatePath = path.join(GATES_DIR, `${gateName}.gate.mjs`);
  const args = ['--json', ...extraArgs];
  try {
    const stdout = execFileSync('node', [gatePath, ...args], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 120000,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const parsed = JSON.parse(stdout);
    return {
      gate: gateName,
      pass: parsed.pass === true,
      evidence: parsed.evidence || {},
      fixHint: parsed.fixHint || null,
      raw: stdout.trim(),
    };
  } catch (err) {
    return {
      gate: gateName,
      pass: false,
      evidence: { error: err.message },
      fixHint: `Run 'node scripts/agent-customization/gates/${gateName}.gate.mjs --json' to see details.`,
      raw: err.stdout || err.message,
    };
  }
}

/**
 * Classify severity using the specialist-review-severity gate.
 * @param {string[]} files - changed file paths
 * @returns {{ severity: string, specialistCount: number }}
 */
function classifySeverity(files) {
  const gatePath = path.join(GATES_DIR, 'specialist-review-severity.gate.mjs');
  try {
    const stdout = execFileSync(
      'node',
      [gatePath, '--json', `--input=${files.join(',')}`],
      { cwd: REPO_ROOT, encoding: 'utf8', timeout: 30000 },
    );
    const parsed = JSON.parse(stdout);
    const classification = parsed.evidence?.classification || {};
    return {
      severity: classification.severity || 'FULL',
      specialistCount: classification.specialistCount ?? 1,
    };
  } catch {
    return { severity: 'FULL', specialistCount: 1 };
  }
}

// Parse changed files
const changedFiles = changedFilesRaw
  .split(',')
  .map((f) => f.trim())
  .filter(Boolean);

// Step 1: Classify severity
const { severity, specialistCount } = classifySeverity(changedFiles);

// Step 2: Run gates based on severity
// TRIVIAL gates always run
const trivialGates = [
  'plan-sync',
  'step-packet',
  'plan-slice-quality',
  'plan-command-lint',
];

// FULL gates run in addition to TRIVIAL gates
const fullGates = ['shared-validation', 'code-coverage', 'specialist-review'];

const gatesToRun = [...trivialGates];
if (severity === 'FULL') {
  gatesToRun.push(...fullGates);
}

// Run all applicable gates
const results = gatesToRun.map((gateName) => {
  const extraArgs = [];
  if (gateName === 'shared-validation' && changedFiles.length > 0) {
    extraArgs.push(`--changed-files=${changedFiles.join(',')}`);
  }
  if (gateName === 'code-coverage' && changedFiles.length > 0) {
    extraArgs.push(`--changed-files=${changedFiles.join(',')}`);
  }
  return runSubGate(gateName, extraArgs);
});

// Aggregate
const allPassed = results.every((r) => r.pass);
const failedGates = results.filter((r) => !r.pass);

const result = {
  pass: allPassed,
  evidence: {
    gate: 'slice-advancement',
    tier: 1,
    sliceId,
    severity,
    specialistCount,
    gatesRun: gatesToRun,
    gateCount: gatesToRun.length,
    results: results.map((r) => ({
      gate: r.gate,
      pass: r.pass,
      fixHint: r.fixHint,
    })),
    failedGates: failedGates.map((r) => r.gate),
  },
  fixHint: allPassed
    ? `All ${gatesToRun.length} gates passed for slice ${sliceId} (${severity}).`
    : `Failed gates: ${failedGates.map((r) => r.gate).join(', ')}. Fix the issues and re-run. Details: ${failedGates.map((r) => r.fixHint).join(' | ')}`,
  owner: 'orchestrator (Agent Zero)',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(
    `${allPassed ? 'PASS' : 'FAIL'} slice-advancement gate — slice ${sliceId} (${severity}, ${gatesToRun.length} gates)`,
  );
  if (!allPassed) {
    for (const r of results) {
      const status = r.pass ? 'PASS' : 'FAIL';
      console.log(`  [${status}] ${r.gate}`);
      if (!r.pass && r.fixHint) {
        console.log(`         ${r.fixHint}`);
      }
    }
  }
}

process.exitCode = allPassed ? 0 : 1;
