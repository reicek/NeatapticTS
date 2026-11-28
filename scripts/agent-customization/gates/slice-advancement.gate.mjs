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
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { extractStatus, parseArgs } from '../customization-utils.mjs';
import {
  GATE_MAX_BUFFER_BYTES,
  GATE_TIMEOUT_OVERHEAD_MS,
  isTimeoutError,
  resolveSubGateTimeoutMs,
  SUB_GATE_TIMEOUT_BUDGETS,
} from '../gate-timeout-budgets.mjs';
import { summarizeSubGateResults } from './slice-advancement-aggregation.mjs';

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

const SESSION_OVERRIDE_PATH = path.join(
  REPO_ROOT,
  'data',
  'mcp-session-override.json',
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
const planArg = (() => {
  const arg = rawArgv.find((a) => a.startsWith('--plan='));
  return arg ? arg.slice('--plan='.length).trim() : '';
})();

if (!options.json && options.help !== undefined) {
  console.log(`slice-advancement gate — master consolidated gate

Usage:
  node slice-advancement.gate.mjs --json --slice-id=<id> --changed-files=f1,f2

Options:
  --json               Output JSON gate contract
  --slice-id=<id>      Slice identifier (required)
  --changed-files=f1,f2  Comma-separated changed file paths
  --plan=<path>        Forward the plan to plan-command-lint and
                       specialist-review (auto-resolved from
                       data/mcp-session-override.json or a unique [WIP]
                       plan when omitted)
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
  process.exit(1);
}

/**
 * Run a sub-gate and return its parsed JSON result with graceful degradation.
 *
 * Distinguishes two failure modes:
 * - **Content failure** (`gate_error: false`, `pass: false`): the sub-gate
 *   ran successfully but found a real issue (test failure, lint error). These
 *   count toward the consolidated `pass` boolean.
 * - **Tooling error** (`gate_error: true`): the sub-gate crashed or produced
 *   unparseable output. These are logged as warnings and do NOT cause the
 *   consolidated gate to hard-fail.
 *
 * A **timeout** is reported separately (`timed_out: true`). Unlike a generic
 * tooling error, a timeout means the sub-gate was KILLED before it produced
 * a verdict, so the slice was never validated. Timeouts therefore DO fail the
 * consolidated gate — a silent skip would report a green slice that no
 * validation actually covered.
 *
 * @param {string} gateName - gate script filename (without .gate.mjs)
 * @param {string[]} extraArgs - additional CLI args
 * @returns {{ gate: string, pass: boolean, evidence: object, fixHint: string|null, gate_error: boolean, timed_out: boolean, timeoutMs: number, raw: string }}
 */
function runSubGate(gateName, extraArgs = []) {
  const gatePath = path.join(GATES_DIR, `${gateName}.gate.mjs`);
  const args = ['--json', ...extraArgs];
  const timeoutMs = resolveSubGateTimeoutMs(gateName);
  try {
    const stdout = execFileSync('node', [gatePath, ...args], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: timeoutMs,
      maxBuffer: GATE_MAX_BUFFER_BYTES,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    const parsed = JSON.parse(stdout);
    return {
      gate: gateName,
      pass: parsed.pass === true,
      evidence: parsed.evidence || {},
      fixHint: parsed.fixHint || null,
      gate_error: false,
      timed_out: false,
      timeoutMs,
      raw: stdout.trim(),
    };
  } catch (err) {
    // execFileSync throws on non-zero exit. Sub-gates set exitCode=1 when
    // pass:false (content failure), which is NOT a tooling error. Try to
    // parse the captured stdout first.
    const stdout = typeof err.stdout === 'string' ? err.stdout : '';
    if (stdout) {
      try {
        const parsed = JSON.parse(stdout);
        return {
          gate: gateName,
          pass: parsed.pass === true,
          evidence: parsed.evidence || {},
          fixHint: parsed.fixHint || null,
          gate_error: false,
          timed_out: false,
          timeoutMs,
          raw: stdout.trim(),
        };
      } catch {
        // stdout exists but is not valid JSON — genuine tooling error.
      }
    }
    // Genuine tooling failure: exception, crash, timeout, or missing script.
    const timedOut = isTimeoutError(err);
    return {
      gate: gateName,
      pass: false,
      evidence: {
        error: err.message,
        ...(timedOut ? { timeoutMs, timedOut: true } : {}),
      },
      fixHint: timedOut
        ? `Sub-gate '${gateName}' exceeded its ${timeoutMs} ms budget and was killed before it produced a verdict. Re-run it directly (node scripts/agent-customization/gates/${gateName}.gate.mjs --json) to see where it stalls, or narrow the slice.`
        : `Run 'node scripts/agent-customization/gates/${gateName}.gate.mjs --json' to see details.`,
      gate_error: true,
      timed_out: timedOut,
      timeoutMs,
      raw: stdout || err.message,
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
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 30000,
        maxBuffer: GATE_MAX_BUFFER_BYTES,
      },
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

/**
 * Resolve the plan path forwarded to the plan-command-lint and
 * specialist-review sub-gates.
 *
 * Without a forwarded plan, plan-command-lint silently falls back to its own
 * hard-coded DEFAULT_PLAN (plans/orchestration-fixes.plans.md), linting a
 * stale plan instead of the active one. The same hazard applies to
 * specialist-review: it must validate the ACTIVE plan's specialist-review
 * evidence, not a completed plan that happens to carry the legacy
 * VALIDATION_EVIDENCE token. Precedence:
 *   1. Explicit --plan CLI argument.
 *   2. Active plan recorded in data/mcp-session-override.json (plan_path).
 *   3. A unique top-level [WIP] plan in plans/ (mirrors the sibling
 *      plan-slice-quality / step-packet scan convention).
 * Returns null when nothing resolves, in which case each sub-gate keeps
 * its documented default rather than guessing.
 *
 * @returns {Promise<string|null>} Repo-relative plan path, or null.
 */
async function resolveActivePlanPath() {
  if (planArg) {
    return planArg;
  }

  try {
    const override = JSON.parse(await readFile(SESSION_OVERRIDE_PATH, 'utf8'));
    if (typeof override?.plan_path === 'string' && override.plan_path.trim()) {
      return override.plan_path.trim();
    }
  } catch {
    // Missing, malformed, or unreadable override file — fall through.
  }

  try {
    const entries = await readdir(path.join(REPO_ROOT, 'plans'));
    const wipPlans = [];
    for (const name of entries) {
      if (!name.endsWith('.plans.md')) continue;
      const planPath = `plans/${name}`;
      const text = await readFile(path.join(REPO_ROOT, planPath), 'utf8');
      if (extractStatus(text) === 'WIP') {
        wipPlans.push(planPath);
      }
    }
    if (wipPlans.length === 1) {
      return wipPlans[0];
    }
  } catch {
    // plans/ unreadable — fall through to null.
  }

  return null;
}

// Parse changed files
const changedFiles = changedFilesRaw
  .split(',')
  .map((f) => f.trim())
  .filter(Boolean);

// Step 1: Classify severity
const { severity, specialistCount } = classifySeverity(changedFiles);

// Step 1b: Resolve the plan forwarded to plan-command-lint and specialist-review.
const activePlanPath = await resolveActivePlanPath();

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
    extraArgs.push('--exemptions=coverage/coverage-exemptions.json');
  }
  if (gateName === 'plan-command-lint' && activePlanPath) {
    extraArgs.push(`--plan=${activePlanPath}`);
  }
  // Forward the ACTIVE plan to specialist-review as well. Without it, the
  // sub-gate falls back to its own DEFAULT plan (a COMPLETED plan under
  // plans/completed/), so the consolidated gate would validate a stale
  // artifact and report a green specialist-review that never touched the
  // plan being advanced.
  if (gateName === 'specialist-review' && activePlanPath) {
    extraArgs.push(`--plan=${activePlanPath}`);
  }
  return runSubGate(gateName, extraArgs);
});

// Aggregate with graceful degradation.
// gate_error sub-gates (tooling failures) are excluded from the pass
// calculation; only content failures (pass: false, gate_error: false) cause
// the consolidated gate to fail.
//
// A TIMED-OUT sub-gate is the one exception: it was killed before it could
// produce a verdict, so the slice it covers was never validated. Reporting
// such a slice as green would be a false green, so timeouts fail the
// consolidated gate.
const {
  allPassed,
  failedGates,
  erroredGates,
  timedOutGates,
  sub_gates,
} = summarizeSubGateResults(results);

const result = {
  pass: allPassed,
  sub_gates,
  evidence: {
    gate: 'slice-advancement',
    tier: 1,
    sliceId,
    severity,
    specialistCount,
    planCommandLintPlan: activePlanPath,
    specialistReviewPlan: activePlanPath,
    gatesRun: gatesToRun,
    gateCount: gatesToRun.length,
    results: sub_gates,
    failedGates,
    erroredGates,
    timedOutGates,
    timeoutBudgetMs: SUB_GATE_TIMEOUT_BUDGETS,
    timeoutOverheadMs: GATE_TIMEOUT_OVERHEAD_MS,
  },
  fixHint: allPassed
    ? erroredGates.length > 0
      ? `All content gates passed for slice ${sliceId} (${severity}). ${erroredGates.length} gate(s) errored (tooling failure) and were skipped: ${erroredGates.join(', ')}.`
      : `All ${gatesToRun.length} gates passed for slice ${sliceId} (${severity}).`
    : timedOutGates.length > 0
      ? `Timed out: ${timedOutGates.join(', ')}. The sub-gate(s) were killed before producing a verdict, so the slice was NOT validated. Details: ${results.filter((r) => r.timed_out).map((r) => r.fixHint).join(' | ')}`
      : `Failed gates: ${failedGates.join(', ')}. Fix the issues and re-run. Details: ${results.filter((r) => !r.gate_error && !r.pass).map((r) => r.fixHint).join(' | ')}`,
  owner: 'orchestrator (Agent Zero)',
};

if (options.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(
    `${allPassed ? 'PASS' : 'FAIL'} slice-advancement gate — slice ${sliceId} (${severity}, ${gatesToRun.length} gates)`,
  );
  for (const r of results) {
    if (r.timed_out) {
      console.log(`  [TIMEOUT] ${r.gate} — killed after ${r.timeoutMs} ms`);
      if (r.fixHint) console.log(`          ${r.fixHint}`);
    } else if (r.gate_error) {
      console.log(`  [ERROR] ${r.gate} — tooling failure, skipped`);
      if (r.fixHint) console.log(`          ${r.fixHint}`);
    } else if (!r.pass) {
      console.log(`  [FAIL] ${r.gate}`);
      if (r.fixHint) console.log(`         ${r.fixHint}`);
    }
  }
}

process.exitCode = allPassed ? 0 : 1;
